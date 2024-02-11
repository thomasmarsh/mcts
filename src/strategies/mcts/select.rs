use std::sync::atomic::Ordering::Relaxed;

use super::*;
use crate::game::Action;

pub struct SelectContext<'a, A: Action> {
    pub q_init: node::UnvisitedValueEstimate,
    pub current_id: index::Id,
    pub player: usize,
    pub index: &'a TreeIndex<A>,
    pub rng: &'a mut FastRng,
}

const PRIMES: [usize; 16] = [
    14323, 18713, 19463, 30553, 33469, 45343, 50221, 51991, 53201, 56923, 64891, 72763, 74471,
    81647, 92581, 94693,
];

// This function is adapted from from minimax-rs.
#[inline]
fn random_best_index<T, U, F>(
    rng: &mut FastRng,
    set: &[Option<T>],
    init: U,
    default: U,
    mut score_fn: F,
) -> usize
where
    F: Fn(&T) -> U,
    U: PartialOrd + Copy,
{
    // To make the choice more uniformly random among the best moves, start
    // at a random offset and stride by a random amount. The stride must be
    // coprime with n, so pick from a set of 5 digit primes.

    let n = set.len();

    // Combine both random numbers into a single rng call.
    let r = rng.gen_range(0..n * PRIMES.len());
    let mut i = r / PRIMES.len();
    let stride = PRIMES[r % PRIMES.len()];

    let mut best_score = init;
    let mut best_index = i;
    for _ in 0..n {
        let score = set[i].as_ref().map_or(default, &mut score_fn);
        if score > best_score {
            best_score = score;
            best_index = i;
        }
        i = (i + stride) % n;
    }

    best_index
}

pub trait SelectStrategy {
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize;
}

#[derive(Default)]
pub struct RobustChild;

impl SelectStrategy for RobustChild {
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        let current = ctx.index.get(ctx.current_id);
        // Score by (num_visits, expected_score)
        random_best_index(
            ctx.rng,
            current.children(),
            (-1i64, f64::NEG_INFINITY),
            (0, 0.),
            |&child_id| {
                let child = ctx.index.get(child_id);
                (
                    child.stats.num_visits as i64,
                    child.stats.expected_score(ctx.player),
                )
            },
        )
    }
}

#[derive(Default)]
pub struct MaxAvgScore;

impl SelectStrategy for MaxAvgScore {
    #[inline]
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        let current = ctx.index.get(ctx.current_id);
        random_best_index(
            ctx.rng,
            current.children(),
            f64::NEG_INFINITY,
            current
                .stats
                .value_estimate_unvisited(ctx.player, ctx.q_init),
            |&child_id| ctx.index.get(child_id).stats.expected_score(ctx.player),
        )
    }
}

pub struct Ucb1 {
    pub exploration_constant: f64,
}

impl Default for Ucb1 {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl SelectStrategy for Ucb1 {
    #[inline]
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        let current = ctx.index.get(ctx.current_id);
        let parent_log = ((current.stats.num_visits as f64).max(1.)).ln();
        let unvisited_value = current
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        random_best_index(
            ctx.rng,
            current.children(),
            f64::NEG_INFINITY,
            unvisited_value + 2. * self.exploration_constant * parent_log.sqrt(),
            |&child_id| {
                let child = ctx.index.get(child_id);
                let exploit = child.stats.exploitation_score(ctx.player);
                let num_visits =
                    child.stats.num_visits + child.stats.num_visits_virtual.load(Relaxed);
                let explore = (2. * parent_log / num_visits as f64).sqrt();
                exploit + self.exploration_constant * explore
            },
        )
    }
}

pub struct Ucb1Tuned {
    pub exploration_constant: f64,
}

impl Default for Ucb1Tuned {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

const VARIANCE_UPPER_BOUND: f64 = 1.;

#[inline]
fn ucb1_tuned(
    exploration_constant: f64,
    exploit: f64,
    sample_variance: f64,
    visits_fraction: f64,
) -> f64 {
    exploit
        + (visits_fraction * VARIANCE_UPPER_BOUND.min(sample_variance)
            + exploration_constant * visits_fraction.sqrt())
}

impl SelectStrategy for Ucb1Tuned {
    #[inline]
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        let current = ctx.index.get(ctx.current_id);
        let parent_log = ((current.stats.num_visits as f64).max(1.)).ln();
        let unvisited_value = current
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        random_best_index(
            ctx.rng,
            current.children(),
            f64::NEG_INFINITY,
            ucb1_tuned(
                self.exploration_constant,
                unvisited_value,
                VARIANCE_UPPER_BOUND,
                parent_log,
            ),
            |&child_id| {
                let child = ctx.index.get(child_id);
                let exploit = child.stats.exploitation_score(ctx.player);
                let num_visits =
                    child.stats.num_visits + child.stats.num_visits_virtual.load(Relaxed);
                let sample_variance = 0f64.max(
                    child.stats.sum_squared_scores[ctx.player] / num_visits as f64
                        - exploit * exploit,
                );
                let visits_fraction = parent_log / num_visits as f64;

                ucb1_tuned(
                    self.exploration_constant,
                    exploit,
                    sample_variance,
                    visits_fraction,
                )
            },
        )
    }
}

pub struct McGrave {
    // Called ref in the RAVE paper.
    pub threshold: u32,
    pub bias: f64,
    // TODO: thread local
    pub current_ref_id: Option<index::Id>,
}

impl Default for McGrave {
    fn default() -> Self {
        Self {
            threshold: 100,
            bias: 10.0e-6,
            current_ref_id: None,
        }
    }
}

impl SelectStrategy for McGrave {
    #[inline]
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        let current = ctx.index.get(ctx.current_id);
        let unvisited_value = current
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        if self.current_ref_id.is_none()
            || current.stats.num_visits > self.threshold
            || current.is_root()
        {
            self.current_ref_id = Some(ctx.current_id);
        }

        let current_ref = ctx.index.get(self.current_ref_id.unwrap());

        let grave_value = |beta: f64, mean_score: f64, mean_amaf: f64| {
            (1. - beta) * mean_score + beta * mean_amaf
        };

        random_best_index(
            ctx.rng,
            current.children(),
            f64::NEG_INFINITY,
            grave_value(0., unvisited_value, 0.),
            |&child_id| {
                let child = ctx.index.get(child_id);
                let mean_score = child.stats.exploitation_score(ctx.player);
                let (mean_amaf, beta) = match current_ref
                    .stats
                    .grave_stats
                    .get(&(current.actions()[child.action_idx]))
                {
                    None => (0., 0.),
                    Some(grave_stats) => {
                        let grave_score = grave_stats.score;
                        let grave_visits = grave_stats.num_visits as f64;
                        let child_visits = (child.stats.num_visits
                            + child.stats.num_visits_virtual.load(Relaxed))
                            as f64;
                        let mean_amaf = grave_score / grave_visits;
                        let beta = grave_visits
                            / (grave_visits
                                + child_visits
                                + self.bias * grave_visits * child_visits);

                        (mean_amaf, beta)
                    }
                };

                grave_value(beta, mean_score, mean_amaf)
            },
        )
    }
}

pub struct McBrave {
    pub bias: f64,
}

impl Default for McBrave {
    fn default() -> Self {
        Self { bias: 10.0e-6 }
    }
}

impl SelectStrategy for McBrave {
    #[inline]
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        let current = ctx.index.get(ctx.current_id);
        let unvisited_value = current
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        let grave_value = |beta: f64, mean_score: f64, mean_amaf: f64| {
            (1. - beta) * mean_score + beta * mean_amaf
        };

        random_best_index(
            ctx.rng,
            current.children(),
            f64::NEG_INFINITY,
            grave_value(0., unvisited_value, 0.),
            |&child_id| {
                let child = ctx.index.get(child_id);
                let mean_score = child.stats.exploitation_score(ctx.player);

                let mut accum_visits = 0;
                let mut accum_score = 0.0;

                let mut rave_node_id = ctx.current_id;
                loop {
                    let rave_node = ctx.index.get(rave_node_id);

                    if let Some(grave_stats) =
                        rave_node.stats.grave_stats.get(&child.action(ctx.index))
                    {
                        accum_score += grave_stats.score;
                        accum_visits += grave_stats.num_visits;
                    }

                    if rave_node.is_root() {
                        break;
                    }
                    rave_node_id = rave_node.parent_id;
                }

                let mean_amaf: f64;
                let beta: f64;
                if accum_visits == 0 {
                    mean_amaf = 0.;
                    beta = 0.;
                } else {
                    let child_visits = (child.stats.num_visits
                        + child.stats.num_visits_virtual.load(Relaxed))
                        as f64;

                    mean_amaf = accum_score / accum_visits as f64;
                    beta = accum_visits as f64
                        / (accum_visits as f64
                            + child_visits
                            + self.bias * accum_visits as f64 * child_visits);
                }
                grave_value(beta, mean_score, mean_amaf)
            },
        )
    }
}

// This one was found in some implementations of RAVE. It seems strong, but I
// can't find references to it in the literature.
pub struct ScalarAmaf {
    pub exploration_constant: f64,
    pub bias: f64,
}

impl Default for ScalarAmaf {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
            bias: 700.0,
        }
    }
}

impl SelectStrategy for ScalarAmaf {
    #[inline]
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        let current = ctx.index.get(ctx.current_id);
        let parent_log = ((current.stats.num_visits as f64).max(1.)).ln();
        let unvisited_value = current
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        random_best_index(
            ctx.rng,
            current.children(),
            f64::NEG_INFINITY,
            unvisited_value + self.exploration_constant * parent_log.sqrt(),
            |&child_id| {
                let child = ctx.index.get(child_id);
                let exploit = child.stats.exploitation_score(ctx.player);
                let num_visits =
                    child.stats.num_visits + child.stats.num_visits_virtual.load(Relaxed);
                let explore = (parent_log / num_visits as f64).sqrt();
                let uct_value = exploit + self.exploration_constant * explore;

                let amaf_value = if num_visits > 0 {
                    child.stats.scalar_amaf.score / child.stats.num_visits as f64
                } else {
                    0.
                };

                let beta = self.bias / (self.bias + num_visits as f64);

                (1. - beta) * uct_value + beta * amaf_value
            },
        )
    }
}

pub struct Ucb1Grave {
    // Called ref in the RAVE paper.
    pub threshold: u32,
    pub bias: f64,
    pub exploration_constant: f64,
    // TODO: thread local
    pub current_ref_id: Option<index::Id>,
}

impl Default for Ucb1Grave {
    fn default() -> Self {
        Self {
            threshold: 100,
            bias: 10.0e-6,
            exploration_constant: 2f64.sqrt(),
            current_ref_id: None,
        }
    }
}

impl SelectStrategy for Ucb1Grave {
    #[inline]
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        let current = ctx.index.get(ctx.current_id);
        let parent_log = ((current.stats.num_visits as f64).max(1.)).ln();
        let unvisited_value = current
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        if self.current_ref_id.is_none()
            || current.stats.num_visits > self.threshold
            || current.is_root()
        {
            self.current_ref_id = Some(ctx.current_id);
        }

        let current_ref = ctx.index.get(self.current_ref_id.unwrap());

        let ucb1_grave_value = |beta: f64, mean_score: f64, mean_amaf: f64, explore| {
            let grave_value = (1. - beta) * mean_score + beta * mean_amaf;
            grave_value + self.exploration_constant * explore
        };

        random_best_index(
            ctx.rng,
            current.children(),
            f64::NEG_INFINITY,
            ucb1_grave_value(0., unvisited_value, 0., parent_log.sqrt()),
            |&child_id| {
                let child = ctx.index.get(child_id);
                let mean_score = child.stats.exploitation_score(ctx.player);
                let child_visits =
                    (child.stats.num_visits + child.stats.num_visits_virtual.load(Relaxed)) as f64;
                let (mean_amaf, beta) = match current_ref
                    .stats
                    .grave_stats
                    .get(&(current.actions()[child.action_idx]))
                {
                    None => (0., 0.),
                    Some(grave_stats) => {
                        let grave_score = grave_stats.score;
                        let grave_visits = grave_stats.num_visits as f64;
                        let mean_amaf = grave_score / grave_visits;
                        let beta = grave_visits
                            / (grave_visits
                                + child_visits
                                + self.bias * grave_visits * child_visits);

                        (mean_amaf, beta)
                    }
                };

                let explore = (parent_log / child_visits).sqrt();

                ucb1_grave_value(beta, mean_score, mean_amaf, explore)
            },
        )
    }
}

// A greedy epsilon wrapper around any other strategy.
// Q: Traditional epsilon greedy == `EpsilonGreedy<Max>`?
pub struct EpsilonGreedy<S: SelectStrategy> {
    epsilon: f64,
    select: S,
}

impl<S: SelectStrategy> SelectStrategy for EpsilonGreedy<S> {
    fn best_child<A: Action>(&mut self, ctx: &mut SelectContext<'_, A>) -> usize {
        if ctx.rng.gen::<f64>() < self.epsilon {
            ctx.rng
                .gen_range(0..ctx.index.get(ctx.current_id).children().len())
        } else {
            self.select.best_child(ctx)
        }
    }
}
