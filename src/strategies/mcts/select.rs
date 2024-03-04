use super::index::Id;
use super::node::{self, Edge, NodeStats};
use super::table::TranspositionTable;
use super::*;
use crate::game::Game;
use crate::strategies::Search;
use crate::util::random_best;

use rand::rngs::SmallRng;
use rand::Rng;
use rustc_hash::FxHashMap;

pub struct SelectContext<'a, G: Game> {
    pub q_init: node::QInit,
    pub stack: Vec<Id>,
    pub state: &'a G::S,
    pub root_stats: &'a NodeStats,
    pub player: usize,
    pub player_to_move: usize,
    pub index: &'a TreeIndex<G::A>,
    pub table: &'a TranspositionTable,
    pub grave: &'a FxHashMap<u64, Vec<FxHashMap<G::A, node::ActionStats>>>,
    pub use_transpositions: bool,
}

////////////////////////////////////////////////////////////////////////////////

impl<'a, G: Game> SelectContext<'a, G> {
    fn parent_id(&self) -> Id {
        debug_assert!(!self.index.get(self.current_id()).is_root());
        self.stack.get(self.stack.len() - 2).cloned().unwrap()
    }

    fn current_id(&self) -> Id {
        debug_assert!(!self.stack.is_empty());
        *self.stack.last().unwrap()
    }

    #[inline]
    fn current_stats(&self) -> &NodeStats {
        if self.index.get(self.current_id()).is_root() {
            self.root_stats
        } else {
            let action_idx = self.index.get(self.current_id()).action_idx;
            debug_assert_ne!(self.parent_id(), self.current_id());
            let parent_id = self.parent_id();
            let parent = self.index.get(parent_id);
            debug_assert!(parent.edges().len() > action_idx);
            &self.index.get(parent_id).edges()[action_idx].stats
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait SelectStrategy<G: Game>: Sized + Clone + Sync + Send + Default {
    type Score: PartialOrd + Copy;
    type Aux: Copy;

    /// If the strategy wants to lift any calculations out of the inner select
    /// loop, then they can provide this here.
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> Self::Aux;

    /// Default implementation should be sufficient for all cases.
    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.current_id());
        random_best_index(current.edges(), self, ctx, rng)
    }

    /// Given a child index, calculate a score.
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        edge: &Edge<G::A>,
        aux: Self::Aux,
    ) -> Self::Score;

    /// Provide a score for any value that is not yet visited.
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, aux: Self::Aux) -> Self::Score;

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(0)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct EpsilonGreedy<G: Game, S: SelectStrategy<G>> {
    pub epsilon: f64,
    pub inner: S,
    pub marker: std::marker::PhantomData<G>,
}

impl<G, S> EpsilonGreedy<G, S>
where
    G: Game,
    S: SelectStrategy<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn inner(mut self, inner: S) -> Self {
        self.inner = inner;
        self
    }
}

impl<G, S> Default for EpsilonGreedy<G, S>
where
    G: Game,
    S: SelectStrategy<G> + Default,
{
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            inner: S::default(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<G, S> SelectStrategy<G> for EpsilonGreedy<G, S>
where
    G: Game,
    S: SelectStrategy<G>,
{
    type Score = S::Score;
    type Aux = S::Aux;

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        if rng.gen::<f64>() < self.epsilon {
            let current = ctx.index.get(ctx.current_id());
            let n = current.edges().len();
            rng.gen_range(0..n)
        } else {
            self.inner.best_child(ctx, rng)
        }
    }

    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> Self::Aux {
        self.inner.setup(ctx)
    }

    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        edge: &Edge<G::A>,
        aux: Self::Aux,
    ) -> Self::Score {
        self.inner.score_child(ctx, child_id, edge, aux)
    }

    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, aux: Self::Aux) -> Self::Score {
        self.inner.unvisited_value(ctx, aux)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.inner.backprop_flags()
    }
}

////////////////////////////////////////////////////////////////////////////////

const PRIMES: [usize; 16] = [
    14323, 18713, 19463, 30553, 33469, 45343, 50221, 51991, 53201, 56923, 64891, 72763, 74471,
    81647, 92581, 94693,
];

// This function is adapted from from minimax-rs.
#[inline]
fn random_best_index<S, G>(
    set: &[Edge<G::A>],
    strategy: &mut S,
    ctx: &SelectContext<'_, G>,
    rng: &mut SmallRng,
) -> usize
where
    S: SelectStrategy<G>,
    G: Game,
{
    // To make the choice more uniformly random among the best moves, start
    // at a random offset and stride by a random amount. The stride must be
    // coprime with n, so pick from a set of 5 digit primes.

    // Combine both random numbers into a single rng call.
    let n = set.len();
    let r = rng.gen_range(0..n * PRIMES.len());
    let mut i = r / PRIMES.len();
    let stride = PRIMES[r % PRIMES.len()];

    let aux = strategy.setup(ctx);
    let unvisited_value = strategy.unvisited_value(ctx, aux);

    let child_value = |i: usize| {
        if let Some(child_id) = &set[i].node_id {
            strategy.score_child(ctx, *child_id, &set[i], aux)
        } else {
            unvisited_value
        }
    };

    let mut best_score = child_value(i);

    let mut best_index = i;
    for _ in 1..n {
        i = (i + stride) % n;

        let score = child_value(i);

        if score > best_score {
            best_score = score;
            best_index = i;
        }
    }

    best_index
}

////////////////////////////////////////////////////////////////////////////////

/// Select the most visited root child.
#[derive(Default, Clone)]
pub struct RobustChild;

impl<G: Game> SelectStrategy<G> for RobustChild {
    type Score = (i64, f64);
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        _: Self::Aux,
    ) -> (i64, f64) {
        (
            edge.stats.num_visits as i64,
            edge.stats.expected_score(ctx.player),
        )
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> (i64, f64) {
        let q = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        (0, q)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Select the root child with the highest reward.
#[derive(Default, Clone)]
pub struct MaxAvgScore;

impl<G: Game> SelectStrategy<G> for MaxAvgScore {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        _: Self::Aux,
    ) -> f64 {
        edge.stats.expected_score(ctx.player)
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// The secure child is the child that maximizes a lower confidence bound.
#[derive(Clone)]
pub struct SecureChild {
    pub a: f64,
}

impl Default for SecureChild {
    fn default() -> Self {
        // This quantity comes from the Chaslot, Winands progressive strategies paper
        Self { a: 4. }
    }
}

impl<G: Game> SelectStrategy<G> for SecureChild {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        _: Self::Aux,
    ) -> f64 {
        let q = edge.stats.expected_score(ctx.player);
        let n = edge.stats.total_visits();

        q + self.a / (n as f64).sqrt()
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
pub struct ThompsonSampling;

impl<G: Game> SelectStrategy<G> for ThompsonSampling {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline]
    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.current_id());
        // This is just a weighted sampling. Need to implement some stuff for thompson sampling.
        let weights = current
            .edges()
            .iter()
            .map(|edge| {
                edge.node_id
                    .map(|child_id| self.score_child(ctx, child_id, edge, ()))
                    .unwrap_or(self.unvisited_value(ctx, ())) as f32
            })
            .collect::<Vec<_>>();

        use weighted_rand::builder::*;
        let builder = WalkerTableBuilder::new(&weights);
        let wa_table = builder.build();
        wa_table.next_rng(rng)
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        _: Self::Aux,
    ) -> f64 {
        let q = edge.stats.expected_score(ctx.player);
        let n = edge.stats.total_visits();

        q / (n as f64).sqrt()
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Upper Confidence Bounds (UCB1)
#[derive(Clone)]
pub struct Ucb1 {
    pub exploration_constant: f64,
}

impl Ucb1 {
    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }
}

impl Default for Ucb1 {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl<G: Game> SelectStrategy<G> for Ucb1 {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        let stats = ctx.current_stats();
        ((stats.num_visits as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        parent_log: f64,
    ) -> f64 {
        let exploit = edge.stats.exploitation_score(ctx.player);
        let num_visits = edge.stats.total_visits();
        let explore = (parent_log / num_visits as f64).sqrt();
        exploit + self.exploration_constant * explore
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, parent_log: f64) -> f64 {
        let unvisited_value = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        unvisited_value + self.exploration_constant * parent_log.sqrt()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
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

impl Ucb1Tuned {
    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }
}

const VARIANCE_UPPER_BOUND: f64 = 1.;

#[inline(always)]
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

impl<G: Game> SelectStrategy<G> for Ucb1Tuned {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        ((ctx.current_stats().num_visits as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        parent_log: f64,
    ) -> f64 {
        let exploit = edge.stats.exploitation_score(ctx.player);
        let num_visits = edge.stats.total_visits();
        let sample_variance = 0f64.max(
            edge.stats.player[ctx.player].sum_squared_score / num_visits as f64 - exploit * exploit,
        );
        let visits_fraction = parent_log / num_visits as f64;

        ucb1_tuned(
            self.exploration_constant,
            exploit,
            sample_variance,
            visits_fraction,
        )
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, parent_log: f64) -> Self::Score {
        let unvisited_value = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        ucb1_tuned(
            self.exploration_constant,
            unvisited_value,
            VARIANCE_UPPER_BOUND,
            parent_log,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////

// Ameneyro, F.V., Galvan, E., Morales, A.F.K., 2020. Playing Carcassonne with
// Monte Carlo Tree Search.
//
// Cazenave, T., 2015. Generalized Rapid Action Value Estimation, in:
// Proceedings of the Twenty-Fourth International Joint Conference on Artificial
// Intelligence. Presented at the International Joint Conference on Artificial
// Intelligence, Buenos Aires, Argentina.
//
// Gelly, S., Silver, D., 2011. Monte-Carlo tree search and rapid action value
// estimation in computer Go. Artificial Intelligence 175, 1856–1875. https://
// doi.org/10.1016/j.artint.2011.03.007
//
// Rimmel, A., Teytaud, F., Teytaud, O., 2011. Biasing Monte-Carlo
// Simulations through RAVE Values, in: Van Den Herik, H.J., Iida, H.,
// Plaat, A. (Eds.), Computers and Games, Lecture Notes in Computer Science.
// Springer Berlin Heidelberg, Berlin, Heidelberg, pp. 59–68. https://
// doi.org/10.1007/978-3-642-17928-0_6
//
// Sironi, C.F., Winands, M.H.M., 2016. Comparison of rapid action value
// estimation variants for general game playing, in: 2016 IEEE Conference
// on Computational Intelligence and Games (CIG). Presented at the 2016 IEEE
// Conference on Computational Intelligence and Games (CIG), IEEE, Santorini,
// Greece, pp. 1–8. https://doi.org/10.1109/CIG.2016.7860429
//
// Sironi, C.F., Winands, M.H.M., 2018. On-Line Parameter Tuning for Monte-Carlo
// Tree Search in General Game Playing, in: Cazenave, T., Winands, M.H.M.,
// Saffidine, A. (Eds.), Computer Games, Communications in Computer and
// Information Science. Springer International Publishing, Cham, pp. 75–95.
// https://doi.org/10.1007/978-3-319-75931-9_6

#[derive(Clone, Copy)]
pub enum RaveSchedule {
    // HandSelected comes from CadiaPlayer
    // MinMSE and CadiaPlayare are both described in Gelly, Silver 2011
    // k=1000 for go
    HandSelected { k: u32 },
    // TODO: default bias
    MinMSE { bias: f64 },
    // Traditional Rave. I have seen recommendations to start tuning with rave = 700
    Threshold { rave: u32 },
}

impl Default for RaveSchedule {
    fn default() -> Self {
        RaveSchedule::HandSelected { k: 1000 }
    }
}

impl RaveSchedule {
    fn beta(&self, n: u32, amaf_n: u32) -> f64 {
        let n = n as f64;
        let amaf_n = amaf_n as f64;
        match self {
            RaveSchedule::HandSelected { k } => {
                let k = *k as f64;
                (k / (3. * n + k)).sqrt()
            }
            RaveSchedule::MinMSE { bias } => amaf_n / (n + amaf_n + 4. * n * amaf_n * bias * bias),

            RaveSchedule::Threshold { rave } => 0f64.max(*rave as f64 - n) / *rave as f64,
        }
    }
}

#[derive(Clone, Copy)]
pub enum RaveUcb {
    None,
    Ucb1 { exploration_constant: f64 },
    Ucb1Tuned { exploration_constant: f64 },
}
impl Default for RaveUcb {
    fn default() -> Self {
        Self::Ucb1 {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl RaveUcb {
    fn score(&self, parent_log: f64, n: u32, sum_squared_score: f64, exploit: f64) -> f64 {
        match self {
            RaveUcb::None => 0.,
            RaveUcb::Ucb1 {
                exploration_constant,
            } => exploration_constant * (parent_log / n as f64).sqrt(),
            RaveUcb::Ucb1Tuned {
                exploration_constant,
            } => {
                let sample_variance = 0f64.max(sum_squared_score / n as f64 - exploit * exploit);
                let visits_fraction = parent_log / n as f64;
                ucb1_tuned(
                    *exploration_constant,
                    0., // RAVE provides the exploitation term.
                    sample_variance,
                    visits_fraction,
                )
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Rave {
    pub threshold: u32, // 0 == RAVE, inf = HRAVE, else GRAVE
    pub schedule: RaveSchedule,
    pub ucb: RaveUcb,
}

impl Default for Rave {
    fn default() -> Self {
        Self {
            threshold: 700,
            schedule: RaveSchedule::default(),
            ucb: RaveUcb::default(),
        }
    }
}

impl Rave {
    pub fn new(threshold: u32, schedule: RaveSchedule, ucb: RaveUcb) -> Self {
        Self {
            threshold,
            schedule,
            ucb,
        }
    }

    pub fn threshold(mut self, threshold: u32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn schedule(mut self, schedule: RaveSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn ucb(mut self, ucb: RaveUcb) -> Self {
        self.ucb = ucb;
        self
    }
}

struct ReversePairs<'a, T: 'a> {
    stack: &'a [T],
    index: usize,
}

impl<'a, T> ReversePairs<'a, T> {
    fn new(stack: &'a [T]) -> Self {
        Self {
            stack,
            index: stack.len(),
        }
    }
}

impl<'a, T> Iterator for ReversePairs<'a, T> {
    type Item = (&'a T, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= 2 {
            self.index -= 1;
            Some((&self.stack[self.index - 1], &self.stack[self.index]))
        } else {
            None
        }
    }
}

impl Rave {
    fn get_ref<G: Game>(&self, ctx: &SelectContext<'_, G>, node_id: Id) -> Id {
        let mut stack = ctx.stack.clone();
        stack.push(node_id);
        let rev_pairs = ReversePairs::new(&stack);

        // TODO: we can push this down during select descent rather than walking back up.
        for (parent_id, child_id) in rev_pairs {
            if ctx.index.get(*parent_id).edges()[ctx.index.get(*child_id).action_idx]
                .stats
                .total_visits()
                >= self.threshold
            {
                return *child_id;
            }
        }
        stack[0]
    }

    #[inline(always)]
    fn amaf_score(n: u32, q: f64) -> f64 {
        if n == 0 {
            0.
        } else {
            q / n as f64
        }
    }
}

impl<G: Game> SelectStrategy<G> for Rave {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        ((ctx.current_stats().num_visits as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        edge: &Edge<G::A>,
        parent_log: f64,
    ) -> f64 {
        let ref_id = self.get_ref(ctx, child_id);
        let hash = ctx.index.get(ref_id).hash;
        let grave_stats = ctx
            .grave
            .get(&hash)
            .and_then(|player| player[ctx.player].get(&edge.action).cloned())
            .unwrap_or_default();

        let amaf_n = grave_stats.num_visits;
        let amaf_q = grave_stats.score;

        let n = edge.stats.total_visits();
        let exploit = edge.stats.exploitation_score(ctx.player);
        let explore = self.ucb.score(
            parent_log,
            n,
            edge.stats.player[ctx.player].sum_squared_score,
            exploit,
        );

        let b = self.schedule.beta(n, amaf_n);
        let mean_score = edge.stats.expected_score(ctx.player);
        let amaf = Self::amaf_score(amaf_n, amaf_q);

        (1. - b) * mean_score + b * amaf + explore
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: f64) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GRAVE)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct Amaf {
    pub alpha: f64,
    pub exploration_constant: f64,
}

impl Amaf {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
            ..Default::default()
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn exploration_constant(mut self, exploration_constant: f64) -> Self {
        self.exploration_constant = exploration_constant;
        self
    }
}

impl Default for Amaf {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl<G: Game> SelectStrategy<G> for Amaf {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        ((ctx.current_stats().num_visits as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        parent_log: f64,
    ) -> f64 {
        let amaf_n = 1.max(edge.stats.player[ctx.player].amaf.num_visits) as f64;
        let amaf_q = edge.stats.player[ctx.player].amaf.score;
        let amaf = amaf_q / amaf_n;

        let exploit = edge.stats.exploitation_score(ctx.player);
        let num_visits = edge.stats.total_visits();
        let explore = (parent_log / num_visits as f64).sqrt();

        // alpha = 1 is standard AMAF
        // alpha = 0 is standard UCT
        let ucb1 = exploit + self.exploration_constant * explore;
        self.alpha * amaf + (1. - self.alpha) * ucb1
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: f64) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(AMAF)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Quasi Best-First comes from the Chaslot paper on Meta MCTS for opening book
/// generation. This is intended to be used differently than other strategies.
/// For opening book generation, we use the following settings for the higher
/// level MCTS config:
///
/// - expand_threshold: 0 (expand to terminal state during select)
/// - max_iterations: 1 (we only need one PV)
/// - simulate: n/a (ignored, due to max_iteration count)
/// - backprop: n/a (ignored, due to max_iteration count)
///
/// We add an epsilon-greedy parameter since this seems otherwise too greedy
/// a selection strategy and we don't see enough exploration.
///
///
/// > Algorithm 1 The “Quasi Best-First” (QBF) algorithm. λ is the number of machines
/// > available. K is a constant. g is a game, defined as a sequence of game states.
/// > The function “MoGoChoice” asks MOGO to choose a move.
///
/// ```ignore
/// QBF(K, λ)
/// while True do
///   for l = 1..λ, do
///     s =initial state; g = {s}.
///     while s is not a final state do
///       bestScore = K
///       bestMove = Null
///       for m in the set of possible moves in s do
///         score = percentage of won games by playing the move m in s
///         if score > bestScore then
///           bestScore = score
///           bestMove = m
///         end if
///       end for
///       if bestMove = Null then
///         bestMove = MoGoChoice(s) // lower level MCTS
///       end if
///       s = playMove(s, bestMove)
///       g = concat(g, s)
///     end while
///     Add g and the result of the game in the book.
///   end for
/// end while
/// ```
#[derive(Clone)]
pub struct QuasiBestFirst<G: Game, S: Strategy<G>> {
    pub book: book::OpeningBook<G::A>,
    pub search: TreeSearch<G, S>,
    pub epsilon: f64,
    pub k: Vec<f64>,
    pub key_init: Vec<G::A>,
}

impl<G, S> QuasiBestFirst<G, S>
where
    G: Game,
    S: Strategy<G>,
    TreeSearch<G, S>: Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn book(mut self, book: book::OpeningBook<G::A>) -> Self {
        self.book = book;
        self
    }

    pub fn search(mut self, search: TreeSearch<G, S>) -> Self {
        self.search = search;
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn k(mut self, k: Vec<f64>) -> Self {
        self.k = k;
        self
    }

    pub fn key_init(mut self, key_init: Vec<G::A>) -> Self {
        self.key_init = key_init;
        self
    }
}

impl<G, S> Default for QuasiBestFirst<G, S>
where
    G: Game,
    S: Strategy<G>,
    TreeSearch<G, S>: Default,
{
    fn default() -> Self {
        // The default value here is 0.5, but the Chaslot paper noted the difficulty
        // of elevating the black player in go when cold starting, prompting a lower
        // threshold for the initial player.
        // TODO: what about N-player games where N > 2
        let mut k = vec![0.5; G::num_players()];
        if k.len() == 2 {
            k[0] = 0.1;
        }

        Self {
            book: book::OpeningBook::new(G::num_players()),
            search: TreeSearch::default(),
            epsilon: 0.3,
            k,
            key_init: vec![],
        }
    }
}

impl<G, S> SelectStrategy<G> for QuasiBestFirst<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
{
    type Score = f64;
    type Aux = ();

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.current_id());
        let available = current.edges();

        // The stack now contains the action path to the terminal state.
        // TODO: factor this pair iteration out of here
        let mut key_init = vec![];
        for i in 0..ctx.stack.len() - 1 {
            let parent_id = ctx.stack[i];
            let child_id = ctx.stack[i + 1];
            key_init.push(
                ctx.index.get(parent_id).edges()[ctx.index.get(child_id).action_idx]
                    .action
                    .clone(),
            );
        }
        let k_score = self.k[ctx.player_to_move];

        let enumerated = available.iter().cloned().enumerate().collect::<Vec<_>>();
        let best = random_best(
            enumerated.as_slice(),
            rng,
            |(_, edge): &(usize, Edge<G::A>)| {
                let mut key = key_init.clone();
                key.push(edge.action.clone());

                let score = self
                    .book
                    .score(key.as_slice(), ctx.player_to_move)
                    .unwrap_or(f64::NEG_INFINITY);
                if score > k_score {
                    score
                } else {
                    // NOTE: we depend on random_best using this value internally
                    // as an equivalence for None types
                    f64::NEG_INFINITY
                }
            },
        );

        if let Some((best_index, _)) = best {
            *best_index
        } else {
            let action = self.search.choose_action(ctx.state);
            available
                .iter()
                .position(|p| p.action == action.clone())
                .unwrap()
        }
    }

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(&self, _: &SelectContext<'_, G>, _: Id, _: &Edge<G::A>, _: Self::Aux) -> f64 {
        0.
    }

    #[inline(always)]
    fn unvisited_value(&self, _: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        0.
    }
}
