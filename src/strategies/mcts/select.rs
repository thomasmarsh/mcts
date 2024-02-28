use super::index::Id;
use super::node::{self, NodeStats};
use super::table::TranspositionTable;
use super::*;
use crate::game::Game;
use crate::strategies::Search;
use crate::util::random_best;

use rand::rngs::SmallRng;
use rand::Rng;
use std::sync::atomic::Ordering::Relaxed;

pub struct SelectContext<'a, G: Game> {
    pub q_init: node::UnvisitedValueEstimate,
    pub current_id: Id,
    pub stack: Vec<Id>,
    pub state: &'a G::S,
    pub player: usize,
    pub player_to_move: usize,
    pub index: &'a TreeIndex<G::A>,
    pub table: &'a TranspositionTable,
    pub use_transpositions: bool,
}

////////////////////////////////////////////////////////////////////////////////

// Simple Upper Confidence bound for DAGS
//
// This is the "update descent" approach from Saffadine, Cazenave, UCD: Upper
// Confidence bound for rooted Directed acyclic graphs.

impl<'a, G: Game> SelectContext<'a, G> {
    #[inline]
    fn child_state(&self, child_id: Id) -> G::S {
        let child = self.index.get(child_id);
        let action = child.action(self.index);
        G::apply(self.state.clone(), &action)
    }

    #[inline]
    fn get_stats(&self, state: &G::S, node_id: Id) -> NodeStats<G::A> {
        if !self.use_transpositions {
            return self.index.get(node_id).stats.clone();
        }
        let k = G::zobrist_hash(state);
        match self.table.get_const(k) {
            Some(entries) => {
                let stats = entries
                    .iter()
                    .fold(NodeStats::new(G::num_players()), |stats, x| {
                        stats + self.index.get(*x).stats.clone()
                    });
                stats
            }
            None => self.index.get(node_id).stats.clone(),
        }
    }

    #[inline]
    fn current_stats(&self) -> NodeStats<G::A> {
        self.get_stats(self.state, self.current_id)
    }

    #[inline]
    fn child_stats(&self, child_id: Id) -> NodeStats<G::A> {
        self.get_stats(&self.child_state(child_id), child_id)
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait SelectStrategy<G: Game>: Sized + Clone + Sync + Send {
    type Score: PartialOrd + Copy;
    type Aux: Copy;

    /// If the strategy wants to lift any calculations out of the inner select
    /// loop, then they can provide this here.
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> Self::Aux;

    /// Default implementation should be sufficient for all cases.
    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.current_id);
        random_best_index(current.children(), self, ctx, rng)
    }

    /// Given a child index, calculate a score.
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, aux: Self::Aux) -> Self::Score;

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
            let current = ctx.index.get(ctx.current_id);
            let n = current.children().len();
            rng.gen_range(0..n)
        } else {
            self.inner.best_child(ctx, rng)
        }
    }

    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> Self::Aux {
        self.inner.setup(ctx)
    }

    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, aux: Self::Aux) -> Self::Score {
        println!("greedy: score_child");
        self.inner.score_child(ctx, child_id, aux)
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
    set: &[Option<Id>],
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
        if let Some(child_id) = &set[i] {
            strategy.score_child(ctx, *child_id, aux)
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
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, _: Self::Aux) -> (i64, f64) {
        let stats = ctx.child_stats(child_id);
        (stats.num_visits as i64, stats.expected_score(ctx.player))
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
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, _: Self::Aux) -> f64 {
        ctx.child_stats(child_id).expected_score(ctx.player)
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
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, _: Self::Aux) -> f64 {
        let stats = ctx.child_stats(child_id);
        let q = stats.expected_score(ctx.player);
        let n = stats.num_visits + stats.num_visits_virtual.load(Relaxed);

        q + self.a / (n as f64).sqrt()
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
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, parent_log: f64) -> f64 {
        let stats = ctx.child_stats(child_id);
        let exploit = stats.exploitation_score(ctx.player);
        let num_visits = stats.num_visits + stats.num_visits_virtual.load(Relaxed);
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
        let current = ctx.index.get(ctx.current_id);
        ((current.stats.num_visits as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, parent_log: f64) -> f64 {
        let stats = ctx.child_stats(child_id);
        let exploit = stats.exploitation_score(ctx.player);
        let num_visits = stats.num_visits + stats.num_visits_virtual.load(Relaxed);
        let sample_variance =
            0f64.max(stats.sum_squared_scores[ctx.player] / num_visits as f64 - exploit * exploit);
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

#[derive(Clone)]
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
            threshold: 80,
            bias: 10.0e-7,
            current_ref_id: None,
        }
    }
}

#[inline(always)]
fn grave_value(beta: f64, mean_score: f64, mean_amaf: f64) -> f64 {
    (1. - beta) * mean_score + beta * mean_amaf
}

impl<G: Game> SelectStrategy<G> for McGrave {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> Self::Aux {
        assert!(
            !ctx.use_transpositions,
            "GRAVE is incompatible with transposition table usage"
        );
        let current = ctx.index.get(ctx.current_id);

        if self.current_ref_id.is_none()
            || current.stats.num_visits > self.threshold
            || current.is_root()
        {
            self.current_ref_id = Some(ctx.current_id);
        }
    }

    #[inline(always)]
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, _: Self::Aux) -> f64 {
        let t = ctx.index.get(child_id);
        let tref = ctx.index.get(self.current_ref_id.unwrap());
        let p = (t.stats.num_visits + t.stats.num_visits_virtual.load(Relaxed)) as f64;
        let mean = t.stats.exploitation_score(ctx.player);
        let (amaf, beta) = match tref.stats.grave_stats.get(&t.action(ctx.index)) {
            None => (0., 0.),
            Some(stats) => {
                let wa = stats.num_visits as f64;
                let pa = stats.score;
                let beta = pa / (pa + p + self.bias * pa * p);
                let amaf = wa / pa;
                (amaf, beta)
            }
        };

        grave_value(beta, mean, amaf)
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.index
            .get(ctx.current_id)
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GRAVE)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct McBrave {
    pub bias: f64,
}

impl Default for McBrave {
    fn default() -> Self {
        Self { bias: 10.0e-6 }
    }
}

impl<G: Game> SelectStrategy<G> for McBrave {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> Self::Aux {
        assert!(
            !ctx.use_transpositions,
            "BRAVE is incompatible with transposition table usage"
        );
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> Self::Score {
        let current = ctx.index.get(ctx.current_id);
        let unvisited_value = current
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        grave_value(0., unvisited_value, 0.)
    }

    #[inline(always)]
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, _: Self::Aux) -> Self::Score {
        let child = ctx.index.get(child_id);
        let mean_score = child.stats.exploitation_score(ctx.player);

        let mut accum_visits = 0;
        let mut accum_score = 0.0;

        let mut rave_node_id = ctx.current_id;
        loop {
            let rave_node = ctx.index.get(rave_node_id);

            if let Some(grave_stats) = rave_node.stats.grave_stats.get(&child.action(ctx.index)) {
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
            let child_visits =
                (child.stats.num_visits + child.stats.num_visits_virtual.load(Relaxed)) as f64;

            mean_amaf = accum_score / accum_visits as f64;
            beta = accum_visits as f64
                / (accum_visits as f64
                    + child_visits
                    + self.bias * accum_visits as f64 * child_visits);
        }
        grave_value(beta, mean_score, mean_amaf)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GRAVE)
    }
}

////////////////////////////////////////////////////////////////////////////////

// This one was found in some implementations of RAVE. It seems strong, but I
// can't find references to it in the literature.
#[derive(Clone)]
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

impl<G: Game> SelectStrategy<G> for ScalarAmaf {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        ((ctx.current_stats().num_visits as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, parent_log: f64) -> f64 {
        let stats = ctx.child_stats(child_id);
        let exploit = stats.exploitation_score(ctx.player);
        let num_visits = stats.num_visits + stats.num_visits_virtual.load(Relaxed);
        let explore = (parent_log / num_visits as f64).sqrt();
        let uct_value = exploit + self.exploration_constant * explore;

        let amaf_value = if num_visits > 0 {
            stats.scalar_amaf.score / stats.num_visits as f64
        } else {
            0.
        };

        let beta = self.bias / (self.bias + num_visits as f64);

        (1. - beta) * uct_value + beta * amaf_value
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

#[derive(Clone)]
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

#[inline(always)]
fn ucb1_grave_value(
    beta: f64,
    mean_score: f64,
    mean_amaf: f64,
    exploration_constant: f64,
    explore: f64,
) -> f64 {
    grave_value(beta, mean_score, mean_amaf) + exploration_constant * explore
}

impl<G: Game> SelectStrategy<G> for Ucb1Grave {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        assert!(!ctx.use_transpositions);
        let current = ctx.index.get(ctx.current_id);
        if self.current_ref_id.is_none()
            || current.stats.num_visits > self.threshold
            || current.is_root()
        {
            self.current_ref_id = Some(ctx.current_id);
        }

        ((current.stats.num_visits as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(&self, ctx: &SelectContext<'_, G>, child_id: Id, parent_log: f64) -> f64 {
        let current_ref = ctx.index.get(self.current_ref_id.unwrap());
        let child = ctx.index.get(child_id);
        let mean_score = child.stats.exploitation_score(ctx.player);
        let child_visits =
            (child.stats.num_visits + child.stats.num_visits_virtual.load(Relaxed)) as f64;
        let current = ctx.index.get(ctx.current_id);
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
                    / (grave_visits + child_visits + self.bias * grave_visits * child_visits);

                (mean_amaf, beta)
            }
        };

        let explore = (parent_log / child_visits).sqrt();

        ucb1_grave_value(
            beta,
            mean_score,
            mean_amaf,
            self.exploration_constant,
            explore,
        )
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, parent_log: f64) -> Self::Score {
        let current = ctx.index.get(ctx.current_id);
        let unvisited_value = current
            .stats
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        ucb1_grave_value(
            0.,
            unvisited_value,
            0.,
            self.exploration_constant,
            parent_log.sqrt(),
        )
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GRAVE)
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
        let current = ctx.index.get(ctx.current_id);
        let available = current.actions();

        let key_init = ctx
            .stack
            .iter()
            .skip(1)
            .map(|id| ctx.index.get(*id).action(ctx.index))
            .collect::<Vec<_>>();

        let k_score = self.k[ctx.player_to_move];

        let enumerated = available.iter().cloned().enumerate().collect::<Vec<_>>();
        let best = random_best(enumerated.as_slice(), rng, |(_, action): &(usize, G::A)| {
            let mut key = key_init.clone();
            key.push(action.clone());

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
        });

        if let Some((best_index, _)) = best {
            *best_index
        } else {
            let action = self.search.choose_action(ctx.state);
            available.iter().position(|p| *p == action.clone()).unwrap()
        }
    }

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(&self, _: &SelectContext<'_, G>, _: Id, _: Self::Aux) -> f64 {
        0.
    }

    #[inline(always)]
    fn unvisited_value(&self, _: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        0.
    }
}
