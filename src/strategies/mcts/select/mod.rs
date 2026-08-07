pub mod amaf;
pub mod basic;
pub mod quasi;
pub mod rave;
pub mod ucb;

pub use amaf::Amaf;
pub use basic::MaxAvgScore;
pub use basic::RobustChild;
pub use basic::SecureChild;
pub use basic::ThompsonSampling;
pub use quasi::QuasiBestFirst;
pub use rave::Rave;
pub use rave::RaveSchedule;
pub use rave::RaveUcb;
pub use ucb::Ucb1;
pub use ucb::Ucb1Tuned;

use super::index::Id;
use super::node::{self, ChildArray, NodeStats, Proven, StatsRef};
use super::stack::NodeStack;
use super::table::TranspositionTable;
use super::*;
use crate::game::Game;

use rand::rngs::SmallRng;
use rand::Rng;
use rustc_hash::FxHashMap;

pub struct SelectContext<'a, G: Game> {
    pub q_init: node::QInit,
    pub stack: &'a NodeStack<G::A>,
    pub root_stats: &'a NodeStats,
    pub state: &'a G::S,
    pub player: usize,
    pub index: &'a TreeIndex<G::A>,
    pub table: &'a TranspositionTable<G::S>,
    pub grave: &'a FxHashMap<u64, Vec<FxHashMap<G::A, node::ActionStats>>>,
    pub use_transpositions: bool,
}

impl<'a, G: Game> SelectContext<'a, G> {
    fn current_stats(&self) -> StatsRef<'_, G::A> {
        self.stack.current_stats(self.index, self.root_stats)
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
        let current = ctx.index.get(ctx.stack.current_id());
        random_best_index(current.children(), self, ctx, rng)
    }

    /// Given a child index (its position in `children`), calculate a score.
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
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
            let current = ctx.index.get(ctx.stack.current_id());
            let n = current.children().len();
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
        children: &ChildArray<G::A>,
        idx: usize,
        aux: Self::Aux,
    ) -> Self::Score {
        self.inner.score_child(ctx, child_id, children, idx, aux)
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

/// Whether `children[idx]` is a proven loss for `ctx.player` -- a resolved
/// child proven `Win` for the *other* player under the `<= 2`-player
/// scoping the solver is built for (see node.rs's `Proven` doc comment).
/// Always `false` when the solver is off, since backprop never writes
/// anything but `Unproven` in that case.
#[inline]
fn is_proven_loss<G: Game>(
    ctx: &SelectContext<'_, G>,
    children: &ChildArray<G::A>,
    idx: usize,
) -> bool {
    children.node_id(idx).is_some_and(|child_id| {
        matches!(ctx.index.get(child_id).proven(), Proven::Win(w) if w != ctx.player)
    })
}

// This function is adapted from from minimax-rs.
#[inline]
fn random_best_index<S, G>(
    children: &ChildArray<G::A>,
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
    let n = children.len();
    let r = rng.gen_range(0..n * PRIMES.len());
    let mut i = r / PRIMES.len();
    let stride = PRIMES[r % PRIMES.len()];

    let aux = strategy.setup(ctx);
    let unvisited_value = strategy.unvisited_value(ctx, aux);

    let child_value = |i: usize| {
        if let Some(child_id) = children.node_id(i) {
            strategy.score_child(ctx, child_id, children, i, aux)
        } else {
            unvisited_value
        }
    };

    // Proven-loss avoidance (MCTS-Solver): prefer any non-proven-loss
    // sibling, and only fall back to a proven-loss child when every sibling
    // is one -- a graded rule, unlike the proven-win short-circuit (which
    // lives at `select_step`'s call site instead, see search.rs), since it
    // has to interact with each strategy's own scoring rather than bypass
    // it. A no-op scan (never skips) when every edge happens to be a proven
    // loss, or when the solver is off.
    let skip_proven_loss = !(0..n).all(|idx| is_proven_loss(ctx, children, idx));

    let mut best_score = None;
    let mut best_index = i;
    for _ in 0..n {
        if !(skip_proven_loss && is_proven_loss(ctx, children, i)) {
            let score = child_value(i);
            if best_score.is_none_or(|best| score > best) {
                best_score = Some(score);
                best_index = i;
            }
        }
        i = (i + stride) % n;
    }

    best_index
}
