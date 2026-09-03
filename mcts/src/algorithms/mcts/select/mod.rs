pub mod amaf;
pub mod basic;
pub mod bayes;
pub mod gpn;
pub mod history;
pub mod pn;
pub mod quasi;
pub mod rave;
pub mod regularized;
pub mod score_bounded;
pub mod ucb;
pub mod variance;

pub use amaf::Amaf;
pub use basic::MaxAvgScore;
pub use basic::MaxRobustChild;
pub use basic::RobustChild;
pub use basic::SecureChild;
pub use basic::ThompsonSampling;
pub use bayes::BayesUct1;
pub use bayes::BayesUct2;
pub use gpn::GpnBias;
pub use gpn::GpnUct;
pub use history::ProgressiveHistory;
pub use pn::UctPn;
pub use quasi::QuasiBestFirst;
pub use rave::Rave;
pub use rave::RaveSchedule;
pub use rave::RaveUcb;
pub use regularized::GrillAct;
pub use regularized::Ments;
pub use score_bounded::ScoreBoundedUct;
pub use ucb::Ucb1;
pub use variance::KlUcb;
pub use variance::Ucb1Tuned;
pub use variance::UcbV;

use super::config::GraphStats;
use super::config::McgsCorrection;
use super::index::Id;
use super::node::{self, ChildArray, NodeStats, Proven, StatsRef};
use super::search::shared::TreeStats;
use super::stack::NodeStack;
use super::table::TranspositionTable;
use super::*;
use crate::game::Game;
use crate::game::Transform;

use rand::rngs::SmallRng;
use rand::Rng;
use rustc_hash::FxHashMap;

pub struct SelectContext<'a, G: Game> {
    pub q_init: node::QInit,
    pub stack: &'a NodeStack<G::A>,
    pub root_stats: &'a NodeStats,
    /// The root's own literal-board state -- what a caller needing more than
    /// `incoming_sym` (e.g. `QuasiBestFirst::best_child`, which needs every
    /// ancestor's own incoming symmetry, not just `stack.current_id()`'s)
    /// replays real states forward from via `NodeStack::incoming_syms`. See
    /// `crate::symmetry::incoming_sym`'s doc comment for why this can't be cached
    /// per-edge.
    pub root_state: &'a G::S,
    /// Whether nodes can be shared across differing real orientations --
    /// true for explicit `GraphSearch::Dag` and for the legacy
    /// `use_transpositions` table alike (see `crate::symmetry::incoming_sym`'s doc
    /// comment). Gates whether `crate::symmetry::incoming_sym` computes a real
    /// translation or short-circuits to identity.
    pub canonicalizes: bool,
    pub state: &'a G::S,
    pub player: usize,
    pub index: &'a TreeIndex<G::A>,
    pub table: &'a TranspositionTable,
    pub grave: &'a FxHashMap<u64, Vec<FxHashMap<G::A, node::ActionStats>>>,
    pub global: &'a TreeStats<G>,
    pub use_transpositions: bool,
    pub graph_stats: Option<GraphStats>,
    /// `McgsCorrection::RaveBlend` needs to read this during `Ucb1::
    /// score_child` (blending a DAG-merged target's pooled estimate into
    /// the edge's own selection score); every other correction/strategy
    /// ignores this field entirely. `McgsCorrection::Residual`'s own
    /// correction check stays outside `SelectContext` -- it fires from
    /// `search/shared.rs::select_step` after selection has already chosen
    /// `best_idx`, not during `score_child` itself.
    pub mcgs_correction: McgsCorrection,
    /// MCTS-Solver's proven-loss selection threshold `T` -- see
    /// `SearchConfig::solver_loss_threshold`'s doc comment. `0` when the
    /// solver is off, same as everywhere else `Proven` never leaves
    /// `Unproven` in that case (see `is_proven_loss`'s doc comment).
    pub solver_loss_threshold: u32,
    /// The symmetry index of the edge leading into `stack.current_id()` --
    /// i.e. what translates *that* node's own (possibly canonical)
    /// `ChildArray` actions back into `state`'s literal-board orientation
    /// (see `Game::canonical_representation`, `node::real_action`). Not
    /// derivable from `stack`/`index` alone here: it depends on which edge
    /// of `stack.current_id()`'s own *parent* was taken to reach it, a
    /// property of the caller's descent, not of the node itself. `0`
    /// (identity) at the root, and for every game that hasn't overridden
    /// `canonical_representation`.
    pub incoming_sym: Transform,
}

impl<'a, G: Game> SelectContext<'a, G> {
    fn current_stats(&self) -> StatsRef<'_, G::A> {
        self.stack
            .current_stats(self.index, self.root_stats, self.graph_stats)
    }

    pub fn child_snapshot(
        &self,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
    ) -> node::ChildSnapshot {
        if matches!(self.graph_stats, Some(GraphStats::Nodes)) {
            self.index.get(child_id).stats.snapshot(self.player)
        } else {
            children.snapshot(idx, self.player)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait SelectPolicy<G: Game>: Sized + Clone + Sync + Send + Default {
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

    /// Whether this strategy's `score_child` accounts for `ChildArray::
    /// is_growable`'s per-child availability count (Cowling, Powley &
    /// Whitehouse 2012) instead of assuming every child shares the node's
    /// own total visit count -- i.e. whether it gives a correct answer under
    /// ISMCTS (`SearchConfig::ismcts_mode`, either variant), as opposed to a
    /// plain, silently biased UCB computed against a node whose children
    /// aren't all legal on every iteration. `false` by default, matching
    /// every strategy that predates ISMCTS; `Ucb1` is the only override so
    /// far. `SearchConfig::validate` rejects `ismcts_mode` paired with any
    /// strategy that answers `false` here, rather than silently producing a
    /// biased search.
    fn supports_ismcts() -> bool {
        false
    }

    /// This component's `config::Requirements` -- storage it needs and any
    /// hard constraints it places on the game. Defaults to whatever
    /// `backprop_flags` already reports, so every existing `SelectPolicy`
    /// gets a correct answer with no code change; override only when a
    /// component needs to report something `backprop_flags` can't express
    /// (e.g. `UctPn`'s `solver`/`max_players`) -- see `config::Requirements`'s
    /// doc comment.
    fn requirements(&self) -> config::Requirements {
        config::Requirements::from_backprop_flags(self.backprop_flags())
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct EpsilonGreedy<G: Game, S: SelectPolicy<G>> {
    pub epsilon: f64,
    pub inner: S,
    pub marker: std::marker::PhantomData<G>,
}

impl<G, S> EpsilonGreedy<G, S>
where
    G: Game,
    S: SelectPolicy<G> + Default,
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
    S: SelectPolicy<G> + Default,
{
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            inner: S::default(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<G, S> SelectPolicy<G> for EpsilonGreedy<G, S>
where
    G: Game,
    S: SelectPolicy<G>,
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

    /// Delegates to `inner.requirements()` directly, not the default's
    /// `from_backprop_flags(self.backprop_flags())` -- see
    /// `simulate::EpsilonGreedy::requirements`'s doc comment for why.
    fn requirements(&self) -> config::Requirements {
        self.inner.requirements()
    }
}

////////////////////////////////////////////////////////////////////////////////

const PRIMES: [usize; 16] = [
    14323, 18713, 19463, 30553, 33469, 45343, 50221, 51991, 53201, 56923, 64891, 72763, 74471,
    81647, 92581, 94693,
];

/// Whether `children[idx]` is a proven loss for `ctx.player`, and should be
/// excluded from selection as one -- a resolved child proven `Win` for the
/// *other* player under the `<= 2`-player scoping the solver is built for
/// (see node.rs's `Proven` doc comment), whose own visit count has already
/// passed `ctx.solver_loss_threshold` (Kowalski et al. 2023, Section III.B's
/// solver parameter `T` -- see `SearchConfig::solver_loss_threshold`'s doc
/// comment). Always `false` when the solver is off, since backprop never
/// writes anything but `Unproven` in that case; `solver_loss_threshold`
/// itself is also always `0` in that case, so the visit check alone
/// wouldn't be enough to guarantee that on its own.
///
/// `pub(super)` (rather than `mod.rs`-private) so `basic.rs`'s
/// `ThompsonSampling` can reuse this instead of keeping its own,
/// independently-drifting copy of the same rule -- unlike every other
/// `SelectPolicy`, which reaches it indirectly through
/// `random_best_index_by` below.
#[inline]
pub(super) fn is_proven_loss<G: Game>(
    ctx: &SelectContext<'_, G>,
    children: &ChildArray<G::A>,
    idx: usize,
) -> bool {
    children.node_id(idx).is_some_and(|child_id| {
        matches!(ctx.index.get(child_id).proven(), Proven::Win(w) if w != ctx.player)
            && ctx.child_snapshot(child_id, children, idx).num_visits > ctx.solver_loss_threshold
    })
}

/// The exact utility a `Proven` status contributes to `player`'s score:
/// `Win(player)` → +1, any other proven outcome (`Draw`, an opponent win) →
/// its exact utility, `Unproven` → `None`. Factored out so it can be
/// unit-tested without a `SelectContext` (see `select::variance`'s tests).
#[inline]
pub(super) fn proven_to_utility(p: Proven, player: usize) -> Option<f64> {
    match p {
        Proven::Win(w) if w == player => Some(1.0),
        Proven::Win(_) => Some(-1.0),
        Proven::Draw => Some(0.0),
        Proven::Unproven => None,
    }
}

/// The exact utility a `Proven` child contributes to `ctx.player`'s score,
/// or `None` if the child has no tree node yet or is `Unproven`. Used by
/// variance-aware strategies whose exploration term is ill-conditioned at
/// `q̄ ∈ {0, 1}` with small `n` (KL-UCB's bound in particular). The
/// `node_id(idx)?` guard makes it a no-op for the prior-placeholder path
/// (`score_child_or_prior` passing the parent's `Id`). With the solver off,
/// `proven()` is always `Unproven`, so this is a single cheap `None` branch.
#[inline]
pub(super) fn proven_exact_value<G: Game>(
    ctx: &SelectContext<'_, G>,
    children: &ChildArray<G::A>,
    idx: usize,
) -> Option<f64> {
    let child_id = children.node_id(idx)?;
    proven_to_utility(ctx.index.get(child_id).proven(), ctx.player)
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
    S: SelectPolicy<G>,
    G: Game,
{
    let aux = strategy.setup(ctx);
    let unvisited_value = strategy.unvisited_value(ctx, aux);

    random_best_index_by(children, ctx, rng, |i| {
        score_child_or_prior(ctx, strategy, children, i, aux, unvisited_value)
    })
}

/// `random_best_index`'s ISMCTS counterpart: the same tie-broken argmax over
/// `score_child`/`unvisited_value`, but restricted to `legal_idxs` -- this
/// iteration's own `G::determinize`d sample only makes those children
/// reachable, and Cowling et al.'s "restrict to compatible children" step
/// means selection must never even consider one that isn't. Called from
/// `search/shared.rs::select_step` (`IsmctsMode::SingleTree`'s one shared
/// tree) and `search/multi_tree.rs::select_multi_tree` (`IsmctsMode::
/// MultiTree`'s one tree per player, called only for whichever tree belongs
/// to the player about to move). MCTS-Solver's proven-loss skip has no
/// counterpart here: `SearchConfig::validate` requires `use_mcts_solver` off
/// wherever `ismcts_mode` isn't `Off`, so no child reachable through
/// `legal_idxs` is ever proven.
pub(crate) fn ismcts_best_child<S, G>(
    ctx: &SelectContext<'_, G>,
    children: &ChildArray<G::A>,
    legal_idxs: &[usize],
    strategy: &mut S,
    rng: &mut SmallRng,
) -> usize
where
    S: SelectPolicy<G>,
    G: Game,
{
    debug_assert!(!legal_idxs.is_empty());
    let aux = strategy.setup(ctx);
    let unvisited_value = strategy.unvisited_value(ctx, aux);

    let n = legal_idxs.len();
    let r = rng.gen_range(0..n * PRIMES.len());
    let mut i = r / PRIMES.len();
    let stride = PRIMES[r % PRIMES.len()];

    let mut best_score = None;
    let mut best_index = legal_idxs[i];
    for _ in 0..n {
        let idx = legal_idxs[i];
        let score = score_child_or_prior(ctx, strategy, children, idx, aux, unvisited_value);
        if best_score.is_none_or(|best| score > best) {
            best_score = Some(score);
            best_index = idx;
        }
        i = (i + stride) % n;
    }

    best_index
}

/// `random_best_index`'s per-child scoring, factored out so `ThompsonSampling`/
/// `UctPn` (which override `best_child` outright, to compute a rank/weight
/// that needs more than one child's stats at once) can share it instead of
/// re-deriving their own copy.
///
/// A not-yet-created child (`children.node_id(i)` is `None`) with
/// `children.num_visits(i) > 0` has had `prior::PriorPolicy`-seeded
/// pseudo-visits written directly into its `ChildArray` row
/// (`node::ChildArray::seed_prior`, called from `search/shared.rs::expand()`)
/// -- so it's scored exactly like a real child, through `score_child`, with
/// the *parent's own* `Id` passed as a harmless placeholder for the
/// otherwise-nonexistent child `Id`. Sound only because `SelectContext::
/// child_snapshot` never dereferences that `Id` except under
/// `GraphStats::Nodes`, which `SearchConfig::validate` rejects outright
/// whenever a prior policy is active (see its doc comment) -- so this
/// branch is never reached in a configuration where the placeholder would
/// matter. Every other unvisited child (no prior, or `pseudo_visits() == 0`)
/// keeps today's behavior: `unvisited_value`, one constant shared by every
/// still-untouched sibling.
#[inline]
pub(super) fn score_child_or_prior<G, S>(
    ctx: &SelectContext<'_, G>,
    strategy: &S,
    children: &ChildArray<G::A>,
    idx: usize,
    aux: S::Aux,
    unvisited_value: S::Score,
) -> S::Score
where
    G: Game,
    S: SelectPolicy<G>,
{
    match children.node_id(idx) {
        Some(child_id) => strategy.score_child(ctx, child_id, children, idx, aux),
        None if children.num_visits(idx) > 0 => {
            strategy.score_child(ctx, ctx.stack.current_id(), children, idx, aux)
        }
        None => unvisited_value,
    }
}

/// The tie-broken argmax + proven-loss-skip core of `random_best_index`,
/// factored out to take a plain scoring closure instead of a
/// `SelectPolicy` -- what lets a strategy whose per-child score depends on
/// *every* sibling at once (e.g. `UctPn`'s rank, which isn't expressible as
/// a `SelectPolicy::Aux` since that associated type must be `Copy` and a
/// per-child rank table isn't) reuse this instead of re-implementing it.
#[inline]
pub(super) fn random_best_index_by<G, Score>(
    children: &ChildArray<G::A>,
    ctx: &SelectContext<'_, G>,
    rng: &mut SmallRng,
    mut child_value: impl FnMut(usize) -> Score,
) -> usize
where
    G: Game,
    Score: PartialOrd + Copy,
{
    // To make the choice more uniformly random among the best moves, start
    // at a random offset and stride by a random amount. The stride must be
    // coprime with n, so pick from a set of 5 digit primes.

    // Combine both random numbers into a single rng call.
    let n = children.len();
    let r = rng.gen_range(0..n * PRIMES.len());
    let mut i = r / PRIMES.len();
    let stride = PRIMES[r % PRIMES.len()];

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
