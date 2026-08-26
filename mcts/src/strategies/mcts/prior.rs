//! MCTS-IP-E / MCTS-IP-M / MCTS-MS-d-Visit-0 (Baier & Winands): a per-action
//! prior value computed once, at the moment a node is expanded, instead of
//! leaving every one of its not-yet-visited children looking identical (the
//! `QInit`-driven constant `select::SelectStrategy::unvisited_value` uses
//! today). `search/shared.rs::expand()` calls `evaluate_children` right after
//! generating a freshly-expanded node's action list and, if it returns
//! anything, seeds each child's `ChildArray` stats with `pseudo_visits`
//! fictitious visits at that value (`node::ChildArray::seed_prior`) *before*
//! any child `Node` exists -- so ordinary selection math (`score_child`) picks
//! among still-unvisited siblings using the prior from the very first
//! descent, rather than the uniform `unvisited_value` fallback every other
//! `SelectStrategy` otherwise shares.
//!
//! Deliberately its own pluggable component -- not a fifth associated type on
//! `config::Strategy<G>` -- since that trait already has ~20 concrete impls
//! plus the generic `strategy::Compose` escape hatch used broadly across
//! `mcts-bench`/`mcts-tune`/examples, and Rust has no stable way to give an
//! existing trait's associated type a default that every one of those impls
//! would otherwise have to grow by hand. `SearchConfig::prior` instead stores
//! an `Option<Box<dyn PriorStrategyDyn<G>>>` -- a boxed trait object, not a
//! new generic parameter threaded through `TreeSearch<G, S>` and every one of
//! its existing `impl` blocks (`search/core.rs`, `search/parallel.rs`,
//! `search/reroot.rs`, `search/reuse.rs`, `search/compact.rs`,
//! `search_impl.rs`) -- so opting in costs one dynamic dispatch per expansion
//! (rare; nowhere near `select`'s hot per-descent loop) rather than widening
//! this crate's monomorphization surface the way the config-algebra work
//! (`config_ir.rs`) already found expensive at compile time.

use super::config;
use crate::evaluator::Evaluator;
use crate::evaluator::EVAL_MAGNITUDE_LIMIT;
use crate::evaluator::WIN_SCORE;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::TerminalStatus;
use crate::strategies::negamax::Negamax;
use crate::strategies::negamax::NegamaxOptions;

/// `child`'s exact value from its own player-to-move's perspective, `[-1,
/// 1]`, if `child` is terminal -- `None` otherwise. Neither `Evaluator::
/// evaluate` nor `Negamax::bounded_negamax` can be called on a terminal
/// state (the latter asserts a non-empty action list and panics; the former
/// is documented as only meaningful at a depth cutoff, never a genuine
/// terminal), so every `PriorStrategy::evaluate_children` impl here must
/// check this before falling through to its own real evaluation -- a
/// perfectly ordinary outcome, not an edge case: any action a node's
/// `ChildArray` offers can be the move that ends the game.
fn terminal_value<G: Game>(child: &G::S) -> Option<f64> {
    let mover = G::player_to_move(child).to_index();
    match G::terminal_status(child) {
        TerminalStatus::NotTerminal => None,
        TerminalStatus::Draw => Some(0.),
        TerminalStatus::Winner(w) => Some(if w.to_index() == mover { 1. } else { -1. }),
    }
}

/// One node's worth of per-action prior values, computed at expansion time.
/// `evaluate_children` returns values relative to `state`'s own player to
/// move (index `i` answers "how good is `actions[i]` for me"), each clamped
/// to `[-1, 1]` -- the same per-player utility convention `simulate::
/// EvaluatedCutoff`'s cutoff-utility conversion already uses. An empty
/// result (or `pseudo_visits() == 0`) means "no prior for this node", the
/// same as not calling this at all.
pub trait PriorStrategy<G: Game>: Clone + Send + Sync {
    fn evaluate_children(&mut self, state: &G::S, actions: &[G::A]) -> Vec<f64>;

    /// Fictitious visit count each returned prior value is seeded with --
    /// how much a prior outweighs real playout results before they start
    /// dominating it. `0` disables seeding even if `evaluate_children`
    /// returns values (so a strategy can be paired with `pseudo_visits() ==
    /// 0` purely to skip the search-and-seed cost while testing).
    fn pseudo_visits(&self) -> u32;

    /// This component's `config::Requirements` -- see
    /// `select::SelectStrategy::requirements`'s doc comment for the general
    /// contract. Defaults to unconstrained, since [`NoPrior`] has no
    /// requirements of its own; override wherever `evaluate_children`'s
    /// implementation itself assumes something about the game, the way
    /// [`EvaluatorPrior`] and [`NegamaxPrior`] do.
    fn requirements(&self) -> config::Requirements {
        config::Requirements::none()
    }
}

/// The universal default: no prior. `evaluate_children` never allocates
/// (always returns the same empty `Vec`), so a search that never opts in
/// pays nothing beyond the branch in `search/shared.rs::expand()`.
#[derive(Clone, Copy, Default)]
pub struct NoPrior;

impl<G: Game> PriorStrategy<G> for NoPrior {
    fn evaluate_children(&mut self, _state: &G::S, _actions: &[G::A]) -> Vec<f64> {
        Vec::new()
    }

    fn pseudo_visits(&self) -> u32 {
        0
    }
}

/// MCTS-IP-E: a flat `Evaluator` call per candidate action -- apply the
/// action, evaluate the resulting (opponent-to-move) state, negate back to
/// the expanding node's own player. Cheaper than [`NegamaxPrior`] but has no
/// lookahead of its own beyond whatever the evaluator itself encodes.
pub struct EvaluatorPrior<G, E = crate::evaluator::MaterialBlind>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    evaluator: E,
    pseudo_visits: u32,
    marker: std::marker::PhantomData<G>,
}

impl<G, E> Clone for EvaluatorPrior<G, E>
where
    G: Game,
    E: Evaluator<G> + Default + Clone,
{
    fn clone(&self) -> Self {
        Self {
            evaluator: self.evaluator.clone(),
            pseudo_visits: self.pseudo_visits,
            marker: std::marker::PhantomData,
        }
    }
}

impl<G, E> Default for EvaluatorPrior<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    fn default() -> Self {
        Self {
            evaluator: E::default(),
            // Baier & Winands' own experiments cluster around a handful of
            // fictitious visits -- large enough to steer early selection,
            // small enough that a few dozen real playouts already outweigh
            // it. Callers with better domain knowledge should tune via
            // `pseudo_visits`.
            pseudo_visits: 4,
            marker: std::marker::PhantomData,
        }
    }
}

impl<G, E> EvaluatorPrior<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pseudo_visits(mut self, pseudo_visits: u32) -> Self {
        self.pseudo_visits = pseudo_visits;
        self
    }
}

impl<G, E> PriorStrategy<G> for EvaluatorPrior<G, E>
where
    G: Game,
    E: Evaluator<G> + Default + Clone + Send + Sync,
{
    fn evaluate_children(&mut self, state: &G::S, actions: &[G::A]) -> Vec<f64> {
        debug_assert!(
            G::num_players() <= 2,
            "EvaluatorPrior's nega-style utility conversion assumes a \
             two-player zero-sum game"
        );
        actions
            .iter()
            .map(|action| {
                let child = G::apply(state.clone(), action);
                match terminal_value::<G>(&child) {
                    Some(value) => -value,
                    None => {
                        let score = self.evaluator.evaluate(&child) as f64;
                        (-score / EVAL_MAGNITUDE_LIMIT as f64).clamp(-1., 1.)
                    }
                }
            })
            .collect()
    }

    fn pseudo_visits(&self) -> u32 {
        self.pseudo_visits
    }

    /// Same `<= 2`-player scoping as the `debug_assert!` in
    /// `evaluate_children` above, now enforced in release builds too via
    /// `SearchConfig::validate`.
    fn requirements(&self) -> config::Requirements {
        config::Requirements {
            max_players: Some(2),
            ..config::Requirements::none()
        }
    }
}

/// MCTS-IP-M / MCTS-MS-d-Visit-0: a shallow bounded-negamax search from each
/// candidate action's resulting state, negated back to the expanding node's
/// own player -- exact lookahead `depth` plies deep (or to a true terminal,
/// whichever comes first), instead of a single flat evaluation.
pub struct NegamaxPrior<G, E = crate::evaluator::MaterialBlind>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    // No transposition table: each candidate action's resulting state is
    // its own one-off search, never revisited by this same `NegamaxPrior`
    // instance again (the next `evaluate_children` call is a different
    // node entirely) -- same "cold, never-revisited state" reasoning
    // `simulate::MinimaxRollout` already documents for its own `negamax`
    // field.
    negamax: Negamax<G, E>,
    depth: u32,
    pseudo_visits: u32,
}

/// Hand-written for the same reason as `Negamax`'s own `Clone` impl: a
/// derive would add an `E: Clone` bound that `Negamax<G, E>`'s real
/// requirements (an `Arc<E>` internally) don't need.
impl<G, E> Clone for NegamaxPrior<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    fn clone(&self) -> Self {
        Self {
            negamax: self.negamax.clone(),
            depth: self.depth,
            pseudo_visits: self.pseudo_visits,
        }
    }
}

impl<G, E> Default for NegamaxPrior<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    fn default() -> Self {
        Self {
            negamax: Negamax::new_with_options(
                E::default(),
                NegamaxOptions::default().with_table_bits(0),
            ),
            // MS-2 is the literature's own best-performing depth on
            // Breakthrough (Baier & Winands 2015) -- see
            // `examples/strength_breakthrough_hybrid.rs`.
            depth: 2,
            pseudo_visits: 4,
        }
    }
}

impl<G, E> NegamaxPrior<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    pub fn pseudo_visits(mut self, pseudo_visits: u32) -> Self {
        self.pseudo_visits = pseudo_visits;
        self
    }
}

impl<G, E> PriorStrategy<G> for NegamaxPrior<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    fn evaluate_children(&mut self, state: &G::S, actions: &[G::A]) -> Vec<f64> {
        debug_assert!(
            G::num_players() <= 2,
            "NegamaxPrior's nega-style utility conversion assumes a \
             two-player zero-sum game"
        );
        if self.depth == 0 {
            return Vec::new();
        }
        actions
            .iter()
            .map(|action| {
                let child = G::apply(state.clone(), action);
                match terminal_value::<G>(&child) {
                    Some(value) => -value,
                    None => {
                        let (_, score) = self.negamax.bounded_negamax(&child, self.depth);
                        (-(score as f64) / WIN_SCORE as f64).clamp(-1., 1.)
                    }
                }
            })
            .collect()
    }

    fn pseudo_visits(&self) -> u32 {
        self.pseudo_visits
    }

    /// Same `<= 2`-player scoping as the `debug_assert!` in
    /// `evaluate_children` above, now enforced in release builds too via
    /// `SearchConfig::validate`.
    fn requirements(&self) -> config::Requirements {
        config::Requirements {
            max_players: Some(2),
            ..config::Requirements::none()
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Object-safe counterpart of [`PriorStrategy`], letting `SearchConfig` store
/// one behind `Box<dyn PriorStrategyDyn<G>>` without becoming generic over a
/// third strategy type parameter -- see this module's doc comment for why.
/// Blanket-implemented for every `PriorStrategy`; not meant to be implemented
/// directly.
pub trait PriorStrategyDyn<G: Game>: Send + Sync {
    fn evaluate_children(&mut self, state: &G::S, actions: &[G::A]) -> Vec<f64>;
    fn pseudo_visits(&self) -> u32;
    fn requirements(&self) -> config::Requirements;
    fn clone_box(&self) -> Box<dyn PriorStrategyDyn<G>>;
}

impl<G, T> PriorStrategyDyn<G> for T
where
    G: Game,
    T: PriorStrategy<G> + 'static,
{
    fn evaluate_children(&mut self, state: &G::S, actions: &[G::A]) -> Vec<f64> {
        PriorStrategy::evaluate_children(self, state, actions)
    }

    fn pseudo_visits(&self) -> u32 {
        PriorStrategy::pseudo_visits(self)
    }

    fn requirements(&self) -> config::Requirements {
        PriorStrategy::requirements(self)
    }

    fn clone_box(&self) -> Box<dyn PriorStrategyDyn<G>> {
        Box::new(self.clone())
    }
}

impl<G: Game> Clone for Box<dyn PriorStrategyDyn<G>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
