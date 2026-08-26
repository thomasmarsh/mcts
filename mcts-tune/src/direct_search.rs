//! Builds a runnable `Box<dyn Search<G>>` for a [`DirectFamily`] -- the
//! `Direct` counterpart of `config_ir::build_search`. Every `Compose`
//! family resolves through `config_ir`'s four-axis `SearchSpec`; a `Direct`
//! family has no such representation (it's a standalone `Search` impl, not
//! a `Compose<DynSelect<G>, DynSimulate<G>, B, DynSelect<G>>` `TreeSearch`),
//! so this is the one place `G` is monomorphized against its concrete type
//! instead.

use mcts::game::Game;
use mcts::strategies::{flat_mc::FlatMonteCarloStrategy, random::Random, Search};

use crate::family_catalog::DirectFamily;
use crate::SearchBudget;

/// Builds the concrete `Search` impl named by `direct`. `budget` is accepted
/// for symmetry with `config_ir::build_search`'s call shape (and because a
/// future `Direct` family, e.g. `negamax`, reads it), but neither `Random`
/// nor `FlatMc` has a time/iteration/thread budget to apply -- `flat_mc`'s
/// own per-move effort is `samples_per_move`/`max_rollout_depth`, tunable
/// fields rather than a run-level budget.
pub(crate) fn build_direct<G: Game + 'static>(
    direct: &DirectFamily,
    _budget: &SearchBudget,
) -> Box<dyn Search<G = G>> {
    match direct {
        DirectFamily::Random => Box::new(Random::<G>::new()),
        DirectFamily::FlatMc {
            samples_per_move,
            max_rollout_depth,
            ucb1,
        } => Box::new(FlatMonteCarloStrategy::<G> {
            samples_per_move: *samples_per_move,
            max_rollout_depth: *max_rollout_depth,
            ucb1: *ucb1,
            ..FlatMonteCarloStrategy::new()
        }),
    }
}
