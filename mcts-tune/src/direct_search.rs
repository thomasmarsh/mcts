//! Builds a runnable `Box<dyn Search<G>>` for a [`DirectFamily`] -- the
//! `Direct` counterpart of `config_ir::build_search`. Every `Compose`
//! family resolves through `config_ir`'s four-axis `SearchSpec`; a `Direct`
//! family has no such representation (it's a standalone `Search` impl, not
//! a `Compose<DynSelect<G>, DynSimulate<G>, B, DynSelect<G>>` `TreeSearch`),
//! so this is the one place `G` is monomorphized against its concrete type
//! instead.

use mcts::evaluator::MaterialBlind;
use mcts::game::Game;
use mcts::strategies::negamax::{Negamax, NegamaxOptions};
use mcts::strategies::{flat_mc::FlatMonteCarloStrategy, random::Random, Search};

use crate::family_catalog::DirectFamily;
use crate::SearchBudget;

/// Builds the concrete `Search` impl named by `direct`. `budget` is accepted
/// for symmetry with `config_ir::build_search`'s call shape; `Random` and
/// `FlatMc` ignore it (neither has a time/iteration/thread budget to apply --
/// `flat_mc`'s own per-move effort is `samples_per_move`/`max_rollout_depth`,
/// tunable fields rather than a run-level budget), but `Negamax` reads both
/// `threads` (Lazy-SMP root splitting) and `max_time` (iterative-deepening
/// cutoff) from it, the same as any `Compose` family's `SearchSettings`.
pub(crate) fn build_direct<G: Game + 'static>(
    direct: &DirectFamily,
    budget: &SearchBudget,
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
        DirectFamily::Negamax {
            max_depth,
            table_bits,
            replacement,
            aspiration_window,
            principal_variation_search,
            history_heuristic,
            singular_extension,
            countermove_heuristic,
        } => {
            let mut options = NegamaxOptions::default()
                .with_num_threads(budget.threads)
                .with_max_depth(*max_depth)
                .with_table_bits(*table_bits)
                .with_replacement(*replacement)
                .with_principal_variation_search(*principal_variation_search)
                .with_history_heuristic(*history_heuristic)
                .with_singular_extension(*singular_extension)
                .with_countermove_heuristic(*countermove_heuristic);
            if let Some(window) = aspiration_window {
                options = options.with_aspiration_window(*window);
            }
            if let Some(max_time) = budget.max_time {
                options = options.with_max_time(max_time);
            }
            Box::new(Negamax::<G, MaterialBlind>::new_with_options(
                MaterialBlind,
                options,
            ))
        }
    }
}
