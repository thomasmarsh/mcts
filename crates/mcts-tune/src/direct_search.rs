//! Builds a runnable `Box<dyn Search<G>>` for a non-MCTS [`AlgorithmSpec`] --
//! the `random`/`bandit`/`negamax` counterpart of `config_ir::build_search`.
//! Those three are standalone `Search` impls, not a
//! `Mcts<DynSelect<G>, DynSimulate<G>, B, DynSelect<G>>` `TreeSearch`,
//! so this is the one place `G` is monomorphized against its concrete type
//! instead of erased through `config_ir`'s `Dyn*` axes.

use mcts::evaluator::MaterialBlind;
use mcts::game::Game;
use mcts::algorithms::bandit::{self, BanditStrategy};
use mcts::algorithms::negamax::{Negamax, NegamaxOptions};
use mcts::algorithms::{random::Random, Search};

use crate::dispatch::{AlgorithmSpec, BanditPolicySpec};
use crate::SearchBudget;

/// Builds the concrete `Search` impl named by a non-MCTS `algorithm`.
/// `budget` is accepted for symmetry with `config_ir::build_search`'s call
/// shape; `Random` ignores it (it has no compute budget to apply at all),
/// but `Bandit` and `Negamax` both read it: `Bandit` feeds
/// `budget.iteration_limit()` into `BanditStrategy::budget` (its own
/// `algorithm == mcts`-config's `bandit_policy`/`c`/`epsilon` fields still
/// choose the arm-selection rule -- `budget` only bounds how many rollouts
/// that rule gets to spend, the same role `SearchSettings::max_iterations`
/// plays for `algorithm == mcts`), and `Negamax` reads both `threads`
/// (Lazy-SMP root splitting) and `max_time` (iterative-deepening cutoff)
/// from it.
///
/// `BanditStrategy` has no wall-clock awareness at all -- unlike `Negamax`,
/// a `SearchBudget` built from `--max-time-ms` (`max_iterations: None`,
/// `iteration_limit()` reading `usize::MAX`) does *not* cap it; only
/// `--max-iterations` (or the config's own `budget` field, whichever is
/// smaller) actually bounds a `Bandit` candidate's compute.
///
/// [`AlgorithmSpec::Mcts`] is unreachable here: `make_candidate` routes it
/// through `config_ir::build_search` before falling back to this function.
pub(crate) fn build_direct<G: Game + 'static>(
    algorithm: &AlgorithmSpec,
    seed: u64,
    budget: &SearchBudget,
) -> Box<dyn Search<G = G>> {
    match algorithm {
        AlgorithmSpec::Random => Box::new(Random::<G>::new().with_seed(seed)),
        AlgorithmSpec::Bandit {
            budget: rollout_budget,
            max_rollout_depth,
            policy,
        } => {
            let policy: Box<dyn bandit::BanditPolicy + Send + Sync> = match policy {
                BanditPolicySpec::Random => Box::new(bandit::Random),
                BanditPolicySpec::EpsilonGreedy { epsilon } => {
                    Box::new(bandit::EpsilonGreedy { epsilon: *epsilon })
                }
                BanditPolicySpec::Ucb1 { c } => Box::new(bandit::Ucb1 {
                    exploration_constant: *c,
                }),
                BanditPolicySpec::Thompson => Box::new(bandit::ThompsonSampling::default()),
            };
            // `rollout_budget` is the config's own tunable `budget` field;
            // `budget.iteration_limit()` is the operator's run-level
            // `--max-iterations`/`SearchBudget` override, `None` unless one
            // was actually passed. The smaller of the two wins, so a tighter
            // operator override always caps the tunable, but a config that
            // asks for less than the operator's ceiling isn't padded up to
            // it.
            let effective_budget = (*rollout_budget as usize).min(budget.iteration_limit()) as u32;
            Box::new(
                BanditStrategy::<G>::new()
                    .set_budget(effective_budget)
                    .set_max_rollout_depth(*max_rollout_depth)
                    .set_policy(policy)
                    .with_seed(seed),
            )
        }
        AlgorithmSpec::Negamax {
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
        AlgorithmSpec::Mcts(_) => {
            unreachable!("build_direct is only called for a non-MCTS AlgorithmSpec")
        }
    }
}
