//! Generic multi-family MCTS strategy tuning harness shared by every game
//! crate that opts into hyperparameter search. Everything here is generic
//! over `G: Game`: picking a concrete game, a baseline preset, and whether
//! that game has a real `zobrist_hash` is the only per-game glue. See
//! `games/traffic-lights/src/main.rs` for the reference wiring.
//!
//! The tunable search space has two levels: a top-level categorical `family`
//! axis chooses a `Strategy<G>` composition; within that family, the tuner
//! searches its hyperparameters. `q_init` is orthogonal to the strategy and
//! applies to every family. The `final_action` axis applies to every family
//! whose named type does not already fix a different final action.
//!
//! `strategy.rs`'s `QuasiBestFirst` is deliberately not in the catalog. Its
//! opening-book fallback runs a nested `TreeSearch::choose_action` during
//! every outer-search descent when no book entry exists, while this harness
//! uses a shared many-iteration outer budget. It is designed for an outer
//! `max_iterations: 1`, unlike the families represented here.
//!
//! This crate deliberately depends only on `mcts` and `game-host`: building
//! a candidate or baseline `Box<dyn Search<G>>` needs the concrete game at
//! compile time, so that work belongs in each game's crate or binary.

pub mod config_ir;
pub mod config_ir_schema;
mod direct_search;
mod evaluation;
mod family_catalog;
pub mod presets;
mod search;
pub mod trace;
mod tuner_info;

pub use evaluation::{generic_tune_eval, strategy_tune_eval, TuneEvalOutcome};
pub use search::{
    build_search, choose_action_with_report, legacy_analysis_with_report, SearchBudget,
};
pub use tuner_info::{
    dimensions_board_config_schema, square_board_config_schema, strategy_tuner_info,
    strategy_tuner_info_with_mcgs,
};

pub(crate) use search::{resolve_graph_search, META_MCTS_INNER_ITERATIONS};

#[cfg(test)]
pub(crate) use evaluation::cost_from_losses;
#[cfg(test)]
pub(crate) use search::{to_search_spec, EXPAND_THRESHOLD, MAX_ITER, PLAYOUT_DEPTH};

#[cfg(test)]
mod tests;
