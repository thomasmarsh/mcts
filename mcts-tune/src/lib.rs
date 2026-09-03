//! Generic multi-family MCTS strategy tuning harness shared by every game
//! crate that opts into hyperparameter search. Everything here is generic
//! over `G: Game`: picking a concrete game, a baseline preset, and whether
//! that game has a real `zobrist_hash` is the only per-game glue. See
//! `games/traffic-lights/src/main.rs` for the reference wiring.
//!
//! The tunable search space describes a configuration directly: a top-level
//! `algorithm` categorical (`random`/`flat_mc`/`mcts`/`negamax`) and, for
//! `mcts`, the four policy-axis categoricals (`select`/`simulate`/`backprop`/
//! `final_action`) plus each variant's own parameters and the orthogonal
//! `q_init`/`mcgs` engine settings. `dispatch.rs` resolves a params object
//! into a `config_ir::SearchSpec`; `tuner_info.rs` reports the same shape as
//! the per-run schema.
//!
//! `strategy.rs`'s `QuasiBestFirst` is deliberately not exposed. Its
//! opening-book fallback runs a nested `TreeSearch::choose_action` during
//! every outer-search descent when no book entry exists, while this harness
//! uses a shared many-iteration outer budget. It is designed for an outer
//! `max_iterations: 1`, unlike the configurations represented here.
//!
//! This crate deliberately depends only on `mcts` and `game-host`: building
//! a candidate or baseline `Box<dyn Search<G>>` needs the concrete game at
//! compile time, so that work belongs in each game's crate or binary.

pub mod config_ir;
pub mod config_ir_schema;
mod direct_search;
// Algorithm-native construction of a `config_ir::SearchSpec` from the four
// policy-axis categoricals -- `search.rs`'s candidate/opponent builders and
// `tuner_info.rs`'s schema both describe a configuration directly through
// this module rather than a per-composition catalog table.
mod dispatch;
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

pub(crate) use search::resolve_graph_search;

#[cfg(test)]
pub(crate) use evaluation::cost_from_losses;
#[cfg(test)]
pub(crate) use search::MAX_ITER;

#[cfg(test)]
mod tests;
