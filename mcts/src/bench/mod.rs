//! Benchmark / tournament / SMAC3 harness library. Only the `server` process
//! opens `bench.duckdb` directly; `bin/bench` and Python tools communicate
//! via JSONL files and the registry log.

pub mod launch;
pub mod log;
pub mod tournament;
pub mod games;

#[cfg(feature = "duckdb")]
pub mod ingest;
#[cfg(feature = "duckdb")]
pub mod schema;

use std::collections::HashMap;

/// Information about a playable strategy for a game kind.
#[derive(serde::Serialize)]
pub struct StrategyInfo {
    pub id: String,
    pub label: String,
    pub description: String,
}

/// Outcome of a single match between two strategies.
pub struct MatchOutcome {
    /// `None` for draw, `Some(0)` if first strategy won, `Some(1)` if second.
    pub winner: Option<usize>,
    /// Optional extra metadata (moves list, timing, etc.).
    pub extra: Option<serde_json::Value>,
}

/// Per-game-kind trait for the benchmark harness.  One impl per game kind,
/// registered in `registry()`.  Mirrors `GameAdapter`'s shape but for
/// "build these named strategies and play matches" rather than "serve JSON
/// to a browser."
///
/// The concrete impls live in the per-game crates (`games/*`), each backed
/// by its own standalone binary speaking the subprocess protocol.
pub trait BenchGame: Send + Sync {
    /// Machine-readable kind string (e.g. `"druid"`, `"ttt"`).
    fn kind(&self) -> &'static str;

    /// Available strategies for this game kind.
    fn strategies(&self) -> Vec<StrategyInfo>;

    /// Play one match between two strategies identified by their strategy
    /// IDs.  Returns the outcome.
    fn play_match(&self, strategy_a: &str, strategy_b: &str) -> MatchOutcome;
}

/// Register all known `BenchGame` implementations.
///
/// Each game kind is backed by a subprocess game binary discovered
/// relative to the running executable.
pub fn registry() -> HashMap<&'static str, Box<dyn BenchGame>> {
    games::registry()
}