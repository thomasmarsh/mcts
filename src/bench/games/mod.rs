//! Per-game-kind `BenchGame` implementations.  One submodule per game kind,
//! each holding its own concrete `BenchGame` impl.  Registered in
//! `registry()` the same way `server/main.rs` registers `GameAdapter`
//! impls -- mirrors that pattern but for "build named strategies and play a
//! match," not "serve JSON to a browser."

pub mod druid;

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
pub fn registry() -> HashMap<&'static str, Box<dyn BenchGame>> {
    let mut m: HashMap<&'static str, Box<dyn BenchGame>> = HashMap::new();
    m.insert("druid", Box::new(druid::DruidBenchGame));
    m
}