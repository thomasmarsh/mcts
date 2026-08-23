use serde_json::Value;

/// One request read from stdin: a single JSON line.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Request {
    /// Unique request identifier, echoed back in the response.
    pub id: u64,
    /// Method name — maps to a `GameAdapter` method.
    pub method: String,
    /// Method-specific parameters.
    pub params: Value,
}

/// One response written to stdout: a single JSON line.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Response {
    /// Successful method call.
    Success { id: u64, result: Value },
    /// Failed method call.
    Error { id: u64, error: ErrorBody },
}

/// Structured error body within an error response.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ErrorBody {
    /// HTTP-style status code (400, 404, 500, …).
    pub code: u16,
    /// Human-readable error description.
    pub message: String,
}

/// Information about one AI preset exposed by a game.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AiPresetInfo {
    pub id: String,
    pub label: String,
    pub description: String,
}

/// The result of a completed `ai_move`: the chosen move and the resulting
/// state, so the caller can apply both without a second round-trip.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AiMoveResult {
    pub mv: Value,
    pub state: Value,
    /// Final evidence from the action selection. `None` is reserved for
    /// older producers that have not adopted the search-report contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchReport>,
}

/// One candidate root action returned from `analyze`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AnalysisAction {
    pub action: Value,
    pub visits: u32,
    pub mean_value: f64,
    pub is_proven: bool,
}

/// Full analysis returned from `analyze`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Analysis {
    pub actions: Vec<AnalysisAction>,
    pub principal_variation: Vec<Value>,
    pub total_visits: u32,
    pub suggested_move: Option<Value>,
    /// Final evidence from the action selection that produced this analysis.
    /// `None` is reserved for older producers that have not adopted the
    /// search-report contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchReport>,
}

/// Availability of the versioned final-search evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchReportStatus {
    Available,
    Partial,
    Unavailable,
}

/// Why final-search evidence is unavailable or partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchReportReason {
    StrategyUnsupported,
    SearchNotRun,
    RootParallelPvSingleTree,
}

/// The condition that stopped the most recent search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchTermination {
    Iterations,
    Time,
    Solved,
    Unknown,
}

/// The retained search structure represented by a report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchGraphMode {
    Tree,
    Transpositions,
    DagEdges,
    DagNodes,
    DagBoth,
}

/// Non-fatal qualification attached to a final report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchWarning {
    ActionsTruncated,
    PrincipalVariationTruncated,
    StructuralDiagnosticsOmitted,
    RootParallelPvSingleTree,
}

/// One root action's final-search evidence in the game's canonical JSON
/// move format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchActionReport {
    pub action: Value,
    pub visits: u32,
    pub share: f64,
    pub mean_value: f64,
    pub is_proven: bool,
}

/// Version-1 final evidence from one strategy action selection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchReport {
    pub schema_version: u8,
    pub status: SearchReportStatus,
    pub reason: Option<SearchReportReason>,
    pub elapsed_seconds: Option<f64>,
    pub iteration_limit: Option<usize>,
    pub time_limit_seconds: Option<f64>,
    pub completed_iterations: usize,
    pub termination: Option<SearchTermination>,
    pub selected_action: Option<Value>,
    pub actions: Vec<SearchActionReport>,
    pub principal_variation: Vec<Value>,
    pub root_visits: u32,
    pub tree_nodes: usize,
    pub mean_depth: Option<f64>,
    pub max_depth: Option<usize>,
    pub graph_mode: Option<SearchGraphMode>,
    pub tt_reads: usize,
    pub tt_writes: usize,
    pub tt_hits: usize,
    pub tt_hit_ratio: Option<f64>,
    pub iterations_per_second: Option<f64>,
    pub warnings: Vec<SearchWarning>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_report_json_is_stable() {
        let report = SearchReport {
            schema_version: 1,
            status: SearchReportStatus::Partial,
            reason: Some(SearchReportReason::RootParallelPvSingleTree),
            elapsed_seconds: Some(0.25),
            iteration_limit: Some(100),
            time_limit_seconds: None,
            completed_iterations: 80,
            termination: Some(SearchTermination::Time),
            selected_action: Some(serde_json::json!({"ptn": "a1"})),
            actions: vec![SearchActionReport {
                action: serde_json::json!({"ptn": "a1"}),
                visits: 60,
                share: 0.75,
                mean_value: 0.5,
                is_proven: false,
            }],
            principal_variation: vec![serde_json::json!({"ptn": "a1"})],
            root_visits: 80,
            tree_nodes: 91,
            mean_depth: Some(4.0),
            max_depth: Some(7),
            graph_mode: Some(SearchGraphMode::DagBoth),
            tt_reads: 10,
            tt_writes: 8,
            tt_hits: 3,
            tt_hit_ratio: Some(0.3),
            iterations_per_second: Some(320.0),
            warnings: vec![SearchWarning::RootParallelPvSingleTree],
        };

        assert_eq!(
            serde_json::to_value(report).unwrap(),
            serde_json::json!({
                "schema_version": 1,
                "status": "partial",
                "reason": "root_parallel_pv_single_tree",
                "elapsed_seconds": 0.25,
                "iteration_limit": 100,
                "time_limit_seconds": null,
                "completed_iterations": 80,
                "termination": "time",
                "selected_action": {"ptn": "a1"},
                "actions": [{
                    "action": {"ptn": "a1"},
                    "visits": 60,
                    "share": 0.75,
                    "mean_value": 0.5,
                    "is_proven": false
                }],
                "principal_variation": [{"ptn": "a1"}],
                "root_visits": 80,
                "tree_nodes": 91,
                "mean_depth": 4.0,
                "max_depth": 7,
                "graph_mode": "dag_both",
                "tt_reads": 10,
                "tt_writes": 8,
                "tt_hits": 3,
                "tt_hit_ratio": 0.3,
                "iterations_per_second": 320.0,
                "warnings": ["root_parallel_pv_single_tree"]
            })
        );
    }
}

/// Which side the candidate configuration played in one configured match.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfiguredCandidateSide {
    First,
    Second,
}

/// The result of one configured candidate-versus-baseline game.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ConfiguredMatchResult {
    #[serde(rename = "type")]
    pub record_type: String,
    pub seq: u64,
    pub round: u32,
    pub seed: u64,
    pub candidate_side: ConfiguredCandidateSide,
    pub outcome: ConfiguredOutcome,
    pub trace_game_seq: Option<u64>,
    pub plies: u32,
    pub elapsed_ms: u64,
    pub candidate: ConfiguredStrategyMetrics,
    pub baseline: ConfiguredStrategyMetrics,
}

/// One configured strategy's aggregate work in a completed game.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ConfiguredStrategyMetrics {
    pub iterations_total: u64,
    pub iterations_first_half: u64,
    pub move_time_ms: u64,
}

/// A configured match outcome from the candidate's perspective.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfiguredOutcome {
    CandidateWin,
    BaselineWin,
    Draw,
}

/// Aggregate result for a completed configured comparison.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ConfiguredComparisonSummary {
    #[serde(rename = "type")]
    pub record_type: String,
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

// ---------------------------------------------------------------------------
// Tuner metadata (hyperparameter search)
// ---------------------------------------------------------------------------

/// One parameter in the tuner's search space (mirrors the shape of the tuner
/// harness's YAML search space), reported by `tuner()` so a launch form or
/// CLI consumer can render/validate fields without a per-game hardcoded
/// schema. `spec` carries the type-specific keys verbatim (`type`/`bounds`/
/// `default` for `float`/`int`, `type`/`choices`/`default` for
/// `categorical`, `type`/`default` for `bool`, or `type`/`value` for
/// `constant`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunerParameter {
    pub name: String,
    #[serde(flatten)]
    pub spec: Value,
}

/// A conditional activation rule: when `if` matches the trial's active
/// config, every name in `then` also becomes active. `if` is a single-entry
/// object mapping a parent parameter name to either one value or a list of
/// values (mirrors the YAML `if:`/`then:` shape).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunerCondition {
    #[serde(rename = "if")]
    pub if_: Value,
    pub then: Vec<String>,
}

/// Metadata describing a game's tunable strategy search space, as reported
/// by the `tune describe` subcommand -- the parameter space and baseline
/// instances a tuner-style harness needs to run trials, without embedding
/// the actual search/eval logic (that stays behind `tune_eval`).
///
/// `baselines` is a list rather than a single id so a harness can evaluate
/// each trial config against multiple opponent strengths (the tuner's
/// `Scenario(instances=...)` mechanism) instead of one fixed baseline --
/// once a config saturates 100% win rate against an easy baseline, cost
/// floors at `0.0` and a harder second instance is the only way to keep
/// ranking top candidates against each other. Most games report exactly one
/// entry here (a single preset stands in for "the" baseline); a game with a
/// genuine second preset can list it as a second instance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunerInfo {
    pub id: String,
    pub baselines: Vec<String>,
    pub eval_rounds: u32,
    pub parameters: Vec<TunerParameter>,
    pub conditions: Vec<TunerCondition>,
    /// The game's own `default_config()` -- a game-setup axis (e.g. Druid's
    /// board size) that's separate from `parameters` (the strategy search
    /// space) entirely: the tuner never searches over it, `tune_eval`'s
    /// `game_config` argument just pins every trial in a run to it. `{}` for
    /// every game whose board is fixed at compile time (everything but
    /// Druid today) -- a caller should treat that as "nothing to configure",
    /// same as `default_config()` itself already means for `new_state`.
    pub game_config: Value,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct CompareValidationField {
    pub field: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_index: Option<usize>,
}

/// Stable 53-bit SplitMix64-derived seed used by configured comparisons.
/// Inputs are a seed and a zero-based ordinal; the result is safe to carry
/// through JSON and JavaScript without losing integer precision.
pub fn derive_seed(seed: u64, ordinal: u64) -> u64 {
    let mut value = seed.wrapping_add(ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ (value >> 31)) & 9_007_199_254_740_991
}

// ---------------------------------------------------------------------------
// Opening-book metadata (Quasi-Best-First self-play)
// ---------------------------------------------------------------------------

/// Metadata describing a game's opening-book support, as reported by the
/// `book describe` subcommand -- mirrors `TunerInfo`'s shape and reasoning:
/// enough for a generic caller (a launch form, a CLI wrapper script) to
/// know book generation exists and what its default knob values are,
/// without embedding the self-play loop itself (that stays behind
/// `book_build`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BookInfo {
    pub id: String,
    /// Default number of self-play games `book_build` runs when the caller
    /// doesn't override `rounds`.
    pub default_rounds: u32,
    /// The game's own `default_config()` -- same purpose as
    /// `TunerInfo::game_config`: a game-setup axis (e.g. board size)
    /// `book_build`'s `game_config` argument pins the run to, separate from
    /// `rounds`/`seed`. `{}` for a game whose board is fixed at compile
    /// time.
    pub game_config: Value,
}
