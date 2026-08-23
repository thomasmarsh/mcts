use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
pub(crate) struct TuningSessionList {
    pub(crate) schema_version: u32,
    pub(crate) sessions: Vec<TuningSessionListItem>,
}

#[derive(Serialize)]
pub(crate) struct TuningSessionListItem {
    pub(crate) session_id: String,
    pub(crate) game: Option<String>,
    pub(crate) label: Option<String>,
    pub(crate) status: String,
    pub(crate) target_trial_count: Option<i64>,
    pub(crate) counts: TuningTrialCounts,
    pub(crate) created_at: String,
    pub(crate) last_activity_at: String,
    pub(crate) attempts: Vec<TuningAttemptSummary>,
    pub(crate) capabilities: TuningCapabilities,
}

#[derive(Serialize)]
pub(crate) struct TuningAttemptSummary {
    pub(crate) attempt_id: String,
    pub(crate) bench_run_id: Option<String>,
    pub(crate) status: String,
    pub(crate) started_at: String,
    pub(crate) ended_at: Option<String>,
    pub(crate) failure: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TuningSessionSummary {
    pub(crate) session_id: String,
    pub(crate) status: String,
    pub(crate) target_trial_count: Option<i64>,
    pub(crate) counts: TuningTrialCounts,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialCounts {
    pub(crate) total: i64,
    pub(crate) queued: i64,
    pub(crate) running: i64,
    pub(crate) terminal: i64,
    pub(crate) completed: i64,
    pub(crate) failed: i64,
    pub(crate) pruned: i64,
    pub(crate) cancelled: i64,
}

#[derive(Serialize)]
pub(crate) struct TuningAttemptView {
    pub(crate) attempt_id: String,
    pub(crate) bench_run_id: Option<String>,
    pub(crate) status: String,
    pub(crate) started_at: String,
    pub(crate) ended_at: Option<String>,
    pub(crate) failure: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialView {
    pub(crate) trial_id: String,
    pub(crate) trial_number: i64,
    pub(crate) attempt_id: String,
    pub(crate) status: String,
    pub(crate) config: Option<Value>,
    pub(crate) score: Option<f64>,
    pub(crate) mu: Option<f64>,
    pub(crate) sigma: Option<f64>,
    pub(crate) failure: Option<String>,
    pub(crate) pairs: Vec<TuningPairView>,
}

#[derive(Serialize)]
pub(crate) struct TuningOpponentView {
    pub(crate) anchor_id: String,
    pub(crate) config: Value,
    pub(crate) mu: f64,
    pub(crate) sigma: f64,
    pub(crate) label: Option<String>,
    pub(crate) provenance: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TuningRatingView {
    pub(crate) mu: f64,
    pub(crate) sigma: f64,
}

#[derive(Serialize)]
pub(crate) struct TuningStrategyMetricsView {
    pub(crate) iterations_total: u64,
    pub(crate) iterations_first_half: u64,
    pub(crate) move_time_ms: u64,
}

#[derive(Serialize)]
pub(crate) struct TuningGameView {
    pub(crate) game_id: String,
    pub(crate) candidate_side: String,
    pub(crate) outcome: String,
    pub(crate) seed: u64,
    pub(crate) round: u32,
    pub(crate) trace_game_seq: Option<u64>,
    pub(crate) plies: u32,
    pub(crate) elapsed_ms: u64,
    pub(crate) candidate: TuningStrategyMetricsView,
    pub(crate) baseline: TuningStrategyMetricsView,
}

#[derive(Serialize)]
pub(crate) struct TuningPairView {
    pub(crate) pair_id: String,
    pub(crate) pair_index: u32,
    pub(crate) status: String,
    pub(crate) seed: u64,
    pub(crate) round: u32,
    pub(crate) opponent: TuningOpponentView,
    pub(crate) pool_snapshot_fingerprint: String,
    pub(crate) rating_before: TuningRatingView,
    pub(crate) rating_after: Option<TuningRatingView>,
    pub(crate) score: Option<f64>,
    pub(crate) failure: Option<String>,
    pub(crate) games: Vec<TuningGameView>,
}

#[derive(Serialize)]
pub(crate) struct TuningCapabilities {
    pub(crate) has_lifecycle: bool,
    pub(crate) has_pairs: bool,
    pub(crate) has_renderer_trace: bool,
    pub(crate) has_search_reports: bool,
}

#[derive(Serialize)]
pub(crate) struct TuningCursorBoundary {
    pub(crate) session_sequence: i64,
}

#[derive(Serialize)]
pub(crate) struct TuningSessionDetail {
    pub(crate) schema_version: u32,
    pub(crate) summary: TuningSessionSummary,
    pub(crate) attempts: Vec<TuningAttemptView>,
    pub(crate) trials: Vec<TuningTrialView>,
    pub(crate) manifest: Value,
    pub(crate) fingerprint: Option<String>,
    pub(crate) capabilities: TuningCapabilities,
    pub(crate) cursor: TuningCursorBoundary,
}
