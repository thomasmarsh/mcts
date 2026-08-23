use serde::Serialize;
use serde_json::Value;

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
