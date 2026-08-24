//! Logical reads for tuning-session pages and details.
//!
//! Callers consume these records without depending on a database driver.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningSessionRepositoryError {
    Storage(String),
    InvalidData(String),
}

impl std::fmt::Display for TuningSessionRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(message) => write!(f, "tuning session storage failure: {message}"),
            Self::InvalidData(message) => {
                write!(f, "invalid persisted tuning session data: {message}")
            }
        }
    }
}

impl std::error::Error for TuningSessionRepositoryError {}

#[derive(Debug, Clone)]
pub struct TuningSessionListData {
    pub sessions: Vec<TuningSessionListRow>,
    pub attempts: Vec<TuningSessionAttemptRow>,
}

#[derive(Debug, Clone)]
pub struct TuningSessionListRow {
    pub session_id: String,
    pub status: String,
    pub target_trial_count: Option<i64>,
    pub manifest: String,
    pub created_at: String,
    pub last_activity_at: String,
    pub trial_counts: TuningTrialCountsRow,
    pub pair_count: i64,
    pub renderer_trace_count: i64,
    pub search_report_count: i64,
    pub trial_report_count: i64,
    pub control: crate::tuning_command_store::SessionControl,
}

#[derive(Debug, Clone)]
pub struct TuningTrialCountsRow {
    pub total: i64,
    pub queued: i64,
    pub running: i64,
    pub terminal: i64,
    pub completed: i64,
    pub failed: i64,
    pub pruned: i64,
    pub cancelled: i64,
}

#[derive(Debug, Clone)]
pub struct TuningSessionAttemptRow {
    pub session_id: String,
    pub attempt_id: String,
    pub bench_run_id: Option<String>,
    pub status: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub failure: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TuningSessionDetailData {
    pub session: TuningSessionRow,
    pub trial_counts: TuningTrialCountsRow,
    pub attempts: Vec<TuningSessionAttemptRow>,
    pub trials: Vec<TuningSessionTrialRow>,
    pub reports: Vec<TuningSessionTrialReportRow>,
    pub pairs: Vec<TuningSessionPairRow>,
    pub games: Vec<TuningSessionGameRow>,
    pub capabilities: TuningSessionCapabilities,
    pub control: crate::tuning_command_store::SessionControl,
}

#[derive(Debug, Clone)]
pub struct TuningSessionRow {
    pub session_id: String,
    pub status: String,
    pub target_trial_count: Option<i64>,
    pub manifest: String,
    pub fingerprint: Option<String>,
    pub last_sequence: i64,
}

#[derive(Debug, Clone)]
pub struct TuningSessionTrialRow {
    pub trial_id: String,
    pub trial_number: i64,
    pub attempt_id: String,
    pub status: String,
    pub config: Option<String>,
    pub score: Option<f64>,
    pub mu: Option<f64>,
    pub sigma: Option<f64>,
    pub stop_reason: Option<String>,
    pub failure: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TuningSessionTrialReportRow {
    pub trial_id: String,
    pub completed_pairs: u64,
    pub reported_at: String,
    pub mu: f64,
    pub sigma: f64,
    pub score: f64,
    pub score_formula_version: u32,
    pub conservative_k: f64,
    pub outcome: String,
    pub reason: String,
    pub pruning_exempt: bool,
    pub bracket_id: Option<String>,
    pub rung_resource: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TuningSessionPairRow {
    pub trial_id: String,
    pub pair_id: String,
    pub pair_index: u32,
    pub status: String,
    pub seed: u64,
    pub round: u32,
    pub opponent: String,
    pub pool_snapshot_fingerprint: String,
    pub rating_before_mu: f64,
    pub rating_before_sigma: f64,
    pub rating_after_mu: Option<f64>,
    pub rating_after_sigma: Option<f64>,
    pub score: Option<f64>,
    pub failure: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TuningSessionGameRow {
    pub pair_id: String,
    pub game_id: String,
    pub candidate_side: String,
    pub outcome: String,
    pub seed: u64,
    pub round: u32,
    pub trace_game_seq: Option<u64>,
    pub plies: u32,
    pub elapsed_ms: u64,
    pub candidate_metrics: String,
    pub baseline_metrics: String,
}

#[derive(Debug, Clone)]
pub struct TuningSessionCapabilities {
    pub pair_count: i64,
    pub renderer_trace_count: i64,
    pub search_report_count: i64,
    pub trial_report_count: i64,
}

/// Logical storage operations needed to render tuning-session routes.
pub trait TuningSessionRepository {
    fn load_session_list(&self) -> Result<TuningSessionListData, TuningSessionRepositoryError>;

    fn load_session_detail(
        &self,
        session_id: &str,
    ) -> Result<Option<TuningSessionDetailData>, TuningSessionRepositoryError>;

    fn load_session_control(
        &self,
        session_id: &str,
    ) -> Result<Option<crate::tuning_command_store::SessionControl>, TuningSessionRepositoryError>;
}
