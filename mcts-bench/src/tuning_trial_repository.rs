//! Logical reads for tuning-trial pages and details.
//!
//! Callers consume these records without depending on a database driver.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningTrialRepositoryError {
    Storage(String),
    InvalidData(String),
}

impl std::fmt::Display for TuningTrialRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(message) => write!(f, "tuning trial storage failure: {message}"),
            Self::InvalidData(message) => {
                write!(f, "invalid persisted tuning trial data: {message}")
            }
        }
    }
}

impl std::error::Error for TuningTrialRepositoryError {}

#[derive(Debug, Clone)]
pub struct TuningTrialPageData {
    pub session_sequence: i64,
    pub trials: Vec<TuningTrialPageRow>,
}

#[derive(Debug, Clone)]
pub struct TuningTrialPageRow {
    pub trial_id: String,
    pub trial_number: i64,
    pub attempt_id: String,
    pub state: String,
    pub config: Option<String>,
    pub score: Option<f64>,
    pub mu: Option<f64>,
    pub sigma: Option<f64>,
    pub stop_reason: Option<String>,
    pub last_reason: Option<String>,
    pub bracket_id: Option<String>,
    pub resource: Option<u64>,
    pub pair_count: u64,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub elapsed_ms: u64,
    pub search_iterations_total: u64,
    pub search_move_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct TuningTrialDetailData {
    pub session_sequence: i64,
    pub trial: TuningTrialDetailRow,
    pub reports: Vec<TuningTrialReportRow>,
    pub pairs: Vec<TuningTrialPairRow>,
}

#[derive(Debug, Clone)]
pub struct TuningTrialDetailRow {
    pub trial_id: String,
    pub trial_number: i64,
    pub attempt_id: String,
    pub state: String,
    pub config: Option<String>,
    pub score: Option<f64>,
    pub mu: Option<f64>,
    pub sigma: Option<f64>,
    pub stop_reason: Option<String>,
    pub failure: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TuningTrialReportRow {
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
pub struct TuningTrialPairRow {
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
    pub games: Vec<TuningTrialGameRow>,
}

#[derive(Debug, Clone)]
pub struct TuningTrialGameRow {
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
    pub run_id: Option<String>,
    pub has_renderer_trace: bool,
    pub has_search_reports: bool,
}

/// Logical storage operations needed to render tuning-trial routes.
pub trait TuningTrialRepository {
    fn load_trial_page(
        &self,
        session_id: &str,
    ) -> Result<Option<TuningTrialPageData>, TuningTrialRepositoryError>;

    fn load_trial_detail(
        &self,
        session_id: &str,
        trial_id: &str,
    ) -> Result<Option<TuningTrialDetailData>, TuningTrialRepositoryError>;
}
