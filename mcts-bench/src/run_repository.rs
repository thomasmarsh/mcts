//! Logical reads over benchmark runs.
//!
//! Callers depend on this interface instead of a particular database driver,
//! so they can use an in-memory implementation or a test double.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunRepositoryError {
    NotFound,
    Storage(String),
}

impl std::fmt::Display for RunRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "benchmark run was not found"),
            Self::Storage(message) => write!(f, "benchmark run storage failure: {message}"),
        }
    }
}

impl std::error::Error for RunRepositoryError {}

#[derive(Debug, Default)]
pub struct RunListQuery {
    pub game: Option<String>,
    pub experiment_id: Option<String>,
    pub project_id: Option<String>,
}

#[derive(Debug)]
pub struct RunSummary {
    pub run_id: String,
    pub kind: String,
    pub game: Option<String>,
    pub project_id: Option<String>,
    pub experiment_id: Option<String>,
    pub label: Option<String>,
    pub git_sha: String,
    pub git_dirty: bool,
    pub host: String,
    pub pid: Option<i64>,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub status: String,
    pub match_count: i64,
    pub trial_count: i64,
    pub tuning_session_id: Option<String>,
}

#[derive(Debug)]
pub struct RunDetail {
    pub run_id: String,
    pub kind: String,
    pub game: Option<String>,
    pub project_id: Option<String>,
    pub experiment_id: Option<String>,
    pub experiment_spec: Option<Value>,
    pub label: Option<String>,
    pub config: Option<Value>,
    pub git_sha: String,
    pub git_dirty: bool,
    pub host: String,
    pub pid: Option<i64>,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub status: String,
    pub log_path: String,
    pub exit_code: Option<i64>,
    pub match_count: i64,
    pub trial_count: i64,
    pub tuning_session_id: Option<String>,
    pub incumbent: Option<RunIncumbent>,
}

#[derive(Debug)]
pub struct RunIncumbent {
    pub config: Value,
    pub cost: f64,
}

#[derive(Debug, Default)]
pub struct LeaderboardQuery {
    pub game: Option<String>,
    pub git_sha: Option<String>,
    pub since: Option<String>,
}

#[derive(Debug)]
pub struct LeaderboardRow {
    pub strategy: String,
    pub total: i64,
    pub wins: i64,
    pub losses: i64,
    pub draws: i64,
}

#[derive(Debug)]
pub struct ExperimentCell {
    pub cell_id: String,
    pub cell_seed: Option<u64>,
    pub game: String,
    pub game_config: Value,
    pub variant_id: String,
    pub variant_label: String,
    pub candidate_config: Value,
    pub baseline_id: String,
    pub baseline_label: String,
    pub baseline_config: Value,
    pub budget: Value,
    pub rounds: i64,
    pub planned_games: u64,
    pub completed_games: u64,
    pub status: String,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
    pub error: Option<String>,
    pub match_outcomes: Vec<Option<String>>,
}

/// Logical read operations over benchmark runs.
///
/// All arguments and results are ordinary application data. Implementations
/// may use DuckDB, a different durable store, or a test double.
pub trait RunRepository {
    fn load_log_path(&self, run_id: &str) -> Result<String, RunRepositoryError>;
    fn list_runs(&self, query: &RunListQuery) -> Result<Vec<RunSummary>, RunRepositoryError>;
    fn load_run(&self, run_id: &str) -> Result<RunDetail, RunRepositoryError>;
    fn load_leaderboard(
        &self,
        query: &LeaderboardQuery,
    ) -> Result<Vec<LeaderboardRow>, RunRepositoryError>;
    fn load_experiment_cells(
        &self,
        run_id: &str,
    ) -> Result<Vec<ExperimentCell>, RunRepositoryError>;
}
