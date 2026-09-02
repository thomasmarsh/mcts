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

#[derive(Debug, Default)]
pub struct RunTrialsQuery {
    pub limit: Option<i64>,
}

#[derive(Debug)]
pub struct RunTrial {
    pub trial_id: i64,
    pub ts: String,
    pub config: Value,
    pub seed: Option<i64>,
    pub cost: Option<f64>,
    pub extra: Option<Value>,
}

#[derive(Debug, Default)]
pub struct RunGamesQuery {
    pub limit: Option<i64>,
    pub cell_id: Option<String>,
}

#[derive(Debug)]
pub struct RunGame {
    pub game_seq: i64,
    pub match_seq: Option<i64>,
    pub cell_id: Option<String>,
    pub seed: Option<u64>,
    pub metrics: Option<Value>,
    pub ply_count: i64,
    pub started_at: String,
    pub ended_at: String,
    pub strategy_a: Option<String>,
    pub strategy_b: Option<String>,
    pub outcome: Option<String>,
    pub winner: Option<String>,
}

#[derive(Debug)]
pub struct RunGameMove {
    pub game_seq: i64,
    pub ply: i64,
    pub ts: String,
    pub state: Value,
    pub mv: Option<Value>,
    pub player: Option<String>,
    pub search: Option<Value>,
}

#[derive(Debug)]
pub struct RunDeletionInfo {
    pub status: String,
}

/// Logical storage operations over benchmark runs.
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
    fn ensure_run_exists(&self, run_id: &str) -> Result<(), RunRepositoryError>;
    fn load_trials(
        &self,
        run_id: &str,
        query: &RunTrialsQuery,
    ) -> Result<Vec<RunTrial>, RunRepositoryError>;
    fn load_games(
        &self,
        run_id: &str,
        query: &RunGamesQuery,
    ) -> Result<Vec<RunGame>, RunRepositoryError>;
    fn load_game_moves(
        &self,
        run_id: &str,
        game_seq: i64,
        after_ply: Option<i64>,
    ) -> Result<Vec<RunGameMove>, RunRepositoryError>;
    fn load_latest_game_seq(&self, run_id: &str) -> Result<Option<i64>, RunRepositoryError>;
    fn load_deletion_info(&self, run_id: &str) -> Result<RunDeletionInfo, RunRepositoryError>;
    fn delete_run_records(
        &self,
        run_id: &str,
        ingest_log_paths: &[String],
    ) -> Result<(), RunRepositoryError>;
}
