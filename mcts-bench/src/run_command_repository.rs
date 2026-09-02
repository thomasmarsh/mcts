//! Durable command-side mutations for physical benchmark runs.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunCommandRepositoryError {
    NotFound,
    ContradictoryIdentity,
    Conflict,
    Storage(String),
}

impl std::fmt::Display for RunCommandRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "benchmark run was not found"),
            Self::ContradictoryIdentity => write!(f, "benchmark run identity is contradictory"),
            Self::Conflict => write!(f, "benchmark run is assigned to a different process"),
            Self::Storage(message) => write!(f, "benchmark run storage failure: {message}"),
        }
    }
}
impl std::error::Error for RunCommandRepositoryError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuationParent {
    pub logical_run_id: String,
    pub parent_attempt_id: String,
    pub attempt_ordinal: u64,
}

#[derive(Debug, Clone)]
pub struct RecordRunLaunch {
    pub run_id: String,
    pub kind: String,
    pub game: String,
    pub label: Option<String>,
    pub config_json: Option<String>,
    pub git_sha: String,
    pub git_dirty: bool,
    pub host: String,
    pub pid: i64,
    pub started_at: String,
    pub log_path: String,
    pub continuation_parent: Option<ContinuationParent>,
}

pub trait RunCommandRepository {
    fn prepare_continuation(
        &self,
        parent_attempt_id: &str,
    ) -> Result<ContinuationParent, RunCommandRepositoryError>;
    fn record_launch(&self, launch: RecordRunLaunch) -> Result<(), RunCommandRepositoryError>;
    fn backfill_config(
        &self,
        run_id: &str,
        config_json: &str,
    ) -> Result<(), RunCommandRepositoryError>;
    fn mark_crashed(&self, run_id: &str, ended_at: &str) -> Result<(), RunCommandRepositoryError>;
    fn project_legacy_stop(
        &self,
        run_id: &str,
        kind: &str,
        ended_at: &str,
    ) -> Result<(), RunCommandRepositoryError>;
}
