//! Logical reads needed while delivering tuning-session commands.
//!
//! The command routes use these application records instead of issuing SQL
//! themselves, so a different durable store or a test double can provide the
//! same continuation and replay facts.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningCommandRepositoryError {
    Storage(String),
}

impl std::fmt::Display for TuningCommandRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(message) => write!(f, "tuning command storage failure: {message}"),
        }
    }
}

impl std::error::Error for TuningCommandRepositoryError {}

#[derive(Debug, Clone)]
pub struct StoredTuningCommand {
    pub session_id: String,
    pub request_json: String,
}

#[derive(Debug, Clone)]
pub struct TuningContinuationMetadata {
    pub game: String,
    pub config_json: Option<String>,
    pub optimizer_id: String,
    pub lifecycle_path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuningCommandReplayState {
    Launched,
    Reserved,
    Failed,
}

/// Logical reads needed to apply and deliver a tuning-session command.
pub trait TuningCommandRepository {
    fn load_command(
        &self,
        command_id: &str,
    ) -> Result<Option<StoredTuningCommand>, TuningCommandRepositoryError>;

    fn replay_state(
        &self,
        session_id: &str,
        command_id: &str,
        attempt_id: &str,
        physical_run_id: &str,
    ) -> Result<TuningCommandReplayState, TuningCommandRepositoryError>;

    fn load_continuation_metadata(
        &self,
        session_id: &str,
    ) -> Result<Option<TuningContinuationMetadata>, TuningCommandRepositoryError>;

    fn load_attempt_pid(
        &self,
        session_id: &str,
        attempt_id: &str,
    ) -> Result<Option<i64>, TuningCommandRepositoryError>;
}
