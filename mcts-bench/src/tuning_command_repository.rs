//! Durable command operations needed while delivering tuning-session commands.
//!
//! The command routes use these application records instead of issuing SQL
//! themselves, so a different durable store or a test double can provide the
//! same continuation and replay facts.

pub use crate::tuning_command_store::{
    CommandDecision as TuningCommandDecision, CommandKind as TuningCommandKind,
    CommandRequest as TuningCommandRequest, DenialReason as TuningCommandDenialReason,
    LaunchOutcome as TuningLaunchOutcome, LaunchReservation as TuningLaunchReservation,
    SessionCommand as TuningSessionCommand, SessionControl as TuningSessionControl,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningCommandRepositoryError {
    NotFound(String),
    CommandIdReuseMismatch {
        command_id: String,
    },
    ExpectedVersionConflict {
        expected: u64,
        control: Box<TuningSessionControl>,
    },
    ActiveAttempt {
        attempt_id: String,
        control: Box<TuningSessionControl>,
    },
    LaunchReserved {
        attempt_id: String,
        control: Box<TuningSessionControl>,
    },
    InvalidDeltaStart {
        control: Box<TuningSessionControl>,
    },
    ExhaustedResume {
        control: Box<TuningSessionControl>,
    },
    NoncontinuableLegacy {
        control: Box<TuningSessionControl>,
    },
    CommandDenied {
        reason: TuningCommandDenialReason,
        control: Box<TuningSessionControl>,
    },
    TargetOverflow {
        control: Box<TuningSessionControl>,
    },
    MissingReservation {
        command_id: String,
    },
    Storage(String),
}

impl std::fmt::Display for TuningCommandRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(session_id) => write!(f, "tuning session {session_id} was not found"),
            Self::CommandIdReuseMismatch { command_id } => {
                write!(f, "command id {command_id} was reused with different input")
            }
            Self::ExpectedVersionConflict { expected, control } => write!(
                f,
                "expected control version {expected}, found {}",
                control.control_version
            ),
            Self::ActiveAttempt { attempt_id, .. } => write!(f, "attempt {attempt_id} is active"),
            Self::LaunchReserved { attempt_id, .. } => {
                write!(f, "attempt {attempt_id} is already reserved for launch")
            }
            Self::InvalidDeltaStart { .. } => write!(f, "invalid budget delta/start combination"),
            Self::ExhaustedResume { .. } => write!(f, "session budget is exhausted"),
            Self::NoncontinuableLegacy { .. } => write!(f, "legacy session cannot be continued"),
            Self::CommandDenied { reason, .. } => write!(f, "command denied: {reason:?}"),
            Self::TargetOverflow { .. } => write!(f, "target trial count overflow"),
            Self::MissingReservation { command_id } => {
                write!(
                    f,
                    "launch reservation for command {command_id} was not found"
                )
            }
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
    fn apply_command(
        &self,
        session_id: &str,
        request: &TuningCommandRequest,
    ) -> Result<TuningCommandDecision, TuningCommandRepositoryError>;

    fn record_launch_outcome(
        &self,
        session_id: &str,
        command_id: &str,
        outcome: TuningLaunchOutcome,
    ) -> Result<TuningSessionControl, TuningCommandRepositoryError>;

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
