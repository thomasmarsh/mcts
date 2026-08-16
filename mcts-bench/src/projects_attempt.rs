//! Backend-neutral protocol and compatibility rules for typed Projects attempts.

use crate::orchestration::{AttemptPhase, AttemptState, ExitObservation};

pub const START_REQUESTED_KEY: &str = "projects.start-requested";
pub const PROCESS_OBSERVED_KEY: &str = "projects.process-observed";
pub const SPAWN_FAILED_KEY: &str = "projects.spawn-failed";
pub const OPERATOR_STOP_REQUESTED_KEY: &str = "projects.operator-stop-requested";
pub const SIGNAL_OBSERVED_KEY: &str = "projects.signal-observed";
pub const EXIT_OBSERVED_KEY: &str = "projects.exit-observed";
pub const FINAL_OUTPUT_INGESTED_KEY: &str = "projects.final-output-ingested";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectsError {
    NotFound,
    Conflict(String),
    Corrupt(String),
    Storage(String),
}

impl std::fmt::Display for ProjectsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "typed Projects attempt was not found"),
            Self::Conflict(message) => write!(f, "typed Projects conflict: {message}"),
            Self::Corrupt(message) => write!(f, "corrupt typed Projects attempt: {message}"),
            Self::Storage(message) => write!(f, "typed Projects storage failure: {message}"),
        }
    }
}

impl std::error::Error for ProjectsError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Receipt {
    pub state: AttemptState,
    pub version: u64,
    pub replay: bool,
}

impl Receipt {
    pub fn needs_final_output(self) -> bool {
        self.state.phase() == AttemptPhase::Finalizing
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopAuthorization {
    pub signal_process_group: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopTarget {
    pub pid: Option<i64>,
    pub status: String,
    pub kind: String,
    pub typed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivenessTarget {
    pub run_id: String,
    pub pid: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExitAuthorization {
    pub finalize_output: bool,
    pub state: AttemptState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellSummary {
    pub failed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityStatus {
    Completed,
    CompletedWithErrors,
    Stopped,
    Crashed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompatibilityOutcome {
    pub status: CompatibilityStatus,
    pub crash_message: Option<&'static str>,
}

#[must_use]
pub fn compatibility_outcome(
    state: AttemptState,
    cells: CellSummary,
) -> Option<CompatibilityOutcome> {
    match state.phase() {
        AttemptPhase::Completed => Some(CompatibilityOutcome {
            status: if cells.failed == 0 {
                CompatibilityStatus::Completed
            } else {
                CompatibilityStatus::CompletedWithErrors
            },
            crash_message: None,
        }),
        AttemptPhase::Stopped => Some(CompatibilityOutcome {
            status: CompatibilityStatus::Stopped,
            crash_message: None,
        }),
        AttemptPhase::Crashed => Some(CompatibilityOutcome {
            status: CompatibilityStatus::Crashed,
            crash_message: Some(match state.exit_observation() {
                Some(ExitObservation::Lost) => "coordinator disappeared",
                Some(ExitObservation::Exited { .. }) | None => "coordinator exited",
            }),
        }),
        _ => None,
    }
}

pub trait ProjectsRepository {
    fn load_stop_target(&self, run_id: &str) -> Result<StopTarget, ProjectsError>;
    fn load_if_initialized(&self, run_id: &str) -> Result<Option<Receipt>, ProjectsError>;
    fn typed_liveness_targets(&self) -> Result<Vec<LivenessTarget>, ProjectsError>;
    fn create_and_request_start(&self, request: &StartRequest) -> Result<(), ProjectsError>;
    fn observe_process(
        &self,
        run_id: &str,
        pid: i64,
        log_path: &str,
        observed_at: &str,
    ) -> Result<Receipt, ProjectsError>;
    fn observe_spawn_failure(
        &self,
        run_id: &str,
        message: &str,
        observed_at: &str,
    ) -> Result<Receipt, ProjectsError>;
    fn request_operator_stop(
        &self,
        run_id: &str,
        observed_at: &str,
    ) -> Result<StopAuthorization, ProjectsError>;
    fn observe_signal(&self, run_id: &str, observed_at: &str) -> Result<Receipt, ProjectsError>;
    fn observe_exit(
        &self,
        run_id: &str,
        exit: ExitObservation,
        ended_at: &str,
    ) -> Result<ExitAuthorization, ProjectsError>;
    fn finalize_output(&self, run_id: &str, observed_at: &str) -> Result<Receipt, ProjectsError>;
    fn project_stop(&self, run_id: &str, ended_at: &str) -> Result<(), ProjectsError>;
}

#[derive(Debug, Clone)]
pub struct StartRequest {
    pub run_id: String,
    pub game: Option<String>,
    pub project_id: String,
    pub experiment_id: String,
    pub spec_json: String,
    pub label: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub host: String,
    pub started_at: String,
    pub log_path: String,
    pub cells: Vec<CellRequest>,
}

#[derive(Debug, Clone)]
pub struct CellRequest {
    pub cell_id: String,
    pub cell_seed: u64,
    pub game: String,
    pub game_config: String,
    pub variant_id: String,
    pub variant_label: String,
    pub candidate_config: String,
    pub baseline_id: String,
    pub baseline_label: String,
    pub baseline_config: String,
    pub budget: String,
    pub rounds: u32,
    pub planned_games: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::{transition_attempt, AttemptEvent, AttemptState};

    #[test]
    fn compatibility_is_derived_from_typed_state_and_cells() {
        let state = AttemptState::planned();
        assert!(compatibility_outcome(state, CellSummary { failed: 0 }).is_none());
    }

    #[test]
    fn crashed_messages_preserve_exit_evidence() {
        let state = transition_attempt(&AttemptState::planned(), AttemptEvent::StartRequested)
            .unwrap()
            .state()
            .to_owned();
        let state = transition_attempt(&state, AttemptEvent::SpawnFailed)
            .unwrap()
            .state()
            .to_owned();
        let outcome = compatibility_outcome(state, CellSummary { failed: 0 }).unwrap();
        assert_eq!(outcome.status, CompatibilityStatus::Crashed);
        assert_eq!(outcome.crash_message, Some("coordinator exited"));

        let state = transition_attempt(&AttemptState::planned(), AttemptEvent::StartRequested)
            .unwrap()
            .state()
            .to_owned();
        let state = transition_attempt(&state, AttemptEvent::ProcessObserved)
            .unwrap()
            .state()
            .to_owned();
        let state = transition_attempt(
            &state,
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Lost,
            },
        )
        .unwrap()
        .state()
        .to_owned();
        let state = transition_attempt(&state, AttemptEvent::FinalOutputIngested)
            .unwrap()
            .state()
            .to_owned();
        let outcome = compatibility_outcome(state, CellSummary { failed: 0 }).unwrap();
        assert_eq!(outcome.status, CompatibilityStatus::Crashed);
        assert_eq!(outcome.crash_message, Some("coordinator disappeared"));
    }
}
