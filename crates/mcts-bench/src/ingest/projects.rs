use duckdb::Connection;

use crate::launch::iso_timestamp;

use crate::projects_attempt::{self, ProjectsRepository};

use crate::projects_attempt_duckdb;

use crate::supervised_launch::{classify_observation, ObservationDecision};

use super::IngestError;

pub(super) fn observe(conn: &Connection) -> Result<(), IngestError> {
    let repo = projects_attempt_duckdb::Repository::new(conn);
    let mut first_error = None;
    for target in repo.observation_targets()? {
        let decision = classify_observation(
            &target,
            crate::lifecycle::read_journal(&target.journal_path),
        );
        match decision {
            ObservationDecision::Pending => {}
            ObservationDecision::Terminal(exit) => {
                let exit = match exit {
                    crate::lifecycle::ExitEvidence::Code { code } => {
                        crate::orchestration::ExitObservation::Exited { code: Some(code) }
                    }
                    crate::lifecycle::ExitEvidence::Signal { signal } => {
                        crate::orchestration::ExitObservation::Signaled { signal }
                    }
                    crate::lifecycle::ExitEvidence::WaitFailed { .. } => {
                        crate::orchestration::ExitObservation::Unavailable
                    }
                };
                if let Err(error) = repo.observe_exit(&target.attempt_id, exit, &iso_timestamp()) {
                    first_error.get_or_insert(IngestError::Attempt(error));
                }
            }
            ObservationDecision::Invalid(reason) => {
                first_error.get_or_insert(IngestError::Attempt(
                    projects_attempt::ProjectsError::Conflict(format!(
                        "invalid lifecycle observation: {reason:?}"
                    )),
                ));
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}
