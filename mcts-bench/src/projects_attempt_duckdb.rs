//! DuckDB repository for the typed Projects attempt protocol.

use duckdb::{params, Connection, Transaction};
use std::sync::Mutex;

use crate::attempt_store::{self, AttemptStoreError};
use crate::identity;
use crate::orchestration::{
    AttemptAction, AttemptEvent, AttemptState, ExitObservation, StopReason,
};
use crate::projects_attempt::{
    self, CellSummary, CompatibilityStatus, ExitAuthorization, LaunchRecord, LaunchResult,
    LivenessTarget, ProjectsError, ProjectsRepository, Receipt, StartAuthorization, StartRequest,
    StopAuthorization, StopTarget,
};
use crate::supervised_launch::LaunchDescriptor;

mod launch;

pub struct Repository<'a> {
    conn: &'a Connection,
}

impl From<duckdb::Error> for ProjectsError {
    fn from(error: duckdb::Error) -> Self {
        ProjectsError::Storage(error.to_string())
    }
}

impl<'a> Repository<'a> {
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    fn tx(&self) -> Result<Transaction<'_>, ProjectsError> {
        self.conn
            .unchecked_transaction()
            .map_err(|error| ProjectsError::Storage(error.to_string()))
    }
}

fn store_error(error: AttemptStoreError) -> ProjectsError {
    match error {
        AttemptStoreError::MissingAttempt(_) | AttemptStoreError::Uninitialized(_) => {
            ProjectsError::NotFound
        }
        AttemptStoreError::StaleVersion { .. }
        | AttemptStoreError::EventKeyConflict { .. }
        | AttemptStoreError::Transition(_) => ProjectsError::Conflict(error.to_string()),
        AttemptStoreError::Corrupt { .. } | AttemptStoreError::MissingIdentity(_) => {
            ProjectsError::Corrupt(error.to_string())
        }
        AttemptStoreError::DuckDb(_) => ProjectsError::Storage(error.to_string()),
    }
}

fn db_error(error: duckdb::Error) -> ProjectsError {
    ProjectsError::Storage(error.to_string())
}

fn identity_error(error: identity::IdentityError) -> ProjectsError {
    ProjectsError::Storage(error.to_string())
}

fn record(
    tx: &Transaction<'_>,
    run_id: &str,
    key: &str,
    event: AttemptEvent,
    observed_at: &str,
    expected: &[AttemptAction],
) -> Result<Receipt, ProjectsError> {
    let current = attempt_store::load_attempt(tx, run_id).map_err(store_error)?;
    let transition =
        attempt_store::record_attempt_event(tx, run_id, current.version(), key, event, observed_at)
            .map_err(store_error)?;
    if !transition.is_replay() && transition.actions() != expected {
        return Err(ProjectsError::Corrupt(format!(
            "event {key} proposed an unexpected action"
        )));
    }
    Ok(Receipt {
        state: *transition.state(),
        version: transition.version(),
        replay: transition.is_replay(),
    })
}

fn terminal_projection(
    tx: &Transaction<'_>,
    run_id: &str,
    state: AttemptState,
    cells: CellSummary,
    ended_at: &str,
) -> Result<(), ProjectsError> {
    let Some(outcome) = projects_attempt::compatibility_outcome(state, cells) else {
        return Err(ProjectsError::Corrupt(
            "terminal output has nonterminal state".into(),
        ));
    };
    let exit_code = match state.exit_observation() {
        Some(ExitObservation::Exited { code }) => code.map(i64::from),
        Some(ExitObservation::Lost) | None => None,
    };
    match outcome.status {
        CompatibilityStatus::Completed | CompatibilityStatus::CompletedWithErrors => {}
        CompatibilityStatus::Stopped => {
            tx.execute(
                "UPDATE experiment_cells SET status = 'cancelled', ended_at = COALESCE(ended_at, ?1), error = COALESCE(error, 'run stopped') WHERE run_id = ?2 AND status IN ('pending', 'running')",
                params![ended_at, run_id],
            )?;
        }
        CompatibilityStatus::Crashed => {
            let message = outcome.crash_message.unwrap_or("coordinator exited");
            tx.execute(
                "UPDATE experiment_cells SET status = 'failed', error = COALESCE(error, ?1), ended_at = COALESCE(ended_at, ?2) WHERE run_id = ?3 AND status = 'running'",
                params![message, ended_at, run_id],
            )?;
            tx.execute(
                "UPDATE experiment_cells SET status = 'cancelled', error = COALESCE(error, ?1), ended_at = COALESCE(ended_at, ?2) WHERE run_id = ?3 AND status = 'pending'",
                params![message, ended_at, run_id],
            )?;
        }
    }
    let status = match outcome.status {
        CompatibilityStatus::Completed => "completed",
        CompatibilityStatus::CompletedWithErrors => "completed_with_errors",
        CompatibilityStatus::Stopped => "stopped",
        CompatibilityStatus::Crashed => "crashed",
    };
    tx.execute(
        "UPDATE runs SET status = ?1, ended_at = COALESCE(ended_at, ?2), exit_code = ?3 WHERE run_id = ?4",
        params![status, ended_at, exit_code, run_id],
    )?;
    Ok(())
}

impl ProjectsRepository for Repository<'_> {
    fn authorize_start(
        &self,
        request: &StartRequest,
        descriptor: &LaunchDescriptor,
    ) -> Result<StartAuthorization, ProjectsError> {
        launch::authorize_start(self, request, descriptor)
    }

    fn record_launch(
        &self,
        run_id: &str,
        result: &LaunchResult,
        observed_at: &str,
    ) -> Result<LaunchRecord, ProjectsError> {
        launch::record_launch(self, run_id, result, observed_at)
    }

    fn load_stop_target(&self, run_id: &str) -> Result<StopTarget, ProjectsError> {
        let tx = self.tx()?;
        let target = tx
            .query_row(
                "SELECT pid, status, kind, attempt_phase FROM runs WHERE run_id = ?1",
                params![run_id],
                |row| {
                    Ok(StopTarget {
                        pid: row.get(0)?,
                        status: row.get(1)?,
                        kind: row.get(2)?,
                        typed: row.get::<_, String>(2)? == "experiment"
                            && row.get::<_, Option<String>>(3)?.is_some(),
                    })
                },
            )
            .map_err(|error| match error {
                duckdb::Error::QueryReturnedNoRows => ProjectsError::NotFound,
                other => db_error(other),
            })?;
        tx.commit().map_err(db_error)?;
        Ok(target)
    }

    fn load_if_initialized(&self, run_id: &str) -> Result<Option<Receipt>, ProjectsError> {
        let tx = self.tx()?;
        let phase: Option<String> = match tx.query_row(
            "SELECT attempt_phase FROM runs WHERE run_id = ?1 AND kind = 'experiment'",
            params![run_id],
            |row| row.get(0),
        ) {
            Ok(phase) => phase,
            Err(duckdb::Error::QueryReturnedNoRows) => {
                tx.commit().map_err(db_error)?;
                return Ok(None);
            }
            Err(error) => return Err(db_error(error)),
        };
        if phase.is_none() {
            tx.commit().map_err(db_error)?;
            return Ok(None);
        }
        let snapshot = attempt_store::load_attempt(&tx, run_id).map_err(store_error)?;
        tx.commit().map_err(db_error)?;
        Ok(Some(Receipt {
            state: *snapshot.state(),
            version: snapshot.version(),
            replay: false,
        }))
    }

    fn typed_liveness_targets(&self) -> Result<Vec<LivenessTarget>, ProjectsError> {
        let tx = self.tx()?;
        let mut statement = tx
            .prepare(
                "SELECT run_id, pid, attempt_phase FROM runs
                 WHERE kind = 'experiment' AND pid IS NOT NULL",
            )
            .map_err(db_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(db_error)?;
        let mut targets = Vec::new();
        for row in rows {
            let (run_id, pid, phase) = row.map_err(db_error)?;
            if phase.is_none() {
                continue;
            }
            let snapshot = attempt_store::load_attempt(&tx, &run_id).map_err(store_error)?;
            if matches!(
                snapshot.state().phase(),
                crate::orchestration::AttemptPhase::Running
                    | crate::orchestration::AttemptPhase::StopRequested
                    | crate::orchestration::AttemptPhase::AwaitingExit
            ) {
                targets.push(LivenessTarget { run_id, pid });
            }
        }
        drop(statement);
        tx.commit().map_err(db_error)?;
        Ok(targets)
    }

    fn request_operator_stop(
        &self,
        run_id: &str,
        observed_at: &str,
    ) -> Result<StopAuthorization, ProjectsError> {
        let tx = self.tx()?;
        let receipt = record(
            &tx,
            run_id,
            projects_attempt::OPERATOR_STOP_REQUESTED_KEY,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
            observed_at,
            &[AttemptAction::SignalProcessGroup],
        )?;
        tx.commit().map_err(db_error)?;
        Ok(StopAuthorization {
            signal_process_group: !receipt.replay,
        })
    }

    fn observe_signal(&self, run_id: &str, observed_at: &str) -> Result<Receipt, ProjectsError> {
        let tx = self.tx()?;
        let receipt = record(
            &tx,
            run_id,
            projects_attempt::SIGNAL_OBSERVED_KEY,
            AttemptEvent::SignalObserved,
            observed_at,
            &[],
        )?;
        tx.commit().map_err(db_error)?;
        Ok(receipt)
    }

    fn observe_exit(
        &self,
        run_id: &str,
        exit: ExitObservation,
        ended_at: &str,
    ) -> Result<ExitAuthorization, ProjectsError> {
        let tx = self.tx()?;
        let receipt = record(
            &tx,
            run_id,
            projects_attempt::EXIT_OBSERVED_KEY,
            AttemptEvent::ExitObserved { exit },
            ended_at,
            &[AttemptAction::FinalizeOutput],
        )?;
        tx.execute(
            "UPDATE runs SET ended_at = ?1, exit_code = ?2 WHERE run_id = ?3",
            params![
                ended_at,
                match exit {
                    ExitObservation::Exited { code } => code.map(i64::from),
                    ExitObservation::Lost => None,
                },
                run_id
            ],
        )?;
        tx.commit().map_err(db_error)?;
        Ok(ExitAuthorization {
            finalize_output: !receipt.replay,
            state: receipt.state,
        })
    }

    fn finalize_output(&self, run_id: &str, observed_at: &str) -> Result<Receipt, ProjectsError> {
        let tx = self.tx()?;
        let receipt = record(
            &tx,
            run_id,
            projects_attempt::FINAL_OUTPUT_INGESTED_KEY,
            AttemptEvent::FinalOutputIngested,
            observed_at,
            &[],
        )?;
        let failed: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM experiment_cells
                 WHERE run_id = ?1 AND status = 'failed'",
                params![run_id],
                |row| row.get(0),
            )
            .map_err(db_error)?;
        let cells = CellSummary {
            failed: failed as u64,
        };
        terminal_projection(&tx, run_id, receipt.state, cells, observed_at)?;
        tx.commit().map_err(db_error)?;
        Ok(receipt)
    }

    fn project_stop(&self, run_id: &str, ended_at: &str) -> Result<(), ProjectsError> {
        let tx = self.tx()?;
        tx.execute("UPDATE experiment_cells SET status = 'cancelled', ended_at = ?1, error = COALESCE(error, 'run stopped') WHERE run_id = ?2 AND status IN ('pending', 'running')", params![ended_at, run_id])?;
        tx.execute("UPDATE runs SET status = 'stopped', ended_at = ?1 WHERE run_id = ?2 AND status = 'running'", params![ended_at, run_id])?;
        tx.commit().map_err(db_error)
    }
}

fn locked_connection(
    mutex: &Mutex<Connection>,
) -> Result<std::sync::MutexGuard<'_, Connection>, ProjectsError> {
    mutex
        .lock()
        .map_err(|_| ProjectsError::Storage("benchmark database mutex poisoned".into()))
}

impl ProjectsRepository for Mutex<Connection> {
    fn authorize_start(
        &self,
        request: &StartRequest,
        descriptor: &LaunchDescriptor,
    ) -> Result<StartAuthorization, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).authorize_start(request, descriptor)
    }

    fn record_launch(
        &self,
        run_id: &str,
        result: &LaunchResult,
        observed_at: &str,
    ) -> Result<LaunchRecord, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).record_launch(run_id, result, observed_at)
    }

    fn load_stop_target(&self, run_id: &str) -> Result<StopTarget, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).load_stop_target(run_id)
    }

    fn load_if_initialized(&self, run_id: &str) -> Result<Option<Receipt>, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).load_if_initialized(run_id)
    }

    fn typed_liveness_targets(&self) -> Result<Vec<LivenessTarget>, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).typed_liveness_targets()
    }

    fn request_operator_stop(
        &self,
        run_id: &str,
        observed_at: &str,
    ) -> Result<StopAuthorization, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).request_operator_stop(run_id, observed_at)
    }

    fn observe_signal(&self, run_id: &str, observed_at: &str) -> Result<Receipt, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).observe_signal(run_id, observed_at)
    }

    fn observe_exit(
        &self,
        run_id: &str,
        exit: ExitObservation,
        ended_at: &str,
    ) -> Result<ExitAuthorization, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).observe_exit(run_id, exit, ended_at)
    }

    fn finalize_output(&self, run_id: &str, observed_at: &str) -> Result<Receipt, ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).finalize_output(run_id, observed_at)
    }

    fn project_stop(&self, run_id: &str, ended_at: &str) -> Result<(), ProjectsError> {
        let connection = locked_connection(self)?;
        Repository::new(&connection).project_stop(run_id, ended_at)
    }
}
