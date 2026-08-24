//! DuckDB implementation of [`crate::tuning_command_repository::TuningCommandRepository`].

use std::sync::{Arc, Mutex};

use duckdb::{params, Connection};

use crate::tuning_command_repository::{
    StoredTuningCommand, TuningCommandDecision, TuningCommandReplayState, TuningCommandRepository,
    TuningCommandRepositoryError, TuningCommandRequest, TuningContinuationMetadata,
    TuningLaunchOutcome, TuningSessionControl,
};

/// A tuning-command repository backed by a shared DuckDB connection.
#[derive(Clone)]
pub struct SharedDuckDbTuningCommandRepository {
    connection: Arc<Mutex<Connection>>,
}

impl SharedDuckDbTuningCommandRepository {
    pub fn new(connection: Arc<Mutex<Connection>>) -> Self {
        Self { connection }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, TuningCommandRepositoryError> {
        self.connection.lock().map_err(|_| {
            TuningCommandRepositoryError::Storage("benchmark database mutex poisoned".into())
        })
    }
}

impl TuningCommandRepository for SharedDuckDbTuningCommandRepository {
    fn apply_command(
        &self,
        session_id: &str,
        request: &TuningCommandRequest,
    ) -> Result<TuningCommandDecision, TuningCommandRepositoryError> {
        let connection = self.lock()?;
        crate::tuning_command_store::apply_command(&connection, session_id, request)
            .map_err(command_store_error)
    }

    fn record_launch_outcome(
        &self,
        session_id: &str,
        command_id: &str,
        outcome: TuningLaunchOutcome,
    ) -> Result<TuningSessionControl, TuningCommandRepositoryError> {
        let connection = self.lock()?;
        crate::tuning_command_store::record_launch_outcome(
            &connection,
            session_id,
            command_id,
            outcome,
        )
        .map_err(command_store_error)
    }

    fn load_command(
        &self,
        command_id: &str,
    ) -> Result<Option<StoredTuningCommand>, TuningCommandRepositoryError> {
        let connection = self.lock()?;
        load_command(&connection, command_id)
    }

    fn replay_state(
        &self,
        session_id: &str,
        command_id: &str,
        attempt_id: &str,
        physical_run_id: &str,
    ) -> Result<TuningCommandReplayState, TuningCommandRepositoryError> {
        let connection = self.lock()?;
        replay_state(
            &connection,
            session_id,
            command_id,
            attempt_id,
            physical_run_id,
        )
    }

    fn load_continuation_metadata(
        &self,
        session_id: &str,
    ) -> Result<Option<TuningContinuationMetadata>, TuningCommandRepositoryError> {
        let connection = self.lock()?;
        load_continuation_metadata(&connection, session_id)
    }

    fn load_attempt_pid(
        &self,
        session_id: &str,
        attempt_id: &str,
    ) -> Result<Option<i64>, TuningCommandRepositoryError> {
        let connection = self.lock()?;
        load_attempt_pid(&connection, session_id, attempt_id)
    }
}

fn load_command(
    connection: &Connection,
    command_id: &str,
) -> Result<Option<StoredTuningCommand>, TuningCommandRepositoryError> {
    match connection.query_row(
        "SELECT session_id, CAST(request AS TEXT) FROM tuning_session_commands WHERE command_id = ?1",
        params![command_id],
        |row| {
            Ok(StoredTuningCommand {
                session_id: row.get(0)?,
                request_json: row.get(1)?,
            })
        },
    ) {
        Ok(command) => Ok(Some(command)),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(storage(error)),
    }
}

fn replay_state(
    connection: &Connection,
    session_id: &str,
    command_id: &str,
    attempt_id: &str,
    physical_run_id: &str,
) -> Result<TuningCommandReplayState, TuningCommandRepositoryError> {
    let run_exists: bool = connection
        .query_row(
            "SELECT EXISTS (SELECT 1 FROM runs WHERE run_id = ?1)",
            params![physical_run_id],
            |row| row.get(0),
        )
        .map_err(storage)?;
    if run_exists {
        return Ok(TuningCommandReplayState::Launched);
    }
    let reservation_exists: bool = connection
        .query_row(
            "SELECT EXISTS ( \
                 SELECT 1 FROM tuning_launch_reservations \
                 WHERE session_id = ?1 AND command_id = ?2 AND attempt_id = ?3 AND physical_run_id = ?4 \
             )",
            params![session_id, command_id, attempt_id, physical_run_id],
            |row| row.get(0),
        )
        .map_err(storage)?;
    Ok(if reservation_exists {
        TuningCommandReplayState::Reserved
    } else {
        TuningCommandReplayState::Failed
    })
}

fn load_continuation_metadata(
    connection: &Connection,
    session_id: &str,
) -> Result<Option<TuningContinuationMetadata>, TuningCommandRepositoryError> {
    match connection.query_row(
        "SELECT run.game, CAST(run.config AS TEXT), session.optimizer_id, session.lifecycle_path \
         FROM tuning_sessions session \
         JOIN tuning_attempts attempt ON attempt.session_id = session.session_id \
         JOIN runs run ON run.run_id = attempt.bench_run_id \
         WHERE session.session_id = ?1 AND session.optimizer_id IS NOT NULL \
           AND session.lifecycle_path IS NOT NULL \
         ORDER BY attempt.started_at DESC, attempt.attempt_id DESC LIMIT 1",
        params![session_id],
        |row| {
            Ok(TuningContinuationMetadata {
                game: row.get(0)?,
                config_json: row.get(1)?,
                optimizer_id: row.get(2)?,
                lifecycle_path: row.get(3)?,
            })
        },
    ) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(storage(error)),
    }
}

fn load_attempt_pid(
    connection: &Connection,
    session_id: &str,
    attempt_id: &str,
) -> Result<Option<i64>, TuningCommandRepositoryError> {
    match connection.query_row(
        "SELECT run.pid FROM tuning_attempts attempt \
         LEFT JOIN runs run ON run.run_id = attempt.bench_run_id \
         WHERE attempt.session_id = ?1 AND attempt.attempt_id = ?2",
        params![session_id, attempt_id],
        |row| row.get(0),
    ) {
        Ok(pid) => Ok(pid),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(storage(error)),
    }
}

fn storage(error: duckdb::Error) -> TuningCommandRepositoryError {
    TuningCommandRepositoryError::Storage(error.to_string())
}

fn command_store_error(
    error: crate::tuning_command_store::CommandStoreError,
) -> TuningCommandRepositoryError {
    use crate::tuning_command_store::CommandStoreError;

    match error {
        CommandStoreError::DuckDb(message) | CommandStoreError::Serialization(message) => {
            TuningCommandRepositoryError::Storage(message)
        }
        CommandStoreError::SessionNotFound(session_id) => {
            TuningCommandRepositoryError::NotFound(session_id)
        }
        CommandStoreError::CommandIdReuseMismatch { command_id } => {
            TuningCommandRepositoryError::CommandIdReuseMismatch { command_id }
        }
        CommandStoreError::ExpectedVersionConflict { expected, control } => {
            TuningCommandRepositoryError::ExpectedVersionConflict { expected, control }
        }
        CommandStoreError::ActiveAttempt {
            attempt_id,
            control,
        } => TuningCommandRepositoryError::ActiveAttempt {
            attempt_id,
            control,
        },
        CommandStoreError::LaunchReserved {
            attempt_id,
            control,
        } => TuningCommandRepositoryError::LaunchReserved {
            attempt_id,
            control,
        },
        CommandStoreError::InvalidDeltaStart { control } => {
            TuningCommandRepositoryError::InvalidDeltaStart { control }
        }
        CommandStoreError::ExhaustedResume { control } => {
            TuningCommandRepositoryError::ExhaustedResume { control }
        }
        CommandStoreError::NoncontinuableLegacy { control } => {
            TuningCommandRepositoryError::NoncontinuableLegacy { control }
        }
        CommandStoreError::CommandDenied { reason, control } => {
            TuningCommandRepositoryError::CommandDenied { reason, control }
        }
        CommandStoreError::TargetOverflow { control } => {
            TuningCommandRepositoryError::TargetOverflow { control }
        }
        CommandStoreError::MissingReservation { command_id } => {
            TuningCommandRepositoryError::MissingReservation { command_id }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ensure_schema;
    use crate::tuning_command_repository::{TuningLaunchReservation, TuningSessionCommand};

    fn repository() -> SharedDuckDbTuningCommandRepository {
        let connection = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        ensure_schema(&connection.lock().unwrap()).unwrap();
        connection
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence, optimizer_id, lifecycle_path) VALUES ('s', 'idle', '{}', 3, '2026-01-01T00:00:00Z', 0, 'optimizer', '/tmp/lifecycle.jsonl')",
                [],
            )
            .unwrap();
        SharedDuckDbTuningCommandRepository::new(connection)
    }

    fn request(
        command_id: &str,
        expected_version: u64,
        command: TuningSessionCommand,
    ) -> TuningCommandRequest {
        let launch = match command {
            TuningSessionCommand::Resume | TuningSessionCommand::AddBudget { start: true, .. } => {
                Some(TuningLaunchReservation {
                    attempt_id: format!("attempt-{command_id}"),
                    physical_run_id: format!("run-{command_id}"),
                })
            }
            _ => None,
        };
        TuningCommandRequest {
            command_id: command_id.into(),
            expected_version,
            command,
            launch,
            n_workers: None,
            observed_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn applies_replays_and_translates_durable_command_outcomes() {
        let repository = repository();
        let resume = request("resume", 0, TuningSessionCommand::Resume);
        let accepted = repository.apply_command("s", &resume).unwrap();
        assert!(!accepted.replay);
        assert!(accepted.control.launch_reservation.is_some());
        assert!(repository.apply_command("s", &resume).unwrap().replay);

        let mismatch = TuningCommandRequest {
            expected_version: 1,
            ..resume.clone()
        };
        assert!(matches!(
            repository.apply_command("s", &mismatch),
            Err(TuningCommandRepositoryError::CommandIdReuseMismatch { .. })
        ));
        assert!(matches!(
            repository.apply_command("s", &request("stale", 0, TuningSessionCommand::Resume)),
            Err(TuningCommandRepositoryError::ExpectedVersionConflict { .. })
        ));
    }

    #[test]
    fn launch_outcomes_release_failed_reservations_without_rolling_back_budget() {
        let repository = repository();
        let command = request(
            "extend",
            0,
            TuningSessionCommand::AddBudget {
                delta: 2,
                start: true,
            },
        );
        let decision = repository.apply_command("s", &command).unwrap();
        assert_eq!(decision.control.target_trial_count, Some(5));
        let control = repository
            .record_launch_outcome("s", "extend", TuningLaunchOutcome::SpawnFailed)
            .unwrap();
        assert_eq!(control.target_trial_count, Some(5));
        assert!(control.launch_reservation.is_none());
        assert!(matches!(
            repository.record_launch_outcome("s", "missing", TuningLaunchOutcome::Spawned),
            Err(TuningCommandRepositoryError::MissingReservation { .. })
        ));
    }
}
