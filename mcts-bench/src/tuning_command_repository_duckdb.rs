//! DuckDB implementation of [`crate::tuning_command_repository::TuningCommandRepository`].

use std::sync::{Arc, Mutex};

use duckdb::{params, Connection};

use crate::tuning_command_repository::{
    StoredTuningCommand, TuningCommandReplayState, TuningCommandRepository,
    TuningCommandRepositoryError, TuningContinuationMetadata,
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
