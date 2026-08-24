//! DuckDB implementation of [`crate::run_command_repository::RunCommandRepository`].

use crate::run_command_repository::{
    ContinuationParent, RecordRunLaunch, RunCommandRepository, RunCommandRepositoryError,
};
use crate::{identity, tuning_store};
use duckdb::{params, Connection};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct SharedDuckDbRunCommandRepository {
    connection: Arc<Mutex<Connection>>,
}
impl SharedDuckDbRunCommandRepository {
    pub fn new(connection: Arc<Mutex<Connection>>) -> Self {
        Self { connection }
    }
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, RunCommandRepositoryError> {
        self.connection.lock().map_err(|_| {
            RunCommandRepositoryError::Storage("benchmark database mutex poisoned".into())
        })
    }
}
impl RunCommandRepository for SharedDuckDbRunCommandRepository {
    fn prepare_continuation(
        &self,
        parent_attempt_id: &str,
    ) -> Result<ContinuationParent, RunCommandRepositoryError> {
        let connection = self.lock()?;
        identity::prepare_continuation(&connection, parent_attempt_id)
            .map(|parent| ContinuationParent {
                logical_run_id: parent.logical_run_id,
                parent_attempt_id: parent.parent_attempt_id,
                attempt_ordinal: parent.attempt_ordinal,
            })
            .map_err(identity_error)
    }
    fn record_launch(&self, launch: RecordRunLaunch) -> Result<(), RunCommandRepositoryError> {
        let mut connection = self.lock()?;
        let tx = connection.transaction().map_err(storage)?;
        let inserted = tx.execute("INSERT INTO runs (run_id, kind, game, label, config, git_sha, git_dirty, host, pid, started_at, status, log_path) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'running', ?11) ON CONFLICT (run_id) DO NOTHING", params![&launch.run_id, &launch.kind, &launch.game, &launch.label, &launch.config_json, &launch.git_sha, launch.git_dirty, &launch.host, launch.pid, &launch.started_at, &launch.log_path]).map_err(storage)?;
        if inserted == 0 {
            let existing: (String, String, Option<i64>, String) = tx
                .query_row(
                    "SELECT kind, game, pid, log_path FROM runs WHERE run_id = ?1",
                    params![&launch.run_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .map_err(storage)?;
            if existing
                != (
                    launch.kind.clone(),
                    launch.game.clone(),
                    Some(launch.pid),
                    launch.log_path.clone(),
                )
            {
                return Err(RunCommandRepositoryError::Conflict);
            }
        }
        if let Some(parent) = launch.continuation_parent {
            identity::create_child_identity(
                &tx,
                &launch.run_id,
                &identity::ParentIdentity {
                    logical_run_id: parent.logical_run_id,
                    parent_attempt_id: parent.parent_attempt_id,
                    attempt_ordinal: parent.attempt_ordinal,
                },
            )
            .map_err(identity_error)?;
        } else {
            identity::create_root_identity(
                &tx,
                &launch.run_id,
                &launch.kind,
                None,
                None,
                &launch.started_at,
            )
            .map_err(identity_error)?;
        }
        if let Some(source) = launch.tuner_lifecycle_source {
            tuning_store::register_lifecycle_source(&tx, &source, &launch.run_id)
                .map_err(storage)?;
        }
        tx.commit().map_err(storage)
    }
    fn backfill_config(
        &self,
        run_id: &str,
        config_json: &str,
    ) -> Result<(), RunCommandRepositoryError> {
        let connection = self.lock()?;
        connection
            .execute(
                "UPDATE runs SET config = ?1 WHERE run_id = ?2 AND config IS NULL",
                params![config_json, run_id],
            )
            .map_err(storage)?;
        Ok(())
    }
    fn mark_crashed(&self, run_id: &str, ended_at: &str) -> Result<(), RunCommandRepositoryError> {
        let connection = self.lock()?;
        connection.execute("UPDATE runs SET ended_at = ?1, status = 'crashed' WHERE run_id = ?2 AND status = 'running'", params![ended_at, run_id]).map_err(storage)?;
        Ok(())
    }
}
fn storage(error: duckdb::Error) -> RunCommandRepositoryError {
    RunCommandRepositoryError::Storage(error.to_string())
}
fn identity_error(error: identity::IdentityError) -> RunCommandRepositoryError {
    match error {
        identity::IdentityError::MissingRun(_) => RunCommandRepositoryError::NotFound,
        identity::IdentityError::Contradiction(_) | identity::IdentityError::InvalidLinkage(_) => {
            RunCommandRepositoryError::ContradictoryIdentity
        }
        identity::IdentityError::DuckDb(error) => storage(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::CREATE_TABLES;

    fn repository() -> (Arc<Mutex<Connection>>, SharedDuckDbRunCommandRepository) {
        let connection = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        for statement in CREATE_TABLES {
            connection.lock().unwrap().execute(statement, []).unwrap();
        }
        (
            connection.clone(),
            SharedDuckDbRunCommandRepository::new(connection),
        )
    }

    fn launch(run_id: &str) -> RecordRunLaunch {
        RecordRunLaunch {
            run_id: run_id.into(),
            kind: "round_robin".into(),
            game: "druid".into(),
            label: None,
            config_json: None,
            git_sha: "test".into(),
            git_dirty: false,
            host: "test".into(),
            pid: 42,
            started_at: "2026-01-01T00:00:00Z".into(),
            log_path: format!("/tmp/{run_id}.jsonl"),
            continuation_parent: None,
            tuner_lifecycle_source: None,
        }
    }

    #[test]
    fn records_root_child_ingest_races_and_backfills_without_overwrite() {
        let (connection, repository) = repository();
        repository.record_launch(launch("root")).unwrap();
        let parent = repository.prepare_continuation("root").unwrap();
        let mut child = launch("child");
        child.continuation_parent = Some(parent);
        repository.record_launch(child).unwrap();
        assert_eq!(
            connection
                .lock()
                .unwrap()
                .query_row(
                    "SELECT logical_run_id FROM runs WHERE run_id = 'child'",
                    [],
                    |row| row.get::<_, String>(0)
                )
                .unwrap(),
            "root"
        );
        let mut replay = launch("child");
        replay.continuation_parent = Some(repository.prepare_continuation("root").unwrap());
        repository.record_launch(replay).unwrap();
        let mut conflicting = launch("child");
        conflicting.pid = 7;
        assert_eq!(
            repository.record_launch(conflicting),
            Err(RunCommandRepositoryError::Conflict)
        );
        repository.backfill_config("root", r#"{"a":1}"#).unwrap();
        repository.backfill_config("root", r#"{"a":2}"#).unwrap();
        assert_eq!(
            connection
                .lock()
                .unwrap()
                .query_row(
                    "SELECT CAST(config AS TEXT) FROM runs WHERE run_id = 'root'",
                    [],
                    |row| row.get::<_, String>(0)
                )
                .unwrap(),
            r#"{"a":1}"#
        );
    }

    #[test]
    fn tuner_source_is_atomic_and_crash_marking_is_idempotent() {
        let (connection, repository) = repository();
        let mut first = launch("first");
        first.kind = "tuner".into();
        first.tuner_lifecycle_source = Some("/tmp/lifecycle.jsonl".into());
        repository.record_launch(first).unwrap();
        let mut second = launch("second");
        second.kind = "tuner".into();
        second.tuner_lifecycle_source = Some("/tmp/lifecycle.jsonl".into());
        repository.record_launch(second).unwrap();
        repository
            .mark_crashed("first", "2026-01-01T01:00:00Z")
            .unwrap();
        repository
            .mark_crashed("first", "2026-01-01T02:00:00Z")
            .unwrap();
        assert_eq!(
            connection
                .lock()
                .unwrap()
                .query_row(
                    "SELECT status FROM runs WHERE run_id = 'first'",
                    [],
                    |row| row.get::<_, String>(0)
                )
                .unwrap(),
            "crashed"
        );
    }

    #[test]
    fn identity_failure_rolls_back_the_run_insert() {
        let (connection, repository) = repository();
        connection.lock().unwrap().execute(
            "INSERT INTO logical_runs (logical_run_id, kind, created_at, current_attempt_id) VALUES ('broken', 'tuner', CURRENT_TIMESTAMP, 'broken')",
            [],
        ).unwrap();
        assert_eq!(
            repository.record_launch(launch("broken")),
            Err(RunCommandRepositoryError::ContradictoryIdentity)
        );
        let count: i64 = connection
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE run_id = 'broken'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }
}
