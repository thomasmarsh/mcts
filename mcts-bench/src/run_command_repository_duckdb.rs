//! DuckDB implementation of [`crate::run_command_repository::RunCommandRepository`].

use crate::run_command_repository::{
    ContinuationParent, RecordRunLaunch, RecordedTunerLaunch, RunCommandRepository,
    RunCommandRepositoryError, TunerLaunchReservation,
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
    fn verify_tuner_launch_reservation(
        &self,
        reservation: &TunerLaunchReservation,
    ) -> Result<(), RunCommandRepositoryError> {
        let connection = self.lock()?;
        let target = i64::try_from(reservation.target_trial_count)
            .map_err(|_| RunCommandRepositoryError::Conflict)?;
        let found: Option<(String, String, i64)> = connection
            .query_row(
                "SELECT attempt_id, physical_run_id, target_trial_count FROM tuning_launch_reservations WHERE session_id = ?1 AND command_id = ?2",
                params![&reservation.session_id, &reservation.command_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .ok();
        if found
            == Some((
                reservation.attempt_id.clone(),
                reservation.physical_run_id.clone(),
                target,
            ))
        {
            Ok(())
        } else {
            Err(RunCommandRepositoryError::Conflict)
        }
    }
    fn recorded_tuner_launch(
        &self,
        physical_run_id: &str,
    ) -> Result<Option<RecordedTunerLaunch>, RunCommandRepositoryError> {
        let connection = self.lock()?;
        match connection.query_row(
            "SELECT pid, log_path FROM runs WHERE run_id = ?1 AND kind = 'tuner'",
            params![physical_run_id],
            |row| Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, String>(1)?)),
        ) {
            Ok((pid, log_path)) => Ok(Some(RecordedTunerLaunch {
                run_id: physical_run_id.into(),
                pid: u32::try_from(pid.unwrap_or_default()).unwrap_or_default(),
                log_path,
            })),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(storage(error)),
        }
    }
    fn prepare_latest_tuner_continuation(
        &self,
        session_id: &str,
    ) -> Result<ContinuationParent, RunCommandRepositoryError> {
        let connection = self.lock()?;
        let parent_run_id: String = connection.query_row(
            "SELECT bench_run_id FROM tuning_attempts WHERE session_id = ?1 AND bench_run_id IS NOT NULL ORDER BY started_at DESC, attempt_id DESC LIMIT 1",
            params![session_id],
            |row| row.get(0),
        ).map_err(|error| match error {
            duckdb::Error::QueryReturnedNoRows => RunCommandRepositoryError::NotFound,
            error => storage(error),
        })?;
        identity::prepare_continuation(&connection, &parent_run_id)
            .map(|parent| ContinuationParent {
                logical_run_id: parent.logical_run_id,
                parent_attempt_id: parent.parent_attempt_id,
                attempt_ordinal: parent.attempt_ordinal,
            })
            .map_err(identity_error)
    }
    fn record_tuner_attempt_launch(
        &self,
        launch: RecordRunLaunch,
    ) -> Result<(), RunCommandRepositoryError> {
        self.record_launch(launch)
    }
    fn project_legacy_stop(
        &self,
        run_id: &str,
        kind: &str,
        ended_at: &str,
    ) -> Result<(), RunCommandRepositoryError> {
        let mut connection = self.lock()?;
        let tx = connection.transaction().map_err(storage)?;
        tx.execute("UPDATE runs SET status = 'stopped', ended_at = ?1 WHERE run_id = ?2 AND status = 'running'", params![ended_at, run_id]).map_err(storage)?;
        if kind == "experiment" {
            tx.execute("UPDATE experiment_cells SET status = 'cancelled', ended_at = ?1, error = COALESCE(error, 'run stopped') WHERE run_id = ?2 AND status IN ('pending', 'running')", params![ended_at, run_id]).map_err(storage)?;
        }
        tx.commit().map_err(storage)
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

    #[test]
    fn tuner_reservation_replay_continuation_and_legacy_stop_are_durable() {
        let (connection, repository) = repository();
        let db = connection.lock().unwrap();
        db.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session', 'idle', '{}', CURRENT_TIMESTAMP, 0)", []).unwrap();
        db.execute("INSERT INTO tuning_launch_reservations (session_id, command_id, attempt_id, physical_run_id, target_trial_count, reserved_at) VALUES ('session', 'command', 'next', 'child', 8, CURRENT_TIMESTAMP)", []).unwrap();
        drop(db);
        let reservation = TunerLaunchReservation {
            session_id: "session".into(),
            command_id: "command".into(),
            attempt_id: "next".into(),
            physical_run_id: "child".into(),
            target_trial_count: 8,
        };
        repository
            .verify_tuner_launch_reservation(&reservation)
            .unwrap();
        let mut mismatch = reservation.clone();
        mismatch.physical_run_id = "other".into();
        assert_eq!(
            repository.verify_tuner_launch_reservation(&mismatch),
            Err(RunCommandRepositoryError::Conflict)
        );
        let mut missing = reservation.clone();
        missing.command_id = "missing".into();
        assert_eq!(
            repository.verify_tuner_launch_reservation(&missing),
            Err(RunCommandRepositoryError::Conflict)
        );
        assert_eq!(repository.recorded_tuner_launch("child").unwrap(), None);
        assert_eq!(
            repository.prepare_latest_tuner_continuation("missing"),
            Err(RunCommandRepositoryError::NotFound)
        );

        repository.record_launch(launch("parent")).unwrap();
        connection.lock().unwrap().execute("INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at) VALUES ('parent-attempt', 'session', 'parent', 'completed', CURRENT_TIMESTAMP)", []).unwrap();
        let parent = repository
            .prepare_latest_tuner_continuation("session")
            .unwrap();
        let mut child = launch("child");
        child.kind = "tuner".into();
        child.continuation_parent = Some(parent);
        child.tuner_lifecycle_source = Some("/tmp/child.lifecycle".into());
        repository.record_tuner_attempt_launch(child).unwrap();
        assert_eq!(
            repository.recorded_tuner_launch("child").unwrap(),
            Some(RecordedTunerLaunch {
                run_id: "child".into(),
                pid: 42,
                log_path: "/tmp/child.jsonl".into()
            })
        );

        let db = connection.lock().unwrap();
        db.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('experiment', 'experiment', 'druid', 'test', false, 'test', CURRENT_TIMESTAMP, 'running', '/tmp/experiment')", []).unwrap();
        db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games, status) VALUES ('experiment', 'pending', 'druid', '{}', 'v', 'v', '{}', 'b', 'b', '{}', '{}', 1, 1, 'pending'), ('experiment', 'done', 'druid', '{}', 'v', 'v', '{}', 'b', 'b', '{}', '{}', 1, 1, 'completed')", []).unwrap();
        drop(db);
        repository
            .project_legacy_stop("experiment", "experiment", "2026-01-02T00:00:00Z")
            .unwrap();
        let statuses: Vec<String> = connection
            .lock()
            .unwrap()
            .prepare(
                "SELECT status FROM experiment_cells WHERE run_id = 'experiment' ORDER BY cell_id",
            )
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(statuses, ["completed", "cancelled"]);

        repository.record_launch(launch("ordinary")).unwrap();
        repository
            .project_legacy_stop("ordinary", "round_robin", "2026-01-02T00:00:00Z")
            .unwrap();
        assert_eq!(
            connection
                .lock()
                .unwrap()
                .query_row(
                    "SELECT status FROM runs WHERE run_id = 'ordinary'",
                    [],
                    |row| row.get::<_, String>(0)
                )
                .unwrap(),
            "stopped"
        );
    }

    #[test]
    fn failed_lifecycle_source_registration_rolls_back_tuner_run() {
        let (connection, repository) = repository();
        connection
            .lock()
            .unwrap()
            .execute("DROP TABLE tuning_lifecycle_sources", [])
            .unwrap();
        let mut broken = launch("broken-source");
        broken.kind = "tuner".into();
        broken.tuner_lifecycle_source = Some("/tmp/broken.lifecycle".into());
        assert!(matches!(
            repository.record_tuner_attempt_launch(broken),
            Err(RunCommandRepositoryError::Storage(_))
        ));
        assert_eq!(
            connection
                .lock()
                .unwrap()
                .query_row(
                    "SELECT COUNT(*) FROM runs WHERE run_id = 'broken-source'",
                    [],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
    }
}
