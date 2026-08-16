use std::fs;

use super::test_support::{
    start_event, stop_event, typed_projects_fixture, typed_projects_starting_fixture,
};
use super::{ingest_once, process_registry};
use crate::attempt_store;
use crate::log::{LogRecord, RegistryEvent};
use crate::orchestration::{AttemptEvent, StopReason};
use crate::projects_attempt::{self, ProjectsRepository};
use crate::projects_attempt_duckdb;

#[test]
fn typed_projects_registry_start_replays_process_observation() {
    let (fix, log_path) = typed_projects_fixture("typed-start");
    fs::write(
        fix.bench_runs.join("registry.log"),
        start_event(
            "typed-start",
            "experiment",
            "nim",
            std::process::id(),
            &log_path.to_string_lossy(),
        )
        .to_json_line()
            + "\n",
    )
    .unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-start'"),
        "running"
    );
    assert_eq!(
        fix.query_string(
            "SELECT CAST(attempt_version AS TEXT) FROM runs WHERE run_id = 'typed-start'"
        ),
        "2"
    );
    assert_eq!(
        fix.db
            .query_row(
                "SELECT COUNT(*) FROM attempt_events WHERE attempt_id = 'typed-start'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
}

#[test]
fn typed_projects_short_child_stop_before_start_is_reordered_safely() {
    let (fix, log_path) = typed_projects_starting_fixture("typed-short");
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-short'"),
        "starting"
    );
    assert_eq!(fix.count("attempt_events"), 1);
    fs::write(
        fix.bench_runs.join("registry.log"),
        format!(
            "{}\n{}\n",
            stop_event("typed-short", Some(0)).to_json_line(),
            start_event(
                "typed-short",
                "experiment",
                "nim",
                999999999,
                &log_path.to_string_lossy()
            )
            .to_json_line()
        ),
    )
    .unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-short'"),
        "completed"
    );
    assert_eq!(fix.count("attempt_events"), 4);
    assert_eq!(
        fix.query_string("SELECT status FROM runs WHERE run_id = 'typed-short'"),
        "completed"
    );
}

#[test]
fn typed_projects_exit_and_logs_finalize_completed_or_with_errors() {
    for (run_id, log_record, expected_status) in [
        ("typed-complete", None, "completed"),
        (
            "typed-errors",
            Some(LogRecord::CellFailed {
                cell_id: "cell-1".into(),
                completed_games: 1,
                error: "partial".into(),
            }),
            "completed_with_errors",
        ),
    ] {
        let (fix, log_path) = typed_projects_fixture(run_id);
        if let Some(record) = log_record {
            fs::write(&log_path, record.to_json_line() + "\n").unwrap();
        }
        fs::write(
            fix.bench_runs.join("registry.log"),
            format!(
                "{}\n{}\n",
                start_event(
                    run_id,
                    "experiment",
                    "nim",
                    999999999,
                    &log_path.to_string_lossy()
                )
                .to_json_line(),
                stop_event(run_id, Some(0)).to_json_line()
            ),
        )
        .unwrap();
        ingest_once(&fix.db, &fix.bench_runs).unwrap();
        assert_eq!(
            fix.query_string(&format!(
                "SELECT attempt_phase FROM runs WHERE run_id = '{run_id}'"
            )),
            "completed"
        );
        assert_eq!(
            fix.query_string(&format!(
                "SELECT status FROM runs WHERE run_id = '{run_id}'"
            )),
            expected_status
        );
        if expected_status == "completed_with_errors" {
            assert_eq!(
                fix.query_string(&format!(
                    "SELECT status FROM experiment_cells WHERE run_id = '{run_id}'"
                )),
                "failed"
            );
            assert_eq!(
                    fix.db
                        .query_row(
                            &format!(
                                "SELECT completed_games FROM experiment_cells WHERE run_id = '{run_id}'"
                            ),
                            [],
                            |row| row.get::<_, i64>(0),
                        )
                        .unwrap(),
                    1
                );
        }
    }
}

#[test]
fn typed_projects_exit_variants_finalize_crashed_with_partial_evidence() {
    for (run_id, exit_code) in [("typed-nonzero", Some(1)), ("typed-unknown", None)] {
        let (fix, log_path) = typed_projects_fixture(run_id);
        fs::write(
            &log_path,
            LogRecord::CellFinished {
                cell_id: "cell-1".into(),
                completed_games: 1,
            }
            .to_json_line()
                + "\n",
        )
        .unwrap();
        fs::write(
            fix.bench_runs.join("registry.log"),
            format!(
                "{}\n{}\n",
                start_event(
                    run_id,
                    "experiment",
                    "nim",
                    999999999,
                    &log_path.to_string_lossy()
                )
                .to_json_line(),
                stop_event(run_id, exit_code).to_json_line()
            ),
        )
        .unwrap();
        ingest_once(&fix.db, &fix.bench_runs).unwrap();
        assert_eq!(
            fix.query_string(&format!(
                "SELECT attempt_phase FROM runs WHERE run_id = '{run_id}'"
            )),
            "crashed"
        );
        assert_eq!(
            fix.query_string(&format!(
                "SELECT status FROM runs WHERE run_id = '{run_id}'"
            )),
            "crashed"
        );
        assert_eq!(
            fix.query_string(&format!(
                "SELECT attempt_exit_kind FROM runs WHERE run_id = '{run_id}'"
            )),
            "exited"
        );
        assert_eq!(
            fix.db
                .query_row(
                    &format!("SELECT exit_code FROM runs WHERE run_id = '{run_id}'"),
                    [],
                    |row| row.get::<_, Option<i32>>(0),
                )
                .unwrap(),
            exit_code
        );
        assert_eq!(
            fix.query_string(&format!(
                "SELECT status FROM experiment_cells WHERE run_id = '{run_id}'"
            )),
            "completed"
        );
        assert_eq!(
            fix.db
                .query_row(
                    &format!(
                        "SELECT completed_games FROM experiment_cells WHERE run_id = '{run_id}'"
                    ),
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
    }
    let (fix, log_path) = typed_projects_fixture("typed-lost");
    fix.db
        .execute(
            "UPDATE runs SET pid = 999999999 WHERE run_id = 'typed-lost'",
            [],
        )
        .unwrap();
    fs::write(
        &log_path,
        LogRecord::CellStarted {
            cell_id: "cell-1".into(),
        }
        .to_json_line()
            + "\n",
    )
    .unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-lost'"),
        "crashed"
    );
    assert_eq!(
        fix.query_string("SELECT error FROM experiment_cells WHERE run_id = 'typed-lost'"),
        "coordinator disappeared"
    );
    assert_eq!(
        fix.query_string("SELECT attempt_exit_kind FROM runs WHERE run_id = 'typed-lost'"),
        "lost"
    );
    assert_eq!(
        fix.query_string("SELECT status FROM experiment_cells WHERE run_id = 'typed-lost'"),
        "failed"
    );
    assert_eq!(
        fix.db
            .query_row(
                "SELECT completed_games FROM experiment_cells WHERE run_id = 'typed-lost'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
}

#[test]
fn typed_projects_stop_exit_and_final_output_preserve_finished_cells() {
    let (fix, log_path) = typed_projects_fixture("typed-stopped");
    fix.db.execute("UPDATE experiment_cells SET status = 'completed', completed_games = 2 WHERE run_id = 'typed-stopped'", []).unwrap();
    let tx = fix.db.unchecked_transaction().unwrap();
    attempt_store::record_attempt_event(
        &tx,
        "typed-stopped",
        2,
        projects_attempt::OPERATOR_STOP_REQUESTED_KEY,
        AttemptEvent::StopRequested {
            reason: StopReason::Operator,
        },
        "2026-01-01T00:00:02Z",
    )
    .unwrap();
    attempt_store::record_attempt_event(
        &tx,
        "typed-stopped",
        3,
        projects_attempt::SIGNAL_OBSERVED_KEY,
        AttemptEvent::SignalObserved,
        "2026-01-01T00:00:03Z",
    )
    .unwrap();
    tx.commit().unwrap();
    fs::write(
        fix.bench_runs.join("registry.log"),
        format!(
            "{}\n{}\n",
            start_event(
                "typed-stopped",
                "experiment",
                "nim",
                999999999,
                &log_path.to_string_lossy()
            )
            .to_json_line(),
            stop_event("typed-stopped", Some(0)).to_json_line()
        ),
    )
    .unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-stopped'"),
        "stopped"
    );
    assert_eq!(
        fix.query_string("SELECT status FROM experiment_cells WHERE run_id = 'typed-stopped'"),
        "completed"
    );
}

#[test]
fn typed_projects_stop_without_signal_is_reconciled() {
    let (fix, _log_path) = typed_projects_fixture("typed-stopped-lost");
    fix.db
        .execute(
            "UPDATE runs SET pid = 999999999 WHERE run_id = 'typed-stopped-lost'",
            [],
        )
        .unwrap();
    let repository = projects_attempt_duckdb::Repository::new(&fix.db);
    repository
        .request_operator_stop("typed-stopped-lost", "2026-01-01T00:00:02Z")
        .unwrap();
    repository
        .project_stop("typed-stopped-lost", "2026-01-01T00:00:04Z")
        .unwrap();

    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-stopped-lost'"),
        "finalizing"
    );
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-stopped-lost'"),
        "crashed"
    );
    assert_eq!(
        fix.query_string("SELECT attempt_exit_kind FROM runs WHERE run_id = 'typed-stopped-lost'"),
        "lost"
    );
    assert_eq!(
        fix.query_string("SELECT status FROM runs WHERE run_id = 'typed-stopped-lost'"),
        "crashed"
    );
    assert_eq!(
        fix.query_string("SELECT error FROM experiment_cells WHERE run_id = 'typed-stopped-lost'"),
        "run stopped"
    );
}

#[test]
fn typed_projects_finalizing_recovery_tails_and_finishes_once() {
    let (fix, log_path) = typed_projects_fixture("typed-recover");
    fs::write(
        fix.bench_runs.join("registry.log"),
        format!("{}\n", stop_event("typed-recover", Some(0)).to_json_line()),
    )
    .unwrap();
    process_registry(&fix.db, &fix.bench_runs.join("registry.log")).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-recover'"),
        "finalizing"
    );
    assert_eq!(fix.count("attempt_events"), 3);
    fs::write(&log_path, "").unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'typed-recover'"),
        "completed"
    );
    assert_eq!(fix.count("attempt_events"), 4);
}

#[test]
fn typed_projects_exit_conflict_and_corruption_do_not_overwrite_compatibility() {
    let (fix, log_path) = typed_projects_fixture("typed-conflict");
    fs::write(
        fix.bench_runs.join("registry.log"),
        format!("{}\n", stop_event("typed-conflict", Some(0)).to_json_line()),
    )
    .unwrap();
    // A Stop without a Start is still accepted as exact evidence for an
    // already initialized attempt; the next differing delivery conflicts.
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    fs::write(
        fix.bench_runs.join("registry.log"),
        format!(
            "{}\n{}\n",
            stop_event("typed-conflict", Some(0)).to_json_line(),
            RegistryEvent::Stop {
                run_id: "typed-conflict".into(),
                exit_code: Some(1),
                ended_at: "2026-01-01T02:00:00Z".into()
            }
            .to_json_line()
        ),
    )
    .unwrap();
    let before = fix.query_string("SELECT status FROM runs WHERE run_id = 'typed-conflict'");
    assert!(ingest_once(&fix.db, &fix.bench_runs).is_err());
    assert_eq!(
        fix.query_string("SELECT status FROM runs WHERE run_id = 'typed-conflict'"),
        before
    );
    let _ = log_path;

    let (corrupt, corrupt_log) = typed_projects_fixture("typed-corrupt");
    corrupt
        .db
        .execute(
            "UPDATE runs SET attempt_phase = 'not-a-phase' WHERE run_id = 'typed-corrupt'",
            [],
        )
        .unwrap();
    fs::write(
        &corrupt_log,
        LogRecord::CellFailed {
            cell_id: "cell-1".into(),
            completed_games: 0,
            error: "should not apply".into(),
        }
        .to_json_line()
            + "\n",
    )
    .unwrap();
    assert!(ingest_once(&corrupt.db, &corrupt.bench_runs).is_err());
    assert_eq!(
        corrupt.query_string("SELECT status FROM runs WHERE run_id = 'typed-corrupt'"),
        "running"
    );
    assert_eq!(
        corrupt.query_string("SELECT status FROM experiment_cells WHERE run_id = 'typed-corrupt'"),
        "pending"
    );
}
