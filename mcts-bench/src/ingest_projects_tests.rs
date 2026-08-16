use std::fs;

use duckdb::params;

use super::ingest_once;
use super::test_support::{stop_event, typed_projects_fixture};
use crate::lifecycle::{ExitEvidence, LifecycleWriter, OutputClosure, WrapperManifest};

fn journal(
    run_id: &str,
    close: bool,
    exit: ExitEvidence,
) -> (super::test_support::TestFixture, LifecycleWriter) {
    let (fix, log_path) = typed_projects_fixture(run_id);
    let run_dir = fix.bench_runs.join(run_id);
    let stdout = run_dir.join("stdout.log");
    let stderr = run_dir.join("stderr.log");
    let journal_path = run_dir.join("lifecycle.jsonl");
    fix.db.execute(
        "INSERT INTO projects_launches (attempt_id, logical_run_id, launch_nonce, workload_argv, lifecycle_path, stdout_path, stderr_path, wrapper_pid, process_group_id, launch_result) VALUES (?1, ?1, 'nonce', '[\"work\"]', ?2, ?3, ?4, 7, 7, 'ready')",
        params![run_id, journal_path.to_string_lossy(), stdout.to_string_lossy(), stderr.to_string_lossy()],
    ).unwrap();
    let mut writer = LifecycleWriter::create(
        &journal_path,
        WrapperManifest {
            logical_run_id: run_id.into(),
            attempt_id: run_id.into(),
            parent_attempt_id: None,
            argv: vec!["work".into()],
            wrapper_pid: 7,
            process_group_id: 7,
            hostname: "host".into(),
            boot_id: None,
            process_start_id: None,
        },
        "nonce",
        "2026-01-01T00:00:00Z",
    )
    .unwrap();
    writer.child_started(8, "2026-01-01T00:00:01Z").unwrap();
    writer.child_exited(exit, "2026-01-01T00:00:02Z").unwrap();
    if close {
        writer
            .outputs_closed(
                vec![
                    OutputClosure {
                        path: stdout.to_string_lossy().into(),
                        byte_length: Some(0),
                    },
                    OutputClosure {
                        path: stderr.to_string_lossy().into(),
                        byte_length: Some(0),
                    },
                ],
                "2026-01-01T00:00:03Z",
            )
            .unwrap();
    }
    let _ = log_path;
    (fix, writer)
}

#[test]
fn journal_exit_waits_for_output_closure_then_replays_after_restart() {
    let (fix, mut writer) = journal("observed", false, ExitEvidence::Code { code: 0 });
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'observed'"),
        "running"
    );
    let run_dir = fix.bench_runs.join("observed");
    writer
        .outputs_closed(
            vec![
                OutputClosure {
                    path: run_dir.join("stdout.log").to_string_lossy().into(),
                    byte_length: Some(0),
                },
                OutputClosure {
                    path: run_dir.join("stderr.log").to_string_lossy().into(),
                    byte_length: Some(0),
                },
            ],
            "2026-01-01T00:00:03Z",
        )
        .unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'observed'"),
        "completed"
    );
    assert_eq!(fix.count("attempt_events"), 4);
}

#[test]
fn registry_stop_cannot_mutate_a_supervised_projects_attempt() {
    let (fix, _writer) = journal("registry", false, ExitEvidence::Code { code: 0 });
    fs::write(
        fix.bench_runs.join("registry.log"),
        stop_event("registry", Some(1)).to_json_line() + "\n",
    )
    .unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'registry'"),
        "running"
    );
}

#[test]
fn conflicting_journal_identity_is_typed_and_leaves_the_attempt_running() {
    let (fix, _writer) = journal("conflict", true, ExitEvidence::Code { code: 0 });
    fix.db
        .execute(
            "UPDATE projects_launches SET launch_nonce = 'other' WHERE attempt_id = 'conflict'",
            [],
        )
        .unwrap();
    assert!(ingest_once(&fix.db, &fix.bench_runs).is_err());
    assert_eq!(
        fix.query_string("SELECT attempt_phase FROM runs WHERE run_id = 'conflict'"),
        "running"
    );
}

#[test]
fn signal_and_unavailable_evidence_survive_reload_and_finalize() {
    for (run_id, exit, kind, value) in [
        (
            "signal",
            ExitEvidence::Signal { signal: 15 },
            "signaled",
            Some(15),
        ),
        (
            "unavailable",
            ExitEvidence::WaitFailed {
                error: "wait unavailable".into(),
            },
            "unavailable",
            None,
        ),
    ] {
        let (fix, _writer) = journal(run_id, true, exit);
        ingest_once(&fix.db, &fix.bench_runs).unwrap();
        ingest_once(&fix.db, &fix.bench_runs).unwrap();
        assert_eq!(
            fix.query_string(&format!(
                "SELECT attempt_exit_kind FROM runs WHERE run_id = '{run_id}'"
            )),
            kind
        );
        assert_eq!(
            fix.db
                .query_row(
                    &format!("SELECT attempt_exit_code FROM runs WHERE run_id = '{run_id}'"),
                    [],
                    |row| row.get::<_, Option<i32>>(0),
                )
                .unwrap(),
            value
        );
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
    }
}
