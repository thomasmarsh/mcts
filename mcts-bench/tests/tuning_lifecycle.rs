use duckdb::Connection;
use mcts_bench::ingest::ingest_once;
use mcts_bench::schema::ensure_schema;
use mcts_bench::tuning_lifecycle::{TuningLifecycleEvent, TUNING_LIFECYCLE_SCHEMA_VERSION};
use mcts_bench::tuning_store::{apply_event, ApplyDisposition};

fn event(id: &str, sequence: u64, kind: &str, payload: serde_json::Value) -> TuningLifecycleEvent {
    serde_json::from_value(serde_json::json!({
        "schema_version": TUNING_LIFECYCLE_SCHEMA_VERSION,
        "event_id": id,
        "session_id": "session-1",
        "attempt_id": "attempt-1",
        "session_sequence": sequence,
        "timestamp": format!("2026-08-23T00:00:{sequence:02}Z"),
        "event_type": kind,
        "payload": payload,
    }))
    .unwrap()
}

fn apply(conn: &Connection, event: &TuningLifecycleEvent) -> ApplyDisposition {
    let tx = conn.unchecked_transaction().unwrap();
    let result = apply_event(
        &tx,
        event,
        "run-1",
        "lifecycle.jsonl",
        event.session_sequence,
    )
    .unwrap();
    tx.commit().unwrap();
    result
}

fn fixture() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    ensure_schema(&conn).unwrap();
    conn.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('run-1', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'running', '/tmp/run.log')", []).unwrap();
    conn
}

#[test]
fn schema_and_v1_event_shape_are_available() {
    let conn = fixture();
    let tables: i64 = conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('tuning_sessions', 'tuning_attempts', 'tuning_trials', 'tuning_lifecycle_events')", [], |r| r.get(0)).unwrap();
    assert_eq!(tables, 4);
    let invalid = event(
        "e0",
        1,
        "session_started",
        serde_json::json!({"manifest": {}, "manifest_fingerprint": "f"}),
    );
    assert!(invalid.validate_shape().is_ok());
}

#[test]
fn lifecycle_projection_is_idempotent_and_retains_rejected_evidence() {
    let conn = fixture();
    let events = vec![
        event(
            "e1",
            1,
            "session_started",
            serde_json::json!({"manifest": {"game": "nim"}, "manifest_fingerprint": "f", "target_trial_count": 1}),
        ),
        event(
            "e2",
            2,
            "attempt_started",
            serde_json::json!({"run_id": "run-1", "target_trial_count": 1}),
        ),
        event(
            "e3",
            3,
            "trial_created",
            serde_json::json!({"trial_id": "trial-1", "trial_number": 0, "config": {"c": 1}}),
        ),
        event(
            "e4",
            4,
            "trial_started",
            serde_json::json!({"trial_id": "trial-1", "trial_number": 0}),
        ),
        event(
            "e5",
            5,
            "trial_failed",
            serde_json::json!({"trial_id": "trial-1", "error": "worker failed"}),
        ),
        event(
            "e6",
            6,
            "attempt_stopped",
            serde_json::json!({"reason": "cancelled"}),
        ),
    ];
    for item in &events {
        assert_eq!(apply(&conn, item), ApplyDisposition::Applied);
    }
    assert_eq!(apply(&conn, &events[4]), ApplyDisposition::Replay);
    let conflicting = event(
        "e5",
        5,
        "trial_completed",
        serde_json::json!({"trial_id": "trial-1"}),
    );
    assert_eq!(apply(&conn, &conflicting), ApplyDisposition::Conflict);

    let status: String = conn
        .query_row(
            "SELECT status FROM tuning_trials WHERE trial_id = 'trial-1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let session_status: String = conn
        .query_row(
            "SELECT status FROM tuning_sessions WHERE session_id = 'session-1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let attempt_status: String = conn
        .query_row(
            "SELECT status FROM tuning_attempts WHERE attempt_id = 'attempt-1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "failed");
    assert_eq!(session_status, "idle");
    assert_eq!(attempt_status, "stopped");
    assert_eq!(
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM tuning_trials", [], |r| r.get(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM tuning_lifecycle_events", [], |r| r
            .get(0))
            .unwrap(),
        6
    );
}

#[test]
fn invalid_transition_is_audited_without_projection_change() {
    let conn = fixture();
    let start = event(
        "e1",
        1,
        "session_started",
        serde_json::json!({"manifest": {}, "manifest_fingerprint": "f"}),
    );
    assert_eq!(apply(&conn, &start), ApplyDisposition::Applied);
    let invalid = event(
        "e2",
        2,
        "trial_started",
        serde_json::json!({"trial_id": "missing", "trial_number": 1}),
    );
    assert_eq!(apply(&conn, &invalid), ApplyDisposition::Rejected);
    let accepted: bool = conn
        .query_row(
            "SELECT accepted FROM tuning_lifecycle_events WHERE event_id = 'e2'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(!accepted);
    assert_eq!(
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM tuning_trials", [], |r| r.get(0))
            .unwrap(),
        0
    );
}

#[test]
fn lifecycle_artifact_reaches_persistence_without_consuming_a_partial_record() {
    use std::io::Write;

    let root = std::env::temp_dir().join(format!(
        "mcts_tuning_lifecycle_ingest_{}",
        std::process::id()
    ));
    let run_dir = root.join("run-1");
    std::fs::create_dir_all(&run_dir).unwrap();
    let lifecycle_path = run_dir.join("lifecycle.jsonl");
    let log_path = run_dir.join("log.jsonl");
    let conn = Connection::open_in_memory().unwrap();
    ensure_schema(&conn).unwrap();
    conn.execute(
        "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('run-1', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'completed', ?1)",
        duckdb::params![log_path.to_string_lossy().as_ref()],
    )
    .unwrap();

    let records = [
        event(
            "e1",
            1,
            "session_started",
            serde_json::json!({"manifest": {"game": "nim"}, "manifest_fingerprint": "f", "target_trial_count": 1}),
        ),
        event(
            "e2",
            2,
            "attempt_started",
            serde_json::json!({"run_id": "run-1", "target_trial_count": 1}),
        ),
        event(
            "e3",
            3,
            "trial_created",
            serde_json::json!({"trial_id": "trial-1", "trial_number": 0, "config": {"c": 1}}),
        ),
        event(
            "e4",
            4,
            "trial_started",
            serde_json::json!({"trial_id": "trial-1", "trial_number": 0}),
        ),
        event(
            "e5",
            5,
            "trial_completed",
            serde_json::json!({"trial_id": "trial-1", "trial_number": 0, "score": 12.5, "mu": 20.0, "sigma": 2.5}),
        ),
        event(
            "e6",
            6,
            "attempt_completed",
            serde_json::json!({"target_trial_count": 1}),
        ),
    ];
    let mut artifact = std::fs::File::create(&lifecycle_path).unwrap();
    for record in &records[..3] {
        writeln!(artifact, "{}", serde_json::to_string(record).unwrap()).unwrap();
    }
    write!(artifact, "{}", serde_json::to_string(&records[3]).unwrap()).unwrap();
    artifact.flush().unwrap();

    ingest_once(&conn, &root).unwrap();
    assert_eq!(
        conn.query_row::<String, _, _>(
            "SELECT status FROM tuning_trials WHERE trial_id = 'trial-1'",
            [],
            |row| row.get(0),
        )
        .unwrap(),
        "queued"
    );

    let mut artifact = std::fs::OpenOptions::new()
        .append(true)
        .open(&lifecycle_path)
        .unwrap();
    writeln!(artifact).unwrap();
    for record in &records[4..] {
        writeln!(artifact, "{}", serde_json::to_string(record).unwrap()).unwrap();
    }
    artifact.flush().unwrap();
    ingest_once(&conn, &root).unwrap();
    ingest_once(&conn, &root).unwrap();

    let projection: (String, String, i64) = conn
        .query_row(
            "SELECT t.status, s.status, (SELECT COUNT(*) FROM tuning_lifecycle_events) FROM tuning_trials t JOIN tuning_sessions s ON s.session_id = t.session_id WHERE t.trial_id = 'trial-1'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(projection, ("complete".into(), "idle".into(), 6));
    std::fs::remove_dir_all(&root).unwrap();
}
