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
        Some("run-1"),
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
    let tables: i64 = conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('tuning_sessions', 'tuning_attempts', 'tuning_trials', 'tuning_pool_revisions', 'tuning_pool_anchors', 'tuning_evaluation_pairs', 'tuning_games', 'tuning_lifecycle_events')", [], |r| r.get(0)).unwrap();
    assert_eq!(tables, 8);
    let invalid = event(
        "e0",
        1,
        "session_started",
        serde_json::json!({"manifest": {}, "manifest_fingerprint": "f"}),
    );
    assert!(invalid.validate_shape().is_ok());
}

fn started_trial_events() -> Vec<TuningLifecycleEvent> {
    vec![
        event(
            "e1",
            1,
            "session_started",
            serde_json::json!({"manifest": {}, "manifest_fingerprint": "f"}),
        ),
        event(
            "e2",
            2,
            "attempt_started",
            serde_json::json!({"run_id": "run-1"}),
        ),
        event(
            "e3",
            3,
            "trial_created",
            serde_json::json!({"trial_id": "trial-1", "trial_number": 0, "config": {}}),
        ),
        event(
            "e4",
            4,
            "trial_started",
            serde_json::json!({"trial_id": "trial-1", "trial_number": 0}),
        ),
    ]
}

fn pool_revision(
    id: &str,
    sequence: u64,
    fingerprint: &str,
    anchors: serde_json::Value,
) -> TuningLifecycleEvent {
    event(
        id,
        sequence,
        "pool_revised",
        serde_json::json!({
            "pool_snapshot_fingerprint": fingerprint,
            "anchors": anchors,
        }),
    )
}

fn bootstrap_anchor() -> serde_json::Value {
    serde_json::json!({
        "anchor_id": "default",
        "config": {"family": "rave"},
        "mu": 25.0,
        "sigma": 0.5,
        "provenance": "bootstrap_default",
        "insertion_reason": "bootstrap",
        "source_trial_id": null,
    })
}

#[test]
fn pool_revisions_project_full_anchors_in_first_observation_order() {
    let conn = fixture();
    for item in started_trial_events().into_iter().take(2) {
        assert_eq!(apply(&conn, &item), ApplyDisposition::Applied);
    }
    let first = pool_revision(
        "pool-1",
        3,
        "pool-a",
        serde_json::json!([bootstrap_anchor()]),
    );
    let second = pool_revision(
        "pool-2",
        4,
        "pool-b",
        serde_json::json!([
            bootstrap_anchor(),
            {
                "anchor_id": "trial-4",
                "config": {"family": "ucb", "c": 1.4},
                "mu": 30.0,
                "sigma": 2.0,
                "provenance": "trial",
                "insertion_reason": "champion",
                "source_trial_id": "trial-4",
            }
        ]),
    );
    assert_eq!(apply(&conn, &first), ApplyDisposition::Applied);
    assert_eq!(apply(&conn, &second), ApplyDisposition::Applied);
    assert_eq!(apply(&conn, &second), ApplyDisposition::Replay);
    assert_eq!(
        apply(
            &conn,
            &pool_revision(
                "pool-duplicate",
                5,
                "pool-a",
                serde_json::json!([bootstrap_anchor()])
            ),
        ),
        ApplyDisposition::Applied
    );

    let revisions: Vec<(String, u32)> = conn
        .prepare("SELECT pool_snapshot_fingerprint, display_ordinal FROM tuning_pool_revisions ORDER BY display_ordinal")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(revisions, vec![("pool-a".into(), 1), ("pool-b".into(), 2)]);
    let anchor: (String, String, f64, f64, String, String, Option<String>) = conn
        .query_row(
            "SELECT anchor_id, CAST(config AS TEXT), mu, sigma, provenance, insertion_reason, source_trial_id FROM tuning_pool_anchors WHERE pool_snapshot_fingerprint = 'pool-b' AND anchor_ordinal = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?)),
        )
        .unwrap();
    assert_eq!(anchor.0, "trial-4");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&anchor.1).unwrap(),
        serde_json::json!({"family": "ucb", "c": 1.4})
    );
    assert_eq!(anchor.2, 30.0);
    assert_eq!(anchor.3, 2.0);
    assert_eq!(anchor.4, "trial");
    assert_eq!(anchor.5, "champion");
    assert_eq!(anchor.6.as_deref(), Some("trial-4"));
}

#[test]
fn invalid_or_conflicting_pool_revision_is_rejected_without_projection() {
    let conn = fixture();
    for item in started_trial_events().into_iter().take(2) {
        assert_eq!(apply(&conn, &item), ApplyDisposition::Applied);
    }
    let invalid = pool_revision(
        "pool-invalid",
        3,
        "pool-a",
        serde_json::json!([{
            "anchor_id": "trial-1", "config": {}, "mu": 20.0, "sigma": 1.0,
            "provenance": "trial", "insertion_reason": "champion", "source_trial_id": null,
        }]),
    );
    assert_eq!(apply(&conn, &invalid), ApplyDisposition::Rejected);
    assert_eq!(
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM tuning_pool_revisions", [], |row| row
            .get(0))
            .unwrap(),
        0
    );
    assert_eq!(
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM tuning_pool_anchors", [], |row| row
            .get(0))
            .unwrap(),
        0
    );

    let valid = pool_revision(
        "pool-valid",
        3,
        "pool-a",
        serde_json::json!([bootstrap_anchor()]),
    );
    assert_eq!(apply(&conn, &valid), ApplyDisposition::Applied);
    let conflicting = pool_revision(
        "pool-conflict",
        4,
        "pool-a",
        serde_json::json!([{
            "anchor_id": "default", "config": {"family": "ucb"}, "mu": 25.0, "sigma": 0.5,
            "provenance": "bootstrap_default", "insertion_reason": "bootstrap", "source_trial_id": null,
        }]),
    );
    assert_eq!(apply(&conn, &conflicting), ApplyDisposition::Rejected);
    assert_eq!(
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM tuning_pool_anchors", [], |row| row
            .get(0))
            .unwrap(),
        1
    );
}

fn pair_started(sequence: u64) -> TuningLifecycleEvent {
    event(
        "pair-start",
        sequence,
        "pair_started",
        serde_json::json!({
            "trial_id": "trial-1", "pair_id": "pair-1", "pair_index": 0, "seed": 7, "round": 1,
            "opponent": {"anchor_id": "anchor-1", "config": {"family": "ucb"}, "mu": 25.0, "sigma": 1.0},
            "pool_snapshot_fingerprint": "pool-1", "rating_before": {"mu": 24.0, "sigma": 2.0}
        }),
    )
}

fn game_finished(sequence: u64, side: &str) -> TuningLifecycleEvent {
    event(
        &format!("game-{side}-{sequence}"),
        sequence,
        "game_finished",
        serde_json::json!({
            "trial_id": "trial-1", "pair_id": "pair-1", "game_id": format!("pair-1-{side}"),
            "candidate_side": side, "outcome": "candidate_win", "seed": 7, "round": 1,
            "trace_game_seq": 100 + sequence, "plies": 12, "elapsed_ms": 30,
            "candidate": {"iterations_total": 20, "iterations_first_half": 9, "move_time_ms": 14},
            "baseline": {"iterations_total": 18, "iterations_first_half": 8, "move_time_ms": 13}
        }),
    )
}

fn pair_finished(sequence: u64) -> TuningLifecycleEvent {
    event(
        "pair-finished",
        sequence,
        "pair_finished",
        serde_json::json!({
            "trial_id": "trial-1", "pair_id": "pair-1", "pair_index": 0, "rating_before": {"mu": 24.0, "sigma": 2.0},
            "rating_after": {"mu": 25.0, "sigma": 1.5}, "score": 20.5
        }),
    )
}

fn trial_report(
    id: &str,
    sequence: u64,
    completed_pairs: u64,
    outcome: &str,
    reason: &str,
) -> TuningLifecycleEvent {
    event(
        id,
        sequence,
        "trial_reported",
        serde_json::json!({
            "trial_id": "trial-1", "trial_number": 0, "completed_pairs": completed_pairs,
            "mu": 25.0, "sigma": 1.5, "score": 20.5, "score_formula_version": 1,
            "conservative_k": 3.0, "outcome": outcome, "reason": reason,
            "pruning_exempt": false, "bracket_id": null, "rung_resource": null
        }),
    )
}

#[test]
fn trial_reports_project_consecutive_resources_and_replay_idempotently() {
    let conn = fixture();
    for item in started_trial_events() {
        assert_eq!(apply(&conn, &item), ApplyDisposition::Applied);
    }
    let first = trial_report("report-1", 5, 1, "continue", "below_min_pairs");
    let second = trial_report("report-2", 6, 2, "continue", "pruning_disabled");
    assert_eq!(apply(&conn, &first), ApplyDisposition::Applied);
    assert_eq!(apply(&conn, &second), ApplyDisposition::Applied);
    assert_eq!(apply(&conn, &first), ApplyDisposition::Replay);
    assert_eq!(
        apply(
            &conn,
            &trial_report("report-duplicate", 7, 2, "continue", "pruning_disabled")
        ),
        ApplyDisposition::Rejected
    );
    assert_eq!(
        apply(
            &conn,
            &trial_report("report-skipped", 7, 4, "continue", "pruning_disabled")
        ),
        ApplyDisposition::Rejected
    );
    assert_eq!(
        apply(
            &conn,
            &trial_report("report-decreasing", 7, 1, "continue", "pruning_disabled")
        ),
        ApplyDisposition::Rejected
    );
    type TrialReportRow = (u64, String, String, bool, Option<String>, Option<u64>);
    let reports: Vec<TrialReportRow> = conn
        .prepare("SELECT completed_pairs, outcome, reason, pruning_exempt, bracket_id, rung_resource FROM tuning_trial_reports ORDER BY completed_pairs")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        reports,
        vec![
            (
                1,
                "continue".into(),
                "below_min_pairs".into(),
                false,
                None,
                None
            ),
            (
                2,
                "continue".into(),
                "pruning_disabled".into(),
                false,
                None,
                None
            ),
        ]
    );
}

#[test]
fn trial_reports_require_a_running_known_trial_and_valid_decisions() {
    let queued = fixture();
    for item in started_trial_events().into_iter().take(3) {
        apply(&queued, &item);
    }
    assert_eq!(
        apply(
            &queued,
            &trial_report("queued-report", 4, 1, "continue", "below_min_pairs")
        ),
        ApplyDisposition::Rejected
    );

    let conn = fixture();
    for item in started_trial_events() {
        apply(&conn, &item);
    }
    let unknown = event(
        "unknown-report",
        5,
        "trial_reported",
        serde_json::json!({
            "trial_id": "missing", "trial_number": 0, "completed_pairs": 1,
            "mu": 25.0, "sigma": 1.5, "score": 20.5, "score_formula_version": 1,
            "conservative_k": 3.0, "outcome": "continue", "reason": "below_min_pairs",
            "pruning_exempt": false, "bracket_id": null, "rung_resource": null
        }),
    );
    assert_eq!(apply(&conn, &unknown), ApplyDisposition::Rejected);
    for (id, mut report) in [
        (
            "bad-outcome",
            trial_report("bad-outcome", 5, 1, "continue", "confidence"),
        ),
        (
            "bad-formula",
            trial_report("bad-formula", 5, 1, "continue", "below_min_pairs"),
        ),
        (
            "bad-resource",
            trial_report("bad-resource", 5, 0, "continue", "below_min_pairs"),
        ),
        (
            "bad-number",
            trial_report("bad-number", 5, 1, "continue", "below_min_pairs"),
        ),
    ] {
        match id {
            "bad-formula" => report.payload["score_formula_version"] = serde_json::json!(2),
            "bad-number" => report.payload["trial_number"] = serde_json::json!(1),
            _ => {}
        }
        assert_eq!(apply(&conn, &report), ApplyDisposition::Rejected, "{id}");
    }
    let terminal = event(
        "complete",
        5,
        "trial_completed",
        serde_json::json!({"trial_id": "trial-1", "reason": "confidence", "score": 20.5, "mu": 25.0, "sigma": 1.5}),
    );
    assert_eq!(apply(&conn, &terminal), ApplyDisposition::Applied);
    assert_eq!(
        apply(
            &conn,
            &trial_report("post-terminal", 6, 1, "continue", "below_min_pairs")
        ),
        ApplyDisposition::Rejected
    );
}

#[test]
fn terminal_stop_evidence_projects_for_complete_and_pruned_trials() {
    let conn = fixture();
    for item in started_trial_events() {
        apply(&conn, &item);
    }
    let complete = event(
        "complete",
        5,
        "trial_completed",
        serde_json::json!({
            "trial_id": "trial-1", "reason": "max_pairs", "score": 20.5, "mu": 25.0, "sigma": 1.5,
            "completed_pairs": 15, "score_formula_version": 1, "conservative_k": 3.0,
            "pruning_exempt": false, "bracket_id": null, "rung_resource": null
        }),
    );
    assert_eq!(apply(&conn, &complete), ApplyDisposition::Applied);
    let complete_projection: (String, Option<String>, f64) = conn
        .query_row(
            "SELECT status, stop_reason, score FROM tuning_trials WHERE trial_id = 'trial-1'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        complete_projection,
        ("complete".into(), Some("max_pairs".into()), 20.5)
    );

    let pruned = fixture();
    for item in started_trial_events() {
        apply(&pruned, &item);
    }
    let prune = event(
        "pruned",
        5,
        "trial_pruned",
        serde_json::json!({
            "trial_id": "trial-1", "stop_reason": "hyperband_prune", "score": 20.5, "mu": 25.0, "sigma": 1.5,
            "bracket_id": "bracket-1", "rung_resource": 9
        }),
    );
    assert_eq!(apply(&pruned, &prune), ApplyDisposition::Applied);
    let pruned_projection: (String, Option<String>) = pruned
        .query_row(
            "SELECT status, stop_reason FROM tuning_trials WHERE trial_id = 'trial-1'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        pruned_projection,
        ("pruned".into(), Some("hyperband_prune".into()))
    );
}

#[test]
fn pair_replay_projects_two_games_and_their_trace_references() {
    let conn = fixture();
    for item in started_trial_events() {
        assert_eq!(apply(&conn, &item), ApplyDisposition::Applied);
    }
    let events = [
        pair_started(5),
        game_finished(6, "first"),
        game_finished(7, "second"),
        pair_finished(8),
    ];
    for item in &events {
        assert_eq!(apply(&conn, item), ApplyDisposition::Applied);
    }
    assert_eq!(apply(&conn, &events[3]), ApplyDisposition::Replay);
    let projection: (String, i64, i64) = conn.query_row(
        "SELECT p.status, COUNT(g.game_id), COUNT(g.trace_game_seq) FROM tuning_evaluation_pairs p LEFT JOIN tuning_games g USING (session_id, pair_id) WHERE p.pair_id = 'pair-1' GROUP BY p.status",
        [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).unwrap();
    assert_eq!(projection, ("complete".into(), 2, 2));
}

#[test]
fn incomplete_or_duplicate_pair_evidence_is_rejected_without_terminal_projection() {
    let conn = fixture();
    for item in started_trial_events() {
        apply(&conn, &item);
    }
    assert_eq!(apply(&conn, &pair_started(5)), ApplyDisposition::Applied);
    assert_eq!(
        apply(&conn, &game_finished(6, "first")),
        ApplyDisposition::Applied
    );
    assert_eq!(apply(&conn, &pair_finished(7)), ApplyDisposition::Rejected);
    assert_eq!(
        apply(&conn, &game_finished(8, "first")),
        ApplyDisposition::Rejected
    );
    let projection: (String, i64) = conn.query_row(
        "SELECT p.status, COUNT(g.game_id) FROM tuning_evaluation_pairs p LEFT JOIN tuning_games g USING (session_id, pair_id) WHERE p.pair_id = 'pair-1' GROUP BY p.status",
        [], |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(projection, ("running".into(), 1));
}

#[test]
fn unknown_and_post_terminal_pair_events_are_rejected() {
    let conn = fixture();
    for item in started_trial_events() {
        apply(&conn, &item);
    }
    assert_eq!(
        apply(&conn, &game_finished(5, "first")),
        ApplyDisposition::Rejected
    );
    assert_eq!(apply(&conn, &pair_started(5)), ApplyDisposition::Applied);
    assert_eq!(
        apply(&conn, &game_finished(6, "first")),
        ApplyDisposition::Applied
    );
    assert_eq!(
        apply(&conn, &game_finished(7, "second")),
        ApplyDisposition::Applied
    );
    assert_eq!(apply(&conn, &pair_finished(8)), ApplyDisposition::Applied);
    assert_eq!(
        apply(&conn, &game_finished(9, "first")),
        ApplyDisposition::Rejected
    );
    let trial = event(
        "trial-complete",
        9,
        "trial_completed",
        serde_json::json!({"trial_id": "trial-1", "score": 20.5}),
    );
    assert_eq!(apply(&conn, &trial), ApplyDisposition::Applied);
    let late_pair = event("late-pair", 10, "pair_started", pair_started(10).payload);
    assert_eq!(apply(&conn, &late_pair), ApplyDisposition::Rejected);
}

#[test]
fn pair_failure_precedes_trial_failure_without_score_or_rating() {
    let conn = fixture();
    for item in started_trial_events() {
        apply(&conn, &item);
    }
    assert_eq!(apply(&conn, &pair_started(5)), ApplyDisposition::Applied);
    let failed = event(
        "pair-failed",
        6,
        "pair_failed",
        serde_json::json!({"trial_id": "trial-1", "pair_id": "pair-1", "pair_index": 0, "error": "timeout"}),
    );
    assert_eq!(apply(&conn, &failed), ApplyDisposition::Applied);
    let terminal = event(
        "trial-failed",
        7,
        "trial_failed",
        serde_json::json!({"trial_id": "trial-1", "error": "timeout"}),
    );
    assert_eq!(apply(&conn, &terminal), ApplyDisposition::Applied);
    let projection: (String, Option<f64>, Option<f64>, String) = conn.query_row(
        "SELECT p.status, p.score, p.rating_after_mu, t.status FROM tuning_evaluation_pairs p JOIN tuning_trials t ON t.session_id = p.session_id AND t.trial_id = p.trial_id WHERE p.pair_id = 'pair-1'",
        [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    ).unwrap();
    assert_eq!(projection, ("failed".into(), None, None, "failed".into()));
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

#[test]
fn modern_attempts_use_explicit_runs_and_legacy_events_fall_back_to_the_source() {
    let conn = fixture();
    conn.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('explicit-run', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'running', '/tmp/explicit.log'), ('legacy-run', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'running', '/tmp/legacy.log')", []).unwrap();
    let started = event(
        "e1",
        1,
        "session_started",
        serde_json::json!({
            "manifest": {}, "manifest_fingerprint": "f",
            "optimizer_id": "optimizer-1", "lifecycle_path": "/persist/lifecycle.jsonl"
        }),
    );
    assert_eq!(apply(&conn, &started), ApplyDisposition::Applied);
    let mut explicit = event(
        "e2",
        2,
        "attempt_started",
        serde_json::json!({"optimizer_id": "optimizer-1", "bench_run_id": "explicit-run"}),
    );
    explicit.attempt_id = "attempt-explicit".to_owned().into();
    assert_eq!(apply(&conn, &explicit), ApplyDisposition::Applied);
    let mut legacy = event("e3", 3, "attempt_started", serde_json::json!({}));
    legacy.attempt_id = "attempt-legacy".to_owned().into();
    let tx = conn.unchecked_transaction().unwrap();
    assert_eq!(
        apply_event(
            &tx,
            &legacy,
            Some("legacy-run"),
            "/persist/lifecycle.jsonl",
            2
        )
        .unwrap(),
        ApplyDisposition::Applied
    );
    tx.commit().unwrap();

    let attempts: Vec<(String, Option<String>)> = conn
        .prepare("SELECT attempt_id, bench_run_id FROM tuning_attempts ORDER BY attempt_id")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        attempts,
        vec![
            ("attempt-explicit".into(), Some("explicit-run".into())),
            ("attempt-legacy".into(), Some("legacy-run".into())),
        ]
    );
    let session: (Option<String>, Option<String>) = conn.query_row(
        "SELECT optimizer_id, lifecycle_path FROM tuning_sessions WHERE session_id = 'session-1'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(
        session,
        (Some("optimizer-1".into()), Some("lifecycle.jsonl".into()))
    );
}

#[test]
fn registered_source_has_one_cursor_and_survives_its_first_run_directory() {
    use std::io::Write;

    let root = std::env::temp_dir().join(format!(
        "mcts_tuning_registered_source_{}",
        std::process::id()
    ));
    let source = root.join("optimizer").join("lifecycle.jsonl");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    let mut file = std::fs::File::create(&source).unwrap();
    writeln!(file, "{}", serde_json::to_string(&event("e1", 1, "session_started", serde_json::json!({"manifest": {}, "manifest_fingerprint": "f", "optimizer_id": "optimizer-1"}))).unwrap()).unwrap();
    let conn = Connection::open_in_memory().unwrap();
    ensure_schema(&conn).unwrap();
    let source_path = source
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    conn.execute(
        "INSERT INTO tuning_lifecycle_sources (source_path, bench_run_id) VALUES (?1, 'gone-run')",
        duckdb::params![&source_path],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('gone-run', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'completed', '/tmp/gone.log')",
        [],
    )
    .unwrap();
    conn.execute("DELETE FROM runs WHERE run_id = 'gone-run'", [])
        .unwrap();
    std::fs::remove_dir_all(root.join("optimizer")).unwrap();
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    let mut recreated = std::fs::File::create(&source).unwrap();
    writeln!(recreated, "{}", serde_json::to_string(&event("e1", 1, "session_started", serde_json::json!({"manifest": {}, "manifest_fingerprint": "f", "optimizer_id": "optimizer-1"}))).unwrap()).unwrap();
    ingest_once(&conn, &root).unwrap();
    ingest_once(&conn, &root).unwrap();
    assert_eq!(
        conn.query_row::<i64, _, _>(
            "SELECT COUNT(*) FROM _ingest_cursor WHERE log_path = ?1",
            duckdb::params![&source_path],
            |row| row.get(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM tuning_sessions", [], |row| row.get(0))
            .unwrap(),
        1
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
#[ignore = "requires MCTS_TUNING_LIFECYCLE_PATH from the real tuner e2e"]
fn real_tuner_artifact_projects_complete_pairs_and_trace_links() {
    let path = std::env::var("MCTS_TUNING_LIFECYCLE_PATH")
        .expect("MCTS_TUNING_LIFECYCLE_PATH must name the real lifecycle artifact");
    let conn = fixture();
    let source = std::fs::read_to_string(&path).unwrap();
    for (offset, line) in source.lines().enumerate() {
        let item: TuningLifecycleEvent = serde_json::from_str(line).unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        let disposition = apply_event(&tx, &item, Some("run-1"), &path, offset as u64).unwrap();
        assert_eq!(disposition, ApplyDisposition::Applied);
        tx.commit().unwrap();
    }

    let (pairs, games, traces, incomplete): (i64, i64, i64, i64) = conn
        .query_row(
            "SELECT (SELECT COUNT(*) FROM tuning_evaluation_pairs), \
                    (SELECT COUNT(*) FROM tuning_games), \
                    (SELECT COUNT(trace_game_seq) FROM tuning_games), \
                    (SELECT COUNT(*) FROM tuning_evaluation_pairs WHERE status <> 'complete')",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert!((5..=15).contains(&pairs));
    assert_eq!(games, pairs * 2);
    assert_eq!(traces, games);
    assert_eq!(incomplete, 0);
}
