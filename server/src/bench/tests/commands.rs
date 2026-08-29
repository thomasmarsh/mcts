#![allow(unused_imports)]
use super::support::*;
use crate::bench::*;
use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode as HttpStatusCode;
use mcts_bench::experiment::ExperimentSpecV1;
use mcts_bench::launch::LaunchedRun;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

#[test]
fn test_build_command_tuner_includes_config_and_overrides() {
    let artifact_root = canonical_tuner_artifact_root("test-run");
    let cmd = build_command(
        "tuner",
        "traffic-lights",
        &Some(json!({
            "config": "tuner/config/default.yaml",
            "overrides": ["optimizer.n_trials=10", "optimizer.n_workers=2"],
        })),
        "test-run",
    )
    .unwrap();

    // First element is the (unresolved-in-test) bench binary path --
    // First element is the (unresolved-in-test) bench binary path.
    assert_eq!(
        cmd[1..],
        vec![
            "tuner",
            "--game",
            "traffic-lights",
            "--config",
            "tuner/config/default.yaml",
            "--override",
            "optimizer.n_trials=10",
            "--override",
            "optimizer.n_workers=2",
            "--artifact-root",
            artifact_root.to_str().unwrap(),
            "--optimizer-id",
            "tuning-session-test-run",
            "--bench-run-id",
            "test-run",
            "--session-id",
            "tuning-session-test-run",
            "--attempt-id",
            "tuning-attempt-test-run",
            "--lifecycle-path",
            &canonical_tuner_lifecycle_path("tuning-session-test-run".into()),
            "--game-kind",
            "traffic-lights",
        ]
    );
}

#[test]
fn test_build_command_tuner_with_no_config_is_just_game() {
    let artifact_root = canonical_tuner_artifact_root("test-run");
    let cmd = build_command("tuner", "druid", &None, "test-run").unwrap();
    assert_eq!(
        cmd[1..],
        vec![
            "tuner",
            "--game",
            "druid",
            "--override",
            "optimizer.n_trials=1000",
            "--artifact-root",
            artifact_root.to_str().unwrap(),
            "--optimizer-id",
            "tuning-session-test-run",
            "--bench-run-id",
            "test-run",
            "--session-id",
            "tuning-session-test-run",
            "--attempt-id",
            "tuning-attempt-test-run",
            "--lifecycle-path",
            &canonical_tuner_lifecycle_path("tuning-session-test-run".into()),
            "--game-kind",
            "druid",
        ]
    );
}

#[test]
fn test_build_command_tuner_includes_artifact_root_derived_from_run_id() {
    let cmd = build_command(
        "tuner",
        "druid",
        &None,
        "tuner-druid-20260101T000000-abcdef",
    )
    .unwrap();
    let idx = cmd
        .iter()
        .position(|a| a == "--artifact-root")
        .expect("--artifact-root flag present");
    assert_eq!(
        cmd[idx + 1],
        canonical_tuner_artifact_root("tuner-druid-20260101T000000-abcdef")
    );
}

#[test]
fn test_build_command_tuner_includes_game_config() {
    let cmd = build_command(
        "tuner",
        "druid",
        &Some(json!({
            "game_config": {"size": {"w": 9, "h": 9}},
        })),
        "test-run",
    )
    .unwrap();

    let idx = cmd
        .iter()
        .position(|a| a == "--game-config")
        .expect("--game-config flag present");
    assert_eq!(cmd[idx + 1], r#"{"size":{"h":9,"w":9}}"#);
}

#[test]
fn test_build_command_tuner_omits_null_game_config() {
    let cmd = build_command(
        "tuner",
        "druid",
        &Some(json!({
            "game_config": null,
        })),
        "test-run",
    )
    .unwrap();
    assert!(!cmd.iter().any(|a| a == "--game-config"));
}

#[test]
fn test_build_command_tuner_includes_baseline_configs() {
    let artifact_root = canonical_tuner_artifact_root("test-run");
    let cmd = build_command(
        "tuner",
        "nim",
        &Some(json!({
            "overrides": ["optimizer.n_trials=10"],
            "baseline_configs": {
                "ladder1": {"family": "ucb1", "c": 1.5},
            },
        })),
        "test-run",
    )
    .unwrap();

    assert_eq!(
        cmd[1..],
        vec![
            "tuner",
            "--game",
            "nim",
            "--override",
            "optimizer.n_trials=10",
            "--baseline-config",
            r#"ladder1={"c":1.5,"family":"ucb1"}"#,
            "--artifact-root",
            artifact_root.to_str().unwrap(),
            "--optimizer-id",
            "tuning-session-test-run",
            "--bench-run-id",
            "test-run",
            "--session-id",
            "tuning-session-test-run",
            "--attempt-id",
            "tuning-attempt-test-run",
            "--lifecycle-path",
            &canonical_tuner_lifecycle_path("tuning-session-test-run".into()),
            "--game-kind",
            "nim",
        ]
    );
}

#[test]
fn test_tuner_attempt_builder_keeps_session_artifacts_stable_across_three_physical_runs() {
    let journal = canonical_tuner_lifecycle_path("optimizer-session-a".into());
    let config = Some(json!({
        "config": "tuner/config/default.yaml",
        "overrides": [
            "target.rounds=30",
            "optimizer.n_trials=3",
            "optimizer.n_trials=4",
            "optimizer.n_workers=2"
        ],
    }));
    let attempts = [
        ("attempt-1", "physical-1", 10, None),
        ("attempt-2", "physical-2", 25, Some(4)),
        ("attempt-3", "physical-3", 25, None),
    ];

    for (attempt_id, physical_run_id, target, workers) in attempts {
        let built = build_tuner_attempt(&TunerAttemptLaunch {
            game: "druid".into(),
            config: config.clone(),
            session_id: "session-a".into(),
            optimizer_id: "optimizer-session-a".into(),
            lifecycle_path: journal.clone(),
            attempt_id: attempt_id.into(),
            physical_run_id: physical_run_id.into(),
            artifact_root: std::env::current_dir()
                .unwrap()
                .join("bench-runs")
                .join(physical_run_id)
                .join("tuning-artifacts"),
            target_trial_count: target,
            workers,
        })
        .unwrap();
        let command = built.command;
        assert!(!command.iter().any(|argument| argument == "--resume"));
        assert_eq!(
            command[command
                .iter()
                .position(|argument| argument == "--artifact-root")
                .unwrap()
                + 1],
            std::env::current_dir()
                .unwrap()
                .join("bench-runs")
                .join(physical_run_id)
                .join("tuning-artifacts")
                .to_string_lossy()
        );
        assert_eq!(
            command[command
                .iter()
                .position(|argument| argument == "--optimizer-id")
                .unwrap()
                + 1],
            "optimizer-session-a"
        );
        assert_eq!(
            command[command
                .iter()
                .position(|argument| argument == "--session-id")
                .unwrap()
                + 1],
            "session-a"
        );
        assert_eq!(
            command[command
                .iter()
                .position(|argument| argument == "--attempt-id")
                .unwrap()
                + 1],
            attempt_id
        );
        assert_eq!(
            command[command
                .iter()
                .position(|argument| argument == "--bench-run-id")
                .unwrap()
                + 1],
            physical_run_id
        );
        assert_eq!(
            command[command
                .iter()
                .position(|argument| argument == "--lifecycle-path")
                .unwrap()
                + 1],
            journal
        );
        let overrides: Vec<&str> = command
            .windows(2)
            .filter(|arguments| arguments[0] == "--override")
            .map(|arguments| arguments[1].as_str())
            .collect();
        assert_eq!(
            overrides
                .iter()
                .filter(|value| value.starts_with("optimizer.n_trials="))
                .count(),
            1
        );
        assert!(overrides.contains(&format!("optimizer.n_trials={target}").as_str()));
        match workers {
            Some(workers) => {
                assert!(overrides.contains(&format!("optimizer.n_workers={workers}").as_str()))
            }
            None => assert!(overrides.contains(&"optimizer.n_workers=2")),
        }
    }
}

#[test]
fn test_tuner_attempt_builder_selects_nego_external_host() {
    let built = build_tuner_attempt(&TunerAttemptLaunch {
        game: "nego".into(),
        config: None,
        session_id: "session-nego".into(),
        optimizer_id: "optimizer-nego".into(),
        lifecycle_path: "/tmp/nego.lifecycle.jsonl".into(),
        attempt_id: "attempt-nego".into(),
        physical_run_id: "physical-nego".into(),
        artifact_root: std::env::current_dir()
            .unwrap()
            .join("bench-runs")
            .join("physical-nego")
            .join("tuning-artifacts"),
        target_trial_count: 1,
        workers: None,
    })
    .unwrap();

    assert!(built
        .command
        .windows(2)
        .any(|arguments| { arguments == ["--target-binary", "../nego/target/release/nego-host"] }));
}

#[test]
fn test_reserved_tuner_launch_records_once_and_reuses_the_physical_identity() {
    let launches = Arc::new(Mutex::new(0));
    let observed = launches.clone();
    let (app, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, config, git_sha, git_dirty, host, started_at, ended_at, status, log_path) VALUES ('physical-parent', 'tuner', 'druid', '{}', 'sha', false, 'host', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'completed', '/tmp/parent.log')",
                [],
            )
            .unwrap();
            let tx = conn.unchecked_transaction().unwrap();
            mcts_bench::identity::create_root_identity(
                &tx,
                "physical-parent",
                "tuner",
                None,
                None,
                "2026-01-01T00:00:00Z",
            )
            .unwrap();
            tx.commit().unwrap();
            conn.execute(
                "INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence, optimizer_id, lifecycle_path) VALUES ('session-a', 'idle', '{}', 3, CURRENT_TIMESTAMP, 1, 'optimizer-a', '/tmp/session-a.jsonl')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at, ended_at) VALUES ('attempt-parent', 'session-a', 'physical-parent', 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO tuning_launch_reservations (session_id, command_id, attempt_id, physical_run_id, target_trial_count, reserved_at) VALUES ('session-a', 'resume-a', 'attempt-next', 'physical-next', 8, CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
        },
        Arc::new(move |run_id, _command, _kind, _game, _label| {
            *observed.lock().unwrap() += 1;
            Ok(LaunchedRun {
                log_path: PathBuf::from(format!("bench-runs/{run_id}/log.jsonl")),
                log_dir: PathBuf::from(format!("bench-runs/{run_id}")),
                run_id,
                pid: 321,
            })
        }),
    );
    drop(app);
    let launch = TunerAttemptLaunch {
        game: "druid".into(),
        config: Some(json!({"overrides": ["optimizer.n_trials=3"]})),
        session_id: "session-a".into(),
        optimizer_id: "optimizer-a".into(),
        lifecycle_path: "/tmp/session-a.jsonl".into(),
        attempt_id: "attempt-next".into(),
        physical_run_id: "physical-next".into(),
        artifact_root: std::env::current_dir()
            .unwrap()
            .join("bench-runs/physical-next/tuning-artifacts"),
        target_trial_count: 8,
        workers: Some(4),
    };

    let first = launch_reserved_tuner_attempt(&state, "resume-a", launch.clone(), None).unwrap();
    let replay = launch_reserved_tuner_attempt(&state, "resume-a", launch, None).unwrap();
    assert_eq!(first.run_id, "physical-next");
    assert_eq!(replay.run_id, "physical-next");
    assert_eq!(*launches.lock().unwrap(), 1);
    let identity: (String, String, u64) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = 'physical-next'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        identity,
        ("physical-parent".into(), "physical-parent".into(), 2)
    );
}

#[test]
fn test_reserved_tuner_spawn_failure_releases_its_reservation() {
    let (_, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, config, git_sha, git_dirty, host, started_at, ended_at, status, log_path) VALUES ('failure-parent', 'tuner', 'druid', '{}', 'sha', false, 'host', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'completed', '/tmp/parent.log')",
                [],
            )
            .unwrap();
            let tx = conn.unchecked_transaction().unwrap();
            mcts_bench::identity::create_root_identity(
                &tx,
                "failure-parent",
                "tuner",
                None,
                None,
                "2026-01-01T00:00:00Z",
            )
            .unwrap();
            tx.commit().unwrap();
            conn.execute(
                "INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence, optimizer_id, lifecycle_path) VALUES ('session-failure', 'idle', '{}', 3, CURRENT_TIMESTAMP, 1, 'optimizer-failure', '/tmp/session-failure.jsonl')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at, ended_at) VALUES ('attempt-parent-failure', 'session-failure', 'failure-parent', 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO tuning_launch_reservations (session_id, command_id, attempt_id, physical_run_id, target_trial_count, reserved_at) VALUES ('session-failure', 'resume-failure', 'attempt-failure', 'physical-failure', 8, CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
        },
        Arc::new(|_, _, _, _, _| Err(std::io::Error::other("injected spawn failure"))),
    );
    let error = match launch_reserved_tuner_attempt(
        &state,
        "resume-failure",
        TunerAttemptLaunch {
            game: "druid".into(),
            config: None,
            session_id: "session-failure".into(),
            optimizer_id: "optimizer-failure".into(),
            lifecycle_path: "/tmp/session-failure.jsonl".into(),
            attempt_id: "attempt-failure".into(),
            physical_run_id: "physical-failure".into(),
            artifact_root: std::env::current_dir()
                .unwrap()
                .join("bench-runs/physical-failure/tuning-artifacts"),
            target_trial_count: 8,
            workers: None,
        },
        None,
    ) {
        Ok(_) => panic!("injected launcher unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.message.contains("injected spawn failure"));
    let remaining: i64 = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM tuning_launch_reservations WHERE session_id = 'session-failure'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(remaining, 0);
}

#[test]
fn test_build_command_unknown_kind_lists_tuner_as_supported() {
    let err = build_command("nope", "druid", &None, "test-run").unwrap_err();
    assert!(err.message.contains("tuner"));
}

// -------------------------------------------------------------------
// POST /api/bench/launch
// -------------------------------------------------------------------

#[tokio::test]
async fn test_launch_rejects_unknown_kind() {
    let app = seeded_app(|_, _| {}).0;
    let (status, body) = http_post_json(
        app,
        "/api/bench/launch",
        json!({
            "kind": "unknown_kind",
            "game": "druid",
            "config": null
        }),
    )
    .await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    let body = body_json(&body);
    assert_eq!(body["code"], 400);
    assert!(body["error"].as_str().unwrap().contains("unknown_kind"));
}

#[tokio::test]
async fn test_launch_tuner_reaches_the_launcher() {
    // Same shape as test_launch_spawns_bench_and_returns_run_id above,
    // for the "tuner" kind -- proves build_command's tuner arm produces
    // a request the handler accepts and forwards to launch::launch
    // (a 400 here would mean it was rejected as an unknown kind before
    // ever reaching the launcher).
    let app = seeded_app(|_conn, dir| {
        std::fs::create_dir_all(dir).ok();
    })
    .0;

    let (status, body) = http_post_json(
        app,
        "/api/bench/launch",
        json!({
            "kind": "tuner",
            "game": "traffic-lights",
            "config": {
                "config": "tuner/config/default.yaml",
                "overrides": ["optimizer.n_trials=1"]
            }
        }),
    )
    .await;

    assert!(
        status == HttpStatusCode::OK || status == HttpStatusCode::INTERNAL_SERVER_ERROR,
        "tuner launch returned unexpected status {status}: body={}",
        String::from_utf8_lossy(&body),
    );
}

#[tokio::test]
async fn test_fresh_tuner_launch_creates_an_identity_root() {
    let (app, _, state) = seeded_app_with_state(|_, _| {}, injected_general_launcher());

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/launch",
        json!({"kind": "tuner", "game": "traffic-lights", "config": {"overrides": []}}),
    )
    .await;
    assert_eq!(
        status,
        HttpStatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let run_id = body_json(&body)["run_id"].as_str().unwrap().to_owned();
    let identity: (String, Option<String>, u64, String) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT r.logical_run_id, r.parent_attempt_id, r.attempt_ordinal, l.current_attempt_id FROM runs r JOIN logical_runs l ON l.logical_run_id = r.logical_run_id WHERE r.run_id = ?1",
            duckdb::params![&run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(identity, (run_id.clone(), None, 1, run_id.clone()));
    let source: (String, String) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT source_path, bench_run_id FROM tuning_lifecycle_sources WHERE bench_run_id = ?1",
            duckdb::params![&run_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(source.1, run_id);
    assert!(source.0.contains("/optuna_output/tuning-session-"));
    assert!(source.0.ends_with("/lifecycle.jsonl"));
}
