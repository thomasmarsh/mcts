use super::support::*;
use axum::http::StatusCode as HttpStatusCode;
use std::sync::Arc;

fn control_session_seed(conn: &duckdb::Connection, status: &str, attempt_status: &str) {
    conn.execute_batch(&format!(
        "INSERT INTO runs (run_id, kind, game, config, git_sha, git_dirty, host, pid, started_at, status, log_path)
         VALUES ('physical-parent', 'tuner', 'nim', '{{\"config\":\"tuner/config/default.yaml\"}}', 'sha', false, 'host', 4242, CURRENT_TIMESTAMP, '{status}', '/tmp/parent.log');
         INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence, optimizer_id, lifecycle_path)
         VALUES ('control', 'idle', '{{}}', 4, CURRENT_TIMESTAMP, 1, 'optimizer-control', '/tmp/control.jsonl');
         INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at)
         VALUES ('attempt-parent', 'control', 'physical-parent', '{attempt_status}', CURRENT_TIMESTAMP);"
    ))
    .unwrap();
}

#[tokio::test]
async fn tuning_session_stop_is_pending_until_lifecycle_and_replays_without_a_second_signal() {
    let signals = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let signals_for_app = signals.clone();
    let (app, _, state) = seeded_app_with_state_and_signaller(
        |conn, _| control_session_seed(conn, "running", "running"),
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
        Arc::new(move |_| {
            signals_for_app.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"))
        }),
    );
    let request = serde_json::json!({"command_id":"stop-one", "expected_version":0});
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/stop",
        request.clone(),
    )
    .await;
    assert_eq!(status, HttpStatusCode::ACCEPTED);
    let value = body_json(&body);
    assert_eq!(value["status"], "stopping");
    assert_eq!(value["attempt_id"], "attempt-parent");
    assert_eq!(value["signal"], "not_found");
    assert_eq!(
        value["control"]["continuation"]["stop_attempt_id"],
        "attempt-parent"
    );
    assert_eq!(signals.load(std::sync::atomic::Ordering::Relaxed), 1);
    let active: String = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT status FROM tuning_attempts WHERE attempt_id = 'attempt-parent'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(active, "running");

    let (status, body) =
        http_post_json(app, "/api/bench/tuner/sessions/control/stop", request).await;
    assert_eq!(status, HttpStatusCode::ACCEPTED);
    assert_eq!(body_json(&body)["replay"], true);
    assert!(body_json(&body)["signal"].is_null());
    assert_eq!(signals.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[tokio::test]
async fn tuning_session_stop_surfaces_signal_failures_after_recording_the_stop_intent() {
    let (app, _, state) = seeded_app_with_state_and_signaller(
        |conn, _| control_session_seed(conn, "running", "running"),
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
        Arc::new(|_| Err(std::io::Error::other("permission denied"))),
    );
    let (status, body) = http_post_json(
        app,
        "/api/bench/tuner/sessions/control/stop",
        serde_json::json!({"command_id":"stop-error", "expected_version":0}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["code"], 500);
    let reservation_count: i64 = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM tuning_stop_reservations WHERE session_id = 'control'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(reservation_count, 1);
}

#[tokio::test]
async fn tuning_session_stop_reports_a_sent_signal() {
    let app = seeded_app_with_state_and_signaller(
        |conn, _| control_session_seed(conn, "running", "running"),
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
        Arc::new(|_| Ok(())),
    )
    .0;
    let (status, body) = http_post_json(
        app,
        "/api/bench/tuner/sessions/control/stop",
        serde_json::json!({"command_id":"stop-sent", "expected_version":0}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::ACCEPTED);
    assert_eq!(body_json(&body)["signal"], "sent");
}

#[tokio::test]
async fn tuning_session_resume_reserves_one_physical_attempt_and_replays_it() {
    let (app, _, state) = seeded_app_with_state(
        |conn, _| control_session_seed(conn, "completed", "completed"),
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    );
    let request = serde_json::json!({"command_id":"resume-one", "expected_version":0});
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/resume",
        request.clone(),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CREATED);
    let value = body_json(&body);
    assert_eq!(value["status"], "resuming");
    let attempt_id = value["attempt_id"].as_str().unwrap().to_owned();
    let run_id = value["bench_run_id"].as_str().unwrap().to_owned();
    assert!(attempt_id.contains(&run_id));
    assert_eq!(value["control"]["continuation"]["target_trial_count"], 4);
    assert_eq!(value["control"]["version"], 1);
    assert_eq!(
        value["control"]["continuation"]["launch_reservation"]["attempt_id"],
        attempt_id
    );

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/resume",
        serde_json::json!({"command_id":"resume-two", "expected_version":1}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CONFLICT);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("reserved"));

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/resume",
        serde_json::json!({"command_id":"resume-one", "expected_version":1}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CONFLICT);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("reused"));

    let (status, body) =
        http_post_json(app, "/api/bench/tuner/sessions/control/resume", request).await;
    assert_eq!(status, HttpStatusCode::CREATED);
    let replay = body_json(&body);
    assert_eq!(replay["replay"], true);
    assert_eq!(replay["attempt_id"], attempt_id);
    assert_eq!(replay["bench_run_id"], run_id);
    let launches: i64 = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM runs WHERE run_id <> 'physical-parent'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(launches, 1);
}

#[tokio::test]
async fn tuning_session_resume_allows_a_conclusively_dead_recovery_attempt() {
    let app = seeded_app_with(
        |conn, _| control_session_seed(conn, "crashed", "running"),
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    )
    .0;
    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/control").await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(
        body_json(&body)["control"]["continuation"]["recovery_required"],
        true
    );
    assert!(body_json(&body)["control"]["allowed_commands"]
        .as_array()
        .unwrap()
        .iter()
        .any(|command| command["command"] == "resume" && command["allowed"] == true));

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/resume",
        serde_json::json!({"command_id":"recover-one", "expected_version":0}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CREATED);
    assert_eq!(body_json(&body)["status"], "resuming");
}

#[tokio::test]
async fn tuning_session_resume_releases_a_failed_spawn_reservation() {
    let (app, _, state) = seeded_app_with_state(
        |conn, _| control_session_seed(conn, "completed", "completed"),
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        Arc::new(|_, _, _, _, _| Err(std::io::Error::other("injected spawn failure"))),
    );
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/resume",
        serde_json::json!({"command_id":"resume-failure", "expected_version":0}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("failed to launch tuner attempt"));
    let (status, body) = http_post_json(
        app,
        "/api/bench/tuner/sessions/control/resume",
        serde_json::json!({"command_id":"resume-failure", "expected_version":0}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("recorded launch"));
    let (reservations, version): (i64, i64) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT (SELECT COUNT(*) FROM tuning_launch_reservations WHERE session_id = 'control'), control_version
             FROM tuning_sessions WHERE session_id = 'control'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(reservations, 0);
    assert_eq!(version, 2);
}

#[tokio::test]
async fn tuning_session_budget_extends_an_active_attempt_once_without_relaunching_it() {
    let launches = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = launches.clone();
    let (app, _, state) = seeded_app_with_state(
        |conn, _| control_session_seed(conn, "running", "running"),
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        Arc::new(move |_, _, _, _, _| {
            observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(mcts_bench::launch::LaunchedRun {
                run_id: "unexpected".into(),
                pid: 1,
                log_path: std::path::PathBuf::from("/tmp/unexpected.log"),
                log_dir: std::path::PathBuf::from("/tmp"),
            })
        }),
    );
    let request = serde_json::json!({
        "command_id": "active-extend",
        "expected_version": 0,
        "delta": 2,
        "start": false,
    });
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/budget",
        request.clone(),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(value["status"], "extended");
    assert_eq!(
        value["budget"],
        serde_json::json!({"previous_target_trial_count": 4, "delta": 2, "target_trial_count": 6})
    );
    assert_eq!(
        value["control"]["continuation"]["active_attempt_id"],
        "attempt-parent"
    );
    assert_eq!(value["control"]["continuation"]["target_trial_count"], 6);
    assert_eq!(value["control"]["version"], 1);
    assert_eq!(launches.load(std::sync::atomic::Ordering::Relaxed), 0);

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/budget",
        request,
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(body_json(&body)["replay"], true);
    assert_eq!(launches.load(std::sync::atomic::Ordering::Relaxed), 0);

    let (status, _) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/budget",
        serde_json::json!({"command_id":"active-extend", "expected_version":0, "delta":3, "start":false}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CONFLICT);
    let (status, _) = http_post_json(
        app,
        "/api/bench/tuner/sessions/control/budget",
        serde_json::json!({"command_id":"stale-extend", "expected_version":0, "delta":1, "start":false}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CONFLICT);
    let target: i64 = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT target_trial_count FROM tuning_sessions WHERE session_id = 'control'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(target, 6);
}

#[tokio::test]
async fn tuning_session_budget_starts_one_attempt_at_the_new_absolute_target() {
    let launches = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let commands = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_launches = launches.clone();
    let observed_commands = commands.clone();
    let (app, _, _) = seeded_app_with_state(
        |conn, _| control_session_seed(conn, "completed", "completed"),
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        Arc::new(move |run_id, command, _, _, _| {
            observed_launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            observed_commands.lock().unwrap().push(command);
            Ok(mcts_bench::launch::LaunchedRun {
                run_id,
                pid: 11,
                log_path: std::path::PathBuf::from("/tmp/budget.log"),
                log_dir: std::path::PathBuf::from("/tmp"),
            })
        }),
    );
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/budget",
        serde_json::json!({"command_id":"idle-extend", "expected_version":0, "delta":2, "start":false}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(
        body_json(&body)["control"]["continuation"]["remaining_trial_count"],
        6
    );
    assert_eq!(launches.load(std::sync::atomic::Ordering::Relaxed), 0);
    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(body_json(&body)["sessions"][0]["target_trial_count"], 6);
    assert_eq!(
        body_json(&body)["sessions"][0]["control"]["continuation"]["remaining_trial_count"],
        6
    );
    assert_eq!(body_json(&body)["sessions"][0]["control"]["version"], 1);
    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/control/analysis").await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(
        body_json(&body)["control"]["continuation"]["target_trial_count"],
        6
    );
    assert!(body_json(&body)["control"]["allowed_commands"]
        .as_array()
        .unwrap()
        .iter()
        .any(|command| command["command"] == "add_budget" && command["allowed"] == true));

    let request = serde_json::json!({
        "command_id":"extend-and-start",
        "expected_version":1,
        "delta":3,
        "start":true,
        "n_workers":4,
    });
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/budget",
        request.clone(),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CREATED);
    let value = body_json(&body);
    assert_eq!(value["status"], "starting");
    assert_eq!(value["budget"]["previous_target_trial_count"], 6);
    assert_eq!(value["budget"]["target_trial_count"], 9);
    assert_eq!(value["control"]["version"], 2);
    assert_eq!(launches.load(std::sync::atomic::Ordering::Relaxed), 1);
    let command = commands.lock().unwrap()[0].clone();
    let overrides = command
        .windows(2)
        .filter(|arguments| arguments[0] == "--override")
        .map(|arguments| arguments[1].as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        overrides
            .iter()
            .filter(|override_value| **override_value == "optimizer.n_trials=9")
            .count(),
        1
    );
    assert!(overrides.contains(&"optimizer.n_workers=4"));

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/budget",
        request,
    )
    .await;
    assert_eq!(status, HttpStatusCode::CREATED);
    assert_eq!(body_json(&body)["replay"], true);
    assert_eq!(launches.load(std::sync::atomic::Ordering::Relaxed), 1);

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/control").await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(body_json(&body)["summary"]["target_trial_count"], 9);
    assert_eq!(
        body_json(&body)["control"]["continuation"]["remaining_trial_count"],
        9
    );
}

#[tokio::test]
async fn tuning_session_budget_validates_workers_and_releases_failed_starts_for_resume() {
    let launches = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = launches.clone();
    let (app, _, _) = seeded_app_with_state(
        |conn, _| {
            control_session_seed(conn, "completed", "completed");
            conn.execute(
                "UPDATE tuning_sessions SET manifest = '{\"semantic_inputs\":{\"optimizer\":{\"pruning\":{\"enabled\":true}}}}' WHERE session_id = 'control'",
                [],
            )
            .unwrap();
        },
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        Arc::new(move |run_id, _, _, _, _| {
            let call = observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if call == 0 {
                return Err(std::io::Error::other("injected spawn failure"));
            }
            Ok(mcts_bench::launch::LaunchedRun {
                run_id,
                pid: 12,
                log_path: std::path::PathBuf::from("/tmp/recovered.log"),
                log_dir: std::path::PathBuf::from("/tmp"),
            })
        }),
    );
    for request in [
        serde_json::json!({"command_id":"zero", "expected_version":0, "delta":0, "start":false}),
        serde_json::json!({"command_id":"too-many", "expected_version":0, "delta":1_000_001, "start":false}),
        serde_json::json!({"command_id":"workers-idle", "expected_version":0, "delta":1, "start":false, "n_workers":1}),
        serde_json::json!({"command_id":"workers-range", "expected_version":0, "delta":1, "start":true, "n_workers":1025}),
    ] {
        let (status, _) = http_post_json(
            app.clone(),
            "/api/bench/tuner/sessions/control/budget",
            request,
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    }

    let failure = serde_json::json!({"command_id":"failed-start", "expected_version":0, "delta":3, "start":true, "n_workers":2});
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/budget",
        failure.clone(),
    )
    .await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    let value = body_json(&body);
    assert_eq!(value["status"], "launch_failed");
    assert_eq!(value["budget"]["target_trial_count"], 7);
    assert_eq!(value["control"]["version"], 2);
    assert!(value["launch_error"]
        .as_str()
        .unwrap()
        .contains("injected spawn failure"));

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/budget",
        failure,
    )
    .await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["replay"], true);
    assert_eq!(launches.load(std::sync::atomic::Ordering::Relaxed), 1);

    let (status, body) = http_post_json(
        app,
        "/api/bench/tuner/sessions/control/resume",
        serde_json::json!({"command_id":"resume-after-failure", "expected_version":2}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CREATED);
    assert_eq!(
        body_json(&body)["control"]["continuation"]["target_trial_count"],
        7
    );
    assert_eq!(launches.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[tokio::test]
async fn tuning_session_control_rejects_stale_and_exhausted_resume_commands() {
    let app = seeded_app(|conn, _| {
        control_session_seed(conn, "completed", "completed");
        conn.execute(
            "UPDATE tuning_sessions SET target_trial_count = 1 WHERE session_id = 'control'",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at)
             VALUES ('control', 'spent', 'attempt-parent', 1, 'complete', '{}', CURRENT_TIMESTAMP)",
            [],
        )
        .unwrap();
    })
    .0;
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/sessions/control/resume",
        serde_json::json!({"command_id":"exhausted", "expected_version":0}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CONFLICT);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("exhausted"));

    let (status, body) = http_post_json(
        app,
        "/api/bench/tuner/sessions/control/stop",
        serde_json::json!({"command_id":"stale", "expected_version":99}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::CONFLICT);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("version"));
}

#[tokio::test]
async fn tuning_sessions_list_projects_counts_attempts_capabilities_and_order() {
    let app = seeded_app(|conn, dir| {
        default_seed(conn, dir);
        conn.execute(
            "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES \
             ('tuner-projected', 'tuner', 'nim', 'abc', false, 'test', '2026-01-01T00:00:00Z', 'running', '/tmp/projected.log'), \
             ('tuner-legacy', 'tuner', 'nim', 'abc', false, 'test', '2026-01-01T00:00:00Z', 'completed', '/tmp/legacy.log')",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence) VALUES \
             ('session-a', 'idle', '{\"schema_version\":1}', NULL, '2026-01-02T00:00:00Z', 1), \
             ('session-z', 'idle', '{\"schema_version\":1}', NULL, '2026-01-02T00:00:00Z', 1), \
             ('session-main', 'active', '{\"schema_version\":1,\"semantic_inputs\":{\"game\":{\"kind\":\"nim\",\"label\":\"Nim tuning\"}}}', 8, '2026-01-01T00:00:00Z', 9)",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at, ended_at, failure) VALUES \
             ('attempt-old', 'session-main', NULL, 'stopped', '2026-01-01T00:00:00Z', '2026-01-01T00:01:00Z', 'stopped by user'), \
             ('attempt-live', 'session-main', 'tuner-projected', 'running', '2026-01-03T00:00:00Z', NULL, NULL)",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at) VALUES \
             ('session-main', 'trial-queued', 'attempt-live', 0, 'queued', '{}', '2026-01-03T00:00:00Z'), \
             ('session-main', 'trial-running', 'attempt-live', 1, 'running', '{}', '2026-01-03T00:00:00Z'), \
             ('session-main', 'trial-complete', 'attempt-old', 2, 'complete', '{}', '2026-01-01T00:00:00Z'), \
             ('session-main', 'trial-failed', 'attempt-old', 3, 'failed', '{}', '2026-01-01T00:00:00Z'), \
             ('session-main', 'trial-pruned', 'attempt-old', 4, 'pruned', '{}', '2026-01-01T00:00:00Z')",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, started_at) VALUES \
             ('session-main', 'pair-1', 'trial-running', 'attempt-live', 0, 'running', 7, 1, '{\"anchor_id\":\"a\",\"config\":{},\"mu\":25.0,\"sigma\":1.0}', 'pool', 25.0, 1.0, '2026-01-03T00:01:00Z')",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_games (session_id, pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, candidate_metrics, baseline_metrics, finished_at) VALUES \
             ('session-main', 'pair-1', 'game-1', 'first', 'candidate_win', 7, 1, 11, 10, 20, '{\"iterations_total\":1,\"iterations_first_half\":1,\"move_time_ms\":1}', '{\"iterations_total\":1,\"iterations_first_half\":1,\"move_time_ms\":1}', '2026-01-03T00:02:00Z')",
            [],
        ).unwrap();
    }).0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    let sessions = value["sessions"].as_array().unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        sessions
            .iter()
            .map(|session| &session["session_id"])
            .collect::<Vec<_>>(),
        vec!["session-main", "session-z", "session-a"]
    );
    assert_eq!(sessions[0]["game"], "nim");
    assert_eq!(sessions[0]["label"], "Nim tuning");
    assert_eq!(
        sessions[0]["counts"],
        serde_json::json!({"total": 5, "queued": 1, "running": 1, "terminal": 3, "completed": 1, "failed": 1, "pruned": 1, "cancelled": 0})
    );
    assert_eq!(sessions[0]["attempts"][0]["attempt_id"], "attempt-old");
    assert_eq!(
        sessions[0]["attempts"][1]["bench_run_id"],
        "tuner-projected"
    );
    assert_eq!(
        sessions[0]["attempts"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|attempt| attempt["bench_run_id"].as_str())
            .collect::<Vec<_>>(),
        vec!["tuner-projected"]
    );
    assert_eq!(sessions[0]["capabilities"]["has_pairs"], true);
    assert_eq!(sessions[0]["capabilities"]["has_renderer_trace"], false);
    assert_eq!(sessions[0]["capabilities"]["has_search_reports"], false);
    assert_eq!(sessions[0]["capabilities"]["has_trial_reports"], false);
    assert!(sessions[0]["control"]["allowed_commands"]
        .as_array()
        .unwrap()
        .iter()
        .any(|command| command["command"] == "stop" && command["allowed"] == true));
    assert_eq!(
        sessions[1]["control"]["allowed_commands"][1]["denial_reason"],
        "noncontinuable_legacy"
    );
    assert_ne!(sessions[0]["last_activity_at"], sessions[0]["created_at"]);
    assert!(sessions[1]["game"].is_null());
}

#[tokio::test]
async fn tuning_sessions_list_rejects_a_malformed_manifest() {
    let app = seeded_app(|conn, _| {
        conn.execute(
            "INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('broken', 'idle', '{\"semantic_inputs\":{\"game\":\"nim\"}}', CURRENT_TIMESTAMP, 1)",
            [],
        ).unwrap();
    }).0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    assert_eq!(body_json(&body)["code"], 400);
}

#[tokio::test]
async fn tuning_sessions_list_returns_a_structured_storage_error() {
    let (app, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute(
                "INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'idle', '{}', CURRENT_TIMESTAMP, 1)",
                [],
            ).unwrap();
        },
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    );
    state
        .db
        .lock()
        .unwrap()
        .execute_batch("DROP TABLE tuning_pool_decisions; DROP TABLE tuning_trials")
        .unwrap();

    let (status, body) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["code"], 500);
}

#[tokio::test]
async fn tuning_session_detail_projects_counts_attempts_and_capabilities() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, manifest_fingerprint, target_trial_count, created_at, last_sequence) VALUES ('session-1', 'idle', '{\"game\":\"nim\"}', 'fp', 2, CURRENT_TIMESTAMP, 7)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at, ended_at) VALUES ('attempt-1', 'session-1', NULL, 'stopped', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at, score) VALUES ('session-1', 'trial-1', 'attempt-1', 0, 'complete', '{\"c\":1}', CURRENT_TIMESTAMP, 0.5), ('session-1', 'trial-2', 'attempt-1', 1, 'failed', '{\"c\":2}', CURRENT_TIMESTAMP, NULL)", []).unwrap();
    }).0;
    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(value["summary"]["session_id"], "session-1");
    assert_eq!(value["summary"]["status"], "idle");
    assert_eq!(value["summary"]["counts"]["total"], 2);
    assert_eq!(value["summary"]["counts"]["completed"], 1);
    assert_eq!(value["summary"]["counts"]["failed"], 1);
    assert_eq!(value["attempts"][0]["attempt_id"], "attempt-1");
    assert_eq!(value["trials"][1]["status"], "failed");
    assert_eq!(value["fingerprint"], "fp");
    assert_eq!(value["capabilities"]["has_lifecycle"], true);
    assert_eq!(value["capabilities"]["has_pairs"], false);
    assert_eq!(value["capabilities"]["has_search_reports"], false);
    assert_eq!(value["capabilities"]["has_trial_reports"], false);
    assert!(value["policy"].is_null());
    assert_eq!(value["trials"][0]["stop_reason"], serde_json::Value::Null);
    assert_eq!(value["trials"][0]["reports"], serde_json::json!([]));
    assert_eq!(value["cursor"]["session_sequence"], 7);
}

#[tokio::test]
async fn tuning_session_detail_loads_all_reports_without_per_trial_queries() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'idle', '{\"schema_version\":2,\"semantic_inputs\":{\"game\":{\"kind\":\"nim\"},\"optimizer\":{\"resource\":{\"min_pairs\":2,\"max_pairs\":6},\"sampler\":{\"kind\":\"tpe\",\"seed\":4,\"deterministic\":true,\"startup_trials\":3},\"pruning\":{\"enabled\":true,\"kind\":\"hyperband\",\"reduction_factor\":3.0,\"startup_trials\":5}},\"rating\":{\"model\":\"ThurstoneMostellerPart\",\"score\":\"mu_minus_k_sigma\",\"sigma_stop\":2.0,\"conservative_k\":3.0}}}', '2026-01-01T00:00:00Z', 4)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at) VALUES ('attempt-1', 'session-1', 'completed', '2026-01-01T00:00:00Z')", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at, stop_reason) VALUES ('session-1', 'trial-1', 'attempt-1', 1, 'complete', '{}', '2026-01-01T00:00:00Z', 'max_pairs'), ('session-1', 'trial-2', 'attempt-1', 2, 'pruned', '{}', '2026-01-01T00:00:00Z', 'hyperband_prune')", []).unwrap();
        conn.execute("INSERT INTO tuning_trial_reports (session_id, trial_id, trial_number, completed_pairs, event_id, reported_at, mu, sigma, score, score_formula_version, conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource) VALUES ('session-1', 'trial-1', 1, 4, 'report-4', '2026-01-01T00:04:00Z', 26.0, 1.0, 23.0, 1, 3.0, 'complete', 'max_pairs', false, 'bracket-a', 4), ('session-1', 'trial-2', 2, 2, 'report-2', '2026-01-01T00:02:00Z', 24.0, 2.0, 18.0, 1, 3.0, 'prune', 'hyperband_prune', false, NULL, NULL), ('session-1', 'trial-1', 1, 2, 'report-1', '2026-01-01T00:02:00Z', 25.0, 2.0, 19.0, 1, 3.0, 'continue', 'startup_exempt', true, NULL, NULL)", []).unwrap();
    }).0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(
        value["policy"],
        serde_json::json!({
            "resource": {"min_pairs": 2, "max_pairs": 6},
            "rating": {"model": "ThurstoneMostellerPart", "score": "mu_minus_k_sigma", "sigma_stop": 2.0, "conservative_k": 3.0},
            "sampler": {"kind": "tpe", "seed": 4, "deterministic": true, "startup_trials": 3},
            "pruning": {"enabled": true, "kind": "hyperband", "reduction_factor": 3.0, "startup_trials": 5}
        })
    );
    assert_eq!(
        value["trials"],
        serde_json::json!([
            {
                "trial_id": "trial-1", "trial_number": 1, "attempt_id": "attempt-1", "status": "complete", "config": {}, "score": null, "mu": null, "sigma": null, "stop_reason": "max_pairs", "failure": null, "pairs": [],
                "reports": [
                    {"completed_pairs": 2, "rating": {"mu": 25.0, "sigma": 2.0}, "score": 19.0, "score_formula_version": 1, "conservative_k": 3.0, "decision": {"outcome": "continue", "reason": "startup_exempt", "pruning_exempt": true, "bracket_id": null, "rung_resource": null}, "reported_at": "2026-01-01 00:02:00"},
                    {"completed_pairs": 4, "rating": {"mu": 26.0, "sigma": 1.0}, "score": 23.0, "score_formula_version": 1, "conservative_k": 3.0, "decision": {"outcome": "complete", "reason": "max_pairs", "pruning_exempt": false, "bracket_id": "bracket-a", "rung_resource": 4}, "reported_at": "2026-01-01 00:04:00"}
                ]
            },
            {
                "trial_id": "trial-2", "trial_number": 2, "attempt_id": "attempt-1", "status": "pruned", "config": {}, "score": null, "mu": null, "sigma": null, "stop_reason": "hyperband_prune", "failure": null, "pairs": [],
                "reports": [
                    {"completed_pairs": 2, "rating": {"mu": 24.0, "sigma": 2.0}, "score": 18.0, "score_formula_version": 1, "conservative_k": 3.0, "decision": {"outcome": "prune", "reason": "hyperband_prune", "pruning_exempt": false, "bracket_id": null, "rung_resource": null}, "reported_at": "2026-01-01 00:02:00"}
                ]
            }
        ])
    );
    assert_eq!(value["capabilities"]["has_search_reports"], false);
    assert_eq!(value["capabilities"]["has_trial_reports"], true);
}

#[tokio::test]
async fn tuning_session_detail_rejects_a_malformed_new_policy() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'idle', '{\"semantic_inputs\":{\"optimizer\":{}}}', CURRENT_TIMESTAMP, 1)", []).unwrap();
    }).0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    assert_eq!(body_json(&body)["code"], 400);
}

#[tokio::test]
async fn tuning_session_detail_returns_a_storage_error_when_reports_are_unavailable() {
    let (app, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'idle', '{}', CURRENT_TIMESTAMP, 1)", []).unwrap();
        },
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    );
    state
        .db
        .lock()
        .unwrap()
        .execute_batch("DROP TABLE tuning_trial_reports")
        .unwrap();

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["code"], 500);
}

#[tokio::test]
async fn tuning_session_detail_returns_404_without_affecting_legacy_run_endpoint() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/missing").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);
    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(body_json(&body)["run_id"], DEFAULT_RUN_ID);
}

#[tokio::test]
async fn tuning_session_detail_nests_projected_pairs_games_and_trace_capability() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'active', '{}', CURRENT_TIMESTAMP, 8)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at) VALUES ('attempt-1', 'session-1', 'running', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at) VALUES ('session-1', 'trial-1', 'attempt-1', 0, 'running', '{}', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, rating_after_mu, rating_after_sigma, score, started_at, ended_at) VALUES ('session-1', 'pair-1', 'trial-1', 'attempt-1', 0, 'complete', 7, 1, '{\"anchor_id\":\"a\",\"config\":{},\"mu\":25.0,\"sigma\":1.0}', 'pool', 24.0, 2.0, 25.0, 1.5, 20.5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_games (session_id, pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, candidate_metrics, baseline_metrics, finished_at) VALUES ('session-1', 'pair-1', 'game-1', 'first', 'candidate_win', 7, 1, 99, 12, 30, '{\"iterations_total\":20,\"iterations_first_half\":9,\"move_time_ms\":14}', '{\"iterations_total\":18,\"iterations_first_half\":8,\"move_time_ms\":13}', CURRENT_TIMESTAMP), ('session-1', 'pair-1', 'game-2', 'second', 'draw', 7, 1, NULL, 13, 31, '{\"iterations_total\":21,\"iterations_first_half\":10,\"move_time_ms\":15}', '{\"iterations_total\":19,\"iterations_first_half\":9,\"move_time_ms\":14}', CURRENT_TIMESTAMP)", []).unwrap();
    }).0;
    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(value["trials"][0]["pairs"][0]["pair_id"], "pair-1");
    assert_eq!(
        value["trials"][0]["pairs"][0]["games"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        value["trials"][0]["pairs"][0]["games"][0]["trace_game_seq"],
        99
    );
    assert_eq!(value["capabilities"]["has_pairs"], true);
    assert_eq!(value["capabilities"]["has_renderer_trace"], false);
}

#[tokio::test]
async fn tuning_capabilities_require_the_authoritative_run_move_join() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('bench-run', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'completed', '/tmp/bench-run.log')", []).unwrap();
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-join', 'idle', '{}', CURRENT_TIMESTAMP, 1)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at) VALUES ('attempt-join', 'session-join', 'bench-run', 'completed', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at) VALUES ('session-join', 'trial-join', 'attempt-join', 1, 'complete', '{}', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, started_at) VALUES ('session-join', 'pair-join', 'trial-join', 'attempt-join', 1, 'complete', 7, 1, '{\"anchor_id\":\"a\",\"config\":{},\"mu\":25.0,\"sigma\":1.0}', 'pool', 25.0, 1.0, CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_games (session_id, pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, candidate_metrics, baseline_metrics, finished_at) VALUES ('session-join', 'pair-join', 'game-join', 'first', 'draw', 7, 1, 41, 1, 1, '{\"iterations_total\":1,\"iterations_first_half\":1,\"move_time_ms\":1}', '{\"iterations_total\":1,\"iterations_first_half\":1,\"move_time_ms\":1}', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO game_moves (run_id, game_seq, ply, ts, trace_schema_version, state, search_report) VALUES ('bench-run', 41, 0, CURRENT_TIMESTAMP, 1, '{}', NULL), ('bench-run', 41, 1, CURRENT_TIMESTAMP, 1, '{}', '{}')", []).unwrap();
    }).0;

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/session-join").await;
    assert_eq!(status, HttpStatusCode::OK);
    let capabilities = &body_json(&body)["capabilities"];
    assert_eq!(capabilities["has_renderer_trace"], true);
    assert_eq!(capabilities["has_search_reports"], true);
    assert_eq!(capabilities["has_trial_reports"], false);

    let (status, body) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::OK);
    let capabilities = &body_json(&body)["sessions"][0]["capabilities"];
    assert_eq!(capabilities["has_renderer_trace"], true);
    assert_eq!(capabilities["has_search_reports"], true);
    assert_eq!(capabilities["has_trial_reports"], false);
}

#[tokio::test]
async fn tuning_analysis_overview_keeps_full_coverage_and_compact_evidence() {
    let app = seeded_app(|conn, _| {
        conn.execute_batch(
            "INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence)
             VALUES ('analysis', 'idle', '{\"semantic_inputs\":{\"optimizer\":{\"resource\":{\"min_pairs\":2,\"max_pairs\":8},\"sampler\":{\"kind\":\"tpe\",\"seed\":4,\"deterministic\":true,\"startup_trials\":3},\"pruning\":{\"enabled\":true,\"kind\":\"hyperband\",\"reduction_factor\":3.0,\"startup_trials\":5}},\"rating\":{\"model\":\"tm\",\"score\":\"mu_minus_k_sigma\",\"sigma_stop\":null,\"conservative_k\":3.0}}}', '2026-01-01T00:00:00Z', 17);
             INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at)
             VALUES ('attempt', 'analysis', 'completed', '2026-01-01T00:00:00Z');
             INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at, score)
             VALUES ('analysis', 'trial-1', 'attempt', 1, 'complete', '{\"secret\":true}', CURRENT_TIMESTAMP, 30),
                    ('analysis', 'trial-2', 'attempt', 2, 'complete', '{\"secret\":true}', CURRENT_TIMESTAMP, 30),
                    ('analysis', 'trial-3', 'attempt', 3, 'pruned', '{\"secret\":true}', CURRENT_TIMESTAMP, 999),
                    ('analysis', 'trial-4', 'attempt', 4, 'failed', '{\"secret\":true}', CURRENT_TIMESTAMP, NULL),
                    ('analysis', 'trial-5', 'attempt', 5, 'cancelled', '{\"secret\":true}', CURRENT_TIMESTAMP, NULL);
             INSERT INTO tuning_trial_reports (session_id, trial_id, trial_number, completed_pairs, event_id, reported_at, mu, sigma, score, score_formula_version, conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource)
             VALUES ('analysis', 'trial-1', 1, 1, 'report-1', CURRENT_TIMESTAMP, 21, 3, 12, 1, 3, 'continue', 'below_min_pairs', false, NULL, NULL),
                    ('analysis', 'trial-1', 1, 2, 'report-2', CURRENT_TIMESTAMP, 22, 3, 13, 1, 3, 'continue', 'pruning_disabled', false, 'alpha', 99),
                    ('analysis', 'trial-1', 1, 4, 'report-3', CURRENT_TIMESTAMP, 23, 2, 17, 1, 3, 'continue', 'startup_exempt', true, 'alpha', 99),
                    ('analysis', 'trial-1', 1, 6, 'report-4', CURRENT_TIMESTAMP, 24, 2, 19, 1, 3, 'continue', 'hyperband_keep', false, 'alpha', 99),
                    ('analysis', 'trial-1', 1, 8, 'report-5', CURRENT_TIMESTAMP, 25, 1, 30, 1, 3, 'complete', 'confidence', false, 'beta', 100),
                    ('analysis', 'trial-2', 2, 8, 'report-6', CURRENT_TIMESTAMP, 26, 1, 30, 1, 3, 'complete', 'max_pairs', false, 'beta', 100),
                    ('analysis', 'trial-3', 3, 3, 'report-7', CURRENT_TIMESTAMP, 20, 4, 4, 1, 3, 'prune', 'hyperband_prune', false, 'alpha', NULL);
             INSERT INTO tuning_pool_revisions (session_id, pool_snapshot_fingerprint, display_ordinal, first_event_id, first_attempt_id, observed_at)
             VALUES ('analysis', 'pool-a', 1, 'pool-event-a', 'attempt', '2026-01-01T00:00:00Z'),
                    ('analysis', 'pool-b', 2, 'pool-event-b', 'attempt', '2026-01-01T00:01:00Z');
             INSERT INTO tuning_pool_anchors (session_id, pool_snapshot_fingerprint, anchor_ordinal, anchor_id, config, mu, sigma, provenance, insertion_reason, source_trial_id)
             VALUES ('analysis', 'pool-a', 0, 'bootstrap', '{\"family\":\"rave\"}', 25, 1, 'bootstrap_default', 'bootstrap', NULL),
                    ('analysis', 'pool-b', 0, 'champion', '{\"family\":\"ucb\"}', 30, 2, 'trial', 'champion', 'trial-1');
             INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, started_at)
             VALUES ('analysis', 'pair-a', 'trial-1', 'attempt', 1, 'complete', 1, 1, '{\"anchor_id\":\"a\",\"config\":{},\"mu\":25,\"sigma\":1}', 'pool-a', 25, 1, CURRENT_TIMESTAMP),
                    ('analysis', 'pair-b', 'trial-2', 'attempt', 1, 'running', 2, 1, '{\"anchor_id\":\"b\",\"config\":{},\"mu\":25,\"sigma\":1}', 'pool-b', 25, 1, CURRENT_TIMESTAMP),
                    ('analysis', 'pair-orphan', 'trial-3', 'attempt', 1, 'failed', 3, 1, '{\"anchor_id\":\"c\",\"config\":{},\"mu\":25,\"sigma\":1}', 'missing-pool', 25, 1, CURRENT_TIMESTAMP);",
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/analysis/analysis").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        value["objective"],
        serde_json::json!({"metric":"score", "direction":"maximize", "complete_trials_only":true})
    );
    assert_eq!(value["cursor"]["session_sequence"], 17);
    assert_eq!(
        value["coverage"]["trials"],
        serde_json::json!({"total":5, "queued":0, "running":0, "terminal":5, "completed":2, "failed":1, "pruned":1, "cancelled":1})
    );
    assert_eq!(value["coverage"]["reports"], 7);
    assert_eq!(
        value["coverage"]["pairs"],
        serde_json::json!({"total":3, "running":1, "complete":1, "failed":1, "unmatched_pool_revisions":1})
    );
    assert_eq!(
        value["coverage"]["points"],
        serde_json::json!({"total":7, "returned":7, "sampled":false})
    );
    assert_eq!(value["decision_groups"].as_array().unwrap().len(), 7);
    assert!(value["decision_groups"]
        .as_array()
        .unwrap()
        .iter()
        .any(|group| group == &serde_json::json!({"outcome":"continue", "reason":"startup_exempt", "pruning_exempt":true, "reports":1})));
    assert_eq!(
        value["bracket_resources"][0]["bracket_id"],
        serde_json::Value::Null
    );
    let alpha_resources: Vec<_> = value["bracket_resources"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["bracket_id"] == "alpha")
        .map(|row| row["resource"].as_u64().unwrap())
        .collect();
    assert_eq!(alpha_resources, vec![2, 3, 4, 6]);
    assert_eq!(
        value["best"],
        serde_json::json!({"score":30.0, "trial_ids":["trial-1", "trial-2"]})
    );
    assert_eq!(
        value["pool_revisions"][0]["pool_snapshot_fingerprint"],
        "pool-a"
    );
    assert_eq!(value["pool_revisions"][0]["pair_count"], 1);
    assert_eq!(
        value["pool_revisions"][1]["anchors"][0]["source_trial_id"],
        "trial-1"
    );
    assert!(value.get("trials").is_none());
    assert!(value["points"].as_array().unwrap()[0]
        .get("config")
        .is_none());
}

#[tokio::test]
async fn tuning_analysis_overview_handles_empty_missing_and_malformed_policy_sessions() {
    let app = seeded_app(|conn, _| {
        conn.execute_batch(
            "INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence)
             VALUES ('empty', 'idle', '{}', CURRENT_TIMESTAMP, 2),
                    ('malformed', 'idle', '{\"semantic_inputs\":{\"optimizer\":{}}}', CURRENT_TIMESTAMP, 3);",
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/empty/analysis").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert!(value["policy"].is_null());
    assert_eq!(value["coverage"]["reports"], 0);
    assert_eq!(
        value["coverage"]["points"],
        serde_json::json!({"total":0, "returned":0, "sampled":false})
    );
    assert_eq!(value["best"], serde_json::Value::Null);
    assert_eq!(value["pool_revisions"], serde_json::json!([]));

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/missing/analysis").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/malformed/analysis").await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    assert_eq!(body_json(&body)["code"], 400);
}

#[tokio::test]
async fn tuning_analysis_overview_returns_a_structured_error_for_malformed_pool_evidence() {
    let app = seeded_app(|conn, _| {
        conn.execute_batch(
            "INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence)
             VALUES ('corrupt', 'idle', '{}', CURRENT_TIMESTAMP, 1);
             INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at)
             VALUES ('attempt', 'corrupt', 'completed', CURRENT_TIMESTAMP);
             INSERT INTO tuning_pool_revisions (session_id, pool_snapshot_fingerprint, display_ordinal, first_event_id, first_attempt_id, observed_at)
             VALUES ('corrupt', 'pool', 1, 'event', 'attempt', CURRENT_TIMESTAMP);
             DROP TABLE tuning_pool_anchors;
             CREATE TABLE tuning_pool_anchors (
                 session_id TEXT, pool_snapshot_fingerprint TEXT, anchor_ordinal UINTEGER,
                 anchor_id TEXT, config TEXT, mu DOUBLE, sigma DOUBLE, provenance TEXT,
                 insertion_reason TEXT, source_trial_id TEXT
             );
             INSERT INTO tuning_pool_anchors VALUES
             ('corrupt', 'pool', 0, 'bad-anchor', 'not-json', 25, 1, 'trial', 'champion', NULL);",
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/corrupt/analysis").await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["code"], 500);
}

#[tokio::test]
async fn tuning_analysis_overview_caps_points_deterministically_without_losing_rare_outcomes() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('sampled', 'idle', '{}', CURRENT_TIMESTAMP, 1)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at) VALUES ('attempt', 'sampled', 'completed', CURRENT_TIMESTAMP)", []).unwrap();
        for number in 0..2_001_i64 {
            conn.execute(
                "INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at) VALUES (?1, ?2, 'attempt', ?3, 'complete', '{}', CURRENT_TIMESTAMP)",
                duckdb::params!["sampled", format!("trial-{number:04}"), number],
            )
            .unwrap();
            let rare = number == 2_000;
            conn.execute(
                "INSERT INTO tuning_trial_reports (session_id, trial_id, trial_number, completed_pairs, event_id, reported_at, mu, sigma, score, score_formula_version, conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource) VALUES (?1, ?2, ?3, 1, ?4, CURRENT_TIMESTAMP, 25, 1, ?5, 1, 3, ?6, ?7, false, ?8, NULL)",
                duckdb::params!["sampled", format!("trial-{number:04}"), number, format!("report-{number:04}"), number as f64, if rare { "prune" } else { "continue" }, if rare { "hyperband_prune" } else { "below_min_pairs" }, if rare { "rare" } else { "common" }],
            )
            .unwrap();
        }
    })
    .0;

    let (_, first) = http_get(app.clone(), "/api/bench/tuner/sessions/sampled/analysis").await;
    let (status, second) = http_get(app, "/api/bench/tuner/sessions/sampled/analysis").await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(first, second);
    let value = body_json(&second);
    assert_eq!(
        value["coverage"]["points"],
        serde_json::json!({"total":2001, "returned":2000, "sampled":true})
    );
    assert!(value["points"]
        .as_array()
        .unwrap()
        .iter()
        .any(|point| point["outcome"] == "prune"));
}

#[tokio::test]
async fn paged_trial_evidence_filters_sorts_and_binds_cursors() {
    let app = seeded_app(seed_paged_trial_evidence).0;

    let (status, body) =
        http_get(app.clone(), "/api/bench/tuner/sessions/page/trials?limit=2").await;
    assert_eq!(status, HttpStatusCode::OK);
    let first = body_json(&body);
    assert_eq!(first["schema_version"], 1);
    assert_eq!(first["total_count"], 5);
    assert_eq!(first["limit"], 2);
    assert_eq!(first["cursor"]["session_sequence"], 42);
    assert_eq!(trial_ids(&first), vec!["trial-e", "trial-d"]);
    assert!(first["trials"][0].get("config").is_none());
    assert!(first["trials"][0].get("reports").is_none());
    assert!(first["trials"][0].get("pairs").is_none());
    assert_eq!(first["trials"][0]["has_detail"], false);

    let (status, body) = http_get(
        app.clone(),
        "/api/bench/tuner/sessions/page/trials?limit=200&sort=trial&direction=asc",
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let capped = body_json(&body);
    assert_eq!(capped["limit"], 200);
    assert_eq!(capped["trials"][0]["pair_count"], 2);
    assert_eq!(capped["trials"][0]["wins"], 1);
    assert_eq!(capped["trials"][0]["losses"], 1);
    assert_eq!(capped["trials"][0]["draws"], 0);
    assert_eq!(capped["trials"][0]["elapsed_ms"], 25);
    assert_eq!(capped["trials"][0]["search_iterations_total"], 46);
    assert_eq!(capped["trials"][0]["search_move_time_ms"], 30);

    let cursor = first["next_cursor"].as_str().unwrap();
    let (status, body) = http_get(
        app.clone(),
        &format!("/api/bench/tuner/sessions/page/trials?limit=2&cursor={cursor}"),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let second = body_json(&body);
    assert_eq!(trial_ids(&second), vec!["trial-c", "trial-b"]);
    let cursor = second["next_cursor"].as_str().unwrap();
    let (status, body) = http_get(
        app.clone(),
        &format!("/api/bench/tuner/sessions/page/trials?limit=2&cursor={cursor}"),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(trial_ids(&body_json(&body)), vec!["trial-a"]);

    let (status, body) = http_get(
        app.clone(),
        &format!("/api/bench/tuner/sessions/page/trials?state=complete&cursor={cursor}"),
    )
    .await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    assert_eq!(body_json(&body)["code"], 400);

    for (query, expected) in [
        ("state=complete", vec!["trial-b", "trial-a"]),
        ("bracket=alpha", vec!["trial-b", "trial-a"]),
        ("bracket=unassigned", vec!["trial-e", "trial-c"]),
        ("reason=max_pairs", vec!["trial-a"]),
        ("family=ucb", vec!["trial-b", "trial-a"]),
        ("q=UCB", vec!["trial-b", "trial-a"]),
        ("q=trial-c", vec!["trial-c"]),
    ] {
        let (status, body) = http_get(
            app.clone(),
            &format!("/api/bench/tuner/sessions/page/trials?{query}"),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK, "{query}");
        assert_eq!(trial_ids(&body_json(&body)), expected, "{query}");
    }

    for sort in [
        "trial", "state", "score", "mu", "sigma", "resource", "family",
    ] {
        for direction in ["asc", "desc"] {
            let (status, body) = http_get(
                app.clone(),
                &format!("/api/bench/tuner/sessions/page/trials?sort={sort}&direction={direction}"),
            )
            .await;
            assert_eq!(status, HttpStatusCode::OK, "{sort} {direction}");
            assert_eq!(body_json(&body)["trials"].as_array().unwrap().len(), 5);
        }
    }
    let (_, body) = http_get(
        app.clone(),
        "/api/bench/tuner/sessions/page/trials?sort=score&direction=asc",
    )
    .await;
    assert_eq!(
        trial_ids(&body_json(&body)),
        vec!["trial-d", "trial-a", "trial-b", "trial-c", "trial-e"]
    );
    let (_, body) = http_get(
        app.clone(),
        "/api/bench/tuner/sessions/page/trials?sort=score&direction=desc",
    )
    .await;
    assert_eq!(
        trial_ids(&body_json(&body)),
        vec!["trial-c", "trial-a", "trial-b", "trial-d", "trial-e"]
    );

    for invalid in [
        "state=unknown",
        "reason=unknown",
        "sort=unknown",
        "direction=sideways",
        "limit=0",
        "limit=201",
        "cursor=not-a-cursor",
    ] {
        let (status, _) = http_get(
            app.clone(),
            &format!("/api/bench/tuner/sessions/page/trials?{invalid}"),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST, "{invalid}");
    }
}

#[tokio::test]
async fn trial_evidence_detail_uses_exact_snapshots_and_stays_session_scoped() {
    let app = seeded_app(seed_paged_trial_evidence).0;
    let (status, body) =
        http_get(app.clone(), "/api/bench/tuner/sessions/page/trials/trial-a").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["cursor"]["session_sequence"], 42);
    assert_eq!(
        value["trial"]["config"],
        serde_json::json!({"family":"ucb","c":1.2})
    );
    assert_eq!(value["trial"]["reason"], "max_pairs");
    assert_eq!(
        value["trial"]["reports"]
            .as_array()
            .unwrap()
            .iter()
            .map(|report| report["completed_pairs"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![2, 4]
    );
    let pair = &value["trial"]["pairs"][0];
    assert_eq!(
        pair["opponent"]["config"],
        serde_json::json!({"family":"opponent"})
    );
    assert_eq!(
        pair["pool_revision"]["pool_snapshot_fingerprint"],
        "pool-stored"
    );
    assert_eq!(
        pair["pool_revision"]["anchors"][0]["config"],
        serde_json::json!({"family":"revision"})
    );
    assert_eq!(
        pair["games"]
            .as_array()
            .unwrap()
            .iter()
            .map(|game| game["candidate_side"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );
    assert_eq!(pair["games"][0]["replay"]["run_id"], "page-run");
    assert_eq!(pair["games"][0]["replay"]["has_renderer_trace"], true);
    assert_eq!(pair["games"][0]["replay"]["has_search_reports"], true);
    assert!(value["trial"]["pairs"][1]["pool_revision"].is_null());

    let (status, body) = http_get(
        app.clone(),
        "/api/bench/tuner/sessions/other/trials/trial-b",
    )
    .await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);
    let (status, body) =
        http_get(app.clone(), "/api/bench/tuner/sessions/page/trials/trial-e").await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(body_json(&body)["trial"]["pairs"], serde_json::json!([]));

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/page").await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(body_json(&body)["trials"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn trial_evidence_rejects_malformed_persisted_config() {
    let app = seeded_app(|conn, _| {
        conn.execute_batch(
            "INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence)
             VALUES ('bad-config', 'idle', '{}', CURRENT_TIMESTAMP, 1);
             INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at)
             VALUES ('bad-attempt', 'bad-config', 'completed', CURRENT_TIMESTAMP);
             DROP TABLE tuning_pool_decisions;
             DROP TABLE tuning_trials;
             CREATE TABLE tuning_trials (
                 session_id TEXT, trial_id TEXT, attempt_id TEXT, trial_number BIGINT,
                 status TEXT, config TEXT, created_at TIMESTAMP, started_at TIMESTAMP,
                 ended_at TIMESTAMP, score DOUBLE, mu DOUBLE, sigma DOUBLE,
                 stop_reason TEXT, failure TEXT
             );
             INSERT INTO tuning_trials VALUES
             ('bad-config', 'bad-trial', 'bad-attempt', 1, 'complete', 'not-json',
              CURRENT_TIMESTAMP, NULL, NULL, NULL, NULL, NULL, NULL, NULL);",
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/bad-config/trials").await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["code"], 500);

    let (status, body) =
        http_get(app, "/api/bench/tuner/sessions/bad-config/trials/bad-trial").await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["code"], 500);
}

fn trial_ids(value: &serde_json::Value) -> Vec<&str> {
    value["trials"]
        .as_array()
        .unwrap()
        .iter()
        .map(|trial| trial["trial_id"].as_str().unwrap())
        .collect()
}

fn seed_paged_trial_evidence(conn: &duckdb::Connection, _dir: &std::path::Path) {
    conn.execute_batch(
        "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path)
         VALUES ('page-run', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'completed', '/tmp/page.log');
         INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence)
         VALUES ('page', 'idle', '{}', CURRENT_TIMESTAMP, 42),
                ('other', 'idle', '{}', CURRENT_TIMESTAMP, 3);
         INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at)
         VALUES ('page-attempt', 'page', 'page-run', 'completed', CURRENT_TIMESTAMP),
                ('other-attempt', 'other', NULL, 'completed', CURRENT_TIMESTAMP);
         INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at, score, mu, sigma, stop_reason)
         VALUES ('page', 'trial-a', 'page-attempt', 1, 'complete', '{\"family\":\"ucb\",\"c\":1.2}', CURRENT_TIMESTAMP, 10, 30, 2, 'max_pairs'),
                ('page', 'trial-b', 'page-attempt', 2, 'complete', '{\"family\":\"ucb\",\"c\":0.7}', CURRENT_TIMESTAMP, 10, 30, 1, 'confidence'),
                ('page', 'trial-c', 'page-attempt', 3, 'running', '{\"family\":\"rave\"}', CURRENT_TIMESTAMP, NULL, NULL, NULL, NULL),
                ('page', 'trial-d', 'page-attempt', 4, 'failed', '{\"family\":\"random\"}', CURRENT_TIMESTAMP, 5, 20, 4, 'hyperband_prune'),
                ('page', 'trial-e', 'page-attempt', 5, 'queued', NULL, CURRENT_TIMESTAMP, NULL, NULL, NULL, NULL),
                ('other', 'trial-a', 'other-attempt', 1, 'complete', '{}', CURRENT_TIMESTAMP, NULL, NULL, NULL, NULL);
         INSERT INTO tuning_trial_reports (session_id, trial_id, trial_number, completed_pairs, event_id, reported_at, mu, sigma, score, score_formula_version, conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource)
         VALUES ('page', 'trial-a', 1, 2, 'a-2', CURRENT_TIMESTAMP, 28, 3, 19, 1, 3, 'continue', 'below_min_pairs', false, 'alpha', 2),
                ('page', 'trial-a', 1, 4, 'a-4', CURRENT_TIMESTAMP, 30, 2, 24, 1, 3, 'complete', 'max_pairs', false, 'alpha', 4),
                ('page', 'trial-b', 2, 4, 'b-4', CURRENT_TIMESTAMP, 30, 1, 27, 1, 3, 'complete', 'confidence', false, 'alpha', 4),
                ('page', 'trial-c', 3, 1, 'c-1', CURRENT_TIMESTAMP, 24, 3, 15, 1, 3, 'continue', 'below_min_pairs', false, NULL, NULL),
                ('page', 'trial-d', 4, 2, 'd-2', CURRENT_TIMESTAMP, 20, 4, 8, 1, 3, 'prune', 'hyperband_prune', false, 'beta', 2);
         INSERT INTO tuning_pool_revisions (session_id, pool_snapshot_fingerprint, display_ordinal, first_event_id, first_attempt_id, observed_at)
         VALUES ('page', 'pool-stored', 1, 'pool-event', 'page-attempt', CURRENT_TIMESTAMP);
         INSERT INTO tuning_pool_anchors (session_id, pool_snapshot_fingerprint, anchor_ordinal, anchor_id, config, mu, sigma, provenance, insertion_reason, source_trial_id)
         VALUES ('page', 'pool-stored', 0, 'stored-anchor', '{\"family\":\"revision\"}', 25, 1, 'bootstrap_default', 'bootstrap', NULL);
         INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, started_at)
         VALUES ('page', 'pair-a', 'trial-a', 'page-attempt', 0, 'complete', 7, 1, '{\"anchor_id\":\"stored-anchor\",\"config\":{\"family\":\"opponent\"},\"mu\":25,\"sigma\":1}', 'pool-stored', 25, 1, CURRENT_TIMESTAMP),
                ('page', 'pair-b', 'trial-a', 'page-attempt', 1, 'complete', 8, 1, '{\"anchor_id\":\"legacy-anchor\",\"config\":{\"family\":\"legacy\"},\"mu\":25,\"sigma\":1}', 'missing-pool', 25, 1, CURRENT_TIMESTAMP);
         INSERT INTO tuning_games (session_id, pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, candidate_metrics, baseline_metrics, finished_at)
         VALUES ('page', 'pair-a', 'game-first', 'first', 'candidate_win', 7, 1, 9, 11, 12, '{\"iterations_total\":10,\"iterations_first_half\":4,\"move_time_ms\":6}', '{\"iterations_total\":11,\"iterations_first_half\":5,\"move_time_ms\":7}', CURRENT_TIMESTAMP),
                ('page', 'pair-a', 'game-second', 'second', 'baseline_win', 7, 1, NULL, 12, 13, '{\"iterations_total\":12,\"iterations_first_half\":5,\"move_time_ms\":8}', '{\"iterations_total\":13,\"iterations_first_half\":6,\"move_time_ms\":9}', CURRENT_TIMESTAMP);
         INSERT INTO game_moves (run_id, game_seq, ply, ts, trace_schema_version, state, search_report)
         VALUES ('page-run', 9, 0, CURRENT_TIMESTAMP, 1, '{}', '{\"status\":\"complete\"}');",
    )
    .unwrap();
}
