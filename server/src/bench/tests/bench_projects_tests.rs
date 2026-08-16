use super::*;
#[tokio::test]
async fn projects_launch_commits_typed_start_before_launcher_and_observes_process() {
    let observed = Arc::new(Mutex::new(None::<(String, u64, i64, i64, i64)>));
    let observed_by_launcher = observed.clone();
    let holder = Arc::new(Mutex::new(None::<Arc<BenchState>>));
    let holder_by_launcher = holder.clone();
    let spec = route_test_spec();
    let (app, _, state) = seeded_app_with_state(
        move |conn, _| {
            conn.execute("INSERT INTO projects (project_id, name, description, created_at, updated_at) VALUES ('p-order', 'Order', '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", []).unwrap();
            conn.execute("INSERT INTO experiments (experiment_id, project_id, name, description, spec, created_at, updated_at) VALUES ('e-order', 'p-order', 'Order', '', ?1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", duckdb::params![serde_json::to_string(&spec).unwrap()]).unwrap();
        },
        Arc::new(|saved| saved.expand().map(|_| ()).map_err(|error| error.fields)),
        Arc::new(move |run_id, _command, _kind, _game, _label| {
            let state = holder_by_launcher.lock().unwrap().clone().unwrap();
            let db = state.db.lock().unwrap();
            let facts: (String, u64, i64, i64, i64) = db.query_row(
                    "SELECT attempt_phase, attempt_version, (SELECT COUNT(*) FROM attempt_events WHERE attempt_id = ?1), (SELECT COUNT(*) FROM logical_runs WHERE logical_run_id = ?1), (SELECT COUNT(*) FROM experiment_cells WHERE run_id = ?1) FROM runs WHERE run_id = ?1",
                    duckdb::params![&run_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
                ).unwrap();
            *observed_by_launcher.lock().unwrap() = Some(facts);
            Ok(LaunchedRun {
                run_id,
                pid: 4242,
                log_path: "bench-runs/order/log.jsonl".into(),
                log_dir: "bench-runs/order".into(),
            })
        }),
    );
    *holder.lock().unwrap() = Some(state.clone());
    let (status, body) =
        http_post_json(app, "/api/bench/experiments/e-order/runs", json!({})).await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(
        observed.lock().unwrap().take(),
        Some(("starting".into(), 1, 1, 1, 1))
    );
    let run_id = body_json(&body)["run_id"].as_str().unwrap().to_owned();
    let typed: (String, u64, i64) = state.db.lock().unwrap().query_row(
            "SELECT attempt_phase, attempt_version, (SELECT COUNT(*) FROM attempt_events WHERE attempt_id = ?1) FROM runs WHERE run_id = ?1",
            duckdb::params![run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).unwrap();
    assert_eq!(typed, ("running".into(), 2, 2));
}

#[tokio::test]
async fn projects_launch_failure_persists_spawn_failure_before_http_error() {
    let spec = route_test_spec();
    let (app, _, state) = seeded_app_with_state(
        move |conn, _| {
            conn.execute("INSERT INTO projects (project_id, name, description, created_at, updated_at) VALUES ('p-fail', 'Fail', '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", []).unwrap();
            conn.execute("INSERT INTO experiments (experiment_id, project_id, name, description, spec, created_at, updated_at) VALUES ('e-fail', 'p-fail', 'Fail', '', ?1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", duckdb::params![serde_json::to_string(&spec).unwrap()]).unwrap();
        },
        Arc::new(|saved| saved.expand().map(|_| ()).map_err(|error| error.fields)),
        Arc::new(|_, _, _, _, _| Err(std::io::Error::other("mock spawn"))),
    );
    let (status, body) = http_post_json(app, "/api/bench/experiments/e-fail/runs", json!({})).await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert!(String::from_utf8_lossy(&body).contains("failed to launch experiment"));
    let db = state.db.lock().unwrap();
    let facts: (String, i64, i64, i64) = db.query_row(
            "SELECT attempt_phase, (SELECT COUNT(*) FROM attempt_events WHERE attempt_id = r.run_id AND event_type = 'spawn_failed'), (SELECT COUNT(*) FROM attempt_events WHERE attempt_id = r.run_id AND event_type = 'process_observed'), (SELECT COUNT(*) FROM experiment_cells WHERE run_id = r.run_id AND status = 'failed') FROM runs r WHERE experiment_id = 'e-fail'",
            [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        ).unwrap();
    assert_eq!(facts, ("crashed".into(), 1, 0, 1));
}

// -------------------------------------------------------------------
// POST /api/bench/launch

#[tokio::test]
async fn projects_stop_commits_intent_before_signalling_and_is_idempotent() {
    let holder = Arc::new(Mutex::new(None::<Arc<BenchState>>));
    let holder_by_signaller = holder.clone();
    let calls = Arc::new(Mutex::new(0_u32));
    let calls_by_signaller = calls.clone();
    let spec = route_test_spec();
    let (app, _, state) = seeded_app_with_state_and_signaller(
        move |conn, _| {
            conn.execute("INSERT INTO projects (project_id, name, description, created_at, updated_at) VALUES ('p-stop', 'Stop', '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", []).unwrap();
            conn.execute("INSERT INTO experiments (experiment_id, project_id, name, description, spec, created_at, updated_at) VALUES ('e-stop', 'p-stop', 'Stop', '', ?1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", duckdb::params![serde_json::to_string(&spec).unwrap()]).unwrap();
        },
        Arc::new(|saved| saved.expand().map(|_| ()).map_err(|error| error.fields)),
        Arc::new(|run_id, _, _, _, _| {
            Ok(LaunchedRun {
                run_id,
                pid: 4243,
                log_path: "bench-runs/stop/log.jsonl".into(),
                log_dir: "bench-runs/stop".into(),
            })
        }),
        Arc::new(move |_| {
            let state = holder_by_signaller.lock().unwrap().clone().unwrap();
            let db = state.db.lock().unwrap();
            let (status, intents): (String, i64) = db.query_row("SELECT status, (SELECT COUNT(*) FROM attempt_events WHERE attempt_id = r.run_id AND event_type = 'stop_requested') FROM runs r WHERE experiment_id = 'e-stop'", [], |row| Ok((row.get(0)?, row.get(1)?))).unwrap();
            assert_eq!(status, "running");
            assert_eq!(intents, 1);
            *calls_by_signaller.lock().unwrap() += 1;
            Ok(())
        }),
    );
    *holder.lock().unwrap() = Some(state.clone());
    let (_, body) =
        http_post_json(app.clone(), "/api/bench/experiments/e-stop/runs", json!({})).await;
    let run_id = body_json(&body)["run_id"].as_str().unwrap().to_owned();
    let stop_path = format!("/api/bench/runs/{run_id}/stop");
    let (status, _) = http_post_json(app.clone(), &stop_path, json!({})).await;
    assert_eq!(status, HttpStatusCode::OK);
    let phase: (String, String, u64) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT attempt_phase, status, attempt_version FROM runs WHERE run_id = ?1",
            duckdb::params![run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(phase, ("awaiting_exit".into(), "stopped".into(), 4));
    let stop_path = format!("/api/bench/runs/{run_id}/stop");
    let _ = http_post_json(app, &stop_path, json!({})).await;
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn projects_stop_missing_pid_and_signaller_error_preserve_typed_intent() {
    for (error_kind, expected_status) in [
        (std::io::ErrorKind::NotFound, HttpStatusCode::OK),
        (
            std::io::ErrorKind::PermissionDenied,
            HttpStatusCode::INTERNAL_SERVER_ERROR,
        ),
    ] {
        let spec = route_test_spec();
        let holder = Arc::new(Mutex::new(None::<Arc<BenchState>>));
        let holder_for_signaller = holder.clone();
        let (app, _, state) = seeded_app_with_state_and_signaller(
            move |conn, _| {
                conn.execute("INSERT INTO projects (project_id, name, description, created_at, updated_at) VALUES ('p-stop-error', 'Stop', '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", []).unwrap();
                conn.execute("INSERT INTO experiments (experiment_id, project_id, name, description, spec, created_at, updated_at) VALUES ('e-stop-error', 'p-stop-error', 'Stop', '', ?1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", duckdb::params![serde_json::to_string(&spec).unwrap()]).unwrap();
            },
            Arc::new(|saved| saved.expand().map(|_| ()).map_err(|error| error.fields)),
            Arc::new(|run_id, _, _, _, _| {
                Ok(LaunchedRun {
                    run_id,
                    pid: 4244,
                    log_path: "bench-runs/stop-error/log.jsonl".into(),
                    log_dir: "bench-runs/stop-error".into(),
                })
            }),
            Arc::new(move |_| {
                let _ = holder_for_signaller.lock().unwrap().clone().unwrap();
                Err(std::io::Error::new(error_kind, "mock signal"))
            }),
        );
        *holder.lock().unwrap() = Some(state.clone());
        let (_, body) = http_post_json(
            app.clone(),
            "/api/bench/experiments/e-stop-error/runs",
            json!({}),
        )
        .await;
        let run_id = body_json(&body)["run_id"].as_str().unwrap().to_owned();
        let stop_path = format!("/api/bench/runs/{run_id}/stop");
        let (status, _) = http_post_json(app, &stop_path, json!({})).await;
        assert_eq!(status, expected_status);
        let facts: (String, String, i64) = state.db.lock().unwrap().query_row("SELECT attempt_phase, status, (SELECT COUNT(*) FROM attempt_events WHERE attempt_id = ?1 AND event_type = 'signal_observed') FROM runs WHERE run_id = ?1", duckdb::params![run_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).unwrap();
        assert_eq!(facts.0, "stop_requested");
        assert_eq!(facts.2, 0);
        assert_eq!(
            facts.1,
            if expected_status == HttpStatusCode::OK {
                "stopped"
            } else {
                "running"
            }
        );
    }
}

#[test]
fn test_process_group_signaller_targets_the_whole_group() {
    let command = process::process_group_signal_command(12345);
    assert_eq!(command.get_program(), "kill");
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(args, ["-TERM", "-12345"]);
}

// -------------------------------------------------------------------
// Error formatting
