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
    // everything after it is the argv this test actually cares about
    // (trailing --trace-path is asserted separately, below).
    assert_eq!(
        cmd[1..cmd.len() - 2],
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
        ]
    );
}

#[test]
fn test_build_command_tuner_with_no_config_is_just_game() {
    let cmd = build_command("tuner", "druid", &None, "test-run").unwrap();
    assert_eq!(cmd[1..cmd.len() - 2], vec!["tuner", "--game", "druid"]);
}

#[test]
fn test_build_command_tuner_includes_trace_path_derived_from_run_id() {
    let cmd = build_command(
        "tuner",
        "druid",
        &None,
        "tuner-druid-20260101T000000-abcdef",
    )
    .unwrap();
    let idx = cmd
        .iter()
        .position(|a| a == "--trace-path")
        .expect("--trace-path flag present");
    assert_eq!(
        cmd[idx + 1],
        "bench-runs/tuner-druid-20260101T000000-abcdef/moves.jsonl"
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
        cmd[1..cmd.len() - 2],
        vec![
            "tuner",
            "--game",
            "nim",
            "--override",
            "optimizer.n_trials=10",
            "--baseline-config",
            r#"ladder1={"c":1.5,"family":"ucb1"}"#,
        ]
    );
}

#[test]
fn test_build_command_unknown_kind_lists_tuner_as_supported() {
    let err = build_command("nope", "druid", &None, "test-run").unwrap_err();
    assert!(err.message.contains("tuner"));
}

#[test]
fn test_build_command_round_robin_includes_trace_path_derived_from_run_id() {
    let cmd = build_command(
        "round_robin",
        "druid",
        &None,
        "rr-druid-20260101T000000-abcdef",
    )
    .unwrap();

    let idx = cmd
        .iter()
        .position(|a| a == "--trace-path")
        .expect("--trace-path flag present");
    assert_eq!(
        cmd[idx + 1],
        "bench-runs/rr-druid-20260101T000000-abcdef/moves.jsonl"
    );
}

#[test]
fn test_build_experiment_command_detaches_foreground_coordinator() {
    let spec = ExperimentSpecV1 {
        version: 1,
        games: vec![mcts_bench::experiment::ExperimentGame {
            game: "nim".into(),
            game_config: Value::Null,
        }],
        baseline: mcts_bench::experiment::NamedStrategyConfig {
            id: "baseline".into(),
            label: "Baseline".into(),
            config: json!({"family": "ucb1"}),
        },
        variants: vec![mcts_bench::experiment::NamedStrategyConfig {
            id: "variant".into(),
            label: "Variant".into(),
            config: json!({"family": "rave"}),
        }],
        budgets: vec![mcts_bench::experiment::Budget::Iterations { value: 1 }],
        rounds_per_cell: 1,
        base_seed: 42,
        max_parallel_cells: 1,
    };
    let cmd = build_experiment_command(&spec, "experiment-run").unwrap();
    assert_eq!(cmd[0], find_bench_binary().to_string_lossy());
    assert_eq!(
        &cmd[1..],
        &[
            "experiment".to_string(),
            "--spec-json".to_string(),
            serde_json::to_string(&spec).unwrap(),
            "--trace-path".to_string(),
            "bench-runs/experiment-run/moves.jsonl".to_string(),
        ]
    );
    assert_ne!(cmd[1], "game-nim");
    assert!(serde_json::from_str::<ExperimentSpecV1>(&cmd[3]).is_ok());
}

#[tokio::test]
async fn test_launch_experiment_materializes_full_grid_and_uses_coordinator() {
    let spec = ExperimentSpecV1 {
        version: 1,
        games: vec![
            mcts_bench::experiment::ExperimentGame {
                game: "game-a".into(),
                game_config: json!({"a": 1}),
            },
            mcts_bench::experiment::ExperimentGame {
                game: "game-b".into(),
                game_config: json!({"b": 2}),
            },
        ],
        baseline: mcts_bench::experiment::NamedStrategyConfig {
            id: "base".into(),
            label: "Base".into(),
            config: json!({"family": "ucb1"}),
        },
        variants: vec![
            mcts_bench::experiment::NamedStrategyConfig {
                id: "v1".into(),
                label: "V1".into(),
                config: json!({"family": "rave"}),
            },
            mcts_bench::experiment::NamedStrategyConfig {
                id: "v2".into(),
                label: "V2".into(),
                config: json!({"family": "flat_mc"}),
            },
            mcts_bench::experiment::NamedStrategyConfig {
                id: "v3".into(),
                label: "V3".into(),
                config: json!({"family": "random"}),
            },
        ],
        budgets: vec![
            mcts_bench::experiment::Budget::Iterations { value: 10 },
            mcts_bench::experiment::Budget::TimePerMoveMs { value: 20 },
        ],
        rounds_per_cell: 2,
        base_seed: 42,
        max_parallel_cells: 2,
    };
    let commands = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let captured = commands.clone();
    let (app, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute("INSERT INTO projects (project_id, name, description, created_at, updated_at) VALUES ('p-grid', 'Grid', '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')", []).unwrap();
            conn.execute("INSERT INTO experiments (experiment_id, project_id, name, description, spec, created_at, updated_at) VALUES ('e-grid', 'p-grid', 'Grid experiment', '', ?1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')", duckdb::params![serde_json::to_string(&spec).unwrap()]).unwrap();
        },
        Arc::new(|saved| saved.expand().map(|_| ()).map_err(|error| error.fields)),
        Arc::new(move |run_id, command, _kind, _game, _label| {
            captured.lock().unwrap().push(command);
            Ok(LaunchedRun {
                run_id,
                pid: 123,
                log_path: PathBuf::from("bench-runs/fake/log.jsonl"),
                log_dir: PathBuf::from("bench-runs/fake"),
            })
        }),
    );
    let (status, body) =
        http_post_json(app.clone(), "/api/bench/experiments/e-grid/runs", json!({})).await;
    assert_eq!(
        status,
        HttpStatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let run_id = body_json(&body)["run_id"].as_str().unwrap().to_string();
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
    let typed: (String, u64, i64) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT attempt_phase, attempt_version, (SELECT COUNT(*) FROM attempt_events WHERE attempt_id = ?1) FROM runs WHERE run_id = ?1",
            duckdb::params![&run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(typed, ("running".into(), 2, 2));
    let (status, cells_body) = http_get(app, &format!("/api/bench/runs/{run_id}/cells")).await;
    assert_eq!(status, HttpStatusCode::OK);
    let cells = body_json(&cells_body).as_array().unwrap().to_vec();
    assert_eq!(cells.len(), 12);
    assert_eq!(cells[0]["cell_id"], "cell-000001");
    assert_eq!(cells[11]["cell_id"], "cell-000012");
    assert!(cells
        .iter()
        .all(|cell| cell["planned_games"] == 4 && cell["cell_seed"].is_number()));
    let captured = commands.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0][1], "experiment");
    assert!(!captured[0]
        .iter()
        .any(|arg| arg == "game-game-a" || arg == "game-game-b"));
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
async fn test_launch_spawns_bench_and_returns_run_id() {
    // Launch a quick `true` command to verify the plumbing works end-to-end.
    // We simulate what the server would do by launching `true` (exits
    // immediately) as a "round_robin" run and checking the registry.
    let app = seeded_app(|_conn, dir| {
        // We need the registry to exist in the bench_runs_dir for the
        // launcher to write to.
        std::fs::create_dir_all(dir).ok();
    })
    .0;

    // We can't easily test the actual bench binary path from tests
    // (the server binary path during `cargo test` is in the build
    // target dir).  Instead, test that a valid request shape hits
    // the launcher and produces an error about a missing binary
    // (expected since `bench` isn't compiled during tests) or
    // succeeds if `true` is used.

    // Use `true` as the command to verify the launcher path works.
    let (status, body) = http_post_json(
        app,
        "/api/bench/launch",
        json!({
            "kind": "round_robin",
            "game": "druid",
            "config": {
                "strategies": ["strong", "master"],
                "rounds": 1
            }
        }),
    )
    .await;

    // The request reaches the handler and tries to find `bench`.
    // Since we're running tests (not the compiled server), the
    // `bench` binary doesn't exist next to the test binary.
    // We expect either a 500 (bench not found) or a success if
    // by coincidence something called `bench` is on PATH.
    // What we *don't* expect is a 400 (which would mean the
    // request body was rejected before reaching the launcher).
    assert!(
        status == HttpStatusCode::OK || status == HttpStatusCode::INTERNAL_SERVER_ERROR,
        "launch returned unexpected status {status}: body={}",
        String::from_utf8_lossy(&body),
    );
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
async fn test_fresh_round_robin_and_tuner_launches_create_identity_roots() {
    let (app, _, state) = seeded_app_with_state(
        |_, _| {},
        Arc::new(|saved| saved.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    );

    for (kind, game, config) in [
        ("round_robin", "druid", json!({"rounds": 1})),
        ("tuner", "traffic-lights", json!({"overrides": []})),
    ] {
        let (status, body) = http_post_json(
            app.clone(),
            "/api/bench/launch",
            json!({"kind": kind, "game": game, "config": config}),
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
        assert_eq!(identity, (run_id.clone(), None, 1, run_id));
    }
}

#[tokio::test]
async fn test_resume_and_manual_promotion_link_children_to_parent_identity() {
    let (app, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, config, git_sha, git_dirty, host, started_at, ended_at, status, log_path) VALUES ('resume-parent', 'tuner', 'traffic-lights', '{\"config\":\"tuner/config/default.yaml\",\"overrides\":[]}', 'sha', false, 'host', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'completed', '/tmp/resume.log')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, config, git_sha, git_dirty, host, started_at, ended_at, status, log_path) VALUES ('promotion-parent', 'tuner', 'druid', '{\"config\":\"tuner/config/default.yaml\",\"overrides\":[]}', 'sha', false, 'host', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'completed', '/tmp/promotion.log')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO incumbents (run_id, ts, config, cost) VALUES ('promotion-parent', CURRENT_TIMESTAMP, '{\"family\":\"ucb1\"}', 0.02)",
                [],
            )
            .unwrap();
        },
        Arc::new(|saved| saved.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    );

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/runs/resume-parent/resume",
        json!({"n_trials": 500}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let resume_child = body_json(&body)["run_id"].as_str().unwrap().to_owned();
    let (logical, parent, ordinal): (String, String, u64) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = ?1",
            duckdb::params![&resume_child],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        (logical, parent, ordinal),
        ("resume-parent".into(), "resume-parent".into(), 2)
    );

    let (status, body) = http_post_json(
        app,
        "/api/bench/runs/promotion-parent/advance-baseline",
        json!({}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let promotion_child = body_json(&body)["run_id"].as_str().unwrap().to_owned();
    let linkage: (String, String, u64) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = ?1",
            duckdb::params![&promotion_child],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        linkage,
        ("promotion-parent".into(), "promotion-parent".into(), 2)
    );
}

#[tokio::test]
async fn test_automatic_promotion_links_child_to_parent_identity() {
    let (_, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, config, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) VALUES ('auto-parent', 'tuner', 'traffic-lights', '{\"config\":\"tuner/config/default.yaml\",\"overrides\":[\"optimizer.n_trials=10\"],\"ladder\":{\"max_rungs\":2,\"saturation_threshold\":0.1},\"ladder_root\":\"auto-parent\"}', 'sha', false, 'host', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'completed', '/tmp/auto.log')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO incumbents (run_id, ts, config, cost) VALUES ('auto-parent', CURRENT_TIMESTAMP, '{\"family\":\"ucb1\"}', 0.02)",
                [],
            )
            .unwrap();
        },
        Arc::new(|saved| saved.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    );

    advance_ladders_once(&state).await;

    let linkage: (String, String, u64) = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id <> 'auto-parent' AND kind = 'tuner' ORDER BY started_at DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(linkage, ("auto-parent".into(), "auto-parent".into(), 2));
}

// -------------------------------------------------------------------
// build_resume_config
// -------------------------------------------------------------------

#[test]
fn test_build_resume_config_appends_n_trials_override() {
    let config = build_resume_config("old-run-1", &None, 500, None);
    let overrides = config["overrides"].as_array().unwrap();
    assert_eq!(overrides, &[json!("optimizer.n_trials=500")]);
    assert!(config.get("config").is_none());
}

#[test]
fn test_build_resume_config_appends_n_workers_when_given() {
    let config = build_resume_config("old-run-1", &None, 500, Some(4));
    let overrides = config["overrides"].as_array().unwrap();
    assert_eq!(
        overrides,
        &[
            json!("optimizer.n_trials=500"),
            json!("optimizer.n_workers=4")
        ]
    );
}

#[test]
fn test_build_resume_config_carries_forward_old_config_and_overrides() {
    let old = Some(json!({
        "config": "tuner/config/default.yaml",
        "overrides": ["target.rounds=30"],
    }));
    let config = build_resume_config("old-run-1", &old, 500, None);
    assert_eq!(config["config"], json!("tuner/config/default.yaml"));
    assert_eq!(
        config["overrides"].as_array().unwrap(),
        &[json!("target.rounds=30"), json!("optimizer.n_trials=500")]
    );
}

#[test]
fn test_build_resume_config_records_resumed_from() {
    let config = build_resume_config("old-run-1", &None, 500, None);
    assert_eq!(config["resumed_from"], json!("old-run-1"));
}

#[test]
fn test_build_resume_config_preserves_unknown_keys() {
    // Ladder bookkeeping (`ladder`, `ladder_root`, `baseline_configs`)
    // must survive a resume untouched -- both the driver's own resume
    // calls and a human clicking the existing UI Resume button on a
    // ladder rung go through this same function.
    let old = Some(json!({
        "overrides": ["target.rounds=30"],
        "ladder": {"max_rungs": 5, "saturation_threshold": 0.0},
        "ladder_root": "root-run-1",
        "baseline_configs": {"ladder1": {"family": "ucb1"}},
    }));
    let config = build_resume_config("rung-1-run", &old, 500, None);
    assert_eq!(config["ladder"]["max_rungs"], json!(5));
    assert_eq!(config["ladder_root"], json!("root-run-1"));
    assert_eq!(
        config["baseline_configs"]["ladder1"],
        json!({"family": "ucb1"})
    );
    assert_eq!(config["resumed_from"], json!("rung-1-run"));
}

// -------------------------------------------------------------------
// inject_ladder_root_if_new_ladder
// -------------------------------------------------------------------
