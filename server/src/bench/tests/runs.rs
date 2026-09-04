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

#[tokio::test]
async fn test_list_runs_empty() {
    let app = seeded_app(|_, _| {}).0;
    let (status, body) = http_get(app, "/api/bench/runs").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert!(runs.is_empty(), "expected empty list, got {runs:?}");
}

#[tokio::test]
async fn test_list_runs_returns_seeded_run() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app, "/api/bench/runs").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert_eq!(runs.len(), 1, "expected 1 run, got {runs:?}");

    let run = &runs[0];
    assert_eq!(run["run_id"], DEFAULT_RUN_ID);
    assert_eq!(run["kind"], "round_robin");
    assert_eq!(run["game"], "druid");
    assert_eq!(run["status"], "completed");
    assert_eq!(run["match_count"], 2);
    assert_eq!(run["trial_count"], 1);
    assert!(run.get("label").and_then(|v| v.as_str()).is_none());
}

#[tokio::test]
async fn test_list_runs_filter_by_status() {
    let app = seeded_app(|conn, dir| {
        default_seed(conn, dir);
        running_run_seed(conn, dir);
    })
    .0;

    // Filter to running only.
    let (status, body) = http_get(app.clone(), "/api/bench/runs?status=running").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert_eq!(runs.len(), 1, "expected 1 running run");
    assert_eq!(runs[0]["run_id"], "running-run");

    // Filter to completed only.
    let (status, body) = http_get(app.clone(), "/api/bench/runs?status=completed").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert_eq!(runs.len(), 1, "expected 1 completed run");
    assert_eq!(runs[0]["run_id"], DEFAULT_RUN_ID);
}

#[tokio::test]
async fn test_list_runs_filter_by_game() {
    let app = seeded_app(multi_run_seed).0;

    let (status, body) = http_get(app.clone(), "/api/bench/runs?game=druid").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert_eq!(runs.len(), 1, "expected 1 druid run");
    assert_eq!(runs[0]["game"], "druid");

    let (status, body) = http_get(app.clone(), "/api/bench/runs?game=ttt").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert_eq!(runs.len(), 1, "expected 1 ttt run");
    assert_eq!(runs[0]["game"], "ttt");
}

#[tokio::test]
async fn test_list_runs_limit() {
    let app = seeded_app(multi_run_seed).0;

    let (status, body) = http_get(app.clone(), "/api/bench/runs?limit=1").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert_eq!(runs.len(), 1, "expected 1 run with limit=1");
}

#[tokio::test]
async fn test_list_runs_orders_by_started_at_desc() {
    let app = seeded_app(multi_run_seed).0;

    let (status, body) = http_get(app.clone(), "/api/bench/runs").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert_eq!(runs.len(), 2);
    // Most recent first: runs have started_at 2026-02-01 and 2026-01-01.
    assert_eq!(runs[0]["run_id"], "rr-ttt-20260201T000000-def5678");
    assert_eq!(runs[1]["run_id"], DEFAULT_RUN_ID);
}

// -------------------------------------------------------------------
// GET /api/bench/runs/{run_id}
// -------------------------------------------------------------------

#[tokio::test]
async fn test_get_run_returns_detail() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
    assert_eq!(status, HttpStatusCode::OK);
    let run = body_json(&body);

    assert_eq!(run["run_id"], DEFAULT_RUN_ID);
    assert_eq!(run["kind"], "round_robin");
    assert_eq!(run["game"], "druid");
    assert_eq!(run["status"], "completed");
    assert_eq!(run["match_count"], 2);
    assert_eq!(run["trial_count"], 1);
    assert!(run.get("config").and_then(|v| v.as_str()).is_none());
    assert!(run.get("log_path").and_then(|v| v.as_str()).is_some());
    assert_eq!(run["exit_code"], Value::Null);
    assert_eq!(run["incumbent"], Value::Null);
}

#[tokio::test]
async fn test_get_run_includes_incumbent_when_present() {
    let app = seeded_app(|conn, dir| {
        default_seed(conn, dir);
        conn.execute(
            "INSERT INTO incumbents (run_id, ts, config, cost) \
             VALUES (?1, '2026-01-01T00:00:40Z', '{\"select\":\"rave\",\"c\":0.7}', 0.2)",
            duckdb::params![DEFAULT_RUN_ID],
        )
        .unwrap();
    })
    .0;
    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
    assert_eq!(status, HttpStatusCode::OK);
    let run = body_json(&body);

    assert_eq!(run["incumbent"]["cost"], 0.2);
    assert_eq!(run["incumbent"]["config"]["select"], "rave");
}

#[tokio::test]
async fn test_get_run_404_for_unknown_run() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app, "/api/bench/runs/nonexistent").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    let body = body_json(&body);
    assert_eq!(body["code"], 404);
    assert!(body["error"].as_str().unwrap().contains("nonexistent"));
}

// -------------------------------------------------------------------
// GET /api/bench/runs/{run_id}/log
// -------------------------------------------------------------------

#[tokio::test]
async fn test_get_run_log_404_for_unknown_run() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app, "/api/bench/runs/nonexistent/log").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    let body = body_json(&body);
    assert_eq!(body["code"], 404);
}

#[tokio::test]
async fn test_get_run_log_returns_lines_since_offset() {
    // Create a run with a real log file.
    let app = seeded_app(|conn, bench_runs_dir| {
        let run_dir = bench_runs_dir.join("loggy-run");
        std::fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();

        // Write some lines.
        std::fs::write(&log_path, "line1\nline2\nline3\n").unwrap();

        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) \
             VALUES ('loggy-run', 'round_robin', 'druid', 'abc', false, 'h', NULL, \
                     '2026-01-01T00:00:00Z', 'running', ?1)",
            duckdb::params![log_path_str],
        )
        .unwrap();
    })
    .0;

    // Read from offset 0 — get all 3 lines.
    let (status, body) = http_get(app.clone(), "/api/bench/runs/loggy-run/log").await;
    assert_eq!(status, HttpStatusCode::OK);
    let resp = body_json(&body);
    let lines = resp["lines"].as_array().unwrap().clone();
    assert_eq!(lines.len(), 3, "expected 3 lines, got {lines:?}");
    assert_eq!(lines[0], "line1");
    assert_eq!(lines[1], "line2");
    assert_eq!(lines[2], "line3");
    assert!(resp["next_offset"].as_u64().unwrap() > 0);

    // Read from an offset past the end — empty result.
    let last_offset = resp["next_offset"].as_u64().unwrap();
    let (status, body) = http_get(
        app.clone(),
        &format!("/api/bench/runs/loggy-run/log?since={last_offset}"),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let resp = body_json(&body);
    assert!(resp["lines"].as_array().unwrap().is_empty());
    assert_eq!(resp["next_offset"].as_u64().unwrap(), last_offset);
}

// -------------------------------------------------------------------
// GET /api/bench/runs/{run_id}/trials
// -------------------------------------------------------------------

#[tokio::test]
async fn test_get_run_trials_404_for_unknown_run() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app, "/api/bench/runs/nonexistent/trials").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    let body = body_json(&body);
    assert_eq!(body["code"], 404);
}

#[tokio::test]
async fn test_get_run_trials_returns_rows_in_trial_id_order() {
    let app = seeded_app(|conn, dir| {
        default_seed(conn, dir); // seeds trial_id 1 with cost 0.375
        conn.execute(
            "INSERT INTO trials (run_id, trial_id, ts, config, seed, cost, extra) \
             VALUES (?1, 2, '2026-01-01T00:00:40Z', '{\"c\":1.5}', 42, 0.2, '{\"wins\":8}')",
            duckdb::params![DEFAULT_RUN_ID],
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}/trials")).await;
    assert_eq!(status, HttpStatusCode::OK);
    let rows = body_json(&body).as_array().unwrap().clone();
    assert_eq!(rows.len(), 2, "expected 2 trials, got {rows:?}");

    assert_eq!(rows[0]["trial_id"], 1);
    assert_eq!(rows[0]["config"], json!({}));
    assert_eq!(rows[0]["cost"], 0.375);
    assert_eq!(rows[0]["seed"], Value::Null);

    assert_eq!(rows[1]["trial_id"], 2);
    assert_eq!(rows[1]["config"], json!({"c": 1.5}));
    assert_eq!(rows[1]["seed"], 42);
    assert_eq!(rows[1]["cost"], 0.2);
    assert_eq!(rows[1]["extra"], json!({"wins": 8}));
}

#[tokio::test]
async fn test_get_run_trials_respects_limit() {
    let app = seeded_app(|conn, dir| {
        default_seed(conn, dir);
        conn.execute(
            "INSERT INTO trials (run_id, trial_id, ts, config, cost) \
             VALUES (?1, 2, '2026-01-01T00:00:40Z', '{}', 0.2)",
            duckdb::params![DEFAULT_RUN_ID],
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_get(
        app,
        &format!("/api/bench/runs/{DEFAULT_RUN_ID}/trials?limit=1"),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let rows = body_json(&body).as_array().unwrap().clone();
    assert_eq!(rows.len(), 1, "expected 1 trial with limit=1");
    assert_eq!(rows[0]["trial_id"], 1);
}

// -------------------------------------------------------------------
// GET /api/bench/runs/{run_id}/games, .../games/{game_seq}/moves
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// GET /api/bench/tuner/kinds
// -------------------------------------------------------------------
