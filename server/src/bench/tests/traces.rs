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
async fn test_get_run_games_404_for_unknown_run() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app, "/api/bench/runs/nonexistent/games").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);
}

#[tokio::test]
async fn test_get_run_games_empty_when_no_traces() {
    // `default_seed` has match_results but no game_moves rows.
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games")).await;
    assert_eq!(status, HttpStatusCode::OK);
    assert!(body_json(&body).as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_get_run_games_joins_match_results_by_seq() {
    let app = seeded_app(game_moves_seed).0;
    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games")).await;
    assert_eq!(status, HttpStatusCode::OK);
    let games = body_json(&body).as_array().unwrap().clone();
    assert_eq!(games.len(), 1, "expected 1 traced game, got {games:?}");
    assert_eq!(games[0]["game_seq"], 1);
    assert_eq!(games[0]["ply_count"], 2);
    assert_eq!(games[0]["strategy_a"], "strong");
    assert_eq!(games[0]["strategy_b"], "master");
    assert_eq!(games[0]["winner"], "strong");
}

#[tokio::test]
async fn test_get_run_games_filters_by_cell_without_leaking_other_cells() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('grid-games', 'experiment', NULL, 'test', false, 'test', '2026-01-01T00:00:00Z', 'running', '/tmp/grid.log')", []).unwrap();
        for (seq, cell) in [(1, "cell-000001"), (2, "cell-000002")] {
            conn.execute("INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner, cell_id) VALUES ('grid-games', ?1, '2026-01-01T00:00:01Z', 'candidate', 'baseline', 'win_a', 'candidate', ?2)", duckdb::params![seq, cell]).unwrap();
            conn.execute("INSERT INTO game_moves (run_id, game_seq, ply, ts, state) VALUES ('grid-games', ?1, 0, '2026-01-01T00:00:01Z', '{}')", duckdb::params![seq]).unwrap();
        }
    }).0;
    let (status, body) =
        http_get(app, "/api/bench/runs/grid-games/games?cell_id=cell-000002").await;
    assert_eq!(status, HttpStatusCode::OK);
    let body = body_json(&body);
    let games = body.as_array().unwrap();
    assert_eq!(games.len(), 1);
    assert_eq!(games[0]["cell_id"], "cell-000002");
    assert_eq!(games[0]["game_seq"], 2);
}

#[tokio::test]
async fn test_get_run_game_moves_ordered_by_ply() {
    let app = seeded_app(game_moves_seed).0;
    let (status, body) = http_get(
        app,
        &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games/1/moves"),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let moves = body_json(&body).as_array().unwrap().clone();
    assert_eq!(moves.len(), 2);
    assert_eq!(moves[0]["ply"], 0);
    assert_eq!(moves[0]["mv"], Value::Null);
    assert_eq!(moves[0]["state"], json!({"board": []}));
    assert_eq!(moves[1]["ply"], 1);
    assert_eq!(moves[1]["mv"], 4);
    assert_eq!(moves[1]["player"], "strong");
}

#[tokio::test]
async fn test_get_run_game_moves_preserves_the_typed_search_report() {
    let app = seeded_app(|conn, dir| {
        game_moves_seed(conn, dir);
        let report = json!({
            "schema_version": 1, "status": "partial", "reason": "root_parallel_pv_single_tree",
            "elapsed_seconds": 0.25, "iteration_limit": 100, "time_limit_seconds": null,
            "completed_iterations": 80, "termination": "time", "selected_action": {"ptn":"a1"},
            "actions": [{"action":{"ptn":"a1"},"visits":80,"share":1.0,"mean_value":0.5,"is_proven":false}],
            "principal_variation": [{"ptn":"a1"}], "root_visits": 80, "tree_nodes": 91,
            "mean_depth": 4.0, "max_depth": 7, "graph_mode": "dag_both",
            "tt_reads": 10, "tt_writes": 8, "tt_hits": 3, "tt_hit_ratio": 0.3,
            "iterations_per_second": 320.0, "warnings": ["root_parallel_pv_single_tree"]
        });
        conn.execute(
            "UPDATE game_moves SET trace_schema_version = 1, search_report = ?1 WHERE run_id = ?2 AND game_seq = 1 AND ply = 1",
            duckdb::params![report.to_string(), DEFAULT_RUN_ID],
        ).unwrap();
    }).0;

    let (status, body) = http_get(
        app,
        &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games/1/moves"),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let moves = body_json(&body);
    assert!(moves[0]["search"].is_null());
    assert_eq!(moves[1]["search"]["status"], "partial");
    assert_eq!(moves[1]["search"]["completed_iterations"], 80);
    assert_eq!(moves[1]["search"]["selected_action"], json!({"ptn":"a1"}));
}

#[tokio::test]
async fn test_get_run_game_moves_empty_for_unknown_game_seq() {
    let app = seeded_app(game_moves_seed).0;
    let (status, body) = http_get(
        app,
        &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games/999/moves"),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    assert!(body_json(&body).as_array().unwrap().is_empty());
}

// -------------------------------------------------------------------
// DELETE /api/bench/runs/{run_id}
// -------------------------------------------------------------------

#[tokio::test]
async fn test_delete_run_404_for_unknown_run() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_delete(app, "/api/bench/runs/nonexistent").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);
}

#[tokio::test]
async fn test_delete_run_409_while_running() {
    let app = seeded_app(running_run_seed).0;
    let (status, body) = http_delete(app, "/api/bench/runs/running-run").await;
    assert_eq!(status, HttpStatusCode::CONFLICT);
    assert_eq!(body_json(&body)["code"], 409);
}

#[tokio::test]
async fn test_modern_tuning_attempt_links_to_its_session_and_cannot_be_deleted() {
    let (app, _, _) = seeded_app_with_state(
        |conn, _| {
            conn.execute_batch(
                "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, ended_at, status, log_path)
                 VALUES ('session-attempt', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'completed', '/tmp/session.log');
                 INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence, optimizer_id, lifecycle_path)
                 VALUES ('modern-session', 'idle', '{}', 4, CURRENT_TIMESTAMP, 1, 'optimizer', '/tmp/lifecycle.jsonl');
                 INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at, ended_at)
                 VALUES ('attempt-1', 'modern-session', 'session-attempt', 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);",
            )
            .unwrap();
        },
        injected_general_launcher(),
    );

    let (status, body) = http_get(app.clone(), "/api/bench/runs/session-attempt").await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(body_json(&body)["tuning_session_id"], "modern-session");

    let (status, body) = http_delete(app, "/api/bench/runs/session-attempt").await;
    assert_eq!(status, HttpStatusCode::CONFLICT);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("future session Delete workflow"));
}

#[tokio::test]
async fn test_delete_run_removes_all_rows_and_files() {
    let (app, tmp_dir, state) = seeded_app_with_state(game_moves_seed, injected_general_launcher());
    state
        .db
        .lock()
        .unwrap()
        .execute(
            "UPDATE game_moves SET search_report = '{}' WHERE run_id = ?1",
            duckdb::params![DEFAULT_RUN_ID],
        )
        .unwrap();
    state.db.lock().unwrap().execute_batch(&format!(
        "INSERT INTO artifact_roots (physical_run_id, artifact_root, descriptor_watermark, updated_at) \
         VALUES ('{DEFAULT_RUN_ID}', '/tmp/artifact-root', '', CURRENT_TIMESTAMP); \
         INSERT INTO artifact_descriptors (physical_run_id, descriptor_filename, descriptor_path, status) \
         VALUES ('{DEFAULT_RUN_ID}', '0000000000000000001-task-00000000000000000000000000000001.json', '/tmp/descriptor', 'registered'); \
         INSERT INTO artifact_tasks (physical_run_id, task_id, attempt_id, task_sequence, descriptor_path, task_root, trace_path, descriptor_digest, status) \
         VALUES ('{DEFAULT_RUN_ID}', 'task-00000000000000000000000000000001', 'attempt', 1, '/tmp/descriptor', '/tmp/task', '/tmp/task/trace.jsonl', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'incomplete'); \
         INSERT INTO _artifact_trace_cursor (physical_run_id, task_id, trace_path, byte_offset, updated_at) \
         VALUES ('{DEFAULT_RUN_ID}', 'task-00000000000000000000000000000001', '/tmp/task/trace.jsonl', 0, CURRENT_TIMESTAMP);"
    )).unwrap();
    let run_dir = tmp_dir.join("bench-runs").join(DEFAULT_RUN_ID);
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(run_dir.join("log.jsonl"), "{}\n").unwrap();
    std::fs::write(run_dir.join("moves.jsonl"), "{}\n").unwrap();

    let (status, _) = http_delete(app.clone(), &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
    assert_eq!(status, HttpStatusCode::NO_CONTENT);

    let (status, _) = http_get(app.clone(), "/api/bench/runs").await;
    assert_eq!(status, HttpStatusCode::OK);

    let (status, _) = http_get(app.clone(), &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);

    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games")).await;
    assert_eq!(
        status,
        HttpStatusCode::NOT_FOUND,
        "run row itself is gone: {body:?}"
    );

    assert!(!run_dir.exists(), "run directory should be removed");
    let report_rows: i64 = state
        .db
        .lock()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM game_moves WHERE run_id = ?1 AND search_report IS NOT NULL",
            duckdb::params![DEFAULT_RUN_ID],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(report_rows, 0);
    for table in [
        "artifact_roots",
        "artifact_descriptors",
        "artifact_tasks",
        "_artifact_trace_cursor",
    ] {
        let rows: i64 = state
            .db
            .lock()
            .unwrap()
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(rows, 0, "{table} rows should be removed with their run");
    }
}

// -------------------------------------------------------------------
// GET /api/bench/runs/{run_id}/live (SSE)
// -------------------------------------------------------------------

#[tokio::test]
async fn test_live_run_moves_404_for_unknown_run() {
    let app = seeded_app(default_seed).0;
    let (status, _) = http_get(app, "/api/bench/runs/nonexistent/live").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_live_run_moves_opens_sse_stream_for_known_run() {
    let app = seeded_app(game_moves_seed).0;
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/bench/runs/{DEFAULT_RUN_ID}/live"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), HttpStatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
}

#[tokio::test]
async fn test_live_run_moves_accepts_a_pinned_game() {
    let app = seeded_app(game_moves_seed).0;
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/bench/runs/{DEFAULT_RUN_ID}/live?game_seq=7"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), HttpStatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
}

// -------------------------------------------------------------------
// GET /api/bench/runs/{run_id}/chain
// -------------------------------------------------------------------
