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
async fn test_list_runs_collapses_ladder_rungs_into_latest_logical_run() {
    let app = seeded_app(ladder_runs_seed).0;
    let (status, body) = http_get(app.clone(), "/api/bench/runs").await;
    assert_eq!(status, HttpStatusCode::OK);
    let runs = body_json(&body).as_array().unwrap().clone();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0]["run_id"], "rung-2");
    assert_eq!(runs[0]["status"], "running");
    assert_eq!(runs[0]["trial_count"], 3);
    assert_eq!(runs[0]["started_at"], "2026-01-01 00:00:00");

    let (_, body) = http_get(app.clone(), "/api/bench/runs?status=running").await;
    assert_eq!(body_json(&body).as_array().unwrap().len(), 1);
    let (_, body) = http_get(app, "/api/bench/runs?status=stopped").await;
    assert!(body_json(&body).as_array().unwrap().is_empty());
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
             VALUES (?1, '2026-01-01T00:00:40Z', '{\"family\":\"rave\",\"c\":0.7}', 0.2)",
            duckdb::params![DEFAULT_RUN_ID],
        )
        .unwrap();
    })
    .0;
    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
    assert_eq!(status, HttpStatusCode::OK);
    let run = body_json(&body);

    assert_eq!(run["incumbent"]["cost"], 0.2);
    assert_eq!(run["incumbent"]["config"]["family"], "rave");
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

#[tokio::test]
async fn test_get_run_chain_404_for_unknown_run() {
    let app = seeded_app(|_, _| {}).0;
    let (status, body) = http_get(app, "/api/bench/runs/nonexistent/chain").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);
}

fn insert_tuner_run(conn: &duckdb::Connection, run_id: &str, started_at: &str, config: &Value) {
    conn.execute(
        "INSERT INTO runs \
         (run_id, kind, game, config, git_sha, git_dirty, host, pid, \
          started_at, ended_at, status, log_path) \
         VALUES (?1, 'tuner', 'nim', ?2, 'abc1234', false, 'testhost', NULL, \
                 ?3, ?3, 'completed', '/tmp/nope/log.jsonl')",
        duckdb::params![run_id, config.to_string(), started_at],
    )
    .unwrap();
}

#[tokio::test]
async fn test_get_run_chain_single_rung_for_a_plain_run() {
    let app = seeded_app(|conn, _dir| {
        insert_tuner_run(
            conn,
            "root-1",
            "2026-01-01T00:00:00Z",
            &json!({"overrides": []}),
        );
    })
    .0;

    let (status, body) = http_get(app, "/api/bench/runs/root-1/chain").await;
    assert_eq!(status, HttpStatusCode::OK);
    let rows = body_json(&body).as_array().unwrap().clone();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["run_id"], "root-1");
}

#[tokio::test]
async fn test_get_run_chain_orders_every_rung_oldest_first() {
    let app = seeded_app(|conn, _dir| {
        insert_tuner_run(
            conn,
            "root-1",
            "2026-01-01T00:00:00Z",
            &json!({"ladder_root": "root-1"}),
        );
        insert_tuner_run(
            conn,
            "root-1-rung3",
            "2026-01-03T00:00:00Z",
            &json!({"ladder_root": "root-1", "resumed_from": "root-1-rung2"}),
        );
        insert_tuner_run(
            conn,
            "root-1-rung2",
            "2026-01-02T00:00:00Z",
            &json!({"ladder_root": "root-1", "resumed_from": "root-1"}),
        );
        // A run from a *different* chain (different ladder_root) must
        // not leak into this chain's result.
        insert_tuner_run(
            conn,
            "other-root",
            "2026-01-02T12:00:00Z",
            &json!({"ladder_root": "other-root"}),
        );
        conn.execute(
            "INSERT INTO incumbents (run_id, ts, config, cost) \
             VALUES ('root-1', '2026-01-01T00:30:00Z', '{\"family\": \"ucb1\"}', 0.02)",
            duckdb::params![],
        )
        .unwrap();
    })
    .0;

    // Query from the *middle* rung -- the chain must resolve via
    // ladder_root regardless of which rung's run_id is asked for.
    let (status, body) = http_get(app, "/api/bench/runs/root-1-rung2/chain").await;
    assert_eq!(status, HttpStatusCode::OK);
    let rows = body_json(&body).as_array().unwrap().clone();
    assert_eq!(rows.len(), 3, "expected 3 rungs, got {rows:?}");
    assert_eq!(rows[0]["run_id"], "root-1");
    assert_eq!(rows[0]["incumbent"]["cost"], 0.02);
    assert_eq!(rows[1]["run_id"], "root-1-rung2");
    assert_eq!(rows[1]["incumbent"], Value::Null);
    assert_eq!(rows[2]["run_id"], "root-1-rung3");
}

// -------------------------------------------------------------------
// GET /api/bench/leaderboard
// -------------------------------------------------------------------

#[tokio::test]
async fn test_leaderboard_empty_when_no_matches() {
    let app = seeded_app(|_, _| {}).0;
    let (status, body) = http_get(app, "/api/bench/leaderboard").await;
    assert_eq!(status, HttpStatusCode::OK);
    let entries = body_json(&body).as_array().unwrap().clone();
    assert!(
        entries.is_empty(),
        "expected empty leaderboard, got {entries:?}"
    );
}

#[tokio::test]
async fn test_leaderboard_aggregates_correctly() {
    // Seed with two runs that have well-known outcomes.
    let app = seeded_app(|conn, _dir| {
        // Run 1: strong beats master twice.
        conn.execute(
            "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
             VALUES ('run1', 'round_robin', 'druid', 'abc', false, 'h', NULL, '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/1')",
            duckdb::params![],
        ).unwrap();
        conn.execute(
            "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
             VALUES \
               ('run1', 1, '2026-01-01T00:00:10Z', 'strong', 'master', 'win_a', 'strong'),\
               ('run1', 2, '2026-01-01T00:00:20Z', 'master', 'strong', 'win_b', 'strong')",
            duckdb::params![],
        ).unwrap();

        // Run 2: strong draws with easy, easy beats master.
        conn.execute(
            "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
             VALUES ('run2', 'round_robin', 'druid', 'abc', false, 'h', NULL, '2026-01-02T00:00:00Z', '2026-01-02T01:00:00Z', 'completed', '/tmp/2')",
            duckdb::params![],
        ).unwrap();
        conn.execute(
            "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
             VALUES \
               ('run2', 1, '2026-01-02T00:00:10Z', 'strong', 'easy', 'draw', NULL),\
               ('run2', 2, '2026-01-02T00:00:20Z', 'easy', 'master', 'win_a', 'easy')",
            duckdb::params![],
        ).unwrap();
    })
    .0;

    let (status, body) = http_get(app, "/api/bench/leaderboard").await;
    assert_eq!(status, HttpStatusCode::OK);
    let entries = body_json(&body).as_array().unwrap().clone();

    // Three strategies: strong, master, easy.
    // strong: vs master (win+win=2 wins), vs easy (draw) → 3 games, 2 wins, 0 losses, 1 draw
    // master: vs strong (loss+loss=2 losses), vs easy (loss) → 3 games, 0 wins, 3 losses, 0 draws
    // easy: vs strong (draw), vs master (win) → 2 games, 1 win, 0 losses, 1 draw

    let by_strategy: HashMap<&str, &Value> = entries
        .iter()
        .map(|e| (e["strategy"].as_str().unwrap(), e))
        .collect();

    // strong
    let s = by_strategy["strong"];
    assert_eq!(s["total"], 3);
    assert_eq!(s["wins"], 2);
    assert_eq!(s["losses"], 0);
    assert_eq!(s["draws"], 1);
    assert!((s["win_rate"].as_f64().unwrap() - (2.5 / 3.0)).abs() < 1e-9);

    // master
    let m = by_strategy["master"];
    assert_eq!(m["total"], 3);
    assert_eq!(m["wins"], 0);
    assert_eq!(m["losses"], 3);
    assert_eq!(m["draws"], 0);
    assert!((m["win_rate"].as_f64().unwrap() - 0.0).abs() < 1e-9);

    // easy
    let e = by_strategy["easy"];
    assert_eq!(e["total"], 2);
    assert_eq!(e["wins"], 1);
    assert_eq!(e["losses"], 0);
    assert_eq!(e["draws"], 1);
    assert!((e["win_rate"].as_f64().unwrap() - (1.5 / 2.0)).abs() < 1e-9);

    // Wilson CI lower < win_rate < upper for all entries.
    for entry in &entries {
        let wr = entry["win_rate"].as_f64().unwrap();
        let lo = entry["ci_lower"].as_f64().unwrap();
        let hi = entry["ci_upper"].as_f64().unwrap();
        assert!(lo <= wr, "ci_lower {lo} > win_rate {wr}");
        assert!(wr <= hi, "win_rate {wr} > ci_upper {hi}");
    }
}

#[tokio::test]
async fn test_leaderboard_filters_by_game() {
    let app = seeded_app(|conn, _dir| {
        // Druid matches.
        conn.execute(
            "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
             VALUES ('druid-run', 'round_robin', 'druid', 'abc', false, 'h', NULL, '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/d')",
            duckdb::params![],
        ).unwrap();
        conn.execute(
            "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
             VALUES ('druid-run', 1, '2026-01-01T00:00:10Z', 'strong', 'master', 'win_a', 'strong')",
            duckdb::params![],
        ).unwrap();

        // TTT matches.
        conn.execute(
            "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
             VALUES ('ttt-run', 'round_robin', 'ttt', 'abc', false, 'h', NULL, '2026-01-02T00:00:00Z', '2026-01-02T01:00:00Z', 'completed', '/tmp/t')",
            duckdb::params![],
        ).unwrap();
        conn.execute(
            "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
             VALUES ('ttt-run', 1, '2026-01-02T00:00:10Z', 'minimax', 'random', 'win_a', 'minimax')",
            duckdb::params![],
        ).unwrap();
    })
    .0;

    // Filter by druid.
    let (status, body) = http_get(app.clone(), "/api/bench/leaderboard?game=druid").await;
    assert_eq!(status, HttpStatusCode::OK);
    let entries = body_json(&body).as_array().unwrap().clone();
    assert_eq!(
        entries.len(),
        2,
        "expected 2 druid strategies, got {entries:?}"
    );
    let strategies: Vec<&str> = entries
        .iter()
        .map(|e| e["strategy"].as_str().unwrap())
        .collect();
    assert!(strategies.contains(&"strong"));
    assert!(strategies.contains(&"master"));

    // Filter by ttt.
    let (status, body) = http_get(app.clone(), "/api/bench/leaderboard?game=ttt").await;
    assert_eq!(status, HttpStatusCode::OK);
    let entries = body_json(&body).as_array().unwrap().clone();
    assert_eq!(
        entries.len(),
        2,
        "expected 2 ttt strategies, got {entries:?}"
    );
}

// -------------------------------------------------------------------
// GET /api/bench/kinds
// -------------------------------------------------------------------

#[tokio::test]
async fn test_list_kinds_includes_round_robin_and_tuner() {
    let app = seeded_app(|_, _| {}).0;
    let (status, body) = http_get(app, "/api/bench/kinds").await;
    assert_eq!(status, HttpStatusCode::OK);
    let kinds = body_json(&body).as_array().unwrap().clone();
    let kind_names: Vec<&str> = kinds.iter().map(|k| k["kind"].as_str().unwrap()).collect();
    assert!(kind_names.contains(&"round_robin"));
    assert!(kind_names.contains(&"tuner"));
}

// -------------------------------------------------------------------
// GET /api/bench/tuner/kinds
// -------------------------------------------------------------------
