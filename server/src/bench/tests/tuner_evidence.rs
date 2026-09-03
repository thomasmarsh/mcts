//! Endpoint + pump tests for the live evidence tail and SSE stream. No real
//! tuner process: a hand-written `evidence.jsonl` under a journalled run dir,
//! and `pump_evidence` driven directly with millisecond timings and an
//! injected liveness fn.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use mcts_bench::tuner_launch::{self, TunerLaunchRecord};
use serde_json::Value;

use super::super::tuner_evidence::{pump_evidence, StreamTiming};
use super::support::{body_json, default_seed, http_get, seeded_app};

fn line(sequence: u64, event_type: &str) -> String {
    format!(
        r#"{{"schema_version":4,"sequence":{sequence},"type":"{event_type}","payload":{{"n":{sequence}}}}}"#
    )
}

/// Journal a live run and return its `evidence.jsonl` path.
fn live_run(runs_root: &std::path::Path, run_id: &str) -> std::path::PathBuf {
    let run_dir = runs_root.join(run_id);
    std::fs::create_dir_all(&run_dir).unwrap();
    tuner_launch::append_launch(
        runs_root,
        &TunerLaunchRecord {
            run_id: run_id.into(),
            argv: vec!["uv".into()],
            run_dir: run_dir.clone(),
            pid: Some(999_999_999),
            started_at: "2026-01-01T00:00:00Z".into(),
            terminal_outcome: None,
        },
    )
    .unwrap();
    run_dir.join("evidence.jsonl")
}

#[tokio::test]
async fn tail_returns_only_events_past_since_seq() {
    let (app, root) = seeded_app(default_seed);
    let runs_root = root.join("bench-runs");
    let evidence = live_run(&runs_root, "tuner_ev1");
    std::fs::write(
        &evidence,
        format!("{}\n{}\n{}\n", line(1, "pair_started"), line(2, "pair_completed"), line(3, "cohort_completed")),
    )
    .unwrap();

    let (status, body) = http_get(
        app.clone(),
        "/api/bench/tuner/runs/tuner_ev1/evidence?since_seq=1",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let json: Value = body_json(&body);
    let seqs: Vec<u64> = json["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["sequence"].as_u64().unwrap())
        .collect();
    assert_eq!(seqs, [2, 3]);
    assert_eq!(json["next_seq"].as_u64().unwrap(), 3);
    assert_eq!(json["events"][0]["type"].as_str().unwrap(), "pair_completed");

    // Already caught up: no events, next_seq is the log's max.
    let (_, body) = http_get(
        app.clone(),
        "/api/bench/tuner/runs/tuner_ev1/evidence?since_seq=3",
    )
    .await;
    let json: Value = body_json(&body);
    assert!(json["events"].as_array().unwrap().is_empty());
    assert_eq!(json["next_seq"].as_u64().unwrap(), 3);
}

#[tokio::test]
async fn tail_withholds_a_torn_last_line() {
    let (app, root) = seeded_app(default_seed);
    let runs_root = root.join("bench-runs");
    let evidence = live_run(&runs_root, "tuner_ev2");
    // Two complete lines then a partial third the writer has not finished.
    std::fs::write(
        &evidence,
        format!("{}\n{}\n{{\"schema_version\":4,\"sequence\":3", line(1, "pair_started"), line(2, "pair_completed")),
    )
    .unwrap();

    let (_, body) = http_get(app.clone(), "/api/bench/tuner/runs/tuner_ev2/evidence").await;
    let json: Value = body_json(&body);
    assert_eq!(json["events"].as_array().unwrap().len(), 2);
    assert_eq!(json["next_seq"].as_u64().unwrap(), 2);

    // The writer completes the line -> it appears.
    std::fs::write(
        &evidence,
        format!("{}\n{}\n{}\n", line(1, "pair_started"), line(2, "pair_completed"), line(3, "cohort_completed")),
    )
    .unwrap();
    let (_, body) = http_get(
        app.clone(),
        "/api/bench/tuner/runs/tuner_ev2/evidence?since_seq=2",
    )
    .await;
    let json: Value = body_json(&body);
    assert_eq!(json["events"].as_array().unwrap().len(), 1);
    assert_eq!(json["events"][0]["sequence"].as_u64().unwrap(), 3);
}

#[tokio::test]
async fn tail_unknown_run_is_404() {
    let (app, _root) = seeded_app(default_seed);
    let (status, _) = http_get(app, "/api/bench/tuner/runs/nope/evidence").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn pump_streams_appended_lines_then_ends_when_the_run_stops() {
    let dir = std::env::temp_dir().join(format!("mcts_ev_pump_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("evidence.jsonl");
    std::fs::write(&path, format!("{}\n", line(1, "pair_started"))).unwrap();

    let alive = Arc::new(AtomicBool::new(true));
    let alive_probe = alive.clone();
    let is_live: Arc<dyn Fn() -> bool + Send + Sync> =
        Arc::new(move || alive_probe.load(Ordering::SeqCst));
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);

    let timing = StreamTiming {
        poll: Duration::from_millis(10),
        quiet_close: Duration::from_millis(30),
    };
    let pump = tokio::spawn(pump_evidence(path.clone(), 0, is_live, tx, timing));

    // First frame is the catch-up line.
    let _ = rx.recv().await.expect("catch-up event");

    // Append a line while live -> it streams.
    tokio::time::sleep(Duration::from_millis(20)).await;
    std::fs::write(
        &path,
        format!("{}\n{}\n", line(1, "pair_started"), line(2, "pair_completed")),
    )
    .unwrap();
    let _ = rx.recv().await.expect("streamed append");

    // The run stops; after the quiet window the pump emits its final `end`
    // frame and returns, which drops the sender and closes the channel.
    alive.store(false, Ordering::SeqCst);
    let mut trailing = 0;
    while rx.recv().await.is_some() {
        trailing += 1;
        assert!(trailing < 100, "pump should end shortly after the run stops");
    }
    tokio::time::timeout(Duration::from_secs(1), pump)
        .await
        .expect("pump task should finish once the run is no longer live")
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
