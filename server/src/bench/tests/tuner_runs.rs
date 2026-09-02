use std::path::PathBuf;

use axum::http::StatusCode;
use mcts_bench::tuner_launch::{self, TerminalOutcome, TunerLaunchRecord};
use serde_json::json;

use super::support::{body_json, default_seed, http_get, http_post_json, seeded_app};

fn record(runs_root: &std::path::Path, run_id: &str, pid: Option<u32>) {
    tuner_launch::append_launch(
        runs_root,
        &TunerLaunchRecord {
            run_id: run_id.into(),
            argv: vec!["uv".into()],
            run_dir: PathBuf::from(run_id),
            pid,
            started_at: "2026-01-01T00:00:00Z".into(),
            terminal_outcome: None,
        },
    )
    .unwrap();
}

#[tokio::test]
async fn get_reports_liveness_and_legacy_session_routes_are_absent() {
    let (app, root) = seeded_app(default_seed);
    let runs_root = root.join("bench-runs");
    record(&runs_root, "tuner_12a", Some(999_999_999));

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/runs/tuner_12a").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["status"], "unknown");

    tuner_launch::append_terminal(&runs_root, "tuner_12a", TerminalOutcome::Signalled).unwrap();
    let (_, body) = http_get(app.clone(), "/api/bench/tuner/runs/tuner_12a").await;
    assert_eq!(body_json(&body)["status"], "exited");

    let (status, _) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn list_returns_all_records_in_launch_order() {
    let (app, root) = seeded_app(default_seed);
    let runs_root = root.join("bench-runs");
    record(&runs_root, "second", None);
    record(&runs_root, "first", None);

    let (status, body) = http_get(app, "/api/bench/tuner/runs").await;
    assert_eq!(status, StatusCode::OK);
    let rows = body_json(&body);
    assert_eq!(rows[0]["run_id"], "second");
    assert_eq!(rows[1]["run_id"], "first");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn launch_rejects_an_unsafe_run_id() {
    let (app, root) = seeded_app(default_seed);
    let (status, _) = http_post_json(
        app,
        "/api/bench/tuner/runs",
        json!({
            "game_binary": "/games/nim",
            "objective_file": "/objectives/nim.yaml",
            "run_id": "../escape",
            "task_seed": 1,
            "tuning_pair_budget": 4,
            "validation_pair_budget": 4,
            "production_validation_pairs": 4
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn stop_is_idempotent_on_an_already_exited_run() {
    let (app, root) = seeded_app(default_seed);
    let runs_root = root.join("bench-runs");
    record(&runs_root, "done", Some(999_999_999));
    tuner_launch::append_terminal(&runs_root, "done", TerminalOutcome::Exited).unwrap();

    let (status, body) = http_post_json(app.clone(), "/api/bench/tuner/runs/done/stop", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["status"], "exited");

    let (status, _) = http_post_json(app, "/api/bench/tuner/runs/missing/stop", json!({})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    std::fs::remove_dir_all(root).unwrap();
}
