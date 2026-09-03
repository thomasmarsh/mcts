use std::path::PathBuf;

use axum::http::StatusCode;
use mcts_bench::tuner_launch::{self, TerminalOutcome, TunerLaunchRecord};
use serde_json::json;

use super::support::{
    body_json, default_seed, http_delete, http_get, http_post_json, http_put_json, seeded_app,
};

fn objective_body(game_kind: &str) -> serde_json::Value {
    json!({
        "schema_version": 1,
        "objective_id": format!("{game_kind}-editor-v1"),
        "game_kind": game_kind,
        "opponents": [
            {"id": "schema-default", "label": "Schema default", "role": "default",
             "weight": 1, "config": {"source": "schema_default"}},
            {"id": "hist", "label": "Historical", "role": "historical_reference",
             "weight": 2, "config": {"source": "inline", "value": {"c": 1.4}}}
        ],
        "start_distribution": {"kind": "default_only"}
    })
}

fn record(runs_root: &std::path::Path, run_id: &str, pid: Option<u32>) {
    let run_dir = runs_root.join(run_id);
    std::fs::create_dir_all(&run_dir).unwrap();
    tuner_launch::append_launch(
        runs_root,
        &TunerLaunchRecord {
            run_id: run_id.into(),
            argv: vec!["uv".into()],
            run_dir,
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
async fn objectives_route_lists_configured_files_by_key() {
    let (app, root, state) = super::support::seeded_app_with_state(default_seed);
    std::fs::create_dir_all(&state.tuner_objectives_dir).unwrap();
    std::fs::write(
        state.tuner_objectives_dir.join("ttt-smoke-v1.json"),
        r#"{"objective_id": "ttt-smoke-v1", "game_kind": "ttt"}"#,
    )
    .unwrap();
    std::fs::write(state.tuner_objectives_dir.join("notes.txt"), "ignored").unwrap();

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/objectives").await;
    assert_eq!(status, StatusCode::OK);
    let rows = body_json(&body);
    assert_eq!(rows.as_array().unwrap().len(), 1);
    assert_eq!(rows[0]["key"], "ttt-smoke-v1");
    assert_eq!(rows[0]["game_kind"], "ttt");

    // An unknown key is a 400, not a path escape.
    let (status, _) = http_post_json(
        app,
        "/api/bench/tuner/runs",
        json!({
            "game_kind": "ttt",
            "objective_key": "../secret",
            "run_id": "keytest",
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
async fn preflight_reports_launch_problems_and_gates_the_launch() {
    let (app, root) = seeded_app(default_seed);
    let good = json!({
        "game_binary": "/games/nim", "objective_file": "/objectives/nim.yaml",
        "run_id": "pf-ok", "task_seed": 1,
        "tuning_pair_budget": 4, "validation_pair_budget": 4, "production_validation_pairs": 4
    });
    let bad = json!({
        "game_binary": "/games/nim", "objective_file": "/objectives/nim.yaml",
        "run_id": "pf-badcfg", "task_seed": 1,
        "tuning_pair_budget": 4, "validation_pair_budget": 4, "production_validation_pairs": 4
    });

    let (status, body) =
        http_post_json(app.clone(), "/api/bench/tuner/runs/preflight", good).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["ok"], true);

    let (status, body) =
        http_post_json(app.clone(), "/api/bench/tuner/runs/preflight", bad.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["ok"], false);
    assert!(body_json(&body)["errors"][0]
        .as_str()
        .unwrap()
        .contains("cannot exceed production"));

    // The same failing request is refused by the launch route itself.
    let (status, body) = http_post_json(app, "/api/bench/tuner/runs", bad).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("cannot exceed production"));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn extend_validates_the_request_and_relaunches() {
    let (app, root) = seeded_app(default_seed);
    let runs_root = root.join("bench-runs");
    let run_dir = runs_root.join("tuner_ext");
    std::fs::create_dir_all(&run_dir).unwrap();
    tuner_launch::append_launch(
        &runs_root,
        &TunerLaunchRecord {
            run_id: "tuner_ext".into(),
            argv: vec!["true".into()],
            run_dir: run_dir.clone(),
            pid: Some(1),
            started_at: "2026-01-01T00:00:00Z".into(),
            terminal_outcome: Some(TerminalOutcome::Exited),
        },
    )
    .unwrap();

    // A reason is required and at least one delta must be positive.
    let (status, _) = http_post_json(
        app.clone(),
        "/api/bench/tuner/runs/tuner_ext/extend",
        json!({ "tuning_pair_attempts_delta": 6, "reason": "" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = http_post_json(
        app.clone(),
        "/api/bench/tuner/runs/missing/extend",
        json!({ "reason": "fund more" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // A valid request relaunches the run with --resume and the extend flags.
    let (status, body) = http_post_json(
        app,
        "/api/bench/tuner/runs/tuner_ext/extend",
        json!({ "tuning_pair_attempts_delta": 6, "reason": "fund another cohort" }),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let argv = body_json(&body)["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert!(argv.contains(&"--resume".to_string()));
    assert!(argv
        .windows(2)
        .any(|pair| pair == ["--extend-tuning-pairs", "6"]));
    assert!(argv
        .windows(2)
        .any(|pair| pair == ["--extend-reason", "fund another cohort"]));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn seed_copy_never_overwrites_a_user_edit() {
    let dir = std::env::temp_dir().join(format!("mcts_seed_test_{}", std::process::id()));
    let seed = dir.join("seed");
    let user = dir.join("user");
    std::fs::create_dir_all(&seed).unwrap();
    std::fs::create_dir_all(&user).unwrap();
    std::fs::write(seed.join("a.json"), r#"{"objective_id":"a-seed"}"#).unwrap();
    std::fs::write(seed.join("b.json"), r#"{"objective_id":"b-seed"}"#).unwrap();
    std::fs::write(user.join("a.json"), r#"{"objective_id":"a-edited"}"#).unwrap();

    crate::bench::seed_tuner_objectives(&seed, &user);

    assert_eq!(
        std::fs::read_to_string(user.join("a.json")).unwrap(),
        r#"{"objective_id":"a-edited"}"#
    );
    assert_eq!(
        std::fs::read_to_string(user.join("b.json")).unwrap(),
        r#"{"objective_id":"b-seed"}"#
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn objective_crud_round_trips_and_validates() {
    let (app, root, state) = super::support::seeded_app_with_state(default_seed);
    std::fs::create_dir_all(&state.tuner_objectives_dir).unwrap();

    // PUT a valid objective, then GET it back.
    let (status, _) =
        http_put_json(app.clone(), "/api/bench/tuner/objectives/mine-v1", objective_body("ttt")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(state.tuner_objectives_dir.join("mine-v1.json").is_file());

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/objectives/mine-v1").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["content"]["game_kind"], "ttt");
    assert_eq!(body_json(&body)["is_seed"], false);

    // It shows up in the list with an opponent count.
    let (_, body) = http_get(app.clone(), "/api/bench/tuner/objectives").await;
    let rows = body_json(&body);
    assert_eq!(rows[0]["key"], "mine-v1");
    assert_eq!(rows[0]["opponent_count"], 2);

    // The injected validator rejects `game_kind: "reject"`.
    let (status, _) = http_put_json(
        app.clone(),
        "/api/bench/tuner/objectives/bad-v1",
        objective_body("reject"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(!state.tuner_objectives_dir.join("bad-v1.json").exists());

    // schema_version must be 1 (pre-check, no validator call).
    let (status, _) = http_put_json(
        app.clone(),
        "/api/bench/tuner/objectives/nope",
        json!({"schema_version": 2, "game_kind": "ttt"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // A non-object game_config is refused by the pre-check.
    let mut bad_config = objective_body("ttt");
    bad_config["game_config"] = json!(9);
    let (status, _) =
        http_put_json(app.clone(), "/api/bench/tuner/objectives/nope2", bad_config).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Path traversal is refused on every keyed route.
    for uri in [
        "/api/bench/tuner/objectives/..%2Fsecret",
        "/api/bench/tuner/objectives/a.b%2Fc",
    ] {
        let (status, _) = http_get(app.clone(), uri).await;
        assert!(status.is_client_error(), "{uri} -> {status}");
    }

    // DELETE removes it; a second delete is a 404.
    let (status, _) = http_delete(app.clone(), "/api/bench/tuner/objectives/mine-v1").await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = http_delete(app.clone(), "/api/bench/tuner/objectives/mine-v1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = http_get(app, "/api/bench/tuner/objectives/mine-v1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn objective_validate_route_is_a_dry_run() {
    let (app, root, state) = super::support::seeded_app_with_state(default_seed);
    std::fs::create_dir_all(&state.tuner_objectives_dir).unwrap();

    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/objectives/scratch/validate",
        objective_body("ttt"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["ok"], true);
    // Nothing was written.
    assert!(!state.tuner_objectives_dir.join("scratch.json").exists());

    let (_, body) = http_post_json(
        app,
        "/api/bench/tuner/objectives/scratch/validate",
        objective_body("reject"),
    )
    .await;
    assert_eq!(body_json(&body)["ok"], false);

    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn stop_is_idempotent_on_an_already_exited_run() {
    let (app, root) = seeded_app(default_seed);
    let runs_root = root.join("bench-runs");
    record(&runs_root, "done", Some(999_999_999));
    // A run that got far enough to write its manifest before exiting reads
    // back as "exited"; one that died before that reads back as "failed".
    std::fs::write(runs_root.join("done/manifest.json"), "{}").unwrap();
    tuner_launch::append_terminal(&runs_root, "done", TerminalOutcome::Exited).unwrap();

    let (status, body) = http_post_json(app.clone(), "/api/bench/tuner/runs/done/stop", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["status"], "exited");

    let (status, _) = http_post_json(app, "/api/bench/tuner/runs/missing/stop", json!({})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    std::fs::remove_dir_all(root).unwrap();
}
