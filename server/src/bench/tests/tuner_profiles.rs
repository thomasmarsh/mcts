use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use axum::http::StatusCode;
use serde_json::json;

use super::support::{
    body_json, default_seed, http_delete, http_get, http_post_json, http_put_json,
    seeded_app_with_state,
};

/// A stand-in `game-*` binary next to the test executable, so
/// `find_game_binary` resolves a profile's `game_kind`. `profilegame` is not a
/// registered game kind, so resolution falls through to the by-name lookup that
/// checks `current_exe()`'s directory.
struct FakeGameBinary(PathBuf);

impl FakeGameBinary {
    fn install(kind: &str) -> Self {
        let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
        let path = dir.join(kind);
        std::fs::write(&path, b"#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        Self(path)
    }
}

impl Drop for FakeGameBinary {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn profile_body(kind: &str) -> serde_json::Value {
    json!({
        "profile_id": "druid-ucb1-sweep",
        "game_kind": kind,
        "objective_key": "prof-obj",
        "constraints": [{ "set": { "c": { "range": [1.2, 1.8] } } }],
        "efforts": { "tuning": { "kind": "iterations", "value": 100 } },
        "budgets": {
            "tuning_pair_budget": 16,
            "validation_pair_budget": 12,
            "production_validation_pairs": 4
        }
    })
}

fn seed_objective(dir: &std::path::Path, kind: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("prof-obj.json"),
        format!(r#"{{"objective_id": "prof-obj", "game_kind": "{kind}"}}"#),
    )
    .unwrap();
}

#[tokio::test]
async fn profile_crud_round_trips_and_preflights() {
    const KIND: &str = "profilegamecrud";
    let _bin = FakeGameBinary::install(KIND);
    let (app, root, state) = seeded_app_with_state(default_seed);
    seed_objective(&state.tuner_objectives_dir, KIND);

    // Validate is a dry run: 200 { ok }, nothing written.
    let (status, body) = http_post_json(
        app.clone(),
        "/api/bench/tuner/profiles/scratch/validate",
        profile_body(KIND),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["ok"], true);
    assert!(!state.tuner_profiles_dir.join("scratch.json").exists());

    // PUT a valid profile, then GET it back.
    let (status, _) =
        http_put_json(app.clone(), "/api/bench/tuner/profiles/mine-v1", profile_body(KIND)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(state.tuner_profiles_dir.join("mine-v1.json").is_file());

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/profiles/mine-v1").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body)["content"]["objective_key"], "prof-obj");
    assert_eq!(body_json(&body)["is_seed"], false);

    // It lists with a constraint count.
    let (_, body) = http_get(app.clone(), "/api/bench/tuner/profiles").await;
    let rows = body_json(&body);
    assert_eq!(rows.as_array().unwrap().len(), 1);
    assert_eq!(rows[0]["key"], "mine-v1");
    assert_eq!(rows[0]["profile_id"], "druid-ucb1-sweep");
    assert_eq!(rows[0]["constraint_count"], 1);

    // DELETE removes it; a second delete is a 404.
    let (status, _) = http_delete(app.clone(), "/api/bench/tuner/profiles/mine-v1").await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = http_delete(app.clone(), "/api/bench/tuner/profiles/mine-v1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = http_get(app, "/api/bench/tuner/profiles/mine-v1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn profile_validate_rejects_bad_bodies() {
    const KIND: &str = "profilegamereject";
    let _bin = FakeGameBinary::install(KIND);
    let (app, root, state) = seeded_app_with_state(default_seed);
    seed_objective(&state.tuner_objectives_dir, KIND);

    // Missing objective_key -> pre-check 400 (no preflight).
    let mut no_objective = profile_body(KIND);
    no_objective.as_object_mut().unwrap().remove("objective_key");
    let (status, _) = http_post_json(
        app.clone(),
        "/api/bench/tuner/profiles/x/validate",
        no_objective,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Unknown objective_key -> resolution 400.
    let mut unknown = profile_body(KIND);
    unknown["objective_key"] = json!("nope");
    let (status, body) =
        http_post_json(app.clone(), "/api/bench/tuner/profiles/x/validate", unknown).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body_json(&body)["error"]
        .as_str()
        .unwrap()
        .contains("objective_key"));

    // A bad effort kind -> lowering 400.
    let mut bad_effort = profile_body(KIND);
    bad_effort["efforts"]["tuning"]["kind"] = json!("forever");
    let (status, _) =
        http_post_json(app.clone(), "/api/bench/tuner/profiles/x/validate", bad_effort).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Path traversal is refused on the keyed routes.
    let (status, _) = http_get(app, "/api/bench/tuner/profiles/..%2Fsecret").await;
    assert!(status.is_client_error());

    std::fs::remove_dir_all(root).unwrap();
}
