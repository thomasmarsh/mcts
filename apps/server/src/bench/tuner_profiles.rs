//! Launch-profile CRUD.
//!
//! A launch profile is a named, saved bundle `{game, objective, constraints,
//! efforts, budgets}` an operator starts tuner runs from — the launch-form
//! counterpart to a frozen objective. Profiles are *not* objectives: they carry
//! no scoring target or opponent panel, only a reference to an `objective_key`
//! that does. Like objectives, a profile is one JSON file keyed by its filename
//! stem; the absolute path never crosses the API boundary.

use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State as AxumState},
    http::StatusCode,
    response::Json,
};
use mcts_bench::tuner_launch::TunerLaunchRequest;
use serde::Serialize;
use serde_json::Value;

use super::tuner_runs::{
    bad_request, file_updated_at, not_found, resolve_launch_request, seed_stems, valid_file_key,
};
use super::{BenchError, BenchState};

/// One launch-profile file the launch form can offer, keyed by its stem.
#[derive(Serialize)]
pub(crate) struct ProfileFileInfo {
    key: String,
    profile_id: Option<String>,
    game_kind: Option<String>,
    objective_key: Option<String>,
    constraint_count: u32,
    updated_at: Option<String>,
    /// The same stem is also shipped in the read-only seed corpus.
    is_seed: bool,
}

/// The full profile JSON plus its metadata (`GET
/// /api/bench/tuner/profiles/{key}`).
#[derive(Serialize)]
pub(crate) struct ProfileFileDetail {
    key: String,
    content: Value,
    updated_at: Option<String>,
    is_seed: bool,
}

fn profile_path(state: &BenchState, key: &str) -> Result<std::path::PathBuf, BenchError> {
    if !valid_file_key(key) {
        return Err(bad_request(format!("invalid profile key '{key}'")));
    }
    Ok(state.tuner_profiles_dir.join(format!("{key}.json")))
}

fn constraint_count(content: &Value) -> u32 {
    match content.get("constraints") {
        Some(Value::Array(entries)) => entries.len() as u32,
        // The bare `{ name: {...} }` sugar map counts each parameter narrowing.
        Some(Value::Object(map)) => map.len() as u32,
        _ => 0,
    }
}

fn read_profile_files(dir: &std::path::Path, seed_dir: &std::path::Path) -> Vec<ProfileFileInfo> {
    let seeds = seed_stems(seed_dir);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut files: Vec<ProfileFileInfo> = entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| {
            let path = entry.path();
            let key = path.file_stem()?.to_string_lossy().into_owned();
            let parsed: Option<Value> = std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok());
            let field = |name: &str| {
                parsed
                    .as_ref()
                    .and_then(|value| value.get(name))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            };
            let constraint_count = parsed.as_ref().map_or(0, constraint_count);
            Some(ProfileFileInfo {
                is_seed: seeds.contains(&key),
                key,
                profile_id: field("profile_id"),
                game_kind: field("game_kind"),
                objective_key: field("objective_key"),
                constraint_count,
                updated_at: file_updated_at(&path),
            })
        })
        .collect();
    files.sort_by(|a, b| a.key.cmp(&b.key));
    files
}

/// Seed the writable launch-profile directory from the read-only corpus.
pub fn seed_tuner_profiles(seed_dir: &std::path::Path, user_dir: &std::path::Path) {
    super::tuner_runs::seed_json_files(seed_dir, user_dir);
}

/// Cheap in-process pre-check: the body must be a JSON object naming a
/// non-empty `game_kind` and `objective_key`, and any `efforts` / `budgets` it
/// carries must be objects.
fn precheck_profile(body: &Value) -> Result<(), BenchError> {
    let object = body
        .as_object()
        .ok_or_else(|| bad_request("profile body must be a JSON object".into()))?;
    let non_empty_str = |name: &str| {
        object
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| bad_request(format!("profile {name} is required")))
    };
    non_empty_str("game_kind")?;
    non_empty_str("objective_key")?;
    for name in ["efforts", "budgets"] {
        if object.get(name).is_some_and(|value| !value.is_object()) {
            return Err(bad_request(format!("profile {name} must be a JSON object")));
        }
    }
    Ok(())
}

/// Lower a profile body to the launch request the preflight validates. `efforts`
/// map to the phase iteration/time fields; `budgets` to the pair-budget and
/// cohort fields; `constraints` pass through verbatim. Anything the profile
/// omits falls back to the tuner CLI's own default (here: a small budget so the
/// preflight has concrete numbers to check the coherence rules against).
fn profile_to_launch_request(body: &Value) -> Result<TunerLaunchRequest, BenchError> {
    let object = body.as_object().expect("prechecked");
    let mut request = serde_json::Map::new();
    request.insert("game_kind".into(), object["game_kind"].clone());
    request.insert("objective_key".into(), object["objective_key"].clone());
    request.insert("run_id".into(), Value::from("profile-validate"));
    request.insert("task_seed".into(), Value::from(0));

    let budgets = object.get("budgets").and_then(Value::as_object);
    let budget_u64 = |name: &str, default: u64| {
        budgets
            .and_then(|map| map.get(name))
            .and_then(Value::as_u64)
            .unwrap_or(default)
    };
    request.insert(
        "tuning_pair_budget".into(),
        Value::from(budget_u64("tuning_pair_budget", 32)),
    );
    request.insert(
        "validation_pair_budget".into(),
        Value::from(budget_u64("validation_pair_budget", 24)),
    );
    request.insert(
        "production_validation_pairs".into(),
        Value::from(budget_u64("production_validation_pairs", 8)),
    );
    if let Some(map) = budgets {
        for name in [
            "cohort_size",
            "finalists",
            "bootstrap_candidates",
            "random_reserve_candidates",
            "diagnostic_pair_budget",
        ] {
            if let Some(value) = map.get(name) {
                request.insert(name.into(), value.clone());
            }
        }
    }

    if let Some(efforts) = object.get("efforts").and_then(Value::as_object) {
        for phase in ["tuning", "validation", "production"] {
            let Some(effort) = efforts.get(phase).and_then(Value::as_object) else {
                continue;
            };
            let kind = effort.get("kind").and_then(Value::as_str);
            let Some(value) = effort.get("value").cloned() else {
                return Err(bad_request(format!("profile {phase} effort has no value")));
            };
            let field = match kind {
                Some("iterations") => format!("{phase}_max_iterations"),
                Some("time_ms") => format!("{phase}_max_time_ms"),
                _ => {
                    return Err(bad_request(format!(
                        "profile {phase} effort kind must be 'iterations' or 'time_ms'"
                    )))
                }
            };
            request.insert(field, value);
        }
    }

    if let Some(constraints) = object.get("constraints") {
        request.insert("constraints".into(), constraints.clone());
    }

    serde_json::from_value(Value::Object(request))
        .map_err(|error| bad_request(format!("profile cannot be lowered to a launch: {error}")))
}

/// Run the launch preflight against a profile body: resolve its `game_kind` /
/// `objective_key`, then dry-run every check `tuner_cli` applies before it
/// starts a run.
fn preflight_profile(
    state: &BenchState,
    body: &Value,
) -> Result<super::types::LaunchPreflight, BenchError> {
    let mut request = profile_to_launch_request(body)?;
    resolve_launch_request(state, &mut request)?;
    (state.tuner_launch_preflight)(&request).map_err(|error| BenchError {
        status: StatusCode::BAD_GATEWAY,
        message: format!("could not preflight the profile: {error}"),
    })
}

/// `GET /api/bench/tuner/profiles`
pub(crate) async fn list_tuner_profiles(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Json<Vec<ProfileFileInfo>> {
    Json(read_profile_files(
        &state.tuner_profiles_dir,
        &state.tuner_seed_profiles_dir,
    ))
}

/// `GET /api/bench/tuner/profiles/{key}` — the profile's full JSON.
pub(crate) async fn get_tuner_profile(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(key): AxumPath<String>,
) -> Result<Json<ProfileFileDetail>, BenchError> {
    let path = profile_path(&state, &key)?;
    let text = std::fs::read_to_string(&path)
        .map_err(|_| not_found(format!("unknown profile '{key}'")))?;
    let content: Value = serde_json::from_str(&text)
        .map_err(|error| bad_request(format!("profile '{key}' is not valid JSON: {error}")))?;
    Ok(Json(ProfileFileDetail {
        is_seed: seed_stems(&state.tuner_seed_profiles_dir).contains(&key),
        updated_at: file_updated_at(&path),
        key,
        content,
    }))
}

/// `POST /api/bench/tuner/profiles/{key}/validate` — dry-run the launch
/// preflight with the profile's values. Writes nothing.
pub(crate) async fn validate_tuner_profile(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(key): AxumPath<String>,
    Json(body): Json<Value>,
) -> Result<Json<super::types::LaunchPreflight>, BenchError> {
    profile_path(&state, &key)?;
    precheck_profile(&body)?;
    Ok(Json(preflight_profile(&state, &body)?))
}

/// `PUT /api/bench/tuner/profiles/{key}` — create or replace, gated on a
/// successful launch preflight so a saved profile always names a launchable run.
pub(crate) async fn put_tuner_profile(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(key): AxumPath<String>,
    Json(body): Json<Value>,
) -> Result<Json<ProfileFileDetail>, BenchError> {
    let path = profile_path(&state, &key)?;
    precheck_profile(&body)?;
    let preflight = preflight_profile(&state, &body)?;
    if !preflight.ok {
        return Err(bad_request(if preflight.errors.is_empty() {
            "profile rejected by preflight".to_owned()
        } else {
            preflight.errors.join("; ")
        }));
    }
    std::fs::create_dir_all(&state.tuner_profiles_dir)?;
    let pretty =
        serde_json::to_string_pretty(&body).map_err(|error| bad_request(error.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, pretty.as_bytes())?;
    std::fs::rename(&tmp, &path)?;
    Ok(Json(ProfileFileDetail {
        is_seed: seed_stems(&state.tuner_seed_profiles_dir).contains(&key),
        updated_at: file_updated_at(&path),
        key,
        content: body,
    }))
}

/// `DELETE /api/bench/tuner/profiles/{key}`.
pub(crate) async fn delete_tuner_profile(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(key): AxumPath<String>,
) -> Result<StatusCode, BenchError> {
    let path = profile_path(&state, &key)?;
    std::fs::remove_file(&path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => not_found(format!("unknown profile '{key}'")),
        _ => BenchError::from(error),
    })?;
    Ok(StatusCode::NO_CONTENT)
}
