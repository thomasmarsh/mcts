use std::sync::Arc;

use std::io::{BufRead, BufReader, Seek, SeekFrom};

use axum::{
    extract::{Path as AxumPath, Query as AxumQuery, State as AxumState},
    http::StatusCode,
    response::Json,
};
use mcts_bench::tuner_launch::{
    self, BudgetExtension, TerminalOutcome, TunerLaunchRecord, TunerLaunchRequest,
};
use serde::Serialize;

use super::{BenchError, BenchState};

#[derive(Serialize)]
pub(crate) struct TunerRunView {
    run_id: String,
    argv: Vec<String>,
    run_dir: String,
    pid: Option<u32>,
    started_at: String,
    terminal_outcome: Option<TerminalOutcome>,
    /// `live` | `exited` | `failed` | `unknown`. `failed` means the process
    /// ended before it ever wrote a `manifest.json`, so nothing in the
    /// projection will ever describe it -- `error_detail` carries its
    /// `launch.err` so the fleet can say what went wrong.
    status: &'static str,
    /// Tail of `launch.err`, populated only when `status == "failed"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    error_detail: Option<String>,
}

/// Last ~4 KiB of a failed run's `launch.err`, for the fleet's failure card.
fn err_tail(run_dir: &std::path::Path) -> Option<String> {
    let (lines, _) = tail_from(&run_dir.join("launch.err"), 0).ok()?;
    let joined = lines.join("\n");
    let trimmed = joined.trim();
    if trimmed.is_empty() {
        return None;
    }
    let start = trimmed.len().saturating_sub(4096);
    Some(trimmed[start..].to_string())
}

/// The `live | exited | failed | unknown` decision for one launch record.
/// `failed` means the process ended before it ever wrote a `manifest.json`,
/// so the projection will never describe it. Shared by the journal `view`
/// and the evidence stream's close condition.
pub(crate) fn liveness(record: &TunerLaunchRecord) -> &'static str {
    let started_but_never_worked = record.terminal_outcome.is_some()
        && !record.run_dir.join("manifest.json").exists()
        && !matches!(record.terminal_outcome, Some(TerminalOutcome::Signalled));
    if started_but_never_worked {
        "failed"
    } else if record.terminal_outcome.is_some() {
        "exited"
    } else if record.pid.is_some_and(tuner_launch::is_alive) {
        "live"
    } else {
        "unknown"
    }
}

fn view(record: TunerLaunchRecord) -> TunerRunView {
    let status = liveness(&record);
    let started_but_never_worked = status == "failed";
    let error_detail = if started_but_never_worked {
        err_tail(&record.run_dir)
    } else {
        None
    };
    TunerRunView {
        run_id: record.run_id,
        argv: record.argv,
        run_dir: record.run_dir.to_string_lossy().into_owned(),
        pid: record.pid,
        started_at: record.started_at,
        terminal_outcome: record.terminal_outcome,
        status,
        error_detail,
    }
}

/// Locate one launch record by run id, or a 404 / journal 500.
pub(crate) fn find_record(
    state: &BenchState,
    run_id: &str,
) -> Result<TunerLaunchRecord, BenchError> {
    tuner_launch::records(&state.bench_runs_dir)
        .map_err(journal_error)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' not found"),
        })
}

fn journal_error(error: std::io::Error) -> BenchError {
    BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("tuner launch journal error: {error}"),
    }
}

/// One frozen-objective JSON file the tuner launch form can offer, keyed by
/// its filename stem. The absolute path never leaves the server — a launch
/// request carries the `key` and the handler resolves it.
#[derive(Serialize)]
pub(crate) struct ObjectiveFileInfo {
    key: String,
    objective_id: Option<String>,
    game_kind: Option<String>,
    opponent_count: u32,
    updated_at: Option<String>,
    /// The same stem is also shipped in the read-only seed corpus.
    is_seed: bool,
}

/// The full objective JSON plus its metadata (`GET
/// /api/bench/tuner/objectives/{key}`).
#[derive(Serialize)]
pub(crate) struct ObjectiveFileDetail {
    key: String,
    content: serde_json::Value,
    updated_at: Option<String>,
    is_seed: bool,
}

/// `key` must be a bare filename stem — no path separators or traversal.
/// Shared by the objective and launch-profile routes.
pub(super) fn valid_file_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 128
        && !key.contains("..")
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

pub(super) fn seed_stems(seed_dir: &std::path::Path) -> std::collections::HashSet<String> {
    std::fs::read_dir(seed_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| entry.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect()
}

pub(super) fn file_updated_at(path: &std::path::Path) -> Option<String> {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .map(mcts_bench::launch::iso_timestamp_at)
}

fn read_objective_files(dir: &std::path::Path, seed_dir: &std::path::Path) -> Vec<ObjectiveFileInfo> {
    let seeds = seed_stems(seed_dir);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut files: Vec<ObjectiveFileInfo> = entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| {
            let path = entry.path();
            let key = path.file_stem()?.to_string_lossy().into_owned();
            let parsed: Option<serde_json::Value> =
                std::fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str(&text).ok());
            let field = |name: &str| {
                parsed
                    .as_ref()
                    .and_then(|value| value.get(name))
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
            };
            let opponent_count = parsed
                .as_ref()
                .and_then(|value| value.get("opponents"))
                .and_then(|value| value.as_array())
                .map_or(0, |list| list.len() as u32);
            Some(ObjectiveFileInfo {
                is_seed: seeds.contains(&key),
                key,
                objective_id: field("objective_id"),
                game_kind: field("game_kind"),
                opponent_count,
                updated_at: file_updated_at(&path),
            })
        })
        .collect();
    files.sort_by(|a, b| a.key.cmp(&b.key));
    files
}

/// Copy every seed `*.json` whose stem is not already a file in the writable
/// dir. Never overwrites — a user's edits outlive a repo update to the seed.
/// Shared by the objective and launch-profile corpora.
pub(super) fn seed_json_files(seed_dir: &std::path::Path, user_dir: &std::path::Path) {
    if std::fs::create_dir_all(user_dir).is_err() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(seed_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            if let Some(name) = path.file_name() {
                let target = user_dir.join(name);
                if !target.exists() {
                    let _ = std::fs::copy(&path, &target);
                }
            }
        }
    }
}

/// Seed the writable frozen-objective directory from the read-only corpus.
pub fn seed_tuner_objectives(seed_dir: &std::path::Path, user_dir: &std::path::Path) {
    seed_json_files(seed_dir, user_dir);
}

fn objective_path(state: &BenchState, key: &str) -> Result<std::path::PathBuf, BenchError> {
    if !valid_file_key(key) {
        return Err(bad_request(format!("invalid objective key '{key}'")));
    }
    Ok(state.tuner_objectives_dir.join(format!("{key}.json")))
}

pub(super) fn bad_request(message: String) -> BenchError {
    BenchError {
        status: StatusCode::BAD_REQUEST,
        message,
    }
}

pub(super) fn not_found(message: String) -> BenchError {
    BenchError {
        status: StatusCode::NOT_FOUND,
        message,
    }
}

/// `GET /api/bench/tuner/objectives`
///
/// The frozen-objective files a run can be launched against, from the
/// server's writable objectives directory.
pub(crate) async fn list_tuner_objectives(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Json<Vec<ObjectiveFileInfo>> {
    Json(read_objective_files(
        &state.tuner_objectives_dir,
        &state.tuner_seed_objectives_dir,
    ))
}

/// `GET /api/bench/tuner/objectives/{key}` — the objective's full JSON.
pub(crate) async fn get_tuner_objective(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(key): AxumPath<String>,
) -> Result<Json<ObjectiveFileDetail>, BenchError> {
    let path = objective_path(&state, &key)?;
    let text = std::fs::read_to_string(&path)
        .map_err(|_| not_found(format!("unknown objective '{key}'")))?;
    let content: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| bad_request(format!("objective '{key}' is not valid JSON: {error}")))?;
    Ok(Json(ObjectiveFileDetail {
        is_seed: seed_stems(&state.tuner_seed_objectives_dir).contains(&key),
        updated_at: file_updated_at(&path),
        key,
        content,
    }))
}

/// Cheap in-process pre-check before the (slower) validator: the body must be
/// a JSON object declaring `schema_version: 1` and a non-empty `game_kind`.
fn precheck_objective(body: &serde_json::Value) -> Result<String, BenchError> {
    let object = body
        .as_object()
        .ok_or_else(|| bad_request("objective body must be a JSON object".into()))?;
    if object.get("schema_version").and_then(serde_json::Value::as_u64) != Some(1) {
        return Err(bad_request("objective schema_version must be 1".into()));
    }
    if object.get("game_config").is_some_and(|value| !value.is_object()) {
        return Err(bad_request("objective game_config must be a JSON object".into()));
    }
    object
        .get("game_kind")
        .and_then(serde_json::Value::as_str)
        .filter(|kind| !kind.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| bad_request("objective game_kind is required".into()))
}

fn run_objective_validator(
    state: &BenchState,
    game_kind: &str,
    body: &serde_json::Value,
) -> Result<super::types::ObjectiveValidation, BenchError> {
    let scratch = std::env::temp_dir().join(format!(
        "tuner-objective-{}-{}.json",
        std::process::id(),
        SEED_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(
        &scratch,
        serde_json::to_vec(body).map_err(|error| bad_request(error.to_string()))?,
    )?;
    let result = (state.tuner_objective_validator)(game_kind, &scratch);
    let _ = std::fs::remove_file(&scratch);
    result.map_err(|error| BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("objective validator failed: {error}"),
    })
}

static SEED_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `POST /api/bench/tuner/objectives/{key}/validate` — dry-run validation.
pub(crate) async fn validate_tuner_objective(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(key): AxumPath<String>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<super::types::ObjectiveValidation>, BenchError> {
    objective_path(&state, &key)?;
    let game_kind = precheck_objective(&body)?;
    Ok(Json(run_objective_validator(&state, &game_kind, &body)?))
}

/// `PUT /api/bench/tuner/objectives/{key}` — create or replace, validated.
pub(crate) async fn put_tuner_objective(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(key): AxumPath<String>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<ObjectiveFileDetail>, BenchError> {
    let path = objective_path(&state, &key)?;
    let game_kind = precheck_objective(&body)?;
    let validation = run_objective_validator(&state, &game_kind, &body)?;
    if !validation.ok {
        let detail = if validation.errors.is_empty() {
            "objective rejected".to_owned()
        } else {
            validation.errors.join("; ")
        };
        return Err(bad_request(detail));
    }
    std::fs::create_dir_all(&state.tuner_objectives_dir)?;
    let pretty = serde_json::to_string_pretty(&body).map_err(|error| bad_request(error.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, pretty.as_bytes())?;
    std::fs::rename(&tmp, &path)?;
    Ok(Json(ObjectiveFileDetail {
        is_seed: seed_stems(&state.tuner_seed_objectives_dir).contains(&key),
        updated_at: file_updated_at(&path),
        key,
        content: body,
    }))
}

/// `DELETE /api/bench/tuner/objectives/{key}`.
pub(crate) async fn delete_tuner_objective(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(key): AxumPath<String>,
) -> Result<StatusCode, BenchError> {
    let path = objective_path(&state, &key)?;
    std::fs::remove_file(&path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => not_found(format!("unknown objective '{key}'")),
        _ => BenchError::from(error),
    })?;
    Ok(StatusCode::NO_CONTENT)
}

/// Production [`BenchState::tuner_objective_validator`]: resolves the built-in
/// game binary and shells out to `python -m tuner_cli validate-objective` from
/// the repo root. A game kind with no built-in binary is `ok: false`.
pub fn shell_validate_objective(
    game_kind: &str,
    objective_file: &std::path::Path,
) -> std::io::Result<super::types::ObjectiveValidation> {
    let Some(game_binary) = mcts_bench::games::find_game_binary(game_kind) else {
        return Ok(super::types::ObjectiveValidation {
            ok: false,
            errors: vec![format!("no built-in game binary for kind '{game_kind}'")],
            objective_id: None,
            panel_fingerprint: None,
        });
    };
    let output = std::process::Command::new("uv")
        .args([
            "run",
            "--project",
            "tuner",
            "python",
            "-m",
            "tuner_cli",
            "validate-objective",
            "--game-binary",
        ])
        .arg(game_binary)
        .arg("--objective-file")
        .arg(objective_file)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "validate-objective exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        std::io::Error::other(format!("validate-objective produced invalid JSON: {error}"))
    })
}

/// Fill `runs_root` and resolve the caller-friendly `game_kind` /
/// `objective_key` to absolute paths, so no filesystem path is ever part of
/// the API contract. Shared by launch and preflight.
pub(super) fn resolve_launch_request(
    state: &BenchState,
    request: &mut TunerLaunchRequest,
) -> Result<(), BenchError> {
    request.runs_root = state.bench_runs_dir.clone();
    if request.game_binary.as_os_str().is_empty() {
        let kind = request
            .game_kind
            .as_deref()
            .ok_or_else(|| bad_request("game_kind or game_binary is required".into()))?;
        request.game_binary = mcts_bench::games::find_game_binary(kind).ok_or_else(|| {
            bad_request(format!("no built-in game binary found for kind '{kind}'"))
        })?;
    }
    if request.objective_file.as_os_str().is_empty() {
        let key = request
            .objective_key
            .as_deref()
            .ok_or_else(|| bad_request("objective_key or objective_file is required".into()))?;
        if !valid_file_key(key) {
            return Err(bad_request(format!("invalid objective_key '{key}'")));
        }
        let candidate = state.tuner_objectives_dir.join(format!("{key}.json"));
        if !candidate.is_file() {
            return Err(bad_request(format!("unknown objective_key '{key}'")));
        }
        request.objective_file = candidate;
    }
    Ok(())
}

/// `POST /api/bench/tuner/runs/preflight` — dry-run a launch request through
/// every check `tuner_cli` applies before it creates the run dir or plays a
/// game. Returns `{ ok, errors }`; the launch form blocks on `ok: false`.
pub(crate) async fn preflight_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    Json(mut request): Json<TunerLaunchRequest>,
) -> Result<Json<super::types::LaunchPreflight>, BenchError> {
    resolve_launch_request(&state, &mut request)?;
    let result = (state.tuner_launch_preflight)(&request).map_err(|error| BenchError {
        status: StatusCode::BAD_GATEWAY,
        message: format!("could not preflight the launch: {error}"),
    })?;
    Ok(Json(result))
}

/// Production [`BenchState::tuner_launch_preflight`]: shells
/// `python -m tuner_cli preflight` with the request's resolved argv.
pub fn shell_preflight_launch(
    request: &TunerLaunchRequest,
) -> std::io::Result<super::types::LaunchPreflight> {
    let argv = request.preflight_argv();
    let output = std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "preflight exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        std::io::Error::other(format!("preflight produced invalid JSON: {error}"))
    })
}

/// `POST /api/bench/tuner/runs/plan` — resolve a launch request to its
/// concrete shape (opponent panel with expanded configs, tuning space after
/// exclusions/overrides, phase efforts, pair budgets, `game_config`, epoch)
/// plus the embedded preflight `ok`/`errors`. Creates no run and plays no
/// game; the launch form renders this as its read-only "Run plan" panel.
pub(crate) async fn plan_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    Json(mut request): Json<TunerLaunchRequest>,
) -> Result<Json<super::types::RunPlan>, BenchError> {
    resolve_launch_request(&state, &mut request)?;
    let result = (state.tuner_launch_plan)(&request).map_err(|error| BenchError {
        status: StatusCode::BAD_GATEWAY,
        message: format!("could not resolve the launch plan: {error}"),
    })?;
    Ok(Json(result))
}

/// Production [`BenchState::tuner_launch_plan`]: shells
/// `python -m tuner_cli plan` with the request's resolved argv.
pub fn shell_plan_launch(request: &TunerLaunchRequest) -> std::io::Result<super::types::RunPlan> {
    let argv = request.plan_argv();
    let output = std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "plan exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| std::io::Error::other(format!("plan produced invalid JSON: {error}")))
}

pub(crate) async fn launch_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    Json(mut request): Json<TunerLaunchRequest>,
) -> Result<(StatusCode, Json<TunerRunView>), BenchError> {
    resolve_launch_request(&state, &mut request)?;
    // Everything a launch could fail on before it starts a run is the
    // preflight's job -- run it here too so a launch is never accepted for a
    // reason the form could have shown. (The form calls the same check; this
    // is the backstop for a direct API caller.)
    let preflight = (state.tuner_launch_preflight)(&request).map_err(|error| BenchError {
        status: StatusCode::BAD_GATEWAY,
        message: format!("could not preflight the launch: {error}"),
    })?;
    if !preflight.ok {
        return Err(bad_request(if preflight.errors.is_empty() {
            "launch rejected by preflight".to_owned()
        } else {
            preflight.errors.join("; ")
        }));
    }
    let record = tuner_launch::launch(&request).map_err(|error| BenchError {
        status: match error.kind() {
            std::io::ErrorKind::InvalidInput => StatusCode::BAD_REQUEST,
            std::io::ErrorKind::AlreadyExists => StatusCode::CONFLICT,
            // The child spawned but died inside the startup grace window --
            // its own diagnostics are already in the message.
            std::io::ErrorKind::Other => StatusCode::BAD_GATEWAY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        },
        message: format!("failed to launch tuner run: {error}"),
    })?;
    // Kick the headless follower so this run's projection starts advancing
    // now, not on the supervisor's next timer tick.
    if let Some(follower) = &state.projection_follower {
        follower.tick();
    }
    Ok((StatusCode::ACCEPTED, Json(view(record))))
}

pub(crate) async fn list_tuner_runs(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Result<Json<Vec<TunerRunView>>, BenchError> {
    Ok(Json(
        tuner_launch::records(&state.bench_runs_dir)
            .map_err(journal_error)?
            .into_iter()
            .map(view)
            .collect(),
    ))
}

pub(crate) async fn get_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<TunerRunView>, BenchError> {
    let record = tuner_launch::records(&state.bench_runs_dir)
        .map_err(journal_error)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' not found"),
        })?;
    Ok(Json(view(record)))
}

/// `POST /api/bench/tuner/runs/{run_id}/extend`
///
/// Raise one or more of a frozen run's pair budgets and resume it. The tuner
/// records the increase as one append-only `budget_extended` evidence event
/// (`manifest.compute_budget` is never edited) and continues the run; a run
/// that had already completed re-opens at its last cohort boundary.
pub(crate) async fn extend_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Json(extension): Json<BudgetExtension>,
) -> Result<(StatusCode, Json<TunerRunView>), BenchError> {
    let record =
        tuner_launch::extend(&state.bench_runs_dir, &run_id, &extension).map_err(|error| {
            BenchError {
                status: match error.kind() {
                    std::io::ErrorKind::InvalidInput => StatusCode::BAD_REQUEST,
                    std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                },
                message: format!("failed to extend tuner run: {error}"),
            }
        })?;
    Ok((StatusCode::ACCEPTED, Json(view(record))))
}

pub(crate) async fn stop_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<TunerRunView>, BenchError> {
    let record = tuner_launch::records(&state.bench_runs_dir)
        .map_err(journal_error)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' not found"),
        })?;
    if record.terminal_outcome.is_none() {
        if let Some(pid) = record.pid {
            let signalled = match tuner_launch::interrupt(pid) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => return Err(journal_error(error)),
            };
            // Give a fast SIGINT-to-exit a brief window to land in the journal
            // so the response can already show the run terminal. A slower exit
            // still returns `live` with no error; the tuner's reaper writes the
            // terminal record whenever it happens.
            if signalled {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while std::time::Instant::now() < deadline {
                    let terminal = tuner_launch::records(&state.bench_runs_dir)
                        .map_err(journal_error)?
                        .into_iter()
                        .any(|r| r.run_id == run_id && r.terminal_outcome.is_some());
                    if terminal {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
        }
    }
    // A foreground tuner translates SIGINT to exit 130; its reaper writes the
    // terminal record. Until then this response deliberately remains `live`.
    get_tuner_run(AxumState(state), AxumPath(run_id)).await
}

/// `DELETE /api/bench/tuner/runs/{run_id}`
///
/// Permanently remove a **terminal** tuner run: [`tuner_launch::delete`]
/// appends a `run_deleted` tombstone to the journal and removes the run
/// directory, then the SQLite projection is refreshed so its rows for the
/// now-vanished run are pruned. A `live` run is rejected with `409` — stop it
/// first. Returns `204`.
pub(crate) async fn delete_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<StatusCode, BenchError> {
    tuner_launch::delete(&state.bench_runs_dir, &run_id).map_err(|error| BenchError {
        status: match error.kind() {
            std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
            std::io::ErrorKind::InvalidInput => StatusCode::CONFLICT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        },
        message: format!("failed to delete tuner run '{run_id}': {error}"),
    })?;
    // Drop the run's projection rows. The run directory is already gone, so a
    // projection pass prunes it; a failure here is logged, not surfaced -- the
    // authoritative delete (journal tombstone + directory) has already
    // happened, and the follower's next pass prunes the run regardless.
    if let Err(error) =
        (state.tuner_projection_refresh)(&state.bench_runs_dir, &state.tuner_projection_db)
    {
        eprintln!("tuner run delete: projection refresh failed for '{run_id}': {error}");
    }
    Ok(StatusCode::NO_CONTENT)
}

#[derive(serde::Deserialize)]
pub(crate) struct TunerLogParams {
    #[serde(default)]
    since: u64,
}

#[derive(Serialize)]
pub(crate) struct TunerLogResponse {
    /// `launch.out` lines appended since the `since` byte offset.
    lines: Vec<String>,
    /// Byte offset to pass as `since` on the next poll.
    next_offset: u64,
    /// Full contents of `launch.err`, re-sent each poll (it is normally tiny:
    /// a panic backtrace or nothing).
    err_lines: Vec<String>,
}

/// `GET /api/bench/tuner/runs/{run_id}/log?since=<offset>`
///
/// Tail the detached tuner's `launch.out` from a byte offset, plus the full
/// `launch.err`. This reads operational files straight from the run-dir; it is
/// not scientific authority (that is the projection API) — it exists so the UI
/// can show what a run is doing in the seconds before the projection catches
/// up.
pub(crate) async fn get_tuner_run_log(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    AxumQuery(params): AxumQuery<TunerLogParams>,
) -> Result<Json<TunerLogResponse>, BenchError> {
    let record = tuner_launch::records(&state.bench_runs_dir)
        .map_err(journal_error)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' not found"),
        })?;

    let out_path = record.run_dir.join("launch.out");
    let (lines, next_offset) = tail_from(&out_path, params.since)?;
    let (err_lines, _) = tail_from(&record.run_dir.join("launch.err"), 0)?;
    Ok(Json(TunerLogResponse {
        lines,
        next_offset,
        err_lines,
    }))
}

fn tail_from(path: &std::path::Path, since: u64) -> Result<(Vec<String>, u64), BenchError> {
    let io_error = |error: std::io::Error| BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("failed to read {}: {error}", path.display()),
    };
    if !path.exists() {
        return Ok((vec![], 0));
    }
    let file_len = std::fs::metadata(path).map_err(io_error)?.len();
    if file_len <= since {
        return Ok((vec![], since));
    }
    let mut file = std::fs::File::open(path).map_err(io_error)?;
    file.seek(SeekFrom::Start(since)).map_err(io_error)?;
    let mut lines = Vec::new();
    for line in BufReader::new(file).lines() {
        lines.push(line.map_err(io_error)?);
    }
    Ok((lines, file_len))
}
