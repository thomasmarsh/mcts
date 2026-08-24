#![allow(unused_imports)]
use std::collections::HashMap;
use std::convert::Infallible;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path as AxumPath, Query, State as AxumState},
    http::{HeaderValue, Method, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer};

use game_host::TunerInfo;
use mcts_bench::experiment::ExperimentSpecV1;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::log::RegistryEvent;
use mcts_bench::projects_attempt::{CellRequest, ProjectsError, StartRequest};
use mcts_bench::run_command_repository::{RecordRunLaunch, TunerLaunchReservation};
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::tuning_command_repository::{TuningCommandRepository, TuningLaunchOutcome};
use mcts_bench::StrategyInfo;

use super::lifecycle;
use super::types::*;
pub(crate) async fn launch_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    Json(body): Json<LaunchBody>,
) -> Result<Json<LaunchResponse>, BenchError> {
    let label = body
        .config
        .as_ref()
        .and_then(|c| c.get("label").and_then(|v| v.as_str()))
        .map(str::to_owned);
    let resp = launch_and_record(
        &state,
        &body.kind,
        &body.game,
        body.config,
        label.as_deref(),
        None,
    )
    .await?;
    Ok(Json(resp))
}

/// Builds the command, pins a
/// fresh physical `run_id` (baked into a tuner launch's own explicit physical
/// id arguments, not just outer bench-runs bookkeeping),
/// spawns it, and inserts the `runs` row so it appears immediately in the
/// runs list without waiting on the ingest loop.
pub(crate) async fn launch_and_record(
    state: &Arc<BenchState>,
    kind: &str,
    game: &str,
    config: Option<Value>,
    label: Option<&str>,
    resume_from: Option<&str>,
) -> Result<LaunchResponse, BenchError> {
    let run_id = launch::generate_run_id(kind, game, crate::BUILD_INFO);
    let config = if kind == "tuner" {
        prepare_tuner_config(config, &run_id)
    } else {
        config
    };
    let parent_identity = if let Some(parent_id) = resume_from {
        Some(
            state
                .run_command_repository
                .prepare_continuation(parent_id)
                .map_err(run_command_bench_error)?,
        )
    } else {
        None
    };
    let (cmd, config) = if kind == "tuner" {
        let attempt = TunerAttemptLaunch::from_config(
            game,
            config.clone(),
            &run_id,
            tuner_artifact_root(&state.bench_runs_dir, &run_id),
        );
        let built = build_tuner_attempt(&attempt)?;
        (built.command, built.config)
    } else {
        (build_command(kind, game, &config, &run_id)?, config)
    };

    // Preserve the existing launch ordering: the process is spawned before
    // the immediate bookkeeping insert. Parent identity was already
    // validated/anchored above, so a contradictory continuation cannot
    // create an orphan process.
    let LaunchedRun {
        run_id,
        pid,
        log_path,
        log_dir,
    } = match (state.run_launcher)(
        run_id.clone(),
        cmd,
        kind.to_owned(),
        game.to_owned(),
        label.map(str::to_owned),
    ) {
        Ok(launched) => launched,
        Err(error) => {
            return Err(BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("failed to launch run: {error}"),
            });
        }
    };

    let started_at = iso_timestamp_now();
    let config_str = config
        .as_ref()
        .map(|c| serde_json::to_string(c).unwrap_or_default());

    let launched_log_path = log_path.to_string_lossy().into_owned();
    let tuner_lifecycle_source = (kind == "tuner").then(|| {
        let optimizer_id = tuner_optimizer_id(config.as_ref(), &run_id);
        tuner_lifecycle_path_from_config(config.as_ref(), &optimizer_id)
    });
    state
        .run_command_repository
        .record_launch(RecordRunLaunch {
            run_id: run_id.clone(),
            kind: kind.to_owned(),
            game: game.to_owned(),
            label: label.map(str::to_owned),
            config_json: config_str,
            git_sha: crate::BUILD_INFO.git_sha.into(),
            git_dirty: crate::BUILD_INFO.git_dirty,
            host: hostname(),
            pid: pid as i64,
            started_at: started_at.clone(),
            log_path: launched_log_path,
            continuation_parent: parent_identity,
            tuner_lifecycle_source,
        })
        .map_err(run_command_bench_error)?;

    // Store config in the runs table so it survives server restarts.
    // (Separate UPDATE for the rare case the row was created by the
    // ingest loop between the INSERT above and here.)
    if let Some(ref config) = config {
        let config_str = serde_json::to_string(config)?;
        let _ = state
            .run_command_repository
            .backfill_config(&run_id, &config_str);
    }

    // Post-spawn check: give the child 500ms to start and possibly fail
    // (e.g. bad arguments to the bench CLI).  If it's already dead, read
    // stdout.log for the error and return it to the caller.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let launch_error: Option<String> = if !launch::is_alive(pid) {
        let stdout_path = log_dir.join("stdout.log");
        let error_content = std::fs::read_to_string(&stdout_path).unwrap_or_default();
        let trimmed = error_content.trim().to_string();

        // Mark the run as crashed in the database.
        let now = iso_timestamp_now();
        let _ = state.run_command_repository.mark_crashed(&run_id, &now);

        // Append a stop event to the registry log so the ingest loop
        // sees it on its next pass (even though we already updated the
        // DB, the ingest loop's reconciliation pass would eventually catch
        // this too — writing the event keeps registry.log authoritative).
        let event = RegistryEvent::Stop {
            run_id: run_id.clone(),
            exit_code: None,
            ended_at: now,
        };
        let registry_path = state.bench_runs_dir.join("registry.log");
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&registry_path)
        {
            use std::io::Write;
            let mut line = event.to_json_line();
            line.push('\n');
            let _ = file.write_all(line.as_bytes());
        }

        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    } else {
        None
    };

    Ok(LaunchResponse {
        run_id,
        pid,
        log_path: log_path.to_string_lossy().to_string(),
        launch_error,
    })
}

/// Launch a continuation after the command store has reserved its stable
/// attempt and physical-run ids. This is deliberately not an HTTP handler:
/// command routes own idempotency and decide whether a stored replay should
/// call it at all.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn launch_reserved_tuner_attempt(
    state: &Arc<BenchState>,
    command_id: &str,
    launch: TunerAttemptLaunch,
    label: Option<&str>,
) -> Result<LaunchResponse, BenchError> {
    state
        .run_command_repository
        .verify_tuner_launch_reservation(&TunerLaunchReservation {
            session_id: launch.session_id.clone(),
            command_id: command_id.into(),
            attempt_id: launch.attempt_id.clone(),
            physical_run_id: launch.physical_run_id.clone(),
            target_trial_count: launch.target_trial_count,
        })
        .map_err(run_command_bench_error)?;
    if let Some(previous) = state
        .run_command_repository
        .recorded_tuner_launch(&launch.physical_run_id)
        .map_err(run_command_bench_error)?
    {
        record_tuner_launch_outcome(
            state,
            &launch.session_id,
            command_id,
            TuningLaunchOutcome::Spawned,
        )?;
        return Ok(LaunchResponse {
            run_id: previous.run_id,
            pid: previous.pid,
            log_path: previous.log_path,
            launch_error: None,
        });
    }

    let parent_identity = state
        .run_command_repository
        .prepare_latest_tuner_continuation(&launch.session_id)
        .map_err(run_command_bench_error)?;
    let built = build_tuner_attempt(&launch)?;
    let launched = match (state.run_launcher)(
        launch.physical_run_id.clone(),
        built.command,
        "tuner".into(),
        launch.game.clone(),
        label.map(str::to_owned),
    ) {
        Ok(launched) => launched,
        Err(error) => {
            record_tuner_launch_outcome(
                state,
                &launch.session_id,
                command_id,
                TuningLaunchOutcome::SpawnFailed,
            )?;
            return Err(BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("failed to launch tuner attempt: {error}"),
            });
        }
    };
    let started_at = iso_timestamp_now();
    state
        .run_command_repository
        .record_tuner_attempt_launch(RecordRunLaunch {
            run_id: launch.physical_run_id.clone(),
            kind: "tuner".into(),
            game: launch.game.clone(),
            label: label.map(str::to_owned),
            config_json: Some(serde_json::to_string(&built.config)?),
            git_sha: crate::BUILD_INFO.git_sha.into(),
            git_dirty: crate::BUILD_INFO.git_dirty,
            host: hostname(),
            pid: i64::from(launched.pid),
            started_at,
            log_path: launched.log_path.to_string_lossy().into_owned(),
            continuation_parent: Some(parent_identity),
            tuner_lifecycle_source: Some(launch.lifecycle_path.clone()),
        })
        .map_err(run_command_bench_error)?;
    record_tuner_launch_outcome(
        state,
        &launch.session_id,
        command_id,
        TuningLaunchOutcome::Spawned,
    )?;
    Ok(LaunchResponse {
        run_id: launched.run_id,
        pid: launched.pid,
        log_path: launched.log_path.to_string_lossy().into_owned(),
        launch_error: None,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn record_tuner_launch_outcome(
    state: &Arc<BenchState>,
    session_id: &str,
    command_id: &str,
    outcome: TuningLaunchOutcome,
) -> Result<(), BenchError> {
    state
        .tuning_command_repository
        .record_launch_outcome(session_id, command_id, outcome)
        .map(|_| ())
        .map_err(|error| BenchError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("failed to record tuner launch outcome: {error}"),
        })
}

// ---------------------------------------------------------------------------
// Automated ladder driver
// ---------------------------------------------------------------------------

/// Snapshot of one `tuner` run's bookkeeping, as read from `runs`.
pub(crate) async fn stop_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Value>, BenchError> {
    let outcome = lifecycle::stop_run_impl(&state, &run_id, &lifecycle::SystemClock).await?;

    if outcome.prior_status != "running" {
        return Ok(Json(json!({
            "run_id": run_id,
            "status": outcome.prior_status,
            "message": "run is not currently running, no signal sent",
        })));
    }

    if outcome.signal_sent {
        Ok(Json(json!({
            "run_id": run_id,
            "pid": outcome.pid,
            "signal": "SIGTERM",
            "message": "stop signal sent and run marked as stopped",
        })))
    } else {
        Ok(Json(json!({
            "run_id": run_id,
            "pid": outcome.pid,
            "signal": null,
            "message": "run marked as stopped (PID was no longer alive or had no PID)",
        })))
    }
}

// ---------------------------------------------------------------------------
// Command construction
// ---------------------------------------------------------------------------

const DEFAULT_TUNER_TARGET_TRIAL_COUNT: u64 = 1_000;

/// Everything that identifies one physical process while preserving the
/// artifacts owned by its logical tuning session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TunerAttemptLaunch {
    pub(crate) game: String,
    pub(crate) config: Option<Value>,
    pub(crate) session_id: String,
    pub(crate) optimizer_id: String,
    pub(crate) lifecycle_path: String,
    pub(crate) attempt_id: String,
    pub(crate) physical_run_id: String,
    pub(crate) artifact_root: PathBuf,
    pub(crate) target_trial_count: u64,
    pub(crate) workers: Option<u64>,
}

pub(crate) struct BuiltTunerAttempt {
    pub(crate) command: Vec<String>,
    pub(crate) config: Option<Value>,
}

impl TunerAttemptLaunch {
    fn from_config(
        game: &str,
        config: Option<Value>,
        physical_run_id: &str,
        artifact_root: PathBuf,
    ) -> Self {
        let optimizer_id = tuner_optimizer_id(config.as_ref(), physical_run_id);
        let session_id = tuner_session_id(config.as_ref(), &optimizer_id);
        let lifecycle_path = tuner_lifecycle_path_from_config(config.as_ref(), &optimizer_id);
        let target_trial_count = tuner_target_trial_count(config.as_ref());
        Self {
            game: game.to_owned(),
            config,
            session_id,
            optimizer_id,
            lifecycle_path,
            attempt_id: format!("tuning-attempt-{physical_run_id}"),
            physical_run_id: physical_run_id.to_owned(),
            artifact_root,
            target_trial_count,
            workers: None,
        }
    }
}

pub(crate) fn canonical_tuner_artifact_root(physical_run_id: &str) -> PathBuf {
    tuner_artifact_root(&PathBuf::from(launch::BENCH_RUNS_DIR), physical_run_id)
}

pub(crate) fn tuner_artifact_root(bench_runs_dir: &Path, physical_run_id: &str) -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|error| panic!("cannot determine current directory: {error}"))
        .join(bench_runs_dir)
        .join(physical_run_id)
        .join("tuning-artifacts")
}

/// Build a tuner argv from stable session artifacts and one physical attempt.
///
/// The Python tuner remains the authority for resolving semantic configuration
/// and verifying its manifest. Rust only replaces operational controls that a
/// reserved command has already decided: an absolute target and optional
/// worker count.
pub(crate) fn build_tuner_attempt(
    launch: &TunerAttemptLaunch,
) -> Result<BuiltTunerAttempt, BenchError> {
    if launch.game.is_empty()
        || launch.session_id.is_empty()
        || launch.optimizer_id.is_empty()
        || launch.lifecycle_path.is_empty()
        || launch.attempt_id.is_empty()
        || launch.physical_run_id.is_empty()
        || !launch.artifact_root.is_absolute()
        || launch
            .artifact_root
            .file_name()
            .and_then(|name| name.to_str())
            != Some("tuning-artifacts")
        || launch
            .artifact_root
            .parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            != Some(launch.physical_run_id.as_str())
        || launch.target_trial_count == 0
    {
        return Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: "tuner attempt launch requires non-empty ids and a positive target".into(),
        });
    }

    let config = tuner_operational_config(
        launch.config.clone(),
        launch.target_trial_count,
        launch.workers,
    );
    let mut command = vec![
        find_bench_binary().to_string_lossy().into_owned(),
        "tuner".into(),
        "--game".into(),
        launch.game.clone(),
    ];
    append_tuner_config_arguments(&mut command, &config);
    command.push("--artifact-root".into());
    command.push(launch.artifact_root.to_string_lossy().into_owned());
    command.extend([
        "--optimizer-id".into(),
        launch.optimizer_id.clone(),
        "--bench-run-id".into(),
        launch.physical_run_id.clone(),
        "--session-id".into(),
        launch.session_id.clone(),
        "--attempt-id".into(),
        launch.attempt_id.clone(),
        "--lifecycle-path".into(),
        launch.lifecycle_path.clone(),
        "--game-kind".into(),
        launch.game.clone(),
    ]);
    Ok(BuiltTunerAttempt { command, config })
}

fn tuner_operational_config(
    config: Option<Value>,
    target_trial_count: u64,
    workers: Option<u64>,
) -> Option<Value> {
    let mut config = match config {
        Some(Value::Object(values)) => Value::Object(values),
        _ => json!({}),
    };
    let mut overrides = config
        .get("overrides")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    replace_override(
        &mut overrides,
        "optimizer.n_trials",
        target_trial_count.to_string(),
    );
    if let Some(workers) = workers {
        replace_override(&mut overrides, "optimizer.n_workers", workers.to_string());
    }
    config["overrides"] = Value::Array(overrides);
    Some(config)
}

fn replace_override(overrides: &mut Vec<Value>, key: &str, value: String) {
    let replacement = Value::String(format!("{key}={value}"));
    let mut found = false;
    overrides.retain_mut(|override_value| {
        let is_target = override_value
            .as_str()
            .and_then(|raw| raw.split_once('='))
            .is_some_and(|(existing, _)| existing == key);
        if !is_target {
            return true;
        }
        if found {
            return false;
        }
        *override_value = replacement.clone();
        found = true;
        true
    });
    if !found {
        overrides.push(replacement);
    }
}

fn tuner_target_trial_count(config: Option<&Value>) -> u64 {
    config
        .and_then(|value| value.get("overrides"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|override_value| override_value.split_once('='))
        .filter(|(key, _)| *key == "optimizer.n_trials")
        .filter_map(|(_, value)| value.parse::<u64>().ok())
        .rfind(|value| *value > 0)
        .unwrap_or(DEFAULT_TUNER_TARGET_TRIAL_COUNT)
}

pub(crate) fn project_legacy_stop(
    state: &Arc<BenchState>,
    run_id: &str,
    kind: &str,
) -> Result<String, BenchError> {
    let ended_at = iso_timestamp_now();
    state
        .run_command_repository
        .project_legacy_stop(run_id, kind, &ended_at)
        .map_err(run_command_bench_error)?;
    Ok(ended_at)
}

/// Build the command vector from the launch request's kind/game/config.
///
/// Supported kinds:
/// - `"round_robin"` — runs `bench round-robin --game ... --strategies ... --rounds ...`
/// - `"tuner"` — runs `bench tuner --game ... [--config ...] [--override k=v ...]`
///   in the foreground; the server's own `launch::launch` (not `bench tuner`'s
///   own `--background` flag) is what detaches and captures its JSONL output,
///   same as every other launch kind.
///
/// Unknown kinds produce an error.
pub(crate) fn build_command(
    kind: &str,
    game: &str,
    config: &Option<Value>,
    run_id: &str,
) -> Result<Vec<String>, BenchError> {
    let bench_binary = find_bench_binary();

    match kind {
        "tuner" => Ok(build_tuner_attempt(&TunerAttemptLaunch::from_config(
            game,
            config.clone(),
            run_id,
            canonical_tuner_artifact_root(run_id),
        ))?
        .command),
        "round_robin" => {
            let mut cmd = vec![
                bench_binary.to_string_lossy().to_string(),
                "round-robin".into(),
                "--game".into(),
                game.to_owned(),
            ];

            if let Some(ref config) = config {
                if let Some(strategies) = config.get("strategies").and_then(|v| v.as_array()) {
                    for s in strategies {
                        if let Some(name) = s.as_str() {
                            cmd.push("--strategies".into());
                            cmd.push(name.to_owned());
                        }
                    }
                }

                if let Some(rounds) = config.get("rounds").and_then(|v| v.as_u64()) {
                    cmd.push("--rounds".into());
                    cmd.push(rounds.to_string());
                }
            }

            // Always include --verbose so progress bars appear on stderr
            // (the launcher redirects stderr to stdout.log).
            cmd.push("--verbose".into());

            // Move-trace lines go to a dedicated `moves.jsonl` in the run's
            // own directory, not `log.jsonl` -- see `LogRecord::Move`'s doc
            // comment for why a full move trace is kept out of the main
            // log. The path is derivable from `run_id` alone (matches
            // `launch::launch_with_run_id`'s own `bench-runs/<run_id>/`
            // layout), so no round-trip through the launcher is needed.
            cmd.push("--trace-path".into());
            cmd.push(
                std::path::Path::new(launch::BENCH_RUNS_DIR)
                    .join(run_id)
                    .join("moves.jsonl")
                    .to_string_lossy()
                    .to_string(),
            );

            Ok(cmd)
        }
        unknown => Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: format!("unknown run kind '{unknown}'; expected one of: round_robin, tuner"),
        }),
    }
}

pub(crate) fn build_experiment_command(
    spec: &ExperimentSpecV1,
    run_id: &str,
) -> Result<Vec<String>, String> {
    spec.expand().map_err(|error| error.to_string())?;
    let spec_json = serde_json::to_string(spec).map_err(|error| error.to_string())?;
    Ok(vec![
        find_bench_binary().to_string_lossy().into_owned(),
        "experiment".into(),
        "--spec-json".into(),
        spec_json,
        "--trace-path".into(),
        Path::new(launch::BENCH_RUNS_DIR)
            .join(run_id)
            .join("moves.jsonl")
            .to_string_lossy()
            .into_owned(),
    ])
}

fn prepare_tuner_config(config: Option<Value>, run_id: &str) -> Option<Value> {
    let mut config = match config {
        Some(Value::Object(values)) => Value::Object(values),
        _ => json!({}),
    };
    if config.get("optimizer_id").and_then(Value::as_str).is_none() {
        config["optimizer_id"] = Value::String(format!("tuning-session-{run_id}"));
    }
    let optimizer_id = tuner_optimizer_id(Some(&config), run_id);
    if config.get("session_id").and_then(Value::as_str).is_none() {
        config["session_id"] = Value::String(optimizer_id.clone());
    }
    if config
        .get("lifecycle_path")
        .and_then(Value::as_str)
        .is_none()
    {
        config["lifecycle_path"] = Value::String(canonical_tuner_lifecycle_path(optimizer_id));
    }
    Some(config)
}

fn tuner_optimizer_id(config: Option<&Value>, run_id: &str) -> String {
    config
        .and_then(|value| value.get("optimizer_id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("tuning-session-{run_id}"))
}

fn tuner_session_id(config: Option<&Value>, optimizer_id: &str) -> String {
    config
        .and_then(|value| value.get("session_id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| optimizer_id.to_owned())
}

fn tuner_lifecycle_path(optimizer_id: String) -> PathBuf {
    Path::new("optuna_output")
        .join(optimizer_id)
        .join("lifecycle.jsonl")
}

pub(crate) fn canonical_tuner_lifecycle_path(optimizer_id: String) -> String {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(tuner_lifecycle_path(optimizer_id))
        .to_string_lossy()
        .into_owned()
}

fn tuner_lifecycle_path_from_config(config: Option<&Value>, optimizer_id: &str) -> String {
    config
        .and_then(|value| value.get("lifecycle_path"))
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(path)
            }
        })
        .unwrap_or_else(|| PathBuf::from(canonical_tuner_lifecycle_path(optimizer_id.into())))
        .to_string_lossy()
        .into_owned()
}

fn append_tuner_config_arguments(cmd: &mut Vec<String>, config: &Option<Value>) {
    let Some(config) = config else {
        return;
    };
    if let Some(config_path) = config.get("config").and_then(Value::as_str) {
        cmd.extend(["--config".into(), config_path.to_owned()]);
    }
    if let Some(overrides) = config.get("overrides").and_then(Value::as_array) {
        for override_value in overrides.iter().filter_map(Value::as_str) {
            cmd.extend(["--override".into(), override_value.to_owned()]);
        }
    }
    if let Some(baseline_configs) = config.get("baseline_configs").and_then(Value::as_object) {
        for (id, raw_config) in baseline_configs {
            cmd.extend(["--baseline-config".into(), format!("{id}={raw_config}")]);
        }
    }
    if let Some(game_config) = config.get("game_config").filter(|value| !value.is_null()) {
        cmd.extend(["--game-config".into(), game_config.to_string()]);
    }
}

/// Find the `bench` binary, preferring a sibling of the current executable
/// (standard Cargo convention for sibling bins), falling back to a bare
/// `"bench"` on PATH.
pub(crate) fn find_bench_binary() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = if cfg!(target_os = "windows") {
                dir.join("bench.exe")
            } else {
                dir.join("bench")
            };
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from("bench")
}

/// Resolve the hostname of the current machine.  Uses `HOSTNAME` env var
/// first (portable across Unix/Windows), falls back to the `hostname`
/// command, then `"unknown"`.
pub(crate) fn hostname() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| {
        std::process::Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    })
}

// ---------------------------------------------------------------------------
// Timestamp helper (same algorithm as src/bench/launch.rs'
// iso_timestamp, but stands alone to keep the module self-contained)
// ---------------------------------------------------------------------------

pub(crate) fn iso_timestamp_now() -> String {
    let total_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock set before Unix epoch")
        .as_secs();
    let days = total_secs / 86400;
    let time_secs = total_secs % 86400;
    let hh = time_secs / 3600;
    let mm = (time_secs % 3600) / 60;
    let ss = time_secs % 60;

    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

pub(crate) fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ---------------------------------------------------------------------------
