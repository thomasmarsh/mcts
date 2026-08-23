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
use mcts_bench::identity;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::log::RegistryEvent;
use mcts_bench::projects_attempt::{CellRequest, ProjectsError, StartRequest};
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::StrategyInfo;

use super::lifecycle;
use super::{ladder::*, types::*};
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

/// `POST /api/bench/runs/{run_id}/resume` — `{n_trials, n_workers?}`
///
/// Relaunches a finished/stopped tuner run with a bigger trial budget,
/// picking up where it left off rather than starting over: the new process
/// is launched with `--resume <old run_id>` (see `tuner_cli/resume.py`),
/// which seeds its runhistory from the old run's saved state before
/// optimizing, so already-evaluated configs aren't re-evaluated. This is
/// also the only way to change worker count "mid-run" -- tuner has no live
/// API for either, only stop-and-relaunch.
///
/// The old run's stored `config` (its `--config` path and any `--override`
/// list) is carried forward, with `optimizer.n_trials`/`optimizer.n_workers`
/// overrides appended (and so taking precedence -- the Python side's
/// `_apply_overrides` keeps the last value for a repeated key).
pub(crate) async fn resume_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Json(body): Json<ResumeBody>,
) -> Result<Json<LaunchResponse>, BenchError> {
    let (kind, game, config_str): (String, String, Option<String>) = {
        let db = state.db.lock().unwrap();
        match db.query_row(
            "SELECT kind, game, CAST(config AS TEXT) FROM runs WHERE run_id = ?1",
            duckdb::params![&run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ) {
            Ok(row) => row,
            Err(duckdb::Error::QueryReturnedNoRows) => {
                return Err(BenchError {
                    status: StatusCode::NOT_FOUND,
                    message: format!("run '{run_id}' not found"),
                });
            }
            Err(e) => return Err(BenchError::from(e)),
        }
    };

    if kind != "tuner" {
        return Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: format!(
                "run '{run_id}' is a '{kind}' run, not 'tuner' -- only tuner runs support resume"
            ),
        });
    }

    let old_config: Option<Value> = config_str.and_then(|s| serde_json::from_str(&s).ok());
    let new_config = build_resume_config(&run_id, &old_config, body.n_trials, body.n_workers);
    let label = format!("resume of {run_id}");
    let resp = launch_and_record(
        &state,
        "tuner",
        &game,
        Some(new_config),
        Some(&label),
        Some(&run_id),
    )
    .await?;
    Ok(Json(resp))
}

/// Shared by `launch_run` and `resume_run`: builds the command, pins a
/// fresh `run_id` (baked into a tuner launch's own `--run-id`/`--resume`
/// argv, not just the outer bench-runs bookkeeping -- see
/// `launch::launch_with_run_id`'s doc comment for why they must match),
/// spawns it, and inserts the `runs` row so it appears immediately in the
/// runs list without waiting on the ingest loop.
/// If `config.ladder` is present but `config.ladder_root` isn't, injects
/// `ladder_root = run_id` -- this launch is the first rung of a new ladder.
/// Every other config (no `ladder` key, or one that already carries
/// `ladder_root` forward from a resume) passes through unchanged.
///
/// A ladder-enabled launch needs `ladder_root` set to its *own* run_id when
/// it's the first rung -- the caller (an operator hitting `POST
/// /api/bench/launch`) can't supply that itself, since the id doesn't exist
/// until `launch::generate_run_id` runs. A resumed/widened rung already
/// carries `ladder_root` forward via `build_resume_config`, so this only
/// ever fires once per ladder, at its root.
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
        record_floor_baseline_settings(config)
    } else {
        config
    };
    let parent_identity = if let Some(parent_id) = resume_from {
        let db = state.db.lock().unwrap();
        Some(identity::prepare_continuation(&db, parent_id).map_err(identity_bench_error)?)
    } else {
        None
    };
    let mut cmd = build_command(kind, game, &config, &run_id)?;
    let config = inject_ladder_root_if_new_ladder(config, &run_id);

    // `--run-id`/`--resume` are tuner-specific flags (see `tuner_cli`'s
    // `--run-id`/`--resume`); other kinds (round_robin) have no concept of
    // a resumable optimizer run to pin.
    if kind == "tuner" {
        cmd.push("--run-id".into());
        cmd.push(run_id.clone());
        if let Some(resume_id) = resume_from {
            cmd.push("--resume".into());
            cmd.push(resume_id.to_owned());
        }
    }

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

    // Insert the run and its identity in one transaction. If the registry
    // ingestion loop won the race, the identity helper adopts only its
    // provisional self-root for a server continuation; a server-recorded
    // child identity is never overwritten by replay.
    {
        let mut db = state.db.lock().unwrap();
        let transaction = db.transaction()?;
        let launched_log_path = log_path.to_string_lossy().to_string();
        let inserted = transaction.execute(
            "INSERT INTO runs \
             (run_id, kind, game, label, config, git_sha, git_dirty, \
              host, pid, started_at, status, log_path) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'running', ?11) \
             ON CONFLICT (run_id) DO NOTHING",
            duckdb::params![
                &run_id,
                kind,
                game,
                label,
                config_str,
                crate::BUILD_INFO.git_sha,
                crate::BUILD_INFO.git_dirty,
                hostname(),
                pid as i64,
                &started_at,
                &launched_log_path,
            ],
        )?;
        if inserted == 0 {
            let existing: (String, String, Option<i64>, String) = transaction.query_row(
                "SELECT kind, game, pid, log_path FROM runs WHERE run_id = ?1",
                duckdb::params![&run_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;
            if existing
                != (
                    kind.to_owned(),
                    game.to_owned(),
                    Some(pid as i64),
                    launched_log_path,
                )
            {
                return Err(BenchError {
                    status: StatusCode::CONFLICT,
                    message: format!(
                        "run id '{run_id}' is already assigned to a different process"
                    ),
                });
            }
        }
        if let Some(parent) = &parent_identity {
            identity::create_child_identity(&transaction, &run_id, parent)
                .map_err(identity_bench_error)?;
        } else {
            identity::create_root_identity(&transaction, &run_id, kind, None, None, &started_at)
                .map_err(identity_bench_error)?;
        }
        transaction.commit()?;
    }

    // Store config in the runs table so it survives server restarts.
    // (Separate UPDATE for the rare case the row was created by the
    // ingest loop between the INSERT above and here.)
    if let Some(ref config) = config {
        let db = state.db.lock().unwrap();
        let config_str = serde_json::to_string(config)?;
        let _ = db.execute(
            "UPDATE runs SET config = ?1 WHERE run_id = ?2 AND config IS NULL",
            duckdb::params![config_str, &run_id],
        );
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
        {
            let db = state.db.lock().unwrap();
            let _ = db.execute(
                "UPDATE runs SET ended_at = ?1, status = 'crashed' \
                 WHERE run_id = ?2 AND status = 'running'",
                duckdb::params![&now, &run_id],
            );
        }

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

pub(crate) fn project_legacy_stop(
    state: &Arc<BenchState>,
    run_id: &str,
    kind: &str,
) -> Result<String, BenchError> {
    let ended_at = iso_timestamp_now();
    let mut db = state.db.lock().unwrap();
    let tx = db.transaction()?;
    tx.execute(
        "UPDATE runs SET status = 'stopped', ended_at = ?1 WHERE run_id = ?2 AND status = 'running'",
        duckdb::params![&ended_at, run_id],
    )?;
    if kind == "experiment" {
        tx.execute(
            "UPDATE experiment_cells SET status = 'cancelled', ended_at = ?1, error = COALESCE(error, 'run stopped') WHERE run_id = ?2 AND status IN ('pending', 'running')",
            duckdb::params![&ended_at, run_id],
        )?;
    }
    tx.commit()?;
    Ok(ended_at)
}

/// Build the launch `config` JSON for a resumed tuner run: clones the old
/// run's config *wholesale* and patches only `overrides` (old entries plus
/// `optimizer.n_trials`/`optimizer.n_workers`, appended so they win -- the
/// Python side's `_apply_overrides` keeps the last value for a repeated
/// key) and `resumed_from` (this resume's source run id). Any other key the
/// old config carried (`config`, `baseline_configs`, `ladder`,
/// `ladder_root`, ...) survives untouched.
///
/// Cloning wholesale rather than reconstructing from just `overrides`/
/// `config` (the only two keys `LaunchBody.config` needs for a plain
/// resume) is what lets the automated ladder driver's own bookkeeping
/// (`ladder`, `ladder_root`, `baseline_configs`) survive a resume --
/// including a human clicking the existing UI Resume button on a ladder
/// rung, not just the driver's own calls. `resumed_from` itself closes a
/// separate, pre-existing gap: before this, nothing durable recorded which
/// run a resumed run came from (only a human-readable `label = "resume of
/// {run_id}"` string) -- the ladder driver needs to query this
/// programmatically to tell whether a rung already has a child.
pub(crate) fn build_resume_config(
    old_run_id: &str,
    old_config: &Option<Value>,
    n_trials: i64,
    n_workers: Option<i64>,
) -> Value {
    let mut new_config = match old_config.clone() {
        Some(Value::Object(map)) => Value::Object(map),
        _ => json!({}),
    };

    let mut overrides: Vec<Value> = new_config
        .get("overrides")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    overrides.push(json!(format!("optimizer.n_trials={n_trials}")));
    if let Some(n_workers) = n_workers {
        overrides.push(json!(format!("optimizer.n_workers={n_workers}")));
    }
    new_config["overrides"] = json!(overrides);
    new_config["resumed_from"] = json!(old_run_id);

    new_config
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
        "tuner" => {
            let mut cmd = vec![
                bench_binary.to_string_lossy().to_string(),
                "tuner".into(),
                "--game".into(),
                game.to_owned(),
            ];

            if let Some(ref config) = config {
                if let Some(config_path) = config.get("config").and_then(|v| v.as_str()) {
                    cmd.push("--config".into());
                    cmd.push(config_path.to_owned());
                }

                if let Some(overrides) = config.get("overrides").and_then(|v| v.as_array()) {
                    for o in overrides {
                        if let Some(ov) = o.as_str() {
                            cmd.push("--override".into());
                            cmd.push(ov.to_owned());
                        }
                    }
                }

                // Extra baseline instances backed by a raw discovered
                // config rather than a named preset -- how the automated
                // ladder widens a rung's opponent set. `id` (the object
                // key) becomes the `Scenario` instance id; its value is
                // passed through verbatim as the `<json>`
                // half of `--baseline-config <id>=<json>`.
                if let Some(baseline_configs) =
                    config.get("baseline_configs").and_then(|v| v.as_object())
                {
                    for (id, raw_config) in baseline_configs {
                        cmd.push("--baseline-config".into());
                        cmd.push(format!("{id}={raw_config}"));
                    }
                }

                // Game-setup config (e.g. Druid's board size) pinning every
                // trial in this run to a non-default `GameAdapter::
                // default_config()` -- see `game_host::GameAdapter::
                // tune_eval`'s `game_config` parameter. Absent or explicit
                // `null` both mean "use the game's own default", so only a
                // real object is forwarded.
                if let Some(game_config) = config.get("game_config") {
                    if !game_config.is_null() {
                        cmd.push("--game-config".into());
                        cmd.push(game_config.to_string());
                    }
                }
            }

            // Move-trace lines go to a dedicated `moves.jsonl` in the run's
            // own directory, same as round_robin below -- see
            // `LogRecord::Move`'s doc comment for why a full move trace is
            // kept out of the main log. Each trial's game-binary subprocess
            // opens this path in append mode, so every trial in the run
            // accumulates into the same file.
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
