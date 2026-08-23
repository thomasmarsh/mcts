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
use super::{commands::*, types::*};
pub(crate) fn inject_ladder_root_if_new_ladder(
    config: Option<Value>,
    run_id: &str,
) -> Option<Value> {
    let mut config = config;
    if let Some(Value::Object(ref mut map)) = config {
        if map.contains_key("ladder") && !map.contains_key("ladder_root") {
            map.insert("ladder_root".to_string(), json!(run_id));
        }
    }
    config
}

/// Persist the exact settings of a floor baseline alongside the launch
/// request. The tuner runner already resolves these ids to raw params when
/// it invokes `tune eval`; keeping the same params in the run record lets
/// the detail view compare the eventual incumbent with the opponent it was
/// actually evaluated against from the first trial onward.
///
/// This is deliberately display metadata, not `baseline_configs`: adding it
/// to the latter would make the Python runner register the same instance
/// twice (once through `target.baselines`, once through `baseline_configs`).
pub(crate) fn record_floor_baseline_settings(config: Option<Value>) -> Option<Value> {
    let mut config = config?;
    let Some(object) = config.as_object_mut() else {
        return Some(config);
    };
    if object.contains_key("baseline_settings") {
        return Some(config);
    }
    let baselines = object
        .get("overrides")
        .and_then(Value::as_array)
        .and_then(|overrides| {
            overrides.iter().rev().find_map(|override_| {
                let text = override_.as_str()?;
                let raw = text.strip_prefix("target.baselines=")?;
                serde_json::from_str::<Vec<String>>(raw).ok()
            })
        });
    let Some(baselines) = baselines else {
        return Some(config);
    };

    let mut settings = serde_json::Map::new();
    for baseline in baselines {
        let params = match baseline.as_str() {
            "flat_mc" => json!({"family": "flat_mc", "q_init": "Infinity"}),
            "random" => json!({"family": "random", "q_init": "Infinity"}),
            _ => return Some(config),
        };
        settings.insert(baseline, params);
    }
    object.insert("baseline_settings".into(), Value::Object(settings));
    Some(config)
}

pub(crate) struct LadderRunRow {
    pub(crate) run_id: String,
    pub(crate) game: String,
    pub(crate) status: String,
    pub(crate) exit_code: Option<i64>,
    pub(crate) config: Option<Value>,
}

/// One decision `plan_ladder_advances` made: widen this rung's baseline set
/// and relaunch as its child. Carrying the decision as data (rather than
/// calling `launch_and_record` inline) is what lets the decision logic --
/// which run to widen, what its next config looks like -- be unit-tested
/// without spawning a real subprocess, the same separation `build_command`/
/// `build_resume_config` already have from the handlers that call them.
pub(crate) struct LadderAdvance {
    pub(crate) parent_run_id: String,
    pub(crate) game: String,
    pub(crate) widened_config: Value,
    pub(crate) label: String,
}

pub(crate) fn ladder_root_of(r: &LadderRunRow) -> Option<&str> {
    r.config
        .as_ref()
        .and_then(|c| c.get("ladder_root"))
        .and_then(|v| v.as_str())
}

pub(crate) fn resumed_from_of(r: &LadderRunRow) -> Option<&str> {
    r.config
        .as_ref()
        .and_then(|c| c.get("resumed_from"))
        .and_then(|v| v.as_str())
}

/// Sets a widened rung's opponent to *just* the new incumbent -- pure
/// self-play curriculum ("always face the current incumbent"), not an
/// ever-growing accumulation of every prior rung's baseline. Two things a
/// naive `baseline_configs.insert` would leave in place otherwise:
///
/// - Any `baseline_configs` entries inherited from the parent's config
///   (`build_resume_config` carries the whole config forward verbatim) are
///   dropped, not merged into.
/// - Any `target.baselines=[...]` override inherited the same way (e.g. the
///   root rung's own chosen starting baseline) is neutralized with a
///   trailing `target.baselines=[]` override -- `tuner_cli`'s
///   `_apply_overrides` applies overrides as a dict keyed by dotted path,
///   so the last occurrence of a repeated key wins, and `Scenario.
///   instances = [*target.baselines, *baseline_configs]`
///   (`tuner/src/tuner_cli/__main__.py`) would otherwise still include the
///   old named baseline alongside the new incumbent, right back to the
///   multi-instance-averaging problem this ladder redesign exists to avoid.
///
/// The runhistory merge (`--resume`) is untouched -- prior rungs' recorded
/// trial costs keep displaying continuously, only the *live* instance set
/// changes per rung.
pub(crate) fn replace_baseline_with_incumbent(
    widened: &mut Value,
    next_id: &str,
    incumbent_config: &Value,
) {
    widened["baseline_configs"] = json!({ next_id: incumbent_config });
    widened["baseline_settings"] = json!({ next_id: incumbent_config });
    let mut overrides: Vec<Value> = widened
        .get("overrides")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    overrides.push(json!("target.baselines=[]"));
    widened["overrides"] = json!(overrides);
}

/// The last `optimizer.n_trials` override is the effective total budget.
/// Baseline changes resume the same logical optimization and must preserve
/// that total rather than allocating a fresh batch for every physical rung.
pub(crate) fn configured_n_trials(config: &Value) -> Option<i64> {
    config
        .get("overrides")?
        .as_array()?
        .iter()
        .rev()
        .filter_map(Value::as_str)
        .find_map(|text| text.strip_prefix("optimizer.n_trials=")?.parse().ok())
}

/// Scans every active or completed tuner run for a ladder-enabled rung that hasn't
/// been widened yet and decides whether it saturated its current baseline
/// set -- the decision half of an automated stop -> extract incumbent ->
/// widen instances -> resume cycle (`incumbents` is keyed by `run_id`,
/// config already parsed JSON).
///
/// A run opts in by carrying `ladder: {"max_rungs", "saturation_threshold"}`
/// and `ladder_root` (the chain's first rung's own id) in its stored
/// `config` -- see `build_resume_config`'s doc comment for why this rides
/// in the existing free-form `config` JSON rather than a new table or
/// column. A run with no `ladder` key is left alone entirely, so this is a
/// no-op for every pre-existing/non-ladder tuner run.
pub(crate) fn plan_ladder_advances(
    runs: &[LadderRunRow],
    trial_counts: &HashMap<String, i64>,
    incumbents: &HashMap<String, (Value, f64)>,
) -> Vec<LadderAdvance> {
    let has_child = |run_id: &str| runs.iter().any(|r| resumed_from_of(r) == Some(run_id));
    let mut advances = Vec::new();

    for run in runs {
        // A running rung is eligible as soon as its incumbent crosses the
        // configured threshold; the IO wrapper stops it before resuming so
        // its runhistory is fully flushed. An operator's explicit `stop` or
        // a crash must not be silently overridden by reviving the chain.
        if !matches!(run.status.as_str(), "running" | "completed")
            || run.exit_code.is_some_and(|c| c != 0)
        {
            continue;
        }
        let Some(config) = &run.config else {
            continue;
        };
        let Some(ladder) = config.get("ladder") else {
            continue; // not a ladder-enabled run at all
        };
        let (Some(max_rungs), Some(saturation_threshold)) = (
            ladder.get("max_rungs").and_then(|v| v.as_i64()),
            ladder.get("saturation_threshold").and_then(|v| v.as_f64()),
        ) else {
            continue; // malformed `ladder` block -- ignore rather than error
        };
        let Some(ladder_root) = ladder_root_of(run) else {
            continue;
        };

        if has_child(&run.run_id) {
            continue; // already advanced (or already judged done)
        }

        let rung_count = runs
            .iter()
            .filter(|r| ladder_root_of(r) == Some(ladder_root))
            .count() as i64;
        if rung_count >= max_rungs {
            continue; // budget exhausted -- ladder is done
        }

        // Saturation is judged from the durable per-run incumbent (the
        // `incumbents` table, tuner's own tracked best config aggregated
        // across every active instance) -- not `Scenario.
        // termination_cost_threshold`, which only averages the
        // instance-seed pairs recorded so far for a config and so is
        // unsafe to rely on once more than one baseline instance is
        // active: a config could look saturated after being evaluated
        // against only the easiest instance.
        let Some((incumbent_config, incumbent_cost)) = incumbents.get(&run.run_id) else {
            continue; // no incumbent ever reported -- nothing to widen from
        };
        if *incumbent_cost > saturation_threshold {
            continue; // not saturated -- ladder is done here
        }

        // `optimizer.n_trials` is the logical run's total budget. A resumed
        // rung inherits the accumulated runhistory and consumes only the
        // remaining trials; increasing the value here would silently grow
        // the run whenever its baseline changed.
        let root_trial_count = *trial_counts.get(ladder_root).unwrap_or(&0);
        let cumulative_trials: i64 = runs
            .iter()
            .filter(|r| ladder_root_of(r) == Some(ladder_root))
            .map(|r| *trial_counts.get(&r.run_id).unwrap_or(&0))
            .sum();
        let next_n_trials = runs
            .iter()
            .find(|r| r.run_id == ladder_root)
            .and_then(|r| r.config.as_ref())
            .and_then(configured_n_trials)
            .or_else(|| configured_n_trials(config))
            .unwrap_or(cumulative_trials + root_trial_count);

        let next_rung = rung_count + 1;
        let next_id = format!("ladder{next_rung}");

        let mut widened = build_resume_config(&run.run_id, &run.config, next_n_trials, None);
        replace_baseline_with_incumbent(&mut widened, &next_id, incumbent_config);

        advances.push(LadderAdvance {
            parent_run_id: run.run_id.clone(),
            game: run.game.clone(),
            widened_config: widened,
            label: format!("ladder rung {next_rung} of {ladder_root}"),
        });
    }

    advances
}

/// Read every tuner run's ladder-relevant bookkeeping from `runs`. Shared by
/// the automated driver (`advance_ladders_once`) and the manual
/// `advance_baseline` route -- both need the same chain-walking data
/// (`ladder_root`/`resumed_from`/`config`), just with different decision
/// logic layered on top (`plan_ladder_advances` vs. `plan_manual_advance`).
pub(crate) fn fetch_tuner_runs(state: &Arc<BenchState>) -> Result<Vec<LadderRunRow>, BenchError> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT run_id, game, status, exit_code, CAST(config AS TEXT) FROM runs \
         WHERE kind = 'tuner'",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })?
        .filter_map(Result::ok)
        .map(
            |(run_id, game, status, exit_code, config_str)| LadderRunRow {
                run_id,
                game,
                status,
                exit_code,
                config: config_str.and_then(|s| serde_json::from_str(&s).ok()),
            },
        )
        .collect();
    Ok(rows)
}

/// Trial counts per run, keyed by `run_id` -- used to compute a widened
/// rung's cumulative `optimizer.n_trials` budget.
pub(crate) fn fetch_trial_counts(
    state: &Arc<BenchState>,
) -> Result<HashMap<String, i64>, BenchError> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare("SELECT run_id, COUNT(*) FROM trials GROUP BY run_id")?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(Result::ok)
        .collect();
    Ok(rows)
}

/// Latest tracked incumbent per run, keyed by `run_id`.
pub(crate) fn fetch_incumbents(
    state: &Arc<BenchState>,
) -> Result<HashMap<String, (Value, f64)>, BenchError> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare("SELECT run_id, CAST(config AS TEXT), cost FROM incumbents")?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?
        .filter_map(Result::ok)
        .filter_map(|(run_id, config_str, cost)| {
            serde_json::from_str::<Value>(&config_str)
                .ok()
                .map(|config| (run_id, (config, cost)))
        })
        .collect();
    Ok(rows)
}

/// IO wrapper around `plan_ladder_advances`: reads `runs`/`trials`/
/// `incumbents` for every tuner run, then calls `launch_and_record` for
/// each decided widen. Called once per tick from a background poll loop in
/// `main.rs`, the same shape as the existing ingest loop.
pub async fn advance_ladders_once(state: &Arc<BenchState>) {
    let runs = match fetch_tuner_runs(state) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ladder driver: query error: {}", e.message);
            return;
        }
    };
    let trial_counts = match fetch_trial_counts(state) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ladder driver: trial-count query error: {}", e.message);
            return;
        }
    };
    let incumbents = match fetch_incumbents(state) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("ladder driver: incumbents query error: {}", e.message);
            return;
        }
    };

    let advances = plan_ladder_advances(&runs, &trial_counts, &incumbents);
    for advance in advances {
        // Crossing the threshold is allowed to end a rung before its trial
        // budget is exhausted. Stop and reap the process before resuming:
        // `--resume` reads the parent's runhistory from disk, so launching
        // while the old process is still flushing could read a torn file.
        let outcome =
            match lifecycle::stop_run_impl(state, &advance.parent_run_id, &lifecycle::SystemClock)
                .await
            {
                Ok(outcome) => outcome,
                Err(e) => {
                    eprintln!(
                        "ladder driver: failed to stop run {}: {}",
                        advance.parent_run_id, e.message
                    );
                    continue;
                }
            };
        if outcome.prior_status == "running" {
            if let Some(pid_val) = outcome.pid {
                let pid = pid_val as u32;
                let deadline = std::time::Instant::now() + Duration::from_secs(15);
                while launch::is_alive(pid) && std::time::Instant::now() < deadline {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                if launch::is_alive(pid) {
                    eprintln!(
                        "ladder driver: run {} did not exit within 15s; not widening yet",
                        advance.parent_run_id
                    );
                    continue;
                }
            }
        }
        if let Err(e) = launch_and_record(
            state,
            "tuner",
            &advance.game,
            Some(advance.widened_config),
            Some(&advance.label),
            Some(&advance.parent_run_id),
        )
        .await
        {
            eprintln!(
                "ladder driver: failed to widen run {}: {}",
                advance.parent_run_id, e.message
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Manual baseline advance (operator-triggered ladder widen)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub struct AdvanceBaselineBody {
    /// Total trial budget for the widened run (tuner's `optimizer.n_trials`
    /// is cumulative once a runhistory is seeded via `--resume`, same as
    /// `ResumeBody::n_trials`). Defaults to giving the new rung as many
    /// fresh trials as the chain's root rung originally had, mirroring the
    /// automated driver's own default (see `plan_manual_advance`).
    #[serde(default)]
    pub n_trials: Option<i64>,
    #[serde(default)]
    pub n_workers: Option<i64>,
}

/// The result of [`plan_manual_advance`]: what to relaunch, and (for a run
/// that never opted into `ladder` at launch) a retroactive patch to the
/// target run's own stored config so it's discoverable as a chain root by
/// `ladder_root` alone from now on.
#[derive(Debug)]
pub(crate) struct ManualAdvance {
    pub(crate) game: String,
    pub(crate) widened_config: Value,
    pub(crate) label: String,
    pub(crate) root_patch: Option<(String, Value)>,
}

/// Decide how to widen a single, specific run's baseline set on demand --
/// the manual counterpart to `plan_ladder_advances`, which only ever
/// widens a rung that opted into `ladder: {max_rungs, saturation_threshold}`
/// at launch time and only once it judges the rung saturated. This instead
/// works on *any* tuner run, the moment an operator (not the threshold)
/// decides its incumbent is good enough to promote to a baseline -- an
/// operator watching the cost chart approach 0% doesn't need to have
/// pre-configured `ladder` at launch, or wait for `saturation_threshold` to
/// trip, to start a chain.
///
/// A run with no `ladder_root` yet becomes the chain's own root: this
/// function returns a `root_patch` the caller must persist (`UPDATE runs
/// SET config = ...`) so a *later* manual advance of a descendant rung (or
/// the UI's chain walk) can find every rung by `ladder_root` alone, the same
/// property `inject_ladder_root_if_new_ladder` gives an automated ladder's
/// root at launch time.
pub(crate) fn plan_manual_advance(
    runs: &[LadderRunRow],
    trial_counts: &HashMap<String, i64>,
    incumbents: &HashMap<String, (Value, f64)>,
    run_id: &str,
    requested_n_trials: Option<i64>,
    n_workers: Option<i64>,
) -> Result<ManualAdvance, String> {
    let run = runs
        .iter()
        .find(|r| r.run_id == run_id)
        .ok_or_else(|| format!("run '{run_id}' not found among tuner runs"))?;

    let Some((incumbent_config, _incumbent_cost)) = incumbents.get(run_id) else {
        return Err(format!(
            "run '{run_id}' has no incumbent recorded yet -- nothing to promote to a baseline"
        ));
    };

    let effective_root = ladder_root_of(run).unwrap_or(run_id).to_string();
    let in_chain = |r: &&LadderRunRow| {
        ladder_root_of(r) == Some(effective_root.as_str()) || r.run_id == effective_root
    };
    let rung_count = runs.iter().filter(in_chain).count() as i64;
    let cumulative_trials: i64 = runs
        .iter()
        .filter(in_chain)
        .map(|r| *trial_counts.get(&r.run_id).unwrap_or(&0))
        .sum();
    let root_trial_count = *trial_counts.get(&effective_root).unwrap_or(&0);
    let next_n_trials = requested_n_trials
        .or_else(|| run.config.as_ref().and_then(configured_n_trials))
        .unwrap_or(cumulative_trials + root_trial_count);

    let root_patch = if ladder_root_of(run).is_none() {
        let mut root_config = run.config.clone().unwrap_or_else(|| json!({}));
        if let Value::Object(ref mut map) = root_config {
            map.insert("ladder_root".to_string(), json!(effective_root));
        }
        Some((effective_root.clone(), root_config))
    } else {
        None
    };

    let next_rung = rung_count + 1;
    let next_id = format!("ladder{next_rung}");

    let mut widened = build_resume_config(run_id, &run.config, next_n_trials, n_workers);
    if let Value::Object(ref mut map) = widened {
        map.entry("ladder_root").or_insert(json!(effective_root));
    }
    replace_baseline_with_incumbent(&mut widened, &next_id, incumbent_config);

    Ok(ManualAdvance {
        game: run.game.clone(),
        widened_config: widened,
        label: format!("baseline advance from {run_id}"),
        root_patch,
    })
}

/// `POST /api/bench/runs/{run_id}/advance-baseline` — `{n_trials?, n_workers?}`
///
/// Operator-triggered counterpart to the automated ladder driver: promotes
/// this run's current incumbent to a new baseline instance and relaunches
/// with a widened `baseline_configs`, same mechanism as a scheduled ladder
/// widen (`plan_ladder_advances`) but firing on demand rather than once
/// `ladder.saturation_threshold` trips -- and it works on any tuner run, not
/// just one that opted into `ladder` at launch (see `plan_manual_advance`).
///
/// If the run is still `running`, it's stopped first (same SIGTERM-to-
/// process-group as `POST .../stop`) and this waits for the process to
/// actually exit before relaunching -- `--resume` reads the old run's
/// `runhistory.json` from disk (see `tuner_cli/resume.py`), so racing a
/// relaunch against the old process still flushing it on the way out would
/// risk a torn read. This is exactly the ordering an operator doing it by
/// hand (click Stop, wait, click Resume) already gets, just automated.
pub(crate) async fn advance_baseline(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Json(body): Json<AdvanceBaselineBody>,
) -> Result<Json<LaunchResponse>, BenchError> {
    let kind: String = {
        let db = state.db.lock().unwrap();
        match db.query_row(
            "SELECT kind FROM runs WHERE run_id = ?1",
            duckdb::params![&run_id],
            |row| row.get(0),
        ) {
            Ok(k) => k,
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
                "run '{run_id}' is a '{kind}' run, not 'tuner' -- only tuner runs support baseline advance"
            ),
        });
    }

    let outcome = lifecycle::stop_run_impl(&state, &run_id, &lifecycle::SystemClock).await?;
    if outcome.prior_status == "running" {
        if let Some(pid_val) = outcome.pid {
            let pid = pid_val as u32;
            let deadline = std::time::Instant::now() + Duration::from_secs(15);
            while launch::is_alive(pid) && std::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            if launch::is_alive(pid) {
                return Err(BenchError {
                    status: StatusCode::CONFLICT,
                    message: format!(
                        "run '{run_id}' did not exit within 15s of being stopped -- try again once it has"
                    ),
                });
            }
        }
    }

    let runs = fetch_tuner_runs(&state)?;
    let trial_counts = fetch_trial_counts(&state)?;
    let incumbents = fetch_incumbents(&state)?;

    let advance = plan_manual_advance(
        &runs,
        &trial_counts,
        &incumbents,
        &run_id,
        body.n_trials,
        body.n_workers,
    )
    .map_err(|message| BenchError {
        status: StatusCode::BAD_REQUEST,
        message,
    })?;

    if let Some((root_run_id, root_config)) = advance.root_patch {
        let config_str = serde_json::to_string(&root_config)?;
        let db = state.db.lock().unwrap();
        db.execute(
            "UPDATE runs SET config = ?1 WHERE run_id = ?2",
            duckdb::params![config_str, &root_run_id],
        )?;
    }

    let resp = launch_and_record(
        &state,
        "tuner",
        &advance.game,
        Some(advance.widened_config),
        Some(&advance.label),
        Some(&run_id),
    )
    .await?;
    Ok(Json(resp))
}
