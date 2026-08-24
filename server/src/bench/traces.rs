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

use game_host::{SearchReport, TunerInfo};
use mcts_bench::experiment::ExperimentSpecV1;
use mcts_bench::identity;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::log::RegistryEvent;
use mcts_bench::projects_attempt::{CellRequest, ProjectsError, StartRequest};
use mcts_bench::run_repository::{RunGamesQuery, RunRepository, RunTrialsQuery};
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::StrategyInfo;

use super::runs::{run_repository_error, run_repository_error_for_run};
use super::types::*;
pub(crate) async fn get_run_trials(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Query(params): Query<TrialsParams>,
) -> Result<Json<Vec<TrialRow>>, BenchError> {
    state
        .run_repository
        .ensure_run_exists(&run_id)
        .map_err(|error| run_repository_error_for_run(error, &run_id))?;
    let rows = state
        .run_repository
        .load_trials(
            &run_id,
            &RunTrialsQuery {
                limit: params.limit,
            },
        )
        .map_err(run_repository_error)?
        .into_iter()
        .map(|trial| TrialRow {
            trial_id: trial.trial_id,
            ts: trial.ts,
            config: trial.config,
            seed: trial.seed,
            cost: trial.cost,
            extra: trial.extra,
        })
        .collect();

    Ok(Json(rows))
}

/// One game's summary within a run, as reported by `GET
/// /api/bench/runs/{run_id}/games` -- one row per distinct `game_seq` in
/// `game_moves`. `strategy_a`/`strategy_b`/`outcome`/`winner` come from a
/// `LEFT JOIN` onto `match_results`; `None` when a trace has no matching
/// persisted result.
#[derive(Serialize)]
pub struct GameSummary {
    pub game_seq: i64,
    pub match_seq: Option<i64>,
    pub cell_id: Option<String>,
    pub seed: Option<u64>,
    pub metrics: Option<Value>,
    pub ply_count: i64,
    pub started_at: String,
    pub ended_at: String,
    pub strategy_a: Option<String>,
    pub strategy_b: Option<String>,
    pub outcome: Option<String>,
    pub winner: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct GamesParams {
    pub limit: Option<i64>,
    pub cell_id: Option<String>,
}

/// Optional game pin for the live trace stream. Without it the endpoint
/// follows the newest game, which is useful for a compact status display;
/// callers replaying a particular worker/game pass this to avoid being
/// switched to another game when the run starts one.
#[derive(Deserialize, Default)]
pub struct LiveGamesParams {
    pub game_seq: Option<i64>,
}

/// `GET /api/bench/runs/{run_id}/games?limit=`
///
/// Lists every game that has at least one traced ply, most recent
/// (highest `game_seq`) first -- the run-detail page's game picker (Session
/// 4) and "is there a live game" check both read this.
pub(crate) async fn get_run_games(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Query(params): Query<GamesParams>,
) -> Result<Json<Vec<GameSummary>>, BenchError> {
    state
        .run_repository
        .ensure_run_exists(&run_id)
        .map_err(|error| run_repository_error_for_run(error, &run_id))?;
    let rows = state
        .run_repository
        .load_games(
            &run_id,
            &RunGamesQuery {
                limit: params.limit,
                cell_id: params.cell_id,
            },
        )
        .map_err(run_repository_error)?
        .into_iter()
        .map(|game| GameSummary {
            game_seq: game.game_seq,
            match_seq: game.match_seq,
            cell_id: game.cell_id,
            seed: game.seed,
            metrics: game.metrics,
            ply_count: game.ply_count,
            started_at: game.started_at,
            ended_at: game.ended_at,
            strategy_a: game.strategy_a,
            strategy_b: game.strategy_b,
            outcome: game.outcome,
            winner: game.winner,
        })
        .collect();

    Ok(Json(rows))
}

/// One traced ply, as reported by `GET
/// /api/bench/runs/{run_id}/games/{game_seq}/moves` -- `state`/`mv` are the
/// same wire-JSON shape `GameAdapter::ai_move` already produces for
/// round-robin traces, so the UI's existing per-game renderer can draw them
/// with no new code.
#[derive(Serialize)]
pub struct MoveRow {
    pub ply: i64,
    pub ts: String,
    pub state: Value,
    pub mv: Option<Value>,
    pub player: Option<String>,
    pub search: Option<SearchReport>,
}

/// `GET /api/bench/runs/{run_id}/games/{game_seq}/moves`
///
/// A single game's full trace, ordered by ply -- historical replay (Session
/// 4) is a plain fetch of this, no SSE needed. Unknown `run_id`/`game_seq`
/// both just come back an empty list rather than 404, matching
/// `get_run_trials`'s own no-existence-check-beyond-the-run pattern.
pub(crate) async fn get_run_game_moves(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath((run_id, game_seq)): AxumPath<(String, i64)>,
) -> Result<Json<Vec<MoveRow>>, BenchError> {
    let rows = state
        .run_repository
        .load_game_moves(&run_id, game_seq, None)
        .map_err(run_repository_error)?
        .into_iter()
        .map(|move_row| MoveRow {
            ply: move_row.ply,
            ts: move_row.ts,
            state: move_row.state,
            mv: move_row.mv,
            player: move_row.player,
            search: move_row
                .search
                .and_then(|value| serde_json::from_value(value).ok()),
        })
        .collect();

    Ok(Json(rows))
}

/// One `experiment_cells` row, as `get_run_cells`'s query returns it --
/// named so the query's `Vec<(...)>` annotation and the `for (...) in rows`
/// destructure both point at one spelled-out tuple shape instead of two
/// independently-drifting 18-tuples.
#[derive(Serialize)]
pub(crate) struct LiveMoveEvent {
    game_seq: i64,
    ply: i64,
    ts: String,
    state: Value,
    mv: Option<Value>,
    player: Option<String>,
    search: Option<SearchReport>,
}

/// `GET /api/bench/runs/{run_id}/live` (SSE)
///
/// Polls `game_moves` every 750ms for plies newer than the last one sent,
/// on whichever `game_seq` is currently the highest for this run (the
/// "in-flight" game -- a fresh game starting under the same run (next
/// round-robin match / tuner trial) is
/// picked up automatically by the `MAX(game_seq)` jumping, no restart
/// needed. Ends when the client disconnects (the spawned polling task's
/// `tx.send` starts failing once the `Sse` response's stream is dropped).
pub(crate) async fn live_run_moves(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Query(params): Query<LiveGamesParams>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, BenchError> {
    state
        .run_repository
        .ensure_run_exists(&run_id)
        .map_err(|error| run_repository_error_for_run(error, &run_id))?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        let mut current_game_seq: Option<i64> = None;
        let mut last_ply: i64 = -1;
        let mut interval = tokio::time::interval(Duration::from_millis(750));
        loop {
            interval.tick().await;

            let game_seq = match params.game_seq {
                Some(game_seq) => game_seq,
                None => {
                    let max_seq = state
                        .run_repository
                        .load_latest_game_seq(&run_id)
                        .unwrap_or(None);
                    let Some(max_seq) = max_seq else { continue };
                    max_seq
                }
            };

            if current_game_seq != Some(game_seq) {
                current_game_seq = Some(game_seq);
                last_ply = -1;
            }

            let new_rows =
                match state
                    .run_repository
                    .load_game_moves(&run_id, game_seq, Some(last_ply))
                {
                    Ok(rows) => rows,
                    Err(_) => continue,
                };

            for move_row in new_rows {
                last_ply = move_row.ply;
                let payload = LiveMoveEvent {
                    game_seq: move_row.game_seq,
                    ply: move_row.ply,
                    ts: move_row.ts,
                    state: move_row.state,
                    mv: move_row.mv,
                    player: move_row.player,
                    search: move_row
                        .search
                        .and_then(|value| serde_json::from_value(value).ok()),
                };
                let Ok(event) = Event::default().json_data(&payload) else {
                    continue;
                };
                if tx.send(event).await.is_err() {
                    return;
                }
            }
        }
    });

    Ok(Sse::new(ReceiverStream::new(rx).map(Ok)).keep_alive(KeepAlive::default()))
}

/// `DELETE /api/bench/runs/{run_id}`
///
/// Removes a run's rows from every table (`game_moves`, `incumbents`,
/// `trials`, `match_results`, `runs`, in FK-safe child-before-parent order)
/// plus its `_ingest_cursor` entries and its `bench-runs/<run_id>/`
/// directory and its artifact-root and cursor records. This is the only deletion
/// path; there is no automatic retention/pruning of traces. A physical attempt
/// attached to a modern tuning session is retained with its trace and search
/// evidence; session deletion will own that lifecycle. Other running
/// runs are refused with 409 rather than deleting out from a live process.
pub(crate) async fn delete_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<StatusCode, BenchError> {
    let deletion = state
        .run_repository
        .load_deletion_info(&run_id)
        .map_err(|error| run_repository_error_for_run(error, &run_id))?;
    if let Some(session_id) = deletion.tuning_session_id {
        return Err(BenchError {
            status: StatusCode::CONFLICT,
            message: format!(
                "run '{run_id}' belongs to tuning session '{session_id}' and retains its trace evidence -- use the future session Delete workflow"
            ),
        });
    }
    if deletion.status == "running" {
        return Err(BenchError {
            status: StatusCode::CONFLICT,
            message: format!("run '{run_id}' is still running -- stop it before deleting"),
        });
    }

    let run_dir = state.bench_runs_dir.join(&run_id);
    let ingest_log_paths = ["log.jsonl", "moves.jsonl", "stdout.log"]
        .into_iter()
        .map(|file| run_dir.join(file).to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    state
        .run_repository
        .delete_run_records(&run_id, &ingest_log_paths)
        .map_err(run_repository_error)?;

    // Best-effort: reclaim the on-disk trace/log files too. A failure here
    // (e.g. already gone) doesn't roll back the DB deletion above -- the DB
    // is the source of truth the UI reads from.
    let _ = std::fs::remove_dir_all(&run_dir);

    Ok(StatusCode::NO_CONTENT)
}
