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
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::StrategyInfo;

use super::types::*;
pub(crate) async fn get_run_trials(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Query(params): Query<TrialsParams>,
) -> Result<Json<Vec<TrialRow>>, BenchError> {
    let db = state.db.lock().unwrap();

    match db.query_row(
        "SELECT run_id FROM runs WHERE run_id = ?1",
        duckdb::params![&run_id],
        |row| row.get::<_, String>(0),
    ) {
        Ok(_) => {}
        Err(duckdb::Error::QueryReturnedNoRows) => {
            return Err(BenchError {
                status: StatusCode::NOT_FOUND,
                message: format!("run '{run_id}' not found"),
            });
        }
        Err(e) => return Err(BenchError::from(e)),
    }

    let mut sql = String::from(
        "SELECT trial_id, CAST(ts AS TEXT), CAST(config AS TEXT), seed, cost, CAST(extra AS TEXT) \
         FROM trials WHERE run_id = ?1 ORDER BY trial_id ASC",
    );
    if let Some(limit) = params.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }

    let mut stmt = db.prepare(&sql)?;
    let rows: Vec<TrialRow> = stmt
        .query_map(duckdb::params![&run_id], |row| {
            let config_str: String = row.get(2)?;
            let config: Value = serde_json::from_str(&config_str).unwrap_or(Value::Null);
            let extra_str: Option<String> = row.get(5)?;
            let extra = extra_str.and_then(|s| serde_json::from_str(&s).ok());
            Ok(TrialRow {
                trial_id: row.get(0)?,
                ts: row.get(1)?,
                config,
                seed: row.get(3)?,
                cost: row.get(4)?,
                extra,
            })
        })?
        .filter_map(|r| r.ok())
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
    let db = state.db.lock().unwrap();

    match db.query_row(
        "SELECT run_id FROM runs WHERE run_id = ?1",
        duckdb::params![&run_id],
        |row| row.get::<_, String>(0),
    ) {
        Ok(_) => {}
        Err(duckdb::Error::QueryReturnedNoRows) => {
            return Err(BenchError {
                status: StatusCode::NOT_FOUND,
                message: format!("run '{run_id}' not found"),
            });
        }
        Err(e) => return Err(BenchError::from(e)),
    }

    let mut sql = String::from(
        "SELECT g.game_seq, m.seq, m.cell_id, m.seed, CAST(m.metrics AS TEXT), COUNT(*), CAST(MIN(g.ts) AS TEXT), CAST(MAX(g.ts) AS TEXT), \
                m.strategy_a, m.strategy_b, m.outcome, m.winner \
         FROM game_moves g \
         LEFT JOIN match_results m ON m.run_id = g.run_id AND (m.trace_game_seq = g.game_seq OR (m.trace_game_seq IS NULL AND m.seq = g.game_seq)) \
         WHERE g.run_id = ?1 AND (?2 IS NULL OR m.cell_id = ?2) \
         GROUP BY g.game_seq, m.seq, m.cell_id, m.seed, m.metrics, m.strategy_a, m.strategy_b, m.outcome, m.winner \
         ORDER BY g.game_seq DESC",
    );
    if let Some(limit) = params.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }

    let mut stmt = db.prepare(&sql)?;
    let rows: Vec<GameSummary> = stmt
        .query_map(duckdb::params![&run_id, params.cell_id.as_deref()], |row| {
            Ok(GameSummary {
                game_seq: row.get(0)?,
                match_seq: row.get(1)?,
                cell_id: row.get(2)?,
                seed: row.get(3)?,
                metrics: row
                    .get::<_, Option<String>>(4)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
                ply_count: row.get(5)?,
                started_at: row.get(6)?,
                ended_at: row.get(7)?,
                strategy_a: row.get(8)?,
                strategy_b: row.get(9)?,
                outcome: row.get(10)?,
                winner: row.get(11)?,
            })
        })?
        .filter_map(|r| r.ok())
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
    let db = state.db.lock().unwrap();

    let mut stmt = db.prepare(
        "SELECT ply, CAST(ts AS TEXT), CAST(state AS TEXT), CAST(mv AS TEXT), player, CAST(search_report AS TEXT) \
         FROM game_moves WHERE run_id = ?1 AND game_seq = ?2 ORDER BY ply ASC",
    )?;
    let rows: Vec<MoveRow> = stmt
        .query_map(duckdb::params![&run_id, game_seq], |row| {
            let state_str: String = row.get(2)?;
            let mv_str: Option<String> = row.get(3)?;
            let search_str: Option<String> = row.get(5)?;
            Ok(MoveRow {
                ply: row.get(0)?,
                ts: row.get(1)?,
                state: serde_json::from_str(&state_str).unwrap_or(Value::Null),
                mv: mv_str.and_then(|s| serde_json::from_str(&s).ok()),
                player: row.get(4)?,
                search: search_str.and_then(|value| serde_json::from_str(&value).ok()),
            })
        })?
        .filter_map(|r| r.ok())
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
    {
        let db = state.db.lock().unwrap();
        match db.query_row(
            "SELECT run_id FROM runs WHERE run_id = ?1",
            duckdb::params![&run_id],
            |row| row.get::<_, String>(0),
        ) {
            Ok(_) => {}
            Err(duckdb::Error::QueryReturnedNoRows) => {
                return Err(BenchError {
                    status: StatusCode::NOT_FOUND,
                    message: format!("run '{run_id}' not found"),
                });
            }
            Err(e) => return Err(BenchError::from(e)),
        }
    }

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
                    let max_seq: Option<i64> = {
                        let db = state.db.lock().unwrap();
                        db.query_row(
                            "SELECT MAX(game_seq) FROM game_moves WHERE run_id = ?1",
                            duckdb::params![&run_id],
                            |row| row.get(0),
                        )
                        .unwrap_or(None)
                    };
                    let Some(max_seq) = max_seq else { continue };
                    max_seq
                }
            };

            if current_game_seq != Some(game_seq) {
                current_game_seq = Some(game_seq);
                last_ply = -1;
            }

            // (ply, ts, state, mv, player)
            type GameMoveRow = (i64, String, String, Option<String>, Option<String>);
            let new_rows: Vec<GameMoveRow> = {
                let db = state.db.lock().unwrap();
                let stmt = db.prepare(
                    "SELECT ply, CAST(ts AS TEXT), CAST(state AS TEXT), CAST(mv AS TEXT), player \
                     FROM game_moves WHERE run_id = ?1 AND game_seq = ?2 AND ply > ?3 \
                     ORDER BY ply ASC",
                );
                match stmt {
                    Ok(mut stmt) => {
                        let mapped =
                            stmt.query_map(duckdb::params![&run_id, game_seq, last_ply], |row| {
                                Ok((
                                    row.get::<_, i64>(0)?,
                                    row.get::<_, String>(1)?,
                                    row.get::<_, String>(2)?,
                                    row.get::<_, Option<String>>(3)?,
                                    row.get::<_, Option<String>>(4)?,
                                ))
                            });
                        match mapped {
                            Ok(iter) => iter.filter_map(Result::ok).collect(),
                            Err(_) => continue,
                        }
                    }
                    Err(_) => continue,
                }
            };

            for (ply, ts, state_str, mv_str, player) in new_rows {
                last_ply = ply;
                let payload = LiveMoveEvent {
                    game_seq,
                    ply,
                    ts,
                    state: serde_json::from_str(&state_str).unwrap_or(Value::Null),
                    mv: mv_str.and_then(|s| serde_json::from_str(&s).ok()),
                    player,
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
/// directory (`log.jsonl`/`moves.jsonl`/`stdout.log`). This is the only
/// deletion path; there is no automatic retention/pruning of traces. Refuses a still-`running`
/// run with 409 rather than deleting out from under a live process; stop it
/// first.
pub(crate) async fn delete_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<StatusCode, BenchError> {
    let status: String = {
        let db = state.db.lock().unwrap();
        match db.query_row(
            "SELECT status FROM runs WHERE run_id = ?1",
            duckdb::params![&run_id],
            |row| row.get(0),
        ) {
            Ok(s) => s,
            Err(duckdb::Error::QueryReturnedNoRows) => {
                return Err(BenchError {
                    status: StatusCode::NOT_FOUND,
                    message: format!("run '{run_id}' not found"),
                });
            }
            Err(e) => return Err(BenchError::from(e)),
        }
    };
    if status == "running" {
        return Err(BenchError {
            status: StatusCode::CONFLICT,
            message: format!("run '{run_id}' is still running -- stop it before deleting"),
        });
    }

    let run_dir = state.bench_runs_dir.join(&run_id);
    {
        let db = state.db.lock().unwrap();
        db.execute(
            "DELETE FROM game_moves WHERE run_id = ?1",
            duckdb::params![&run_id],
        )?;
        db.execute(
            "DELETE FROM incumbents WHERE run_id = ?1",
            duckdb::params![&run_id],
        )?;
        db.execute(
            "DELETE FROM trials WHERE run_id = ?1",
            duckdb::params![&run_id],
        )?;
        db.execute(
            "DELETE FROM match_results WHERE run_id = ?1",
            duckdb::params![&run_id],
        )?;
        db.execute(
            "DELETE FROM experiment_cells WHERE run_id = ?1",
            duckdb::params![&run_id],
        )?;
        for file in ["log.jsonl", "moves.jsonl", "stdout.log"] {
            let path = run_dir.join(file).to_string_lossy().to_string();
            db.execute(
                "DELETE FROM _ingest_cursor WHERE log_path = ?1",
                duckdb::params![&path],
            )?;
        }
        db.execute(
            "DELETE FROM runs WHERE run_id = ?1",
            duckdb::params![&run_id],
        )?;
    }

    // Best-effort: reclaim the on-disk trace/log files too. A failure here
    // (e.g. already gone) doesn't roll back the DB deletion above -- the DB
    // is the source of truth the UI reads from.
    let _ = std::fs::remove_dir_all(&run_dir);

    Ok(StatusCode::NO_CONTENT)
}
