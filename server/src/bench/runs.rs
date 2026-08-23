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

use super::{ladder::*, types::*};
pub(crate) async fn list_runs(
    AxumState(state): AxumState<Arc<BenchState>>,
    Query(params): Query<ListRunsParams>,
) -> Result<Json<Vec<RunSummary>>, BenchError> {
    let db = state.db.lock().unwrap();

    // Cast TIMESTAMP columns to TEXT so DuckDB's Rust bindings can read
    // them as strings without the `chrono` feature.
    let mut sql = String::from(
        "SELECT r.run_id, r.kind, r.game, r.label, r.git_sha, r.git_dirty, \
                r.host, r.pid, \
                CAST(r.started_at AS TEXT), \
                CAST(r.ended_at AS TEXT), \
                r.status, r.project_id, r.experiment_id, \
                COALESCE(m.match_count, 0), COALESCE(t.trial_count, 0), \
                CAST(r.config AS TEXT) \
         FROM runs r \
         LEFT JOIN (SELECT run_id, COUNT(*) AS match_count FROM match_results GROUP BY run_id) m \
           ON r.run_id = m.run_id \
         LEFT JOIN (SELECT run_id, COUNT(*) AS trial_count FROM trials GROUP BY run_id) t \
           ON r.run_id = t.run_id \
         WHERE 1=1",
    );

    // Build optional WHERE clauses by interpolating values directly into
    // the SQL.  These are internal API query params (status/game strings,
    // integer limit), not user-submitted SQL — injection is not a concern.
    if let Some(ref game) = params.game {
        sql.push_str(&format!(" AND r.game = '{}'", game.replace('\'', "''")));
    }
    if let Some(ref experiment_id) = params.experiment_id {
        sql.push_str(&format!(
            " AND r.experiment_id = '{}'",
            experiment_id.replace('\'', "''")
        ));
    }
    if let Some(ref project_id) = params.project_id {
        sql.push_str(&format!(
            " AND r.project_id = '{}'",
            project_id.replace('\'', "''")
        ));
    }

    sql.push_str(" ORDER BY CAST(r.started_at AS TEXT) DESC");

    let mut stmt = db.prepare(&sql)?;

    let physical_runs: Vec<(RunSummary, Option<Value>)> = stmt
        .query_map([], |row| {
            let run_id: String = row.get(0)?;
            let config = row
                .get::<_, Option<String>>(15)?
                .and_then(|text| serde_json::from_str(&text).ok());
            Ok((
                RunSummary {
                    run_id,
                    kind: row.get(1)?,
                    game: row.get(2)?,
                    project_id: row.get(11)?,
                    experiment_id: row.get(12)?,
                    label: row.get(3)?,
                    git_sha: row.get(4)?,
                    git_dirty: row.get(5)?,
                    host: row.get(6)?,
                    pid: row.get(7)?,
                    started_at: row.get(8)?,
                    ended_at: row.get(9)?,
                    status: row.get(10)?,
                    match_count: row.get(13)?,
                    trial_count: row.get(14)?,
                },
                config,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    // A ladder is one logical run even though each baseline change needs a
    // fresh tuner process and therefore a fresh storage row. Rows arrive
    // newest-first, so retain the newest rung's identity/status while
    // accumulating work from all of its physical rungs.
    let mut logical_runs: Vec<RunSummary> = Vec::new();
    let mut logical_indexes: HashMap<String, usize> = HashMap::new();
    for (run, config) in physical_runs {
        let logical_id = config
            .as_ref()
            .and_then(|value| value.get("ladder_root"))
            .and_then(Value::as_str)
            .unwrap_or(&run.run_id)
            .to_owned();
        if let Some(index) = logical_indexes.get(&logical_id).copied() {
            logical_runs[index].match_count += run.match_count;
            logical_runs[index].trial_count += run.trial_count;
            logical_runs[index].started_at = run.started_at;
        } else {
            logical_indexes.insert(logical_id, logical_runs.len());
            logical_runs.push(run);
        }
    }
    if let Some(ref status) = params.status {
        logical_runs.retain(|run| run.status == *status);
    }
    if let Some(limit) = params.limit {
        logical_runs.truncate(limit.max(0) as usize);
    }

    Ok(Json(logical_runs))
}

/// `GET /api/bench/runs/{run_id}`
pub(crate) async fn get_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<RunDetail>, BenchError> {
    let db = state.db.lock().unwrap();

    let detail = db.query_row(
        "SELECT r.run_id, r.kind, r.game, r.label, \
                CAST(r.config AS TEXT), r.project_id, r.experiment_id, CAST(r.experiment_spec AS TEXT), \
                r.git_sha, r.git_dirty, \
                r.host, r.pid, \
                CAST(r.started_at AS TEXT), \
                CAST(r.ended_at AS TEXT), \
                r.status, r.log_path, r.exit_code, \
                COALESCE(m.match_count, 0), COALESCE(t.trial_count, 0), \
                CAST(i.config AS TEXT), i.cost \
         FROM runs r \
         LEFT JOIN (SELECT run_id, COUNT(*) AS match_count FROM match_results GROUP BY run_id) m \
           ON r.run_id = m.run_id \
         LEFT JOIN (SELECT run_id, COUNT(*) AS trial_count FROM trials GROUP BY run_id) t \
           ON r.run_id = t.run_id \
         LEFT JOIN incumbents i ON r.run_id = i.run_id \
         WHERE r.run_id = ?1",
        duckdb::params![&run_id],
        |row| {
            let config_str: Option<String> = row.get::<_, Option<String>>(4).ok().flatten();
            let config = config_str.and_then(|s| serde_json::from_str(&s).ok());
            let incumbent_config_str: Option<String> =
                row.get::<_, Option<String>>(19).ok().flatten();
            let incumbent_cost: Option<f64> = row.get(20)?;
            let experiment_spec = row.get::<_, Option<String>>(7).ok().flatten().and_then(|s| serde_json::from_str(&s).ok());
            let incumbent =
                incumbent_config_str
                    .zip(incumbent_cost)
                    .map(|(s, cost)| IncumbentInfo {
                        config: serde_json::from_str(&s).unwrap_or(Value::Null),
                        cost,
                    });
            Ok(RunDetail {
                run_id: row.get(0)?,
                kind: row.get(1)?,
                game: row.get(2)?,
                project_id: row.get(5)?,
                experiment_id: row.get(6)?,
                experiment_spec,
                label: row.get(3)?,
                config,
                git_sha: row.get(8)?,
                git_dirty: row.get(9)?,
                host: row.get(10)?,
                pid: row.get(11)?,
                started_at: row.get(12)?,
                ended_at: row.get(13)?,
                status: row.get(14)?,
                log_path: row.get(15)?,
                exit_code: row.get(16)?,
                match_count: row.get(17)?,
                trial_count: row.get(18)?,
                incumbent,
            })
        },
    );

    match detail {
        Ok(run) => Ok(Json(run)),
        Err(duckdb::Error::QueryReturnedNoRows) => Err(BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("run '{run_id}' not found"),
        }),
        Err(e) => Err(BenchError::from(e)),
    }
}

/// `GET /api/bench/runs/{run_id}/stdout`
///
/// Returns the full raw content of the run's `stdout.log` file (stderr
/// output redirected by the launcher).  Unlike `log.jsonl`, this is
/// unstructured human-readable output — clap errors, panic traces, etc.
/// Useful for debugging a crashed run.
pub(crate) async fn get_run_stdout(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<String, BenchError> {
    let db = state.db.lock().unwrap();

    let log_path: String = match db.query_row(
        "SELECT log_path FROM runs WHERE run_id = ?1",
        duckdb::params![&run_id],
        |row| row.get(0),
    ) {
        Ok(p) => p,
        Err(duckdb::Error::QueryReturnedNoRows) => {
            return Err(BenchError {
                status: StatusCode::NOT_FOUND,
                message: format!("run '{run_id}' not found"),
            });
        }
        Err(e) => return Err(BenchError::from(e)),
    };

    // stdout.log is a sibling of log.jsonl.
    let log_path_obj = Path::new(&log_path);
    let stdout_path = log_path_obj
        .parent()
        .map(|p| p.join("stdout.log"))
        .unwrap_or_else(|| PathBuf::from("stdout.log"));

    if !stdout_path.exists() {
        return Ok(String::new());
    }

    Ok(std::fs::read_to_string(&stdout_path)?)
}

/// `GET /api/bench/runs/{run_id}/log?since=<offset>`
///
/// Tail lines from the run's `log.jsonl` since a byte offset.  Returns the
/// lines and the new offset for the caller to use on the next poll.
pub(crate) async fn get_run_log(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Query(params): Query<RunLogParams>,
) -> Result<Json<RunLogResponse>, BenchError> {
    let db = state.db.lock().unwrap();

    // Resolve the log_path from the runs table.
    let log_path: String = match db.query_row(
        "SELECT log_path FROM runs WHERE run_id = ?1",
        duckdb::params![&run_id],
        |row| row.get(0),
    ) {
        Ok(p) => p,
        Err(duckdb::Error::QueryReturnedNoRows) => {
            return Err(BenchError {
                status: StatusCode::NOT_FOUND,
                message: format!("run '{run_id}' not found"),
            });
        }
        Err(e) => return Err(BenchError::from(e)),
    };

    let path = Path::new(&log_path);
    if !path.exists() {
        return Ok(Json(RunLogResponse {
            lines: vec![],
            next_offset: 0,
        }));
    }

    let offset = params.since.unwrap_or(0);
    let file_len = std::fs::metadata(path)?.len();

    if file_len <= offset {
        return Ok(Json(RunLogResponse {
            lines: vec![],
            next_offset: offset,
        }));
    }

    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let reader = BufReader::new(file);

    let mut lines = Vec::new();
    for line_result in reader.lines() {
        let line = line_result?;
        lines.push(line);
    }

    Ok(Json(RunLogResponse {
        next_offset: file_len,
        lines,
    }))
}

/// `GET /api/bench/leaderboard?game=&git_sha=&since=`
///
/// Aggregated win-rate + Wilson CI over `match_results`.  Computed at query
/// time — no materialized view.
pub(crate) async fn get_leaderboard(
    AxumState(state): AxumState<Arc<BenchState>>,
    Query(params): Query<LeaderboardParams>,
) -> Result<Json<Vec<LeaderboardEntry>>, BenchError> {
    let db = state.db.lock().unwrap();

    // Build the SQL with optional WHERE clauses.  DuckDB's Rust bindings
    // use positional parameters ($1, $2, ...).  We chain filters and track
    // the parameter index.
    let mut conditions = String::from("r.status IN ('completed', 'crashed', 'stopped')");

    // Build filter clauses with 1-based parameter indices.  Hardcode
    // indices since there are at most 3 optional params.
    if let Some(ref game) = params.game {
        conditions.push_str(&format!(" AND r.game = '{}'", game.replace('\'', "''")));
    }
    if let Some(ref sha) = params.git_sha {
        conditions.push_str(&format!(" AND r.git_sha = '{}'", sha.replace('\'', "''")));
    }
    if let Some(ref since) = params.since {
        conditions.push_str(&format!(
            " AND r.started_at >= '{}'",
            since.replace('\'', "''")
        ));
    }

    let sql = format!(
        "WITH a_stats AS (
            SELECT mr.strategy_a AS strategy,
                   COUNT(*) AS total,
                   SUM(CASE WHEN mr.outcome = 'win_a' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN mr.outcome = 'win_b' THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN mr.outcome = 'draw' THEN 1 ELSE 0 END) AS draws
            FROM match_results mr
            JOIN runs r ON mr.run_id = r.run_id
            WHERE {conditions}
            GROUP BY mr.strategy_a
        ),
        b_stats AS (
            SELECT mr.strategy_b AS strategy,
                   COUNT(*) AS total,
                   SUM(CASE WHEN mr.outcome = 'win_b' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN mr.outcome = 'win_a' THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN mr.outcome = 'draw' THEN 1 ELSE 0 END) AS draws
            FROM match_results mr
            JOIN runs r ON mr.run_id = r.run_id
            WHERE {conditions}
            GROUP BY mr.strategy_b
        )
        SELECT COALESCE(a.strategy, b.strategy) AS strategy,
               COALESCE(a.total, 0) + COALESCE(b.total, 0) AS total,
               COALESCE(a.wins, 0) + COALESCE(b.wins, 0) AS wins,
               COALESCE(a.losses, 0) + COALESCE(b.losses, 0) AS losses,
               COALESCE(a.draws, 0) + COALESCE(b.draws, 0) AS draws
        FROM a_stats a
        FULL OUTER JOIN b_stats b ON a.strategy = b.strategy
        ORDER BY wins DESC, losses ASC"
    );

    let mut stmt = db.prepare(&sql)?;

    let entries: Vec<LeaderboardEntry> = stmt
        .query_map([], |row| {
            let total_i: i64 = row.get(1)?;
            let wins_i: i64 = row.get(2)?;
            let losses_i: i64 = row.get(3)?;
            let draws_i: i64 = row.get(4)?;

            let total = total_i as usize;
            let wins = wins_i as usize;
            let losses = losses_i as usize;
            let draws = draws_i as usize;
            let score = wins as f64 + 0.5 * draws as f64;
            let (win_rate, (ci_lower, ci_upper)) = if total > 0 {
                (score / total as f64, wilson_interval(score, total, 1.96))
            } else {
                (0.5, (0.0, 1.0))
            };

            Ok(LeaderboardEntry {
                strategy: row.get(0)?,
                total,
                wins,
                losses,
                draws,
                win_rate,
                ci_lower,
                ci_upper,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(Json(entries))
}

/// `GET /api/bench/kinds`
///
/// Returns metadata for every available run kind, including which games
/// and strategies are registered per kind.  Data-driven counterpart to
/// `POST /api/bench/launch` — the UI uses this to populate the launch form
/// dynamically rather than hardcoding one form per kind.
pub(crate) type ExperimentCellRow = (
    String,
    Option<u64>,
    String,
    Value,
    String,
    String,
    Value,
    String,
    String,
    Value,
    Value,
    i64,
    u64,
    u64,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);

pub(crate) async fn get_run_cells(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Vec<CellResponse>>, BenchError> {
    let db = state.db.lock().unwrap();
    let exists: i64 = db.query_row(
        "SELECT COUNT(*) FROM runs WHERE run_id = ?1",
        duckdb::params![run_id],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Err(BenchError {
            status: StatusCode::NOT_FOUND,
            message: "run not found".into(),
        });
    }
    let mut stmt = db.prepare("SELECT cell_id, cell_seed, game, CAST(game_config AS TEXT), variant_id, variant_label, CAST(candidate_config AS TEXT), baseline_id, baseline_label, CAST(baseline_config AS TEXT), CAST(budget AS TEXT), rounds, planned_games, completed_games, status, CAST(started_at AS TEXT), CAST(ended_at AS TEXT), error FROM experiment_cells WHERE run_id = ?1 ORDER BY cell_id")?;
    let rows: Vec<ExperimentCellRow> = stmt
        .query_map(duckdb::params![run_id], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or(Value::Null),
                row.get(4)?,
                row.get(5)?,
                serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or(Value::Null),
                row.get(7)?,
                row.get(8)?,
                serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or(Value::Null),
                serde_json::from_str(&row.get::<_, String>(10)?).unwrap_or(Value::Null),
                row.get(11)?,
                row.get(12)?,
                row.get(13)?,
                row.get(14)?,
                row.get(15)?,
                row.get(16)?,
                row.get(17)?,
            ))
        })?
        .filter_map(Result::ok)
        .collect();
    let mut result = Vec::with_capacity(rows.len());
    for (
        cell_id,
        cell_seed,
        game,
        game_config,
        variant_id,
        variant_label,
        candidate_config,
        baseline_id,
        baseline_label,
        baseline_config,
        budget,
        rounds,
        planned_games,
        completed_games,
        status,
        started_at,
        ended_at,
        error,
    ) in rows
    {
        let mut wins = 0_u64;
        let mut losses = 0_u64;
        let mut draws = 0_u64;
        let mut matches = db.prepare("SELECT CAST(metrics AS TEXT) FROM match_results WHERE run_id = ?1 AND cell_id = ?2 ORDER BY seq")?;
        for row in matches.query_map(duckdb::params![run_id, cell_id], |row| {
            row.get::<_, Option<String>>(0)
        })? {
            if let Ok(Some(row)) = row {
                match serde_json::from_str::<Value>(&row)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("outcome")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .as_deref()
                {
                    Some("candidate_win") => wins += 1,
                    Some("baseline_win") => losses += 1,
                    Some("draw") => draws += 1,
                    _ => {}
                }
            }
        }
        let total = wins + losses + draws;
        let score = wins as f64 + draws as f64 * 0.5;
        let (win_rate, (ci_lower, ci_upper)) = if total == 0 {
            (0.5, (0.0, 1.0))
        } else {
            (
                score / total as f64,
                wilson_interval(score, total as usize, 1.96),
            )
        };
        result.push(CellResponse {
            cell_id,
            cell_seed,
            game,
            game_config,
            variant_id,
            variant_label,
            candidate_config,
            baseline_id,
            baseline_label,
            baseline_config,
            budget,
            rounds,
            planned_games,
            completed_games,
            status,
            started_at,
            ended_at,
            error,
            wins,
            losses,
            draws,
            win_rate,
            ci_lower,
            ci_upper,
        });
    }
    Ok(Json(result))
}

/// One live ply pushed down `GET /api/bench/runs/{run_id}/live`'s SSE
/// stream. `game_seq` is included on every event (not just once) since the
/// currently in-flight game can change mid-stream (one trial/match ends,
/// the next one's moves start arriving) -- the client detects that by
/// watching for a `game_seq` change, no separate "game boundary" event type
/// needed.
/// One rung of a tuner ladder chain, ordered within its baseline history.
#[derive(Serialize)]
pub(crate) struct ChainRung {
    pub(crate) run_id: String,
    pub(crate) label: Option<String>,
    pub(crate) status: String,
    pub(crate) started_at: String,
    pub(crate) ended_at: Option<String>,
    pub(crate) trial_count: i64,
    pub(crate) incumbent: Option<IncumbentInfo>,
}

pub(crate) async fn get_run_chain(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Vec<ChainRung>>, BenchError> {
    let db = state.db.lock().unwrap();

    let config_str: Option<String> = match db.query_row(
        "SELECT CAST(config AS TEXT) FROM runs WHERE run_id = ?1",
        duckdb::params![&run_id],
        |row| row.get(0),
    ) {
        Ok(c) => c,
        Err(duckdb::Error::QueryReturnedNoRows) => {
            return Err(BenchError {
                status: StatusCode::NOT_FOUND,
                message: format!("run '{run_id}' not found"),
            });
        }
        Err(e) => return Err(BenchError::from(e)),
    };
    let config: Option<Value> = config_str.and_then(|s| serde_json::from_str(&s).ok());
    let root = config
        .as_ref()
        .and_then(|c| c.get("ladder_root"))
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| run_id.clone());

    let mut stmt = db.prepare(
        "SELECT r.run_id, r.label, r.status, CAST(r.started_at AS TEXT), \
                CAST(r.ended_at AS TEXT), COALESCE(t.trial_count, 0), \
                CAST(r.config AS TEXT), CAST(i.config AS TEXT), i.cost \
         FROM runs r \
         LEFT JOIN (SELECT run_id, COUNT(*) AS trial_count FROM trials GROUP BY run_id) t \
           ON r.run_id = t.run_id \
         LEFT JOIN incumbents i ON r.run_id = i.run_id \
         WHERE r.kind = 'tuner'",
    )?;
    let mut rungs: Vec<ChainRung> = stmt
        .query_map([], |row| {
            let run_config_str: Option<String> = row.get(6)?;
            let run_config: Option<Value> =
                run_config_str.and_then(|s| serde_json::from_str(&s).ok());
            let incumbent_config_str: Option<String> = row.get(7)?;
            let incumbent_cost: Option<f64> = row.get(8)?;
            let incumbent =
                incumbent_config_str
                    .zip(incumbent_cost)
                    .map(|(s, cost)| IncumbentInfo {
                        config: serde_json::from_str(&s).unwrap_or(Value::Null),
                        cost,
                    });
            Ok((
                run_config,
                ChainRung {
                    run_id: row.get(0)?,
                    label: row.get(1)?,
                    status: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    trial_count: row.get(5)?,
                    incumbent,
                },
            ))
        })?
        .filter_map(|r| r.ok())
        .filter(|(run_config, rung)| {
            rung.run_id == root
                || run_config
                    .as_ref()
                    .and_then(|c| c.get("ladder_root"))
                    .and_then(|v| v.as_str())
                    == Some(root.as_str())
        })
        .map(|(_, rung)| rung)
        .collect();

    rungs.sort_by(|a, b| a.started_at.cmp(&b.started_at));

    Ok(Json(rungs))
}
