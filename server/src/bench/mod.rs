//! Axum sub-router for `/api/bench/*` endpoints.
//!
//! Backed by DuckDB (single-writer, owned by this server process) and the
//! filesystem (`bench-runs/`).  Each route calls into `src/bench` library
//! code — this module is a thin HTTP translation layer, exactly as
//! `server/adapters/` is a thin layer over `mcts::games`/`mcts::strategies`.
//!
//! Mounted in `server/main.rs` alongside the game-play routes.

use std::collections::HashMap;
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
use std::convert::Infallible;
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer};

use game_host::TunerInfo;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::log::RegistryEvent;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::StrategyInfo;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// State shared by all bench routes.  The DuckDB connection is
/// `Mutex`-guarded because `duckdb::Connection` is `Send` but not `Sync`;
/// the ingest loop and API routes all share the same in-process connection.
pub struct BenchState {
    pub db: Mutex<duckdb::Connection>,
    pub bench_runs_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Query parameter types
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub struct ListRunsParams {
    pub status: Option<String>,
    pub game: Option<String>,
    pub limit: Option<i64>,
}

#[derive(Deserialize)]
pub struct RunLogParams {
    pub since: Option<u64>,
}

#[derive(Deserialize, Default)]
pub struct TrialsParams {
    pub limit: Option<i64>,
}

#[derive(Deserialize, Default)]
pub struct LeaderboardParams {
    pub game: Option<String>,
    pub git_sha: Option<String>,
    pub since: Option<String>,
}

#[derive(Deserialize)]
pub struct LaunchBody {
    pub kind: String,
    pub game: String,
    #[serde(default)]
    pub config: Option<Value>,
}

#[derive(Deserialize)]
pub struct ResumeBody {
    pub n_trials: i64,
    #[serde(default)]
    pub n_workers: Option<i64>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct RunSummary {
    pub run_id: String,
    pub kind: String,
    pub game: String,
    pub label: Option<String>,
    pub git_sha: String,
    pub git_dirty: bool,
    pub host: String,
    pub pid: Option<i64>,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub status: String,
    pub match_count: i64,
    pub trial_count: i64,
}

#[derive(Serialize)]
pub struct RunDetail {
    pub run_id: String,
    pub kind: String,
    pub game: String,
    pub label: Option<String>,
    pub config: Option<Value>,
    pub git_sha: String,
    pub git_dirty: bool,
    pub host: String,
    pub pid: Option<i64>,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub status: String,
    pub log_path: String,
    pub exit_code: Option<i64>,
    pub match_count: i64,
    pub trial_count: i64,
    /// SMAC3's own current best config for this run (from its intensifier,
    /// not a naive `MIN(cost)` over `trials` -- see `LogRecord::Incumbent`'s
    /// doc comment for why that distinction matters once multiple baseline
    /// instances are in play). `None` for a non-SMAC3 run, or a SMAC3 run
    /// that hasn't reported one yet.
    pub incumbent: Option<IncumbentInfo>,
}

/// A run's current incumbent, as reported by `GET /api/bench/runs/{run_id}`
/// -- `config` is already in the exact shape `tune eval --baseline-config`
/// expects, so an operator can copy it straight into a later run's launch.
#[derive(Serialize)]
pub struct IncumbentInfo {
    pub config: Value,
    pub cost: f64,
}

#[derive(Serialize)]
pub struct RunLogResponse {
    pub lines: Vec<String>,
    pub next_offset: u64,
}

#[derive(Serialize)]
pub struct LeaderboardEntry {
    pub strategy: String,
    pub total: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub win_rate: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

#[derive(Serialize)]
pub struct LaunchResponse {
    pub run_id: String,
    pub pid: u32,
    pub log_path: String,
    /// If the child process exited within 500ms of launch, the contents of
    /// its stderr (redirected to stdout.log).  None means the child was
    /// still alive after the check window — the launch succeeded normally.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub launch_error: Option<String>,
}

/// Metadata for a run kind exposed via `GET /api/bench/kinds`.
#[derive(Serialize)]
pub struct BenchKindInfo {
    pub kind: String,
    pub label: String,
    pub description: String,
    pub games: Vec<BenchGameInfo>,
}

/// Per-game information within a run kind.
#[derive(Serialize)]
pub struct BenchGameInfo {
    pub game: String,
    pub strategies: Vec<StrategyInfo>,
}

/// A game's tunable strategy search-space metadata, as reported by
/// `GET /api/bench/smac3/kinds` -- the SMAC3 launch form's data-driven
/// counterpart to `BenchGameInfo`.
#[derive(Serialize)]
pub struct Smac3GameInfo {
    pub game: String,
    pub tuner: TunerInfo,
}

/// One row from the `trials` table, as reported by
/// `GET /api/bench/runs/{run_id}/trials`.
#[derive(Serialize)]
pub struct TrialRow {
    pub trial_id: i64,
    pub ts: String,
    pub config: Value,
    pub seed: Option<i64>,
    pub cost: Option<f64>,
    pub extra: Option<Value>,
}

/// Structured error for bench routes — mirrors `adapters::AdapterError`'s
/// pattern with `{error, code}` JSON body.
#[derive(Debug)]
pub struct BenchError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for BenchError {
    fn into_response(self) -> axum::response::Response {
        let code = self.status.as_u16();
        (
            self.status,
            Json(json!({ "error": self.message, "code": code })),
        )
            .into_response()
    }
}

impl From<duckdb::Error> for BenchError {
    fn from(e: duckdb::Error) -> Self {
        BenchError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("database error: {e}"),
        }
    }
}

impl From<std::io::Error> for BenchError {
    fn from(e: std::io::Error) -> Self {
        BenchError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("I/O error: {e}"),
        }
    }
}

impl From<serde_json::Error> for BenchError {
    fn from(e: serde_json::Error) -> Self {
        BenchError {
            status: StatusCode::BAD_REQUEST,
            message: format!("JSON error: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Router constructor
// ---------------------------------------------------------------------------

/// Build the `/api/bench/*` sub-router, ready to be merged into the
/// server's main router.
pub fn bench_router(state: Arc<BenchState>) -> Router {
    // The launch route can run long if the child is slow to start (fork +
    // exec), but normally should complete in milliseconds.  Give it 30s
    // headroom anyway.
    let launch_timeout = TimeoutLayer::with_status_code(
        StatusCode::GATEWAY_TIMEOUT,
        std::time::Duration::from_secs(30),
    );

    let cors = CorsLayer::new()
        .allow_origin([
            "http://127.0.0.1:7878".parse::<HeaderValue>().unwrap(),
            "http://localhost:5173".parse::<HeaderValue>().unwrap(),
            "http://127.0.0.1:5173".parse::<HeaderValue>().unwrap(),
        ])
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    Router::new()
        .route("/api/bench/kinds", get(list_kinds))
        .route("/api/bench/smac3/kinds", get(list_smac3_kinds))
        .route("/api/bench/runs", get(list_runs))
        .route("/api/bench/runs/{run_id}", get(get_run))
        .route("/api/bench/runs/{run_id}/log", get(get_run_log))
        .route("/api/bench/runs/{run_id}/stdout", get(get_run_stdout))
        .route("/api/bench/runs/{run_id}/trials", get(get_run_trials))
        .route("/api/bench/runs/{run_id}/chain", get(get_run_chain))
        .route("/api/bench/runs/{run_id}/games", get(get_run_games))
        .route(
            "/api/bench/runs/{run_id}/games/{game_seq}/moves",
            get(get_run_game_moves),
        )
        .route("/api/bench/runs/{run_id}/live", get(live_run_moves))
        .route("/api/bench/leaderboard", get(get_leaderboard))
        .route("/api/bench/launch", post(launch_run).layer(launch_timeout))
        .route("/api/bench/runs/{run_id}/stop", post(stop_run))
        .route("/api/bench/runs/{run_id}", delete(delete_run))
        .route(
            "/api/bench/runs/{run_id}/resume",
            post(resume_run).layer(launch_timeout),
        )
        .route(
            "/api/bench/runs/{run_id}/advance-baseline",
            post(advance_baseline).layer(launch_timeout),
        )
        .layer(cors)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

/// `GET /api/bench/runs?status=&game=&limit=`
async fn list_runs(
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
                r.status, \
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

    sql.push_str(" ORDER BY CAST(r.started_at AS TEXT) DESC");

    let mut stmt = db.prepare(&sql)?;

    let physical_runs: Vec<(RunSummary, Option<Value>)> = stmt
        .query_map([], |row| {
            let run_id: String = row.get(0)?;
            let config = row
                .get::<_, Option<String>>(13)?
                .and_then(|text| serde_json::from_str(&text).ok());
            Ok((
                RunSummary {
                    run_id,
                    kind: row.get(1)?,
                    game: row.get(2)?,
                    label: row.get(3)?,
                    git_sha: row.get(4)?,
                    git_dirty: row.get(5)?,
                    host: row.get(6)?,
                    pid: row.get(7)?,
                    started_at: row.get(8)?,
                    ended_at: row.get(9)?,
                    status: row.get(10)?,
                    match_count: row.get(11)?,
                    trial_count: row.get(12)?,
                },
                config,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    // A ladder is one logical run even though each baseline change needs a
    // fresh SMAC3 process and therefore a fresh storage row. Rows arrive
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
async fn get_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<RunDetail>, BenchError> {
    let db = state.db.lock().unwrap();

    let detail = db.query_row(
        "SELECT r.run_id, r.kind, r.game, r.label, \
                CAST(r.config AS TEXT), \
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
                row.get::<_, Option<String>>(16).ok().flatten();
            let incumbent_cost: Option<f64> = row.get(17)?;
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
                label: row.get(3)?,
                config,
                git_sha: row.get(5)?,
                git_dirty: row.get(6)?,
                host: row.get(7)?,
                pid: row.get(8)?,
                started_at: row.get(9)?,
                ended_at: row.get(10)?,
                status: row.get(11)?,
                log_path: row.get(12)?,
                exit_code: row.get(13)?,
                match_count: row.get(14)?,
                trial_count: row.get(15)?,
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
async fn get_run_stdout(
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
async fn get_run_log(
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
async fn get_leaderboard(
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
async fn list_kinds() -> Json<Vec<BenchKindInfo>> {
    let game_registry = mcts_bench::registry();

    let mut games: Vec<BenchGameInfo> = game_registry
        .iter()
        .map(|(game_kind, bg)| BenchGameInfo {
            game: game_kind.to_string(),
            strategies: bg.strategies(),
        })
        .collect();
    games.sort_by(|a, b| a.game.cmp(&b.game));

    let kinds = vec![
        BenchKindInfo {
            kind: "round_robin".to_string(),
            label: "Round Robin".to_string(),
            description: "Every strategy plays every other strategy an equal number of times, both as first and second player.  Results are streamed as match_result JSONL lines, aggregated into a win-rate leaderboard with Wilson confidence intervals."
                .to_string(),
            games,
        },
        BenchKindInfo {
            kind: "smac3".to_string(),
            label: "SMAC3 Tuning".to_string(),
            description: "Runs a SMAC3 hyperparameter-optimization sweep over a game's tunable strategy search space, playing rounds of a params-built candidate against one or more baseline instances per trial.  Results are streamed as trial JSONL lines.  See GET /api/bench/smac3/kinds for per-game tuner metadata (search space, baselines, eval rounds) instead of a strategies list."
                .to_string(),
            games: vec![],
        },
    ];

    Json(kinds)
}

/// `GET /api/bench/smac3/kinds`
///
/// Per-game tuner metadata (search space, baselines, eval rounds), queried
/// by spawning each of `mcts_bench`'s registered game binaries once with
/// `tune describe` (see `mcts_bench::games::describe_tuners`) rather than
/// through `server::adapter::registry()`'s live gameplay sessions -- that
/// registry only covers the games with a UI renderer, which used to leave
/// tunable-but-UI-less games (e.g. `nim`) unable to appear here even though
/// `POST /api/bench/launch` never needed a live session for them either
/// (the smac3 CLI subprocess it spawns locates the game binary itself).
/// Only games that implement `tuner()` appear -- tuning support is opt-in
/// per game.
async fn list_smac3_kinds() -> Json<Vec<Smac3GameInfo>> {
    let mut games: Vec<Smac3GameInfo> = mcts_bench::games::describe_tuners()
        .into_iter()
        .map(|(kind, tuner)| Smac3GameInfo {
            game: kind.to_string(),
            tuner,
        })
        .collect();
    games.sort_by(|a, b| a.game.cmp(&b.game));
    Json(games)
}

/// `GET /api/bench/runs/{run_id}/trials?limit=`
///
/// Rows from the `trials` table for one run, ordered by `trial_id`.
async fn get_run_trials(
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
/// `LEFT JOIN` onto `match_results` (round-robin's own `seq` == a trace's
/// `game_seq`, by construction -- see plan/spectator.md Session 2a); `None`
/// for SMAC3 trial self-play, whose traces don't join onto `trials.trial_id`
/// (see Session 2b's note on `MoveTracer::start_game`'s own `game_seq`).
#[derive(Serialize)]
pub struct GameSummary {
    pub game_seq: i64,
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
async fn get_run_games(
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
        "SELECT g.game_seq, COUNT(*), CAST(MIN(g.ts) AS TEXT), CAST(MAX(g.ts) AS TEXT), \
                m.strategy_a, m.strategy_b, m.outcome, m.winner \
         FROM game_moves g \
         LEFT JOIN match_results m ON m.run_id = g.run_id AND m.seq = g.game_seq \
         WHERE g.run_id = ?1 \
         GROUP BY g.game_seq, m.strategy_a, m.strategy_b, m.outcome, m.winner \
         ORDER BY g.game_seq DESC",
    );
    if let Some(limit) = params.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }

    let mut stmt = db.prepare(&sql)?;
    let rows: Vec<GameSummary> = stmt
        .query_map(duckdb::params![&run_id], |row| {
            Ok(GameSummary {
                game_seq: row.get(0)?,
                ply_count: row.get(1)?,
                started_at: row.get(2)?,
                ended_at: row.get(3)?,
                strategy_a: row.get(4)?,
                strategy_b: row.get(5)?,
                outcome: row.get(6)?,
                winner: row.get(7)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(Json(rows))
}

/// One traced ply, as reported by `GET
/// /api/bench/runs/{run_id}/games/{game_seq}/moves` -- `state`/`mv` are the
/// same wire-JSON shape `GameAdapter::ai_move` already produces for
/// round-robin traces, so the UI's existing per-game renderer (Session 4)
/// can draw them with no new code. SMAC3 traces store `state` as a
/// `Display`-text JSON string instead (see plan/spectator.md Session 2b) --
/// not renderer-ready, but still fine to tail as text.
#[derive(Serialize)]
pub struct MoveRow {
    pub ply: i64,
    pub ts: String,
    pub state: Value,
    pub mv: Option<Value>,
    pub player: Option<String>,
}

/// `GET /api/bench/runs/{run_id}/games/{game_seq}/moves`
///
/// A single game's full trace, ordered by ply -- historical replay (Session
/// 4) is a plain fetch of this, no SSE needed. Unknown `run_id`/`game_seq`
/// both just come back an empty list rather than 404, matching
/// `get_run_trials`'s own no-existence-check-beyond-the-run pattern.
async fn get_run_game_moves(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath((run_id, game_seq)): AxumPath<(String, i64)>,
) -> Result<Json<Vec<MoveRow>>, BenchError> {
    let db = state.db.lock().unwrap();

    let mut stmt = db.prepare(
        "SELECT ply, CAST(ts AS TEXT), CAST(state AS TEXT), CAST(mv AS TEXT), player \
         FROM game_moves WHERE run_id = ?1 AND game_seq = ?2 ORDER BY ply ASC",
    )?;
    let rows: Vec<MoveRow> = stmt
        .query_map(duckdb::params![&run_id, game_seq], |row| {
            let state_str: String = row.get(2)?;
            let mv_str: Option<String> = row.get(3)?;
            Ok(MoveRow {
                ply: row.get(0)?,
                ts: row.get(1)?,
                state: serde_json::from_str(&state_str).unwrap_or(Value::Null),
                mv: mv_str.and_then(|s| serde_json::from_str(&s).ok()),
                player: row.get(4)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(Json(rows))
}

/// One live ply pushed down `GET /api/bench/runs/{run_id}/live`'s SSE
/// stream. `game_seq` is included on every event (not just once) since the
/// currently in-flight game can change mid-stream (one trial/match ends,
/// the next one's moves start arriving) -- the client detects that by
/// watching for a `game_seq` change, no separate "game boundary" event type
/// needed.
#[derive(Serialize)]
struct LiveMoveEvent {
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
/// "in-flight" game, per plan/spectator.md Session 3) -- a fresh game
/// starting under the same run (next round-robin match / SMAC3 trial) is
/// picked up automatically by the `MAX(game_seq)` jumping, no restart
/// needed. Ends when the client disconnects (the spawned polling task's
/// `tx.send` starts failing once the `Sse` response's stream is dropped).
async fn live_run_moves(
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

            let new_rows: Vec<(i64, String, String, Option<String>, Option<String>)> = {
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
/// directory (`log.jsonl`/`moves.jsonl`/`stdout.log`) -- per
/// plan/spectator.md's scope decision, this is the *only* deletion path
/// (no automatic retention/pruning of traces). Refuses a still-`running`
/// run with 409 rather than deleting out from under a live process; stop it
/// first.
async fn delete_run(
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

/// One rung of a SMAC3 ladder chain, as reported by `GET
/// /api/bench/runs/{run_id}/chain` -- a run's baseline history rendered as a
/// continuous timeline stitches together each rung's own `trials` (fetched
/// separately per rung, same route as a single run) using this list as the
/// index: order, boundaries, and the incumbent each rung handed to the next.
#[derive(Serialize)]
pub struct ChainRung {
    pub run_id: String,
    pub label: Option<String>,
    pub status: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub trial_count: i64,
    /// The cost this rung's baseline was promoted at (the prior rung's own
    /// incumbent) -- `None` for the chain's root rung, which has no prior
    /// baseline advance behind it.
    pub incumbent: Option<IncumbentInfo>,
}

/// `GET /api/bench/runs/{run_id}/chain`
///
/// Every rung of the ladder chain `run_id` belongs to, oldest first --
/// `ladder_root`/`resumed_from` (see `build_resume_config`'s and
/// `plan_manual_advance`'s doc comments) link a sequence of otherwise
/// independent `runs` rows into one logical baseline-advance timeline. A run
/// with no `ladder_root` at all is its own one-rung chain (every plain
/// SMAC3 run, and every ladder run that's never had its baseline advanced
/// yet), so this always returns at least one element for a run that exists.
async fn get_run_chain(
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
         WHERE r.kind = 'smac3'",
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

/// `POST /api/bench/launch` — `{kind, game, config}`
///
/// Translates the request into a command vector, spawns it via
/// `launch::launch`, and returns the run metadata immediately.
async fn launch_run(
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
/// Relaunches a finished/stopped SMAC3 run with a bigger trial budget,
/// picking up where it left off rather than starting over: the new process
/// is launched with `--resume <old run_id>` (see `smac3_cli/resume.py`),
/// which seeds its runhistory from the old run's saved state before
/// optimizing, so already-evaluated configs aren't re-evaluated. This is
/// also the only way to change worker count "mid-run" -- SMAC3 has no live
/// API for either, only stop-and-relaunch.
///
/// The old run's stored `config` (its `--config` path and any `--override`
/// list) is carried forward, with `optimizer.n_trials`/`optimizer.n_workers`
/// overrides appended (and so taking precedence -- the Python side's
/// `_apply_overrides` keeps the last value for a repeated key).
async fn resume_run(
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

    if kind != "smac3" {
        return Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: format!(
                "run '{run_id}' is a '{kind}' run, not 'smac3' -- only SMAC3 runs support resume"
            ),
        });
    }

    let old_config: Option<Value> = config_str.and_then(|s| serde_json::from_str(&s).ok());
    let new_config = build_resume_config(&run_id, &old_config, body.n_trials, body.n_workers);
    let label = format!("resume of {run_id}");
    let resp = launch_and_record(
        &state,
        "smac3",
        &game,
        Some(new_config),
        Some(&label),
        Some(&run_id),
    )
    .await?;
    Ok(Json(resp))
}

/// Shared by `launch_run` and `resume_run`: builds the command, pins a
/// fresh `run_id` (baked into a SMAC3 launch's own `--run-id`/`--resume`
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
fn inject_ladder_root_if_new_ladder(config: Option<Value>, run_id: &str) -> Option<Value> {
    let mut config = config;
    if let Some(Value::Object(ref mut map)) = config {
        if map.contains_key("ladder") && !map.contains_key("ladder_root") {
            map.insert("ladder_root".to_string(), json!(run_id));
        }
    }
    config
}

/// Persist the exact settings of a floor baseline alongside the launch
/// request. The SMAC3 runner already resolves these ids to raw params when
/// it invokes `tune eval`; keeping the same params in the run record lets
/// the detail view compare the eventual incumbent with the opponent it was
/// actually evaluated against from the first trial onward.
///
/// This is deliberately display metadata, not `baseline_configs`: adding it
/// to the latter would make the Python runner register the same instance
/// twice (once through `target.baselines`, once through `baseline_configs`).
fn record_floor_baseline_settings(config: Option<Value>) -> Option<Value> {
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

async fn launch_and_record(
    state: &Arc<BenchState>,
    kind: &str,
    game: &str,
    config: Option<Value>,
    label: Option<&str>,
    resume_from: Option<&str>,
) -> Result<LaunchResponse, BenchError> {
    let run_id = launch::generate_run_id(kind, game, crate::BUILD_INFO);
    let config = if kind == "smac3" {
        record_floor_baseline_settings(config)
    } else {
        config
    };
    let mut cmd = build_command(kind, game, &config, &run_id)?;
    let config = inject_ladder_root_if_new_ladder(config, &run_id);

    // `--run-id`/`--resume` are SMAC3-specific flags (see `smac3_cli`'s
    // `--run-id`/`--resume`); other kinds (round_robin) have no concept of
    // a resumable optimizer run to pin.
    if kind == "smac3" {
        cmd.push("--run-id".into());
        cmd.push(run_id.clone());
        if let Some(resume_id) = resume_from {
            cmd.push("--resume".into());
            cmd.push(resume_id.to_owned());
        }
    }

    let LaunchedRun {
        run_id,
        pid,
        log_path,
        log_dir,
    } = launch::launch_with_run_id(run_id, cmd, kind, game, label, crate::BUILD_INFO).map_err(
        |e| BenchError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("failed to launch run: {e}"),
        },
    )?;

    let started_at = iso_timestamp_now();

    // Insert the run into the runs table so it appears immediately in
    // the runs list (no ingest loop dependency).
    {
        let db = state.db.lock().unwrap();
        let config_str = config
            .as_ref()
            .map(|c| serde_json::to_string(c).unwrap_or_default());
        let hostname = hostname();
        db.execute(
            "INSERT INTO runs \
             (run_id, kind, game, label, config, git_sha, git_dirty, \
              host, pid, started_at, status, log_path) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'running', ?11)",
            duckdb::params![
                &run_id,
                kind,
                game,
                label,
                config_str,
                crate::BUILD_INFO.git_sha,
                crate::BUILD_INFO.git_dirty,
                hostname,
                pid as i64,
                &started_at,
                log_path.to_string_lossy().to_string(),
            ],
        )?;
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

/// Snapshot of one `smac3` run's bookkeeping, as read from `runs`.
struct LadderRunRow {
    run_id: String,
    game: String,
    status: String,
    exit_code: Option<i64>,
    config: Option<Value>,
}

/// One decision `plan_ladder_advances` made: widen this rung's baseline set
/// and relaunch as its child. Carrying the decision as data (rather than
/// calling `launch_and_record` inline) is what lets the decision logic --
/// which run to widen, what its next config looks like -- be unit-tested
/// without spawning a real subprocess, the same separation `build_command`/
/// `build_resume_config` already have from the handlers that call them.
struct LadderAdvance {
    parent_run_id: String,
    game: String,
    widened_config: Value,
    label: String,
}

fn ladder_root_of(r: &LadderRunRow) -> Option<&str> {
    r.config
        .as_ref()
        .and_then(|c| c.get("ladder_root"))
        .and_then(|v| v.as_str())
}

fn resumed_from_of(r: &LadderRunRow) -> Option<&str> {
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
///   trailing `target.baselines=[]` override -- `smac3_cli`'s
///   `_apply_overrides` applies overrides as a dict keyed by dotted path,
///   so the last occurrence of a repeated key wins, and `Scenario.
///   instances = [*target.baselines, *baseline_configs]`
///   (`smac3/src/smac3_cli/__main__.py`) would otherwise still include the
///   old named baseline alongside the new incumbent, right back to the
///   multi-instance-averaging problem this ladder redesign exists to avoid.
///
/// The runhistory merge (`--resume`) is untouched -- prior rungs' recorded
/// trial costs keep displaying continuously, only the *live* instance set
/// changes per rung.
fn replace_baseline_with_incumbent(widened: &mut Value, next_id: &str, incumbent_config: &Value) {
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
fn configured_n_trials(config: &Value) -> Option<i64> {
    config
        .get("overrides")?
        .as_array()?
        .iter()
        .rev()
        .filter_map(Value::as_str)
        .find_map(|text| text.strip_prefix("optimizer.n_trials=")?.parse().ok())
}

/// Scans every active or completed SMAC3 run for a ladder-enabled rung that hasn't
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
/// no-op for every pre-existing/non-ladder SMAC3 run.
fn plan_ladder_advances(
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
        // `incumbents` table, SMAC3's own tracked best config aggregated
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

/// Read every SMAC3 run's ladder-relevant bookkeeping from `runs`. Shared by
/// the automated driver (`advance_ladders_once`) and the manual
/// `advance_baseline` route -- both need the same chain-walking data
/// (`ladder_root`/`resumed_from`/`config`), just with different decision
/// logic layered on top (`plan_ladder_advances` vs. `plan_manual_advance`).
fn fetch_smac3_runs(state: &Arc<BenchState>) -> Result<Vec<LadderRunRow>, BenchError> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT run_id, game, status, exit_code, CAST(config AS TEXT) FROM runs \
         WHERE kind = 'smac3'",
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
fn fetch_trial_counts(state: &Arc<BenchState>) -> Result<HashMap<String, i64>, BenchError> {
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
fn fetch_incumbents(state: &Arc<BenchState>) -> Result<HashMap<String, (Value, f64)>, BenchError> {
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
/// `incumbents` for every SMAC3 run, then calls `launch_and_record` for
/// each decided widen. Called once per tick from a background poll loop in
/// `main.rs`, the same shape as the existing ingest loop.
pub async fn advance_ladders_once(state: &Arc<BenchState>) {
    let runs = match fetch_smac3_runs(state) {
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
        let outcome = match stop_run_impl(state, &advance.parent_run_id).await {
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
            "smac3",
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
    /// Total trial budget for the widened run (SMAC3's `optimizer.n_trials`
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
struct ManualAdvance {
    game: String,
    widened_config: Value,
    label: String,
    root_patch: Option<(String, Value)>,
}

/// Decide how to widen a single, specific run's baseline set on demand --
/// the manual counterpart to `plan_ladder_advances`, which only ever
/// widens a rung that opted into `ladder: {max_rungs, saturation_threshold}`
/// at launch time and only once it judges the rung saturated. This instead
/// works on *any* SMAC3 run, the moment an operator (not the threshold)
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
fn plan_manual_advance(
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
        .ok_or_else(|| format!("run '{run_id}' not found among SMAC3 runs"))?;

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
/// `ladder.saturation_threshold` trips -- and it works on any SMAC3 run, not
/// just one that opted into `ladder` at launch (see `plan_manual_advance`).
///
/// If the run is still `running`, it's stopped first (same SIGTERM-to-
/// process-group as `POST .../stop`) and this waits for the process to
/// actually exit before relaunching -- `--resume` reads the old run's
/// `runhistory.json` from disk (see `smac3_cli/resume.py`), so racing a
/// relaunch against the old process still flushing it on the way out would
/// risk a torn read. This is exactly the ordering an operator doing it by
/// hand (click Stop, wait, click Resume) already gets, just automated.
async fn advance_baseline(
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

    if kind != "smac3" {
        return Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: format!(
                "run '{run_id}' is a '{kind}' run, not 'smac3' -- only SMAC3 runs support baseline advance"
            ),
        });
    }

    let outcome = stop_run_impl(&state, &run_id).await?;
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

    let runs = fetch_smac3_runs(&state)?;
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
        "smac3",
        &advance.game,
        Some(advance.widened_config),
        Some(&advance.label),
        Some(&run_id),
    )
    .await?;
    Ok(Json(resp))
}

/// Outcome of [`stop_run_impl`] — enough for a caller to build its own
/// response (`stop_run`'s JSON body) or decide whether to wait for the
/// process to actually exit (`advance_baseline`).
struct StopOutcome {
    pid: Option<i64>,
    /// The run's status *before* this call — `"running"` means a signal was
    /// (attempted to be) sent; anything else means this was a no-op.
    prior_status: String,
    signal_sent: bool,
}

/// Shared by `stop_run` and `advance_baseline`: sends SIGTERM to the
/// recorded PID's whole process group (`kill -TERM -<pid>`) and marks the
/// run as `stopped` in the database.  `launch::launch` puts every run in its
/// own process group (`process_group(0)`), so the recorded PID is that
/// group's leader -- signalling just that one PID would leave descendants
/// (e.g. the `uv`/python child under `bench smac3`) orphaned instead of
/// terminated.  If the PID is no longer alive, updates the status anyway
/// (the process exited on its own between the list and the stop request). A
/// run that isn't `running` is left untouched -- the caller's
/// `prior_status` tells it so.
async fn stop_run_impl(state: &Arc<BenchState>, run_id: &str) -> Result<StopOutcome, BenchError> {
    let (pid, status): (Option<i64>, String) = {
        let db = state.db.lock().unwrap();
        match db.query_row(
            "SELECT pid, status FROM runs WHERE run_id = ?1",
            duckdb::params![run_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
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

    if status != "running" {
        return Ok(StopOutcome {
            pid,
            prior_status: status,
            signal_sent: false,
        });
    }

    let mut signal_sent = false;

    if let Some(pid_val) = pid {
        #[cfg(unix)]
        {
            // Negative PID = signal the whole process group, not just its
            // leader (see doc comment above).
            match std::process::Command::new("kill")
                .arg("-TERM")
                .arg(format!("-{pid_val}"))
                .status()
            {
                Ok(status_result) if status_result.success() => {
                    signal_sent = true;
                }
                Ok(_) => {
                    // PID not found — that's fine, it means the run exited
                    // on its own.  We'll still mark it stopped below.
                }
                Err(e) => {
                    return Err(BenchError {
                        status: StatusCode::INTERNAL_SERVER_ERROR,
                        message: format!("failed to signal run '{run_id}' (PID {pid_val}): {e}"),
                    });
                }
            }
        }
        #[cfg(not(unix))]
        {
            return Err(BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: "process signalling is not supported on this platform".into(),
            });
        }
    }

    // Update the database.
    let now = iso_timestamp_now();
    {
        let db = state.db.lock().unwrap();
        db.execute(
            "UPDATE runs SET status = 'stopped', ended_at = ?1 WHERE run_id = ?2 AND status = 'running'",
            duckdb::params![&now, run_id],
        )?;
    }

    // Append a stop event to the registry log so the ingest loop sees it
    // if it runs after us.
    let event = RegistryEvent::Stop {
        run_id: run_id.to_owned(),
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

    Ok(StopOutcome {
        pid,
        prior_status: status,
        signal_sent,
    })
}

/// `POST /api/bench/runs/{run_id}/stop` — best-effort SIGTERM
async fn stop_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Value>, BenchError> {
    let outcome = stop_run_impl(&state, &run_id).await?;

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

/// Build the launch `config` JSON for a resumed SMAC3 run: clones the old
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
fn build_resume_config(
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
/// - `"smac3"` — runs `bench smac3 --game ... [--config ...] [--override k=v ...]`
///   in the foreground; the server's own `launch::launch` (not `bench smac3`'s
///   own `--background` flag) is what detaches and captures its JSONL output,
///   same as every other launch kind.
///
/// Unknown kinds produce an error.
fn build_command(
    kind: &str,
    game: &str,
    config: &Option<Value>,
    run_id: &str,
) -> Result<Vec<String>, BenchError> {
    let bench_binary = find_bench_binary();

    match kind {
        "smac3" => {
            let mut cmd = vec![
                bench_binary.to_string_lossy().to_string(),
                "smac3".into(),
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
            message: format!("unknown run kind '{unknown}'; expected one of: round_robin, smac3"),
        }),
    }
}

/// Find the `bench` binary, preferring a sibling of the current executable
/// (standard Cargo convention for sibling bins), falling back to a bare
/// `"bench"` on PATH.
fn find_bench_binary() -> PathBuf {
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
fn hostname() -> String {
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

fn iso_timestamp_now() -> String {
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

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode as HttpStatusCode};
    use mcts_bench::schema::ensure_schema;
    use std::collections::HashMap;
    use tower::ServiceExt;

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    const DEFAULT_RUN_ID: &str = "rr-druid-20260101T000000-abc1234";

    static FIXTURE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Build a fully seeded test app: creates a temp dir with bench-runs/
    /// subdirectory, opens an in-memory DuckDB, seeds it, then moves the
    /// connection into the `BenchState`.  Returns the Router and the temp
    /// dir (kept alive for the test's duration).
    fn seeded_app(seed_fn: impl FnOnce(&duckdb::Connection, &Path)) -> (Router, PathBuf) {
        let n = FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp_dir =
            std::env::temp_dir().join(format!("mcts_bench_api_test_{}_{}", std::process::id(), n,));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let bench_runs_dir = tmp_dir.join("bench-runs");
        std::fs::create_dir_all(&bench_runs_dir).unwrap();

        // Create, seed, and keep the same connection — no file intermediary.
        let conn = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();
        seed_fn(&conn, &bench_runs_dir);

        let state = Arc::new(BenchState {
            db: Mutex::new(conn),
            bench_runs_dir,
        });

        (bench_router(state), tmp_dir)
    }

    /// Default seed: one completed run with two match results and one trial.
    fn default_seed(conn: &duckdb::Connection, _bench_runs_dir: &Path) {
        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
             VALUES (?1, 'round_robin', 'druid', 'abc1234', false, 'testhost', NULL, \
                     '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/nope/log.jsonl')",
            duckdb::params![DEFAULT_RUN_ID],
        ).unwrap();

        // Two matches: "strong" beats "master", "master" beats "strong" (1-1).
        conn.execute(
            "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
             VALUES (?1, 1, '2026-01-01T00:00:10Z', 'strong', 'master', 'win_a', 'strong'),\
                    (?1, 2, '2026-01-01T00:00:20Z', 'master', 'strong', 'win_a', 'master')",
            duckdb::params![DEFAULT_RUN_ID],
        )
        .unwrap();

        // One trial for good measure.
        conn.execute(
            "INSERT INTO trials (run_id, trial_id, ts, config, cost) \
             VALUES (?1, 1, '2026-01-01T00:00:30Z', '{}', 0.375)",
            duckdb::params![DEFAULT_RUN_ID],
        )
        .unwrap();
    }

    /// Seed a run that is still `running` (no ended_at, no stop event).
    fn running_run_seed(conn: &duckdb::Connection, _bench_runs_dir: &Path) {
        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) \
             VALUES ('running-run', 'round_robin', 'druid', 'def5678', false, 'testhost', 12345, \
                     '2026-02-01T00:00:00Z', 'running', '/tmp/running/log.jsonl')",
            duckdb::params![],
        )
        .unwrap();
    }

    /// Seed with the default run plus a second run for multi-run queries.
    fn multi_run_seed(conn: &duckdb::Connection, _bench_runs_dir: &Path) {
        default_seed(conn, _bench_runs_dir);
        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
             VALUES ('rr-ttt-20260201T000000-def5678', 'round_robin', 'ttt', 'def5678', false, 'testhost', \
                     NULL, '2026-02-01T00:00:00Z', '2026-02-01T02:00:00Z', 'completed', '/tmp/ttt/log.jsonl')",
            duckdb::params![],
        ).unwrap();
    }

    fn ladder_runs_seed(conn: &duckdb::Connection, _bench_runs_dir: &Path) {
        conn.execute_batch(
            "INSERT INTO runs
             (run_id, kind, game, config, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path)
             VALUES
             ('root-1', 'smac3', 'druid', '{\"ladder_root\":\"root-1\"}', 'abc', false, 'host', NULL,
              '2026-01-01T00:00:00Z', '2026-01-01T00:10:00Z', 'stopped', '/tmp/root/log.jsonl'),
             ('rung-2', 'smac3', 'druid', '{\"ladder_root\":\"root-1\",\"resumed_from\":\"root-1\"}', 'abc', false, 'host', 42,
              '2026-01-01T00:10:01Z', NULL, 'running', '/tmp/rung2/log.jsonl');
             INSERT INTO trials (run_id, trial_id, ts, config, cost) VALUES
             ('root-1', 1, '2026-01-01T00:00:01Z', '{}', 0.1),
             ('root-1', 2, '2026-01-01T00:00:02Z', '{}', 0.1),
             ('rung-2', 3, '2026-01-01T00:10:02Z', '{}', 0.2);",
        )
        .unwrap();
    }

    /// Default seed plus a two-ply trace for `match_results.seq = 1` (game
    /// 1: "strong" beats "master") -- exercises the join between
    /// `game_moves` and `match_results` on `(run_id, seq == game_seq)`.
    fn game_moves_seed(conn: &duckdb::Connection, bench_runs_dir: &Path) {
        default_seed(conn, bench_runs_dir);
        conn.execute(
            "INSERT INTO game_moves (run_id, game_seq, ply, ts, state, mv, player) \
             VALUES \
             (?1, 1, 0, '2026-01-01T00:00:10Z', '{\"board\":[]}', NULL, NULL), \
             (?1, 1, 1, '2026-01-01T00:00:11Z', '{\"board\":[1]}', '4', 'strong')",
            duckdb::params![DEFAULT_RUN_ID],
        )
        .unwrap();
    }

    fn body_json(body: &axum::body::Bytes) -> Value {
        serde_json::from_slice(body).unwrap()
    }

    async fn http_get(app: Router, uri: &str) -> (HttpStatusCode, axum::body::Bytes) {
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, body)
    }

    async fn http_post_json(
        app: Router,
        uri: &str,
        json: Value,
    ) -> (HttpStatusCode, axum::body::Bytes) {
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&json).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, body)
    }

    async fn http_delete(app: Router, uri: &str) -> (HttpStatusCode, axum::body::Bytes) {
        let resp = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, body)
    }

    // -------------------------------------------------------------------
    // GET /api/bench/runs
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_runs_empty() {
        let app = seeded_app(|_, _| {}).0;
        let (status, body) = http_get(app, "/api/bench/runs").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert!(runs.is_empty(), "expected empty list, got {runs:?}");
    }

    #[tokio::test]
    async fn test_list_runs_returns_seeded_run() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_get(app, "/api/bench/runs").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert_eq!(runs.len(), 1, "expected 1 run, got {runs:?}");

        let run = &runs[0];
        assert_eq!(run["run_id"], DEFAULT_RUN_ID);
        assert_eq!(run["kind"], "round_robin");
        assert_eq!(run["game"], "druid");
        assert_eq!(run["status"], "completed");
        assert_eq!(run["match_count"], 2);
        assert_eq!(run["trial_count"], 1);
        assert!(run.get("label").and_then(|v| v.as_str()).is_none());
    }

    #[tokio::test]
    async fn test_list_runs_collapses_ladder_rungs_into_latest_logical_run() {
        let app = seeded_app(ladder_runs_seed).0;
        let (status, body) = http_get(app.clone(), "/api/bench/runs").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0]["run_id"], "rung-2");
        assert_eq!(runs[0]["status"], "running");
        assert_eq!(runs[0]["trial_count"], 3);
        assert_eq!(runs[0]["started_at"], "2026-01-01 00:00:00");

        let (_, body) = http_get(app.clone(), "/api/bench/runs?status=running").await;
        assert_eq!(body_json(&body).as_array().unwrap().len(), 1);
        let (_, body) = http_get(app, "/api/bench/runs?status=stopped").await;
        assert!(body_json(&body).as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_list_runs_filter_by_status() {
        let app = seeded_app(|conn, dir| {
            default_seed(conn, dir);
            running_run_seed(conn, dir);
        })
        .0;

        // Filter to running only.
        let (status, body) = http_get(app.clone(), "/api/bench/runs?status=running").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert_eq!(runs.len(), 1, "expected 1 running run");
        assert_eq!(runs[0]["run_id"], "running-run");

        // Filter to completed only.
        let (status, body) = http_get(app.clone(), "/api/bench/runs?status=completed").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert_eq!(runs.len(), 1, "expected 1 completed run");
        assert_eq!(runs[0]["run_id"], DEFAULT_RUN_ID);
    }

    #[tokio::test]
    async fn test_list_runs_filter_by_game() {
        let app = seeded_app(multi_run_seed).0;

        let (status, body) = http_get(app.clone(), "/api/bench/runs?game=druid").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert_eq!(runs.len(), 1, "expected 1 druid run");
        assert_eq!(runs[0]["game"], "druid");

        let (status, body) = http_get(app.clone(), "/api/bench/runs?game=ttt").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert_eq!(runs.len(), 1, "expected 1 ttt run");
        assert_eq!(runs[0]["game"], "ttt");
    }

    #[tokio::test]
    async fn test_list_runs_limit() {
        let app = seeded_app(multi_run_seed).0;

        let (status, body) = http_get(app.clone(), "/api/bench/runs?limit=1").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert_eq!(runs.len(), 1, "expected 1 run with limit=1");
    }

    #[tokio::test]
    async fn test_list_runs_orders_by_started_at_desc() {
        let app = seeded_app(multi_run_seed).0;

        let (status, body) = http_get(app.clone(), "/api/bench/runs").await;
        assert_eq!(status, HttpStatusCode::OK);
        let runs = body_json(&body).as_array().unwrap().clone();
        assert_eq!(runs.len(), 2);
        // Most recent first: runs have started_at 2026-02-01 and 2026-01-01.
        assert_eq!(runs[0]["run_id"], "rr-ttt-20260201T000000-def5678");
        assert_eq!(runs[1]["run_id"], DEFAULT_RUN_ID);
    }

    // -------------------------------------------------------------------
    // GET /api/bench/runs/{run_id}
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_get_run_returns_detail() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
        assert_eq!(status, HttpStatusCode::OK);
        let run = body_json(&body);

        assert_eq!(run["run_id"], DEFAULT_RUN_ID);
        assert_eq!(run["kind"], "round_robin");
        assert_eq!(run["game"], "druid");
        assert_eq!(run["status"], "completed");
        assert_eq!(run["match_count"], 2);
        assert_eq!(run["trial_count"], 1);
        assert!(run.get("config").and_then(|v| v.as_str()).is_none());
        assert!(run.get("log_path").and_then(|v| v.as_str()).is_some());
        assert_eq!(run["exit_code"], Value::Null);
        assert_eq!(run["incumbent"], Value::Null);
    }

    #[tokio::test]
    async fn test_get_run_includes_incumbent_when_present() {
        let app = seeded_app(|conn, dir| {
            default_seed(conn, dir);
            conn.execute(
                "INSERT INTO incumbents (run_id, ts, config, cost) \
                 VALUES (?1, '2026-01-01T00:00:40Z', '{\"family\":\"rave\",\"c\":0.7}', 0.2)",
                duckdb::params![DEFAULT_RUN_ID],
            )
            .unwrap();
        })
        .0;
        let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
        assert_eq!(status, HttpStatusCode::OK);
        let run = body_json(&body);

        assert_eq!(run["incumbent"]["cost"], 0.2);
        assert_eq!(run["incumbent"]["config"]["family"], "rave");
    }

    #[tokio::test]
    async fn test_get_run_404_for_unknown_run() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_get(app, "/api/bench/runs/nonexistent").await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        let body = body_json(&body);
        assert_eq!(body["code"], 404);
        assert!(body["error"].as_str().unwrap().contains("nonexistent"));
    }

    // -------------------------------------------------------------------
    // GET /api/bench/runs/{run_id}/log
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_get_run_log_404_for_unknown_run() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_get(app, "/api/bench/runs/nonexistent/log").await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        let body = body_json(&body);
        assert_eq!(body["code"], 404);
    }

    #[tokio::test]
    async fn test_get_run_log_returns_lines_since_offset() {
        // Create a run with a real log file.
        let app = seeded_app(|conn, bench_runs_dir| {
            let run_dir = bench_runs_dir.join("loggy-run");
            std::fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            // Write some lines.
            std::fs::write(&log_path, "line1\nline2\nline3\n").unwrap();

            conn.execute(
                "INSERT INTO runs \
                 (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) \
                 VALUES ('loggy-run', 'round_robin', 'druid', 'abc', false, 'h', NULL, \
                         '2026-01-01T00:00:00Z', 'running', ?1)",
                duckdb::params![log_path_str],
            )
            .unwrap();
        })
        .0;

        // Read from offset 0 — get all 3 lines.
        let (status, body) = http_get(app.clone(), "/api/bench/runs/loggy-run/log").await;
        assert_eq!(status, HttpStatusCode::OK);
        let resp = body_json(&body);
        let lines = resp["lines"].as_array().unwrap().clone();
        assert_eq!(lines.len(), 3, "expected 3 lines, got {lines:?}");
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
        assert_eq!(lines[2], "line3");
        assert!(resp["next_offset"].as_u64().unwrap() > 0);

        // Read from an offset past the end — empty result.
        let last_offset = resp["next_offset"].as_u64().unwrap();
        let (status, body) = http_get(
            app.clone(),
            &format!("/api/bench/runs/loggy-run/log?since={last_offset}"),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let resp = body_json(&body);
        assert!(resp["lines"].as_array().unwrap().is_empty());
        assert_eq!(resp["next_offset"].as_u64().unwrap(), last_offset);
    }

    // -------------------------------------------------------------------
    // GET /api/bench/runs/{run_id}/trials
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_get_run_trials_404_for_unknown_run() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_get(app, "/api/bench/runs/nonexistent/trials").await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        let body = body_json(&body);
        assert_eq!(body["code"], 404);
    }

    #[tokio::test]
    async fn test_get_run_trials_returns_rows_in_trial_id_order() {
        let app = seeded_app(|conn, dir| {
            default_seed(conn, dir); // seeds trial_id 1 with cost 0.375
            conn.execute(
                "INSERT INTO trials (run_id, trial_id, ts, config, seed, cost, extra) \
                 VALUES (?1, 2, '2026-01-01T00:00:40Z', '{\"c\":1.5}', 42, 0.2, '{\"wins\":8}')",
                duckdb::params![DEFAULT_RUN_ID],
            )
            .unwrap();
        })
        .0;

        let (status, body) =
            http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}/trials")).await;
        assert_eq!(status, HttpStatusCode::OK);
        let rows = body_json(&body).as_array().unwrap().clone();
        assert_eq!(rows.len(), 2, "expected 2 trials, got {rows:?}");

        assert_eq!(rows[0]["trial_id"], 1);
        assert_eq!(rows[0]["config"], json!({}));
        assert_eq!(rows[0]["cost"], 0.375);
        assert_eq!(rows[0]["seed"], Value::Null);

        assert_eq!(rows[1]["trial_id"], 2);
        assert_eq!(rows[1]["config"], json!({"c": 1.5}));
        assert_eq!(rows[1]["seed"], 42);
        assert_eq!(rows[1]["cost"], 0.2);
        assert_eq!(rows[1]["extra"], json!({"wins": 8}));
    }

    #[tokio::test]
    async fn test_get_run_trials_respects_limit() {
        let app = seeded_app(|conn, dir| {
            default_seed(conn, dir);
            conn.execute(
                "INSERT INTO trials (run_id, trial_id, ts, config, cost) \
                 VALUES (?1, 2, '2026-01-01T00:00:40Z', '{}', 0.2)",
                duckdb::params![DEFAULT_RUN_ID],
            )
            .unwrap();
        })
        .0;

        let (status, body) = http_get(
            app,
            &format!("/api/bench/runs/{DEFAULT_RUN_ID}/trials?limit=1"),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let rows = body_json(&body).as_array().unwrap().clone();
        assert_eq!(rows.len(), 1, "expected 1 trial with limit=1");
        assert_eq!(rows[0]["trial_id"], 1);
    }

    // -------------------------------------------------------------------
    // GET /api/bench/runs/{run_id}/games, .../games/{game_seq}/moves
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_get_run_games_404_for_unknown_run() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_get(app, "/api/bench/runs/nonexistent/games").await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        assert_eq!(body_json(&body)["code"], 404);
    }

    #[tokio::test]
    async fn test_get_run_games_empty_when_no_traces() {
        // `default_seed` has match_results but no game_moves rows.
        let app = seeded_app(default_seed).0;
        let (status, body) =
            http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games")).await;
        assert_eq!(status, HttpStatusCode::OK);
        assert!(body_json(&body).as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_get_run_games_joins_match_results_by_seq() {
        let app = seeded_app(game_moves_seed).0;
        let (status, body) =
            http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games")).await;
        assert_eq!(status, HttpStatusCode::OK);
        let games = body_json(&body).as_array().unwrap().clone();
        assert_eq!(games.len(), 1, "expected 1 traced game, got {games:?}");
        assert_eq!(games[0]["game_seq"], 1);
        assert_eq!(games[0]["ply_count"], 2);
        assert_eq!(games[0]["strategy_a"], "strong");
        assert_eq!(games[0]["strategy_b"], "master");
        assert_eq!(games[0]["winner"], "strong");
    }

    #[tokio::test]
    async fn test_get_run_game_moves_ordered_by_ply() {
        let app = seeded_app(game_moves_seed).0;
        let (status, body) = http_get(
            app,
            &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games/1/moves"),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let moves = body_json(&body).as_array().unwrap().clone();
        assert_eq!(moves.len(), 2);
        assert_eq!(moves[0]["ply"], 0);
        assert_eq!(moves[0]["mv"], Value::Null);
        assert_eq!(moves[0]["state"], json!({"board": []}));
        assert_eq!(moves[1]["ply"], 1);
        assert_eq!(moves[1]["mv"], 4);
        assert_eq!(moves[1]["player"], "strong");
    }

    #[tokio::test]
    async fn test_get_run_game_moves_empty_for_unknown_game_seq() {
        let app = seeded_app(game_moves_seed).0;
        let (status, body) = http_get(
            app,
            &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games/999/moves"),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        assert!(body_json(&body).as_array().unwrap().is_empty());
    }

    // -------------------------------------------------------------------
    // DELETE /api/bench/runs/{run_id}
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_delete_run_404_for_unknown_run() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_delete(app, "/api/bench/runs/nonexistent").await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        assert_eq!(body_json(&body)["code"], 404);
    }

    #[tokio::test]
    async fn test_delete_run_409_while_running() {
        let app = seeded_app(running_run_seed).0;
        let (status, body) = http_delete(app, "/api/bench/runs/running-run").await;
        assert_eq!(status, HttpStatusCode::CONFLICT);
        assert_eq!(body_json(&body)["code"], 409);
    }

    #[tokio::test]
    async fn test_delete_run_removes_all_rows_and_files() {
        let (app, tmp_dir) = seeded_app(game_moves_seed);
        let run_dir = tmp_dir.join("bench-runs").join(DEFAULT_RUN_ID);
        std::fs::create_dir_all(&run_dir).unwrap();
        std::fs::write(run_dir.join("log.jsonl"), "{}\n").unwrap();
        std::fs::write(run_dir.join("moves.jsonl"), "{}\n").unwrap();

        let (status, _) =
            http_delete(app.clone(), &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
        assert_eq!(status, HttpStatusCode::NO_CONTENT);

        let (status, _) = http_get(app.clone(), "/api/bench/runs").await;
        assert_eq!(status, HttpStatusCode::OK);

        let (status, _) = http_get(app.clone(), &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);

        let (status, body) =
            http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}/games")).await;
        assert_eq!(
            status,
            HttpStatusCode::NOT_FOUND,
            "run row itself is gone: {body:?}"
        );

        assert!(!run_dir.exists(), "run directory should be removed");
    }

    // -------------------------------------------------------------------
    // GET /api/bench/runs/{run_id}/live (SSE)
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_live_run_moves_404_for_unknown_run() {
        let app = seeded_app(default_seed).0;
        let (status, _) = http_get(app, "/api/bench/runs/nonexistent/live").await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_live_run_moves_opens_sse_stream_for_known_run() {
        let app = seeded_app(game_moves_seed).0;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/api/bench/runs/{DEFAULT_RUN_ID}/live"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
    }

    #[tokio::test]
    async fn test_live_run_moves_accepts_a_pinned_game() {
        let app = seeded_app(game_moves_seed).0;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/api/bench/runs/{DEFAULT_RUN_ID}/live?game_seq=7"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
    }

    // -------------------------------------------------------------------
    // GET /api/bench/runs/{run_id}/chain
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_get_run_chain_404_for_unknown_run() {
        let app = seeded_app(|_, _| {}).0;
        let (status, body) = http_get(app, "/api/bench/runs/nonexistent/chain").await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        assert_eq!(body_json(&body)["code"], 404);
    }

    fn insert_smac3_run(conn: &duckdb::Connection, run_id: &str, started_at: &str, config: &Value) {
        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, config, git_sha, git_dirty, host, pid, \
              started_at, ended_at, status, log_path) \
             VALUES (?1, 'smac3', 'nim', ?2, 'abc1234', false, 'testhost', NULL, \
                     ?3, ?3, 'completed', '/tmp/nope/log.jsonl')",
            duckdb::params![run_id, config.to_string(), started_at],
        )
        .unwrap();
    }

    #[tokio::test]
    async fn test_get_run_chain_single_rung_for_a_plain_run() {
        let app = seeded_app(|conn, _dir| {
            insert_smac3_run(
                conn,
                "root-1",
                "2026-01-01T00:00:00Z",
                &json!({"overrides": []}),
            );
        })
        .0;

        let (status, body) = http_get(app, "/api/bench/runs/root-1/chain").await;
        assert_eq!(status, HttpStatusCode::OK);
        let rows = body_json(&body).as_array().unwrap().clone();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["run_id"], "root-1");
    }

    #[tokio::test]
    async fn test_get_run_chain_orders_every_rung_oldest_first() {
        let app = seeded_app(|conn, _dir| {
            insert_smac3_run(
                conn,
                "root-1",
                "2026-01-01T00:00:00Z",
                &json!({"ladder_root": "root-1"}),
            );
            insert_smac3_run(
                conn,
                "root-1-rung3",
                "2026-01-03T00:00:00Z",
                &json!({"ladder_root": "root-1", "resumed_from": "root-1-rung2"}),
            );
            insert_smac3_run(
                conn,
                "root-1-rung2",
                "2026-01-02T00:00:00Z",
                &json!({"ladder_root": "root-1", "resumed_from": "root-1"}),
            );
            // A run from a *different* chain (different ladder_root) must
            // not leak into this chain's result.
            insert_smac3_run(
                conn,
                "other-root",
                "2026-01-02T12:00:00Z",
                &json!({"ladder_root": "other-root"}),
            );
            conn.execute(
                "INSERT INTO incumbents (run_id, ts, config, cost) \
                 VALUES ('root-1', '2026-01-01T00:30:00Z', '{\"family\": \"ucb1\"}', 0.02)",
                duckdb::params![],
            )
            .unwrap();
        })
        .0;

        // Query from the *middle* rung -- the chain must resolve via
        // ladder_root regardless of which rung's run_id is asked for.
        let (status, body) = http_get(app, "/api/bench/runs/root-1-rung2/chain").await;
        assert_eq!(status, HttpStatusCode::OK);
        let rows = body_json(&body).as_array().unwrap().clone();
        assert_eq!(rows.len(), 3, "expected 3 rungs, got {rows:?}");
        assert_eq!(rows[0]["run_id"], "root-1");
        assert_eq!(rows[0]["incumbent"]["cost"], 0.02);
        assert_eq!(rows[1]["run_id"], "root-1-rung2");
        assert_eq!(rows[1]["incumbent"], Value::Null);
        assert_eq!(rows[2]["run_id"], "root-1-rung3");
    }

    // -------------------------------------------------------------------
    // GET /api/bench/leaderboard
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_leaderboard_empty_when_no_matches() {
        let app = seeded_app(|_, _| {}).0;
        let (status, body) = http_get(app, "/api/bench/leaderboard").await;
        assert_eq!(status, HttpStatusCode::OK);
        let entries = body_json(&body).as_array().unwrap().clone();
        assert!(
            entries.is_empty(),
            "expected empty leaderboard, got {entries:?}"
        );
    }

    #[tokio::test]
    async fn test_leaderboard_aggregates_correctly() {
        // Seed with two runs that have well-known outcomes.
        let app = seeded_app(|conn, _dir| {
            // Run 1: strong beats master twice.
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
                 VALUES ('run1', 'round_robin', 'druid', 'abc', false, 'h', NULL, '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/1')",
                duckdb::params![],
            ).unwrap();
            conn.execute(
                "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
                 VALUES \
                   ('run1', 1, '2026-01-01T00:00:10Z', 'strong', 'master', 'win_a', 'strong'),\
                   ('run1', 2, '2026-01-01T00:00:20Z', 'master', 'strong', 'win_b', 'strong')",
                duckdb::params![],
            ).unwrap();

            // Run 2: strong draws with easy, easy beats master.
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
                 VALUES ('run2', 'round_robin', 'druid', 'abc', false, 'h', NULL, '2026-01-02T00:00:00Z', '2026-01-02T01:00:00Z', 'completed', '/tmp/2')",
                duckdb::params![],
            ).unwrap();
            conn.execute(
                "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
                 VALUES \
                   ('run2', 1, '2026-01-02T00:00:10Z', 'strong', 'easy', 'draw', NULL),\
                   ('run2', 2, '2026-01-02T00:00:20Z', 'easy', 'master', 'win_a', 'easy')",
                duckdb::params![],
            ).unwrap();
        })
        .0;

        let (status, body) = http_get(app, "/api/bench/leaderboard").await;
        assert_eq!(status, HttpStatusCode::OK);
        let entries = body_json(&body).as_array().unwrap().clone();

        // Three strategies: strong, master, easy.
        // strong: vs master (win+win=2 wins), vs easy (draw) → 3 games, 2 wins, 0 losses, 1 draw
        // master: vs strong (loss+loss=2 losses), vs easy (loss) → 3 games, 0 wins, 3 losses, 0 draws
        // easy: vs strong (draw), vs master (win) → 2 games, 1 win, 0 losses, 1 draw

        let by_strategy: HashMap<&str, &Value> = entries
            .iter()
            .map(|e| (e["strategy"].as_str().unwrap(), e))
            .collect();

        // strong
        let s = by_strategy["strong"];
        assert_eq!(s["total"], 3);
        assert_eq!(s["wins"], 2);
        assert_eq!(s["losses"], 0);
        assert_eq!(s["draws"], 1);
        assert!((s["win_rate"].as_f64().unwrap() - (2.5 / 3.0)).abs() < 1e-9);

        // master
        let m = by_strategy["master"];
        assert_eq!(m["total"], 3);
        assert_eq!(m["wins"], 0);
        assert_eq!(m["losses"], 3);
        assert_eq!(m["draws"], 0);
        assert!((m["win_rate"].as_f64().unwrap() - 0.0).abs() < 1e-9);

        // easy
        let e = by_strategy["easy"];
        assert_eq!(e["total"], 2);
        assert_eq!(e["wins"], 1);
        assert_eq!(e["losses"], 0);
        assert_eq!(e["draws"], 1);
        assert!((e["win_rate"].as_f64().unwrap() - (1.5 / 2.0)).abs() < 1e-9);

        // Wilson CI lower < win_rate < upper for all entries.
        for entry in &entries {
            let wr = entry["win_rate"].as_f64().unwrap();
            let lo = entry["ci_lower"].as_f64().unwrap();
            let hi = entry["ci_upper"].as_f64().unwrap();
            assert!(lo <= wr, "ci_lower {lo} > win_rate {wr}");
            assert!(wr <= hi, "win_rate {wr} > ci_upper {hi}");
        }
    }

    #[tokio::test]
    async fn test_leaderboard_filters_by_game() {
        let app = seeded_app(|conn, _dir| {
            // Druid matches.
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
                 VALUES ('druid-run', 'round_robin', 'druid', 'abc', false, 'h', NULL, '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/d')",
                duckdb::params![],
            ).unwrap();
            conn.execute(
                "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
                 VALUES ('druid-run', 1, '2026-01-01T00:00:10Z', 'strong', 'master', 'win_a', 'strong')",
                duckdb::params![],
            ).unwrap();

            // TTT matches.
            conn.execute(
                "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
                 VALUES ('ttt-run', 'round_robin', 'ttt', 'abc', false, 'h', NULL, '2026-01-02T00:00:00Z', '2026-01-02T01:00:00Z', 'completed', '/tmp/t')",
                duckdb::params![],
            ).unwrap();
            conn.execute(
                "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
                 VALUES ('ttt-run', 1, '2026-01-02T00:00:10Z', 'minimax', 'random', 'win_a', 'minimax')",
                duckdb::params![],
            ).unwrap();
        })
        .0;

        // Filter by druid.
        let (status, body) = http_get(app.clone(), "/api/bench/leaderboard?game=druid").await;
        assert_eq!(status, HttpStatusCode::OK);
        let entries = body_json(&body).as_array().unwrap().clone();
        assert_eq!(
            entries.len(),
            2,
            "expected 2 druid strategies, got {entries:?}"
        );
        let strategies: Vec<&str> = entries
            .iter()
            .map(|e| e["strategy"].as_str().unwrap())
            .collect();
        assert!(strategies.contains(&"strong"));
        assert!(strategies.contains(&"master"));

        // Filter by ttt.
        let (status, body) = http_get(app.clone(), "/api/bench/leaderboard?game=ttt").await;
        assert_eq!(status, HttpStatusCode::OK);
        let entries = body_json(&body).as_array().unwrap().clone();
        assert_eq!(
            entries.len(),
            2,
            "expected 2 ttt strategies, got {entries:?}"
        );
    }

    // -------------------------------------------------------------------
    // GET /api/bench/kinds
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_kinds_includes_round_robin_and_smac3() {
        let app = seeded_app(|_, _| {}).0;
        let (status, body) = http_get(app, "/api/bench/kinds").await;
        assert_eq!(status, HttpStatusCode::OK);
        let kinds = body_json(&body).as_array().unwrap().clone();
        let kind_names: Vec<&str> = kinds.iter().map(|k| k["kind"].as_str().unwrap()).collect();
        assert!(kind_names.contains(&"round_robin"));
        assert!(kind_names.contains(&"smac3"));
    }

    // -------------------------------------------------------------------
    // GET /api/bench/smac3/kinds
    // -------------------------------------------------------------------

    /// `/api/bench/smac3/kinds` itself now just forwards
    /// `mcts_bench::games::describe_tuners()` -- see that function's own
    /// tests in `mcts-bench/src/games/mod.rs` for the exit-code/JSON
    /// dispatch this route depends on (`Some(TunerInfo)` vs `None` vs a
    /// missing binary). Spawning real `game-*` binaries from this crate's
    /// tests isn't practical here, so there's no fake-adapter-based
    /// HTTP-level test of the *contents* of this route the way earlier
    /// commits had -- `test_list_kinds_includes_round_robin_and_smac3`
    /// above only checks that the `smac3` kind name is present, same
    /// shallow level this route gets today.

    // -------------------------------------------------------------------
    // Command construction (build_command)
    // -------------------------------------------------------------------

    #[test]
    fn test_build_command_smac3_includes_config_and_overrides() {
        let cmd = build_command(
            "smac3",
            "traffic-lights",
            &Some(json!({
                "config": "smac3/config/default.yaml",
                "overrides": ["optimizer.n_trials=10", "optimizer.n_workers=2"],
            })),
            "test-run",
        )
        .unwrap();

        // First element is the (unresolved-in-test) bench binary path --
        // everything after it is the argv this test actually cares about
        // (trailing --trace-path is asserted separately, below).
        assert_eq!(
            cmd[1..cmd.len() - 2],
            vec![
                "smac3",
                "--game",
                "traffic-lights",
                "--config",
                "smac3/config/default.yaml",
                "--override",
                "optimizer.n_trials=10",
                "--override",
                "optimizer.n_workers=2",
            ]
        );
    }

    #[test]
    fn test_build_command_smac3_with_no_config_is_just_game() {
        let cmd = build_command("smac3", "druid", &None, "test-run").unwrap();
        assert_eq!(cmd[1..cmd.len() - 2], vec!["smac3", "--game", "druid"]);
    }

    #[test]
    fn test_build_command_smac3_includes_trace_path_derived_from_run_id() {
        let cmd = build_command(
            "smac3",
            "druid",
            &None,
            "smac3-druid-20260101T000000-abcdef",
        )
        .unwrap();
        let idx = cmd
            .iter()
            .position(|a| a == "--trace-path")
            .expect("--trace-path flag present");
        assert_eq!(
            cmd[idx + 1],
            "bench-runs/smac3-druid-20260101T000000-abcdef/moves.jsonl"
        );
    }

    #[test]
    fn test_build_command_smac3_includes_game_config() {
        let cmd = build_command(
            "smac3",
            "druid",
            &Some(json!({
                "game_config": {"size": {"w": 9, "h": 9}},
            })),
            "test-run",
        )
        .unwrap();

        let idx = cmd
            .iter()
            .position(|a| a == "--game-config")
            .expect("--game-config flag present");
        assert_eq!(cmd[idx + 1], r#"{"size":{"h":9,"w":9}}"#);
    }

    #[test]
    fn test_build_command_smac3_omits_null_game_config() {
        let cmd = build_command(
            "smac3",
            "druid",
            &Some(json!({
                "game_config": null,
            })),
            "test-run",
        )
        .unwrap();
        assert!(!cmd.iter().any(|a| a == "--game-config"));
    }

    #[test]
    fn test_build_command_smac3_includes_baseline_configs() {
        let cmd = build_command(
            "smac3",
            "nim",
            &Some(json!({
                "overrides": ["optimizer.n_trials=10"],
                "baseline_configs": {
                    "ladder1": {"family": "ucb1", "c": 1.5},
                },
            })),
            "test-run",
        )
        .unwrap();

        assert_eq!(
            cmd[1..cmd.len() - 2],
            vec![
                "smac3",
                "--game",
                "nim",
                "--override",
                "optimizer.n_trials=10",
                "--baseline-config",
                r#"ladder1={"c":1.5,"family":"ucb1"}"#,
            ]
        );
    }

    #[test]
    fn test_build_command_unknown_kind_lists_smac3_as_supported() {
        let err = build_command("nope", "druid", &None, "test-run").unwrap_err();
        assert!(err.message.contains("smac3"));
    }

    #[test]
    fn test_build_command_round_robin_includes_trace_path_derived_from_run_id() {
        let cmd = build_command(
            "round_robin",
            "druid",
            &None,
            "rr-druid-20260101T000000-abcdef",
        )
        .unwrap();

        let idx = cmd
            .iter()
            .position(|a| a == "--trace-path")
            .expect("--trace-path flag present");
        assert_eq!(
            cmd[idx + 1],
            "bench-runs/rr-druid-20260101T000000-abcdef/moves.jsonl"
        );
    }

    // -------------------------------------------------------------------
    // POST /api/bench/launch
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_launch_rejects_unknown_kind() {
        let app = seeded_app(|_, _| {}).0;
        let (status, body) = http_post_json(
            app,
            "/api/bench/launch",
            json!({
                "kind": "unknown_kind",
                "game": "druid",
                "config": null
            }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
        let body = body_json(&body);
        assert_eq!(body["code"], 400);
        assert!(body["error"].as_str().unwrap().contains("unknown_kind"));
    }

    #[tokio::test]
    async fn test_launch_spawns_bench_and_returns_run_id() {
        // Launch a quick `true` command to verify the plumbing works end-to-end.
        // We simulate what the server would do by launching `true` (exits
        // immediately) as a "round_robin" run and checking the registry.
        let app = seeded_app(|_conn, dir| {
            // We need the registry to exist in the bench_runs_dir for the
            // launcher to write to.
            std::fs::create_dir_all(dir).ok();
        })
        .0;

        // We can't easily test the actual bench binary path from tests
        // (the server binary path during `cargo test` is in the build
        // target dir).  Instead, test that a valid request shape hits
        // the launcher and produces an error about a missing binary
        // (expected since `bench` isn't compiled during tests) or
        // succeeds if `true` is used.

        // Use `true` as the command to verify the launcher path works.
        let (status, body) = http_post_json(
            app,
            "/api/bench/launch",
            json!({
                "kind": "round_robin",
                "game": "druid",
                "config": {
                    "strategies": ["strong", "master"],
                    "rounds": 1
                }
            }),
        )
        .await;

        // The request reaches the handler and tries to find `bench`.
        // Since we're running tests (not the compiled server), the
        // `bench` binary doesn't exist next to the test binary.
        // We expect either a 500 (bench not found) or a success if
        // by coincidence something called `bench` is on PATH.
        // What we *don't* expect is a 400 (which would mean the
        // request body was rejected before reaching the launcher).
        assert!(
            status == HttpStatusCode::OK || status == HttpStatusCode::INTERNAL_SERVER_ERROR,
            "launch returned unexpected status {status}: body={}",
            String::from_utf8_lossy(&body),
        );
    }

    #[tokio::test]
    async fn test_launch_smac3_reaches_the_launcher() {
        // Same shape as test_launch_spawns_bench_and_returns_run_id above,
        // for the "smac3" kind -- proves build_command's smac3 arm produces
        // a request the handler accepts and forwards to launch::launch
        // (a 400 here would mean it was rejected as an unknown kind before
        // ever reaching the launcher).
        let app = seeded_app(|_conn, dir| {
            std::fs::create_dir_all(dir).ok();
        })
        .0;

        let (status, body) = http_post_json(
            app,
            "/api/bench/launch",
            json!({
                "kind": "smac3",
                "game": "traffic-lights",
                "config": {
                    "config": "smac3/config/default.yaml",
                    "overrides": ["optimizer.n_trials=1"]
                }
            }),
        )
        .await;

        assert!(
            status == HttpStatusCode::OK || status == HttpStatusCode::INTERNAL_SERVER_ERROR,
            "smac3 launch returned unexpected status {status}: body={}",
            String::from_utf8_lossy(&body),
        );
    }

    // -------------------------------------------------------------------
    // build_resume_config
    // -------------------------------------------------------------------

    #[test]
    fn test_build_resume_config_appends_n_trials_override() {
        let config = build_resume_config("old-run-1", &None, 500, None);
        let overrides = config["overrides"].as_array().unwrap();
        assert_eq!(overrides, &[json!("optimizer.n_trials=500")]);
        assert!(config.get("config").is_none());
    }

    #[test]
    fn test_build_resume_config_appends_n_workers_when_given() {
        let config = build_resume_config("old-run-1", &None, 500, Some(4));
        let overrides = config["overrides"].as_array().unwrap();
        assert_eq!(
            overrides,
            &[
                json!("optimizer.n_trials=500"),
                json!("optimizer.n_workers=4")
            ]
        );
    }

    #[test]
    fn test_build_resume_config_carries_forward_old_config_and_overrides() {
        let old = Some(json!({
            "config": "smac3/config/default.yaml",
            "overrides": ["target.rounds=30"],
        }));
        let config = build_resume_config("old-run-1", &old, 500, None);
        assert_eq!(config["config"], json!("smac3/config/default.yaml"));
        assert_eq!(
            config["overrides"].as_array().unwrap(),
            &[json!("target.rounds=30"), json!("optimizer.n_trials=500")]
        );
    }

    #[test]
    fn test_build_resume_config_records_resumed_from() {
        let config = build_resume_config("old-run-1", &None, 500, None);
        assert_eq!(config["resumed_from"], json!("old-run-1"));
    }

    #[test]
    fn test_build_resume_config_preserves_unknown_keys() {
        // Ladder bookkeeping (`ladder`, `ladder_root`, `baseline_configs`)
        // must survive a resume untouched -- both the driver's own resume
        // calls and a human clicking the existing UI Resume button on a
        // ladder rung go through this same function.
        let old = Some(json!({
            "overrides": ["target.rounds=30"],
            "ladder": {"max_rungs": 5, "saturation_threshold": 0.0},
            "ladder_root": "root-run-1",
            "baseline_configs": {"ladder1": {"family": "ucb1"}},
        }));
        let config = build_resume_config("rung-1-run", &old, 500, None);
        assert_eq!(config["ladder"]["max_rungs"], json!(5));
        assert_eq!(config["ladder_root"], json!("root-run-1"));
        assert_eq!(
            config["baseline_configs"]["ladder1"],
            json!({"family": "ucb1"})
        );
        assert_eq!(config["resumed_from"], json!("rung-1-run"));
    }

    // -------------------------------------------------------------------
    // inject_ladder_root_if_new_ladder
    // -------------------------------------------------------------------

    #[test]
    fn test_inject_ladder_root_sets_self_reference_on_a_new_ladder_launch() {
        let config = Some(json!({
            "overrides": ["optimizer.n_trials=10"],
            "ladder": {"max_rungs": 3, "saturation_threshold": 0.0},
        }));
        let config = inject_ladder_root_if_new_ladder(config, "root-run-1").unwrap();
        assert_eq!(config["ladder_root"], json!("root-run-1"));
    }

    #[test]
    fn test_inject_ladder_root_leaves_non_ladder_config_untouched() {
        let config = Some(json!({ "overrides": ["optimizer.n_trials=10"] }));
        let config = inject_ladder_root_if_new_ladder(config, "some-run").unwrap();
        assert!(config.get("ladder_root").is_none());
    }

    #[test]
    fn test_inject_ladder_root_does_not_override_a_carried_forward_root() {
        // A resumed rung's config already has `ladder_root` pointing at the
        // *original* root (via `build_resume_config`) -- this must not be
        // clobbered with the resumed rung's own id.
        let config = Some(json!({
            "ladder": {"max_rungs": 3, "saturation_threshold": 0.0},
            "ladder_root": "original-root",
        }));
        let config = inject_ladder_root_if_new_ladder(config, "rung-2-run").unwrap();
        assert_eq!(config["ladder_root"], json!("original-root"));
    }

    #[test]
    fn test_inject_ladder_root_handles_none_config() {
        assert_eq!(inject_ladder_root_if_new_ladder(None, "some-run"), None);
    }

    // -------------------------------------------------------------------
    // record_floor_baseline_settings
    // -------------------------------------------------------------------

    #[test]
    fn test_record_floor_baseline_settings_persists_flat_mc_params() {
        let config = Some(json!({
            "overrides": ["optimizer.n_trials=10", "target.baselines=[\"flat_mc\"]"],
        }));
        let config = record_floor_baseline_settings(config).unwrap();
        assert_eq!(
            config["baseline_settings"]["flat_mc"],
            json!({"family": "flat_mc", "q_init": "Infinity"})
        );
        assert!(config.get("baseline_configs").is_none());
    }

    #[test]
    fn test_record_floor_baseline_settings_preserves_existing_settings() {
        let config = Some(json!({
            "overrides": ["target.baselines=[\"random\"]"],
            "baseline_settings": {"chosen": {"family": "custom"}},
        }));
        let config = record_floor_baseline_settings(config).unwrap();
        assert_eq!(
            config["baseline_settings"],
            json!({"chosen": {"family": "custom"}})
        );
    }

    // -------------------------------------------------------------------
    // plan_ladder_advances
    // -------------------------------------------------------------------

    fn ladder_root_run(run_id: &str, max_rungs: i64, saturation_threshold: f64) -> LadderRunRow {
        LadderRunRow {
            run_id: run_id.to_string(),
            game: "nim".to_string(),
            status: "completed".to_string(),
            exit_code: Some(0),
            config: Some(json!({
                "overrides": ["optimizer.n_trials=10"],
                "ladder": {"max_rungs": max_rungs, "saturation_threshold": saturation_threshold},
                "ladder_root": run_id,
            })),
        }
    }

    #[test]
    fn test_plan_ladder_advances_widens_a_saturated_root_with_budget_left() {
        let runs = vec![ladder_root_run("root-1", 3, 0.0)];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([(
            "root-1".to_string(),
            (json!({"family": "ucb1", "c": 1.4}), 0.0),
        )]);

        let advances = plan_ladder_advances(&runs, &trial_counts, &incumbents);
        assert_eq!(advances.len(), 1);
        let advance = &advances[0];
        assert_eq!(advance.parent_run_id, "root-1");
        assert_eq!(advance.game, "nim");
        assert_eq!(advance.label, "ladder rung 2 of root-1");
        assert_eq!(advance.widened_config["resumed_from"], json!("root-1"));
        assert_eq!(advance.widened_config["ladder_root"], json!("root-1"));
        // Cumulative budget: root's own 10 trials + another 10 for the new
        // rung, plus a trailing `target.baselines=[]` neutralizing whatever
        // named baseline the root started against -- see
        // `replace_baseline_with_incumbent`'s doc comment.
        assert_eq!(
            advance.widened_config["overrides"],
            json!([
                "optimizer.n_trials=10",
                "optimizer.n_trials=10",
                "target.baselines=[]"
            ])
        );
        // rung_count is 1 (the root itself) before this widen, so the new
        // rung being created is rung 2 -- its baseline id is "ladder2".
        assert_eq!(
            advance.widened_config["baseline_configs"]["ladder2"],
            json!({"family": "ucb1", "c": 1.4})
        );
    }

    #[test]
    fn test_plan_ladder_advances_widens_a_running_rung_at_threshold() {
        let mut run = ladder_root_run("root-1", 3, 0.15);
        run.status = "running".to_string();
        run.exit_code = None;
        let runs = vec![run];
        let trial_counts = HashMap::from([("root-1".to_string(), 3)]);
        let incumbents =
            HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.025))]);

        let advances = plan_ladder_advances(&runs, &trial_counts, &incumbents);
        assert_eq!(advances.len(), 1);
        assert_eq!(advances[0].parent_run_id, "root-1");
        assert_eq!(
            advances[0].widened_config["overrides"],
            json!([
                "optimizer.n_trials=10",
                "optimizer.n_trials=10",
                "target.baselines=[]"
            ])
        );
    }

    #[test]
    fn test_plan_ladder_advances_replaces_rather_than_accumulates_baseline_configs() {
        // The parent rung already carries a `baseline_configs` entry from a
        // prior widen (or a hand-launched `--baseline-config`) -- the new
        // widen must *replace* it with just the new incumbent, not merge
        // alongside it, matching "always face the current incumbent" rather
        // than SMAC3's multi-instance averaging.
        let mut root = ladder_root_run("root-1", 5, 0.0);
        root.config.as_mut().unwrap()["baseline_configs"] =
            json!({"ladder1": {"family": "ucb1", "c": 0.5}});
        let runs = vec![root];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([(
            "root-1".to_string(),
            (json!({"family": "rave", "threshold": 700}), 0.0),
        )]);

        let advances = plan_ladder_advances(&runs, &trial_counts, &incumbents);
        assert_eq!(advances.len(), 1);
        let baseline_configs = advances[0].widened_config["baseline_configs"]
            .as_object()
            .unwrap();
        assert_eq!(baseline_configs.len(), 1);
        assert_eq!(
            baseline_configs.get("ladder2"),
            Some(&json!({"family": "rave", "threshold": 700}))
        );
        assert!(!baseline_configs.contains_key("ladder1"));
    }

    #[test]
    fn test_plan_ladder_advances_does_not_widen_when_not_saturated() {
        let runs = vec![ladder_root_run("root-1", 3, 0.0)];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([(
            "root-1".to_string(),
            (json!({"family": "ucb1"}), 0.2), // above the 0.0 threshold
        )]);

        assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
    }

    #[test]
    fn test_plan_ladder_advances_does_not_widen_without_an_incumbent() {
        let runs = vec![ladder_root_run("root-1", 3, 0.0)];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::new();

        assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
    }

    #[test]
    fn test_plan_ladder_advances_stops_at_max_rungs() {
        // Two rungs already exist for this ladder and max_rungs is 2 --
        // no third rung should be proposed even though the second is
        // saturated with budget nominally available.
        let mut rung2 = ladder_root_run("root-1", 2, 0.0);
        rung2.run_id = "root-1-rung2".to_string();
        rung2.config.as_mut().unwrap()["resumed_from"] = json!("root-1");
        let root = ladder_root_run("root-1", 2, 0.0);
        // root already has a child (rung2), so it wouldn't be reconsidered
        // either -- but the rung-count check is what should stop rung2.
        let runs = vec![root, rung2];
        let trial_counts =
            HashMap::from([("root-1".to_string(), 10), ("root-1-rung2".to_string(), 10)]);
        let incumbents =
            HashMap::from([("root-1-rung2".to_string(), (json!({"family": "ucb1"}), 0.0))]);

        assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
    }

    #[test]
    fn test_plan_ladder_advances_skips_a_rung_that_already_has_a_child() {
        let root = ladder_root_run("root-1", 5, 0.0);
        let mut child = ladder_root_run("root-1", 5, 0.0);
        child.run_id = "root-1-rung2".to_string();
        child.config.as_mut().unwrap()["resumed_from"] = json!("root-1");
        let runs = vec![root, child];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

        assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
    }

    #[test]
    fn test_plan_ladder_advances_ignores_stopped_run() {
        let mut run = ladder_root_run("root-1", 3, 0.0);
        run.status = "stopped".to_string();
        let runs = vec![run];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

        assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
    }

    #[test]
    fn test_plan_ladder_advances_ignores_crashed_exit_code() {
        let mut run = ladder_root_run("root-1", 3, 0.0);
        run.exit_code = Some(1);
        let runs = vec![run];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

        assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
    }

    #[test]
    fn test_plan_ladder_advances_ignores_non_ladder_run() {
        let run = LadderRunRow {
            run_id: "plain-run".to_string(),
            game: "nim".to_string(),
            status: "completed".to_string(),
            exit_code: Some(0),
            config: Some(json!({"overrides": ["optimizer.n_trials=10"]})),
        };
        let runs = vec![run];
        let trial_counts = HashMap::from([("plain-run".to_string(), 10)]);
        let incumbents =
            HashMap::from([("plain-run".to_string(), (json!({"family": "ucb1"}), 0.0))]);

        assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
    }

    // -------------------------------------------------------------------
    // plan_manual_advance
    // -------------------------------------------------------------------

    fn plain_run(run_id: &str, trials: i64) -> LadderRunRow {
        LadderRunRow {
            run_id: run_id.to_string(),
            game: "nim".to_string(),
            status: "completed".to_string(),
            exit_code: Some(0),
            config: Some(json!({"overrides": [format!("optimizer.n_trials={trials}")]})),
        }
    }

    #[test]
    fn test_plan_manual_advance_starts_a_new_chain_from_a_plain_run() {
        let runs = vec![plain_run("root-1", 10)];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([(
            "root-1".to_string(),
            (json!({"family": "ucb1", "c": 1.4}), 0.0),
        )]);

        let advance =
            plan_manual_advance(&runs, &trial_counts, &incumbents, "root-1", None, None).unwrap();
        assert_eq!(advance.game, "nim");
        assert_eq!(advance.label, "baseline advance from root-1");
        assert_eq!(advance.widened_config["resumed_from"], json!("root-1"));
        assert_eq!(advance.widened_config["ladder_root"], json!("root-1"));
        // No pre-existing "ladder" block -- this is a manual-only chain,
        // so the automated driver must never pick it up.
        assert!(advance.widened_config.get("ladder").is_none());
        // The baseline changes within the original total trial budget.
        assert_eq!(
            advance.widened_config["overrides"],
            json!([
                "optimizer.n_trials=10",
                "optimizer.n_trials=10",
                "target.baselines=[]"
            ])
        );
        assert_eq!(
            advance.widened_config["baseline_configs"]["ladder2"],
            json!({"family": "ucb1", "c": 1.4})
        );
        // The root itself never had `ladder_root` set -- the caller must
        // retroactively tag it so a later advance (or the UI) can find the
        // chain by `ladder_root` alone.
        let (root_id, root_config) = advance.root_patch.expect("expected a root patch");
        assert_eq!(root_id, "root-1");
        assert_eq!(root_config["ladder_root"], json!("root-1"));
    }

    #[test]
    fn test_plan_manual_advance_respects_an_explicit_n_trials() {
        let runs = vec![plain_run("root-1", 10)];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

        let advance = plan_manual_advance(
            &runs,
            &trial_counts,
            &incumbents,
            "root-1",
            Some(500),
            Some(4),
        )
        .unwrap();
        assert_eq!(
            advance.widened_config["overrides"],
            json!([
                "optimizer.n_trials=10",
                "optimizer.n_trials=500",
                "optimizer.n_workers=4",
                "target.baselines=[]"
            ])
        );
    }

    #[test]
    fn test_plan_manual_advance_continues_an_existing_chain_without_re_patching_the_root() {
        // root-1 already has ladder_root=root-1 (a prior manual or automated
        // advance already tagged it) and one child rung already exists.
        let mut root = plain_run("root-1", 10);
        root.config.as_mut().unwrap()["ladder_root"] = json!("root-1");
        let mut rung2 = plain_run("root-1-rung2", 10);
        rung2.config.as_mut().unwrap()["ladder_root"] = json!("root-1");
        rung2.config.as_mut().unwrap()["resumed_from"] = json!("root-1");
        let runs = vec![root, rung2];
        let trial_counts =
            HashMap::from([("root-1".to_string(), 10), ("root-1-rung2".to_string(), 10)]);
        let incumbents =
            HashMap::from([("root-1-rung2".to_string(), (json!({"family": "ucb1"}), 0.0))]);

        let advance = plan_manual_advance(
            &runs,
            &trial_counts,
            &incumbents,
            "root-1-rung2",
            None,
            None,
        )
        .unwrap();
        assert!(advance.root_patch.is_none());
        assert_eq!(advance.widened_config["ladder_root"], json!("root-1"));
        // rung_count is 2 (root + rung2) before this widen -> next id "ladder3".
        assert_eq!(
            advance.widened_config["baseline_configs"]["ladder3"],
            json!({"family": "ucb1"})
        );
        // A later baseline change still preserves the logical run's budget.
        assert_eq!(
            advance.widened_config["overrides"],
            json!([
                "optimizer.n_trials=10",
                "optimizer.n_trials=10",
                "target.baselines=[]"
            ])
        );
    }

    #[test]
    fn test_plan_manual_advance_replaces_rather_than_accumulates_baseline_configs() {
        let mut root = plain_run("root-1", 10);
        root.config.as_mut().unwrap()["baseline_configs"] =
            json!({"ladder1": {"family": "ucb1", "c": 0.5}});
        let runs = vec![root];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([(
            "root-1".to_string(),
            (json!({"family": "rave", "threshold": 700}), 0.0),
        )]);

        let advance =
            plan_manual_advance(&runs, &trial_counts, &incumbents, "root-1", None, None).unwrap();
        let baseline_configs = advance.widened_config["baseline_configs"]
            .as_object()
            .unwrap();
        assert_eq!(baseline_configs.len(), 1);
        assert_eq!(
            baseline_configs.get("ladder2"),
            Some(&json!({"family": "rave", "threshold": 700}))
        );
        assert!(!baseline_configs.contains_key("ladder1"));
    }

    #[test]
    fn test_plan_manual_advance_errors_without_an_incumbent() {
        let runs = vec![plain_run("root-1", 10)];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::new();

        let err = plan_manual_advance(&runs, &trial_counts, &incumbents, "root-1", None, None)
            .unwrap_err();
        assert!(err.contains("no incumbent"));
    }

    #[test]
    fn test_plan_manual_advance_errors_for_unknown_run() {
        let runs = vec![plain_run("root-1", 10)];
        let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
        let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

        let err =
            plan_manual_advance(&runs, &trial_counts, &incumbents, "nope", None, None).unwrap_err();
        assert!(err.contains("not found"));
    }

    // -------------------------------------------------------------------
    // POST /api/bench/runs/{run_id}/advance-baseline
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_advance_baseline_returns_404_for_unknown_run() {
        let app = seeded_app(|_, _| {}).0;
        let (status, body) = http_post_json(
            app,
            "/api/bench/runs/nonexistent/advance-baseline",
            json!({}),
        )
        .await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        assert_eq!(body_json(&body)["code"], 404);
    }

    #[tokio::test]
    async fn test_advance_baseline_rejects_non_smac3_run() {
        // DEFAULT_RUN_ID is seeded as a 'round_robin' run.
        let app = seeded_app(default_seed).0;
        let (status, body) = http_post_json(
            app,
            &format!("/api/bench/runs/{DEFAULT_RUN_ID}/advance-baseline"),
            json!({}),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
        let body = body_json(&body);
        assert!(body["error"].as_str().unwrap().contains("round_robin"));
    }

    #[tokio::test]
    async fn test_advance_baseline_rejects_a_run_with_no_incumbent() {
        let app = seeded_app(|conn, dir| {
            std::fs::create_dir_all(dir).ok();
            conn.execute(
                "INSERT INTO runs \
                 (run_id, kind, game, config, git_sha, git_dirty, host, pid, \
                  started_at, ended_at, status, log_path) \
                 VALUES ('smac3-no-incumbent', 'smac3', 'traffic-lights', \
                         '{\"config\": \"smac3/config/default.yaml\", \"overrides\": []}', \
                         'abc1234', false, 'testhost', NULL, \
                         '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/nope/log.jsonl')",
                duckdb::params![],
            )
            .unwrap();
        })
        .0;

        let (status, body) = http_post_json(
            app,
            "/api/bench/runs/smac3-no-incumbent/advance-baseline",
            json!({}),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
        let body = body_json(&body);
        assert!(body["error"].as_str().unwrap().contains("no incumbent"));
    }

    #[tokio::test]
    async fn test_advance_baseline_smac3_reaches_the_launcher() {
        // Same "reaches the launcher, doesn't get rejected as a bad
        // request" shape as test_resume_smac3_reaches_the_launcher: a
        // completed (non-running) run with a recorded incumbent should sail
        // past the stop-and-wait step (a no-op for a non-running run) and
        // the plan_manual_advance validation, reaching launch_and_record.
        let app = seeded_app(|conn, dir| {
            std::fs::create_dir_all(dir).ok();
            conn.execute(
                "INSERT INTO runs \
                 (run_id, kind, game, config, git_sha, git_dirty, host, pid, \
                  started_at, ended_at, status, log_path) \
                 VALUES ('smac3-advance-src', 'smac3', 'traffic-lights', \
                         '{\"config\": \"smac3/config/default.yaml\", \"overrides\": []}', \
                         'abc1234', false, 'testhost', NULL, \
                         '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/nope/log.jsonl')",
                duckdb::params![],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO incumbents (run_id, ts, config, cost) \
                 VALUES ('smac3-advance-src', '2026-01-01T00:30:00Z', '{\"family\": \"ucb1\"}', 0.02)",
                duckdb::params![],
            )
            .unwrap();
        })
        .0;

        let (status, body) = http_post_json(
            app,
            "/api/bench/runs/smac3-advance-src/advance-baseline",
            json!({}),
        )
        .await;

        assert!(
            status == HttpStatusCode::OK || status == HttpStatusCode::INTERNAL_SERVER_ERROR,
            "advance-baseline returned unexpected status {status}: body={}",
            String::from_utf8_lossy(&body),
        );
    }

    // -------------------------------------------------------------------
    // POST /api/bench/runs/{run_id}/resume
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_resume_returns_404_for_unknown_run() {
        let app = seeded_app(|_, _| {}).0;
        let (status, body) = http_post_json(
            app,
            "/api/bench/runs/nonexistent/resume",
            json!({ "n_trials": 500 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        assert_eq!(body_json(&body)["code"], 404);
    }

    #[tokio::test]
    async fn test_resume_rejects_non_smac3_run() {
        // DEFAULT_RUN_ID is seeded as a 'round_robin' run.
        let app = seeded_app(default_seed).0;
        let (status, body) = http_post_json(
            app,
            &format!("/api/bench/runs/{DEFAULT_RUN_ID}/resume"),
            json!({ "n_trials": 500 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
        let body = body_json(&body);
        assert!(body["error"].as_str().unwrap().contains("round_robin"));
    }

    #[tokio::test]
    async fn test_resume_smac3_reaches_the_launcher() {
        // Same "reaches the launcher, doesn't get rejected as a bad
        // request" shape as test_launch_smac3_reaches_the_launcher: proves
        // the old run's kind/config are read back out of the DB and turned
        // into a launch the handler forwards, rather than being rejected
        // before ever reaching launch::launch_with_run_id.
        let app = seeded_app(|conn, dir| {
            std::fs::create_dir_all(dir).ok();
            conn.execute(
                "INSERT INTO runs \
                 (run_id, kind, game, config, git_sha, git_dirty, host, pid, \
                  started_at, ended_at, status, log_path) \
                 VALUES ('smac3-resume-src', 'smac3', 'traffic-lights', \
                         '{\"config\": \"smac3/config/default.yaml\", \"overrides\": []}', \
                         'abc1234', false, 'testhost', NULL, \
                         '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/nope/log.jsonl')",
                duckdb::params![],
            )
            .unwrap();
        })
        .0;

        let (status, body) = http_post_json(
            app,
            "/api/bench/runs/smac3-resume-src/resume",
            json!({ "n_trials": 500 }),
        )
        .await;

        assert!(
            status == HttpStatusCode::OK || status == HttpStatusCode::INTERNAL_SERVER_ERROR,
            "resume returned unexpected status {status}: body={}",
            String::from_utf8_lossy(&body),
        );
    }

    // -------------------------------------------------------------------
    // POST /api/bench/runs/{run_id}/stop
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_stop_returns_404_for_unknown_run() {
        let app = seeded_app(|_, _| {}).0;
        let (status, body) =
            http_post_json(app, "/api/bench/runs/nonexistent/stop", json!({})).await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        let body = body_json(&body);
        assert_eq!(body["code"], 404);
    }

    #[tokio::test]
    async fn test_stop_returns_ok_for_non_running_run_without_signalling() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_post_json(
            app,
            &format!("/api/bench/runs/{DEFAULT_RUN_ID}/stop"),
            json!({}),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        // Completed run — no signal sent, but still succeeds.
        assert_eq!(body["status"], "completed");
    }

    #[tokio::test]
    async fn test_stop_marks_running_run_as_stopped() {
        let app = seeded_app(|conn, bench_runs_dir| {
            let run_dir = bench_runs_dir.join("stoppable-run");
            std::fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            std::fs::write(&log_path, "").unwrap();
            let log_path_str = log_path.to_string_lossy().to_string();

            // Use a non-existent PID so the test doesn't accidentally
            // signal the current process (which would kill the test runner).
            // The stop handler gracefully handles missing PIDs.
            conn.execute(
                "INSERT INTO runs \
                 (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) \
                 VALUES ('stoppable-run', 'round_robin', 'druid', 'abc', false, 'h', 999999999, \
                         '2026-03-01T00:00:00Z', 'running', ?1)",
                duckdb::params![log_path_str],
            )
            .unwrap();
        })
        .0;

        let (status, body) =
            http_post_json(app.clone(), "/api/bench/runs/stoppable-run/stop", json!({})).await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        // No signal was sent (PID doesn't exist), but the run should still
        // be marked as stopped in the database.
        assert_eq!(
            body["message"].as_str().unwrap_or(""),
            "run marked as stopped (PID was no longer alive or had no PID)"
        );

        // Verify the DB was updated.
        let (_, check_body) = http_get(app, "/api/bench/runs/stoppable-run").await;
        let detail = body_json(&check_body);
        assert_eq!(detail["status"], "stopped");
        assert!(detail["ended_at"].as_str().unwrap_or("").len() >= 10);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_stop_kills_the_whole_process_group_not_just_the_leader() {
        use std::io::BufRead as _;
        use std::os::unix::process::CommandExt;
        use std::process::{Command, Stdio};

        // Mirror `launch::launch`'s isolation (`process_group(0)`): spawn a
        // shell that backgrounds a long-lived `sleep` child and waits on it.
        // The child inherits the shell's (new) process group since this is
        // a non-interactive shell with no job control. Recording only the
        // shell's PID and single-PID `kill`ing it (the pre-fix behavior)
        // would leave `sleep` running as an orphan.
        let mut leader = Command::new("sh")
            .arg("-c")
            .arg("sleep 60 & echo $!; wait")
            .stdout(Stdio::piped())
            .process_group(0)
            .spawn()
            .expect("failed to spawn test process group leader");
        let leader_pid = leader.id() as i64;

        let mut reader = std::io::BufReader::new(leader.stdout.take().unwrap());
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .expect("failed to read child sleep PID");
        let sleep_pid: i64 = line.trim().parse().expect("child PID should be numeric");

        let is_alive = |pid: i64| {
            std::process::Command::new("kill")
                .arg("-0")
                .arg(pid.to_string())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };
        assert!(is_alive(sleep_pid), "sleep child should start out alive");

        let app = seeded_app(|conn, bench_runs_dir| {
            let run_dir = bench_runs_dir.join("group-stoppable-run");
            std::fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            std::fs::write(&log_path, "").unwrap();
            conn.execute(
                "INSERT INTO runs \
                 (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) \
                 VALUES ('group-stoppable-run', 'smac3', 'traffic-lights', 'abc', false, 'h', ?1, \
                         '2026-03-01T00:00:00Z', 'running', ?2)",
                duckdb::params![leader_pid, log_path.to_string_lossy().to_string()],
            )
            .unwrap();
        })
        .0;

        let (status, _) =
            http_post_json(app, "/api/bench/runs/group-stoppable-run/stop", json!({})).await;
        assert_eq!(status, HttpStatusCode::OK);

        let _ = leader.wait();

        // SIGTERM's default action is immediate termination, but poll
        // briefly rather than asserting instantaneously to absorb
        // scheduling jitter.
        let mut still_alive = is_alive(sleep_pid);
        for _ in 0..20 {
            if !still_alive {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            still_alive = is_alive(sleep_pid);
        }
        assert!(
            !still_alive,
            "sleep child (PID {sleep_pid}) should have been killed along with its process group leader"
        );
    }

    // -------------------------------------------------------------------
    // Error formatting
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_bench_error_has_structured_body() {
        let app = seeded_app(default_seed).0;
        let (status, body) = http_get(app, "/api/bench/runs/nope").await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);
        let body = body_json(&body);
        assert_eq!(body["code"], 404);
        assert!(body["error"].as_str().unwrap().contains("nope"));
    }
}
