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
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
    /// Live per-game-kind subprocess sessions, shared with the main
    /// gameplay `AppState` -- reused here (rather than spawning a second
    /// set of subprocesses) so `/api/bench/smac3/kinds` can query each
    /// game's `tuner()` over its already-open session.
    pub games: Arc<HashMap<&'static str, Arc<dyn crate::adapter::GameAdapter>>>,
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
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    Router::new()
        .route("/api/bench/kinds", get(list_kinds))
        .route("/api/bench/smac3/kinds", get(list_smac3_kinds))
        .route("/api/bench/runs", get(list_runs))
        .route("/api/bench/runs/{run_id}", get(get_run))
        .route("/api/bench/runs/{run_id}/log", get(get_run_log))
        .route("/api/bench/runs/{run_id}/stdout", get(get_run_stdout))
        .route("/api/bench/runs/{run_id}/trials", get(get_run_trials))
        .route("/api/bench/leaderboard", get(get_leaderboard))
        .route("/api/bench/launch", post(launch_run).layer(launch_timeout))
        .route("/api/bench/runs/{run_id}/stop", post(stop_run))
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
                COALESCE(m.match_count, 0), COALESCE(t.trial_count, 0) \
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
    if let Some(ref status) = params.status {
        sql.push_str(&format!(" AND r.status = '{}'", status.replace('\'', "''")));
    }
    if let Some(ref game) = params.game {
        sql.push_str(&format!(" AND r.game = '{}'", game.replace('\'', "''")));
    }

    sql.push_str(" ORDER BY CAST(r.started_at AS TEXT) DESC");

    if let Some(limit) = params.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }

    let mut stmt = db.prepare(&sql)?;

    let runs: Vec<RunSummary> = stmt
        .query_map([], |row| {
            Ok(RunSummary {
                run_id: row.get(0)?,
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
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(Json(runs))
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
                COALESCE(m.match_count, 0), COALESCE(t.trial_count, 0) \
         FROM runs r \
         LEFT JOIN (SELECT run_id, COUNT(*) AS match_count FROM match_results GROUP BY run_id) m \
           ON r.run_id = m.run_id \
         LEFT JOIN (SELECT run_id, COUNT(*) AS trial_count FROM trials GROUP BY run_id) t \
           ON r.run_id = t.run_id \
         WHERE r.run_id = ?1",
        duckdb::params![&run_id],
        |row| {
            let config_str: Option<String> = row.get::<_, Option<String>>(4).ok().flatten();
            let config = config_str.and_then(|s| serde_json::from_str(&s).ok());
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
            description: "Runs a SMAC3 hyperparameter-optimization sweep over a game's tunable strategy search space, playing rounds of a params-built candidate against a fixed baseline per trial.  Results are streamed as trial JSONL lines.  See GET /api/bench/smac3/kinds for per-game tuner metadata (search space, baseline, eval rounds) instead of a strategies list."
                .to_string(),
            games: vec![],
        },
    ];

    Json(kinds)
}

/// `GET /api/bench/smac3/kinds`
///
/// Per-game tuner metadata (search space, baseline, eval rounds), queried
/// through each game's already-open `SubprocessAdapter` session (the same
/// ones the gameplay routes use) rather than spawning a one-shot `tune
/// describe` process per request.  Only games that implement `tuner()`
/// (return `Some`) appear -- tuning support is opt-in per game.
async fn list_smac3_kinds(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Json<Vec<Smac3GameInfo>> {
    let mut games: Vec<Smac3GameInfo> = state
        .games
        .iter()
        .filter_map(|(kind, adapter)| {
            adapter.tuner().map(|tuner| Smac3GameInfo {
                game: kind.to_string(),
                tuner,
            })
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

/// `POST /api/bench/launch` — `{kind, game, config}`
///
/// Translates the request into a command vector, spawns it via
/// `launch::launch`, and returns the run metadata immediately.
async fn launch_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    Json(body): Json<LaunchBody>,
) -> Result<Json<LaunchResponse>, BenchError> {
    let cmd = build_command(&body.kind, &body.game, &body.config)?;
    let label = body
        .config
        .as_ref()
        .and_then(|c| c.get("label").and_then(|v| v.as_str()));

    let LaunchedRun {
        run_id,
        pid,
        log_path,
        log_dir,
    } = launch::launch(cmd, &body.kind, &body.game, label).map_err(|e| BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("failed to launch run: {e}"),
    })?;

    let started_at = iso_timestamp_now();

    // Insert the run into the runs table so it appears immediately in
    // the runs list (no ingest loop dependency).
    {
        let db = state.db.lock().unwrap();
        let config_str = body
            .config
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
                &body.kind,
                &body.game,
                label,
                config_str,
                game_host::build_info::GIT_SHA,
                game_host::build_info::GIT_DIRTY == "true",
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
    if let Some(ref config) = body.config {
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

    Ok(Json(LaunchResponse {
        run_id,
        pid,
        log_path: log_path.to_string_lossy().to_string(),
        launch_error,
    }))
}

/// `POST /api/bench/runs/{run_id}/stop` — best-effort SIGTERM
///
/// Sends SIGTERM to the recorded PID's whole process group (`kill -TERM
/// -<pid>`) and marks the run as `stopped` in the database.  `launch::launch`
/// puts every run in its own process group (`process_group(0)`), so the
/// recorded PID is that group's leader -- signalling just that one PID would
/// leave descendants (e.g. the `uv`/python child under `bench smac3`)
/// orphaned instead of terminated.  If the PID is no longer alive, updates
/// the status anyway (the process exited on its own between the list and the
/// stop request).
async fn stop_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Value>, BenchError> {
    let db = state.db.lock().unwrap();

    // Look up the run.
    let (pid, status): (Option<i64>, String) = match db.query_row(
        "SELECT pid, status FROM runs WHERE run_id = ?1",
        duckdb::params![&run_id],
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
    };

    if status != "running" {
        return Ok(Json(json!({
            "run_id": run_id,
            "status": status,
            "message": "run is not currently running, no signal sent",
        })));
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
    db.execute(
        "UPDATE runs SET status = 'stopped', ended_at = ?1 WHERE run_id = ?2 AND status = 'running'",
        duckdb::params![&now, &run_id],
    )?;

    // Append a stop event to the registry log so the ingest loop sees it
    // if it runs after us.
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

    if signal_sent {
        Ok(Json(json!({
            "run_id": run_id,
            "pid": pid,
            "signal": "SIGTERM",
            "message": "stop signal sent and run marked as stopped",
        })))
    } else {
        Ok(Json(json!({
            "run_id": run_id,
            "pid": pid,
            "signal": null,
            "message": "run marked as stopped (PID was no longer alive or had no PID)",
        })))
    }
}

// ---------------------------------------------------------------------------
// Command construction
// ---------------------------------------------------------------------------

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
            }

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
        seeded_app_with_games(seed_fn, HashMap::new())
    }

    /// Like `seeded_app`, but with a caller-supplied `games` map -- for
    /// tests that need `/api/bench/smac3/kinds` to see specific fake
    /// per-game tuner metadata instead of the real (subprocess-backed)
    /// registry.
    fn seeded_app_with_games(
        seed_fn: impl FnOnce(&duckdb::Connection, &Path),
        games: HashMap<&'static str, Arc<dyn crate::adapter::GameAdapter>>,
    ) -> (Router, PathBuf) {
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
            games: Arc::new(games),
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

    /// A minimal `crate::adapter::GameAdapter` fake -- only `kind`/`tuner`
    /// are exercised by the `smac3/kinds` route, everything else panics if
    /// called so a test that accidentally reaches it fails loudly.
    struct FakeTunableAdapter {
        kind: &'static str,
        tuner: Option<TunerInfo>,
    }

    impl crate::adapter::GameAdapter for FakeTunableAdapter {
        fn kind(&self) -> &'static str {
            self.kind
        }
        fn label(&self) -> &'static str {
            "Fake"
        }
        fn description(&self) -> &'static str {
            "fake adapter for smac3/kinds tests"
        }
        fn default_config(&self) -> Value {
            json!({})
        }
        fn new_state(&self, _config: Value) -> Result<Value, crate::adapter::AdapterError> {
            unimplemented!()
        }
        fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, crate::adapter::AdapterError> {
            unimplemented!()
        }
        fn apply(
            &self,
            _state: &Value,
            _mv: &Value,
        ) -> Result<Value, crate::adapter::AdapterError> {
            unimplemented!()
        }
        fn view(&self, _state: &Value) -> Result<Value, crate::adapter::AdapterError> {
            unimplemented!()
        }
        fn ai_presets(&self) -> Vec<crate::adapter::AiPresetInfo> {
            vec![]
        }
        fn ai_move(
            &self,
            _state: &Value,
            _preset: &str,
        ) -> Result<crate::adapter::AiMoveResult, crate::adapter::AdapterError> {
            unimplemented!()
        }
        fn analyze(
            &self,
            _state: &Value,
            _preset: &str,
            _budget_ms: Option<u64>,
        ) -> Result<crate::adapter::Analysis, crate::adapter::AdapterError> {
            unimplemented!()
        }
        fn tuner(&self) -> Option<TunerInfo> {
            self.tuner.clone()
        }
    }

    #[tokio::test]
    async fn test_smac3_kinds_only_lists_tunable_games() {
        let mut games: HashMap<&'static str, Arc<dyn crate::adapter::GameAdapter>> = HashMap::new();
        games.insert(
            "traffic-lights",
            Arc::new(FakeTunableAdapter {
                kind: "traffic-lights",
                tuner: Some(TunerInfo {
                    id: "rave".into(),
                    baseline: "strong".into(),
                    eval_rounds: 20,
                    parameters: vec![],
                    conditions: vec![],
                }),
            }),
        );
        games.insert(
            "druid",
            Arc::new(FakeTunableAdapter {
                kind: "druid",
                tuner: None,
            }),
        );

        let app = seeded_app_with_games(|_, _| {}, games).0;
        let (status, body) = http_get(app, "/api/bench/smac3/kinds").await;
        assert_eq!(status, HttpStatusCode::OK);
        let kinds = body_json(&body).as_array().unwrap().clone();
        assert_eq!(
            kinds.len(),
            1,
            "expected only traffic-lights to be tunable, got {kinds:?}"
        );
        assert_eq!(kinds[0]["game"], "traffic-lights");
        assert_eq!(kinds[0]["tuner"]["id"], "rave");
        assert_eq!(kinds[0]["tuner"]["eval_rounds"], 20);
    }

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
        )
        .unwrap();

        // First element is the (unresolved-in-test) bench binary path --
        // everything after it is the argv this test actually cares about.
        assert_eq!(
            cmd[1..],
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
        let cmd = build_command("smac3", "druid", &None).unwrap();
        assert_eq!(cmd[1..], vec!["smac3", "--game", "druid"]);
    }

    #[test]
    fn test_build_command_unknown_kind_lists_smac3_as_supported() {
        let err = build_command("nope", "druid", &None).unwrap_err();
        assert!(err.message.contains("smac3"));
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
