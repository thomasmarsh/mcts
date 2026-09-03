#![allow(unused_imports)]
use std::collections::HashMap;
use std::convert::Infallible;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;
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
use mcts_bench::projects_attempt::{CellRequest, ProjectsError, ProjectsRepository, StartRequest};
use mcts_bench::run_command_repository::RunCommandRepository;
use mcts_bench::run_repository::RunRepository;
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;

use super::lifecycle;
use super::process;
// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// State shared by all bench routes.
pub struct BenchState {
    #[cfg(test)]
    pub(crate) db: TestDatabase,
    pub projects_repository: Arc<dyn ProjectsRepository + Send + Sync>,
    pub run_repository: Arc<dyn RunRepository + Send + Sync>,
    pub run_command_repository: Arc<dyn RunCommandRepository + Send + Sync>,
    pub bench_runs_dir: PathBuf,
    /// Writable directory of frozen-objective JSON files a tuner run can be
    /// launched against, and which the objective editor manages. The launch
    /// and objective APIs take an `objective_key` (a file stem) and resolve
    /// it here, so no filesystem path crosses the API boundary.
    pub tuner_objectives_dir: PathBuf,
    /// Read-only corpus of objective files shipped with the repo. On start-up
    /// any stem not already present in `tuner_objectives_dir` is copied over;
    /// user edits then live only in the writable dir.
    pub tuner_seed_objectives_dir: PathBuf,
    /// Validates an objective file out of band (production: shells out to
    /// `python -m tuner_cli validate-objective`; tests inject a stub).
    pub tuner_objective_validator: ObjectiveValidator,
    pub process_group_signaller: ProcessGroupSignaller,
    /// Read-only SQLite projection of version-4 tuner runs, served by the
    /// `tuner_api` handlers. Built and refreshed by the `tuner-project` tool
    /// (`POST /api/bench/tuner/projection/refresh`).
    pub tuner_projection_db: PathBuf,
    /// Runs the projector for `POST /api/bench/tuner/projection/refresh` and
    /// returns `[projected, skipped, ingest_errors, pruned]`. Production uses
    /// [`super::tuner_api::shell_refresh`]; tests inject a stub.
    pub tuner_projection_refresh: ProjectionRefresher,
    /// Dry-runs a launch request through every check `tuner_cli` applies
    /// before it starts a run (production: shells `python -m tuner_cli
    /// preflight`; tests inject a stub). The launch form calls this so a
    /// launch is never started for a knowable reason.
    pub tuner_launch_preflight: LaunchPreflighter,
}

/// Dry-runs a resolved launch request: `&TunerLaunchRequest ->
/// {ok, errors}`. Production shells `tuner_cli preflight`; the request's
/// `game_binary` / `objective_file` are already resolved to absolute paths.
pub type LaunchPreflighter = Arc<
    dyn Fn(&mcts_bench::tuner_launch::TunerLaunchRequest) -> std::io::Result<LaunchPreflight>
        + Send
        + Sync,
>;

/// Result of [`LaunchPreflighter`] — mirrors the JSON line from
/// `tuner_cli preflight`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchPreflight {
    pub ok: bool,
    #[serde(default)]
    pub errors: Vec<String>,
}

/// Refreshes the tuner projection out of band: `(bench_runs_dir, projection_db)
/// -> [projected, skipped, ingest_errors, pruned]`.
pub type ProjectionRefresher =
    Arc<dyn Fn(&Path, &Path) -> std::io::Result<[i64; 4]> + Send + Sync>;

/// Validates an objective file for a game: `(game_kind, objective_file) ->
/// ObjectiveValidation`. The production impl resolves the built-in game binary
/// itself; a missing binary surfaces as `ok: false`, not an `Err`.
pub type ObjectiveValidator =
    Arc<dyn Fn(&str, &Path) -> std::io::Result<ObjectiveValidation> + Send + Sync>;

/// Result of [`ObjectiveValidator`] — mirrors the JSON line emitted by
/// `tuner_cli validate-objective`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveValidation {
    pub ok: bool,
    #[serde(default)]
    pub errors: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel_fingerprint: Option<String>,
}

#[cfg(test)]
pub(crate) struct TestDatabase(Option<Arc<Mutex<duckdb::Connection>>>);

#[cfg(test)]
impl TestDatabase {
    pub(crate) fn shared(connection: Arc<Mutex<duckdb::Connection>>) -> Self {
        Self(Some(connection))
    }

    pub(crate) fn unavailable() -> Self {
        Self(None)
    }

    pub(crate) fn lock(&self) -> Result<std::sync::MutexGuard<'_, duckdb::Connection>, ()> {
        self.0.as_ref().ok_or(())?.lock().map_err(|_| ())
    }
}

pub type ProcessGroupSignaller = Arc<dyn Fn(i64) -> std::io::Result<()> + Send + Sync>;

/// Signal one detached run's process group through the process adapter.
pub fn signal_process_group(pid: i64) -> std::io::Result<()> {
    process::signal_process_group(pid)
}

// ---------------------------------------------------------------------------
// Query parameter types
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub struct ListRunsParams {
    pub status: Option<String>,
    pub game: Option<String>,
    pub limit: Option<i64>,
    pub experiment_id: Option<String>,
    pub project_id: Option<String>,
}

#[derive(Deserialize)]
pub struct RunLogParams {
    pub since: Option<u64>,
}

#[derive(Deserialize, Default)]
pub struct TrialsParams {
    pub limit: Option<i64>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct RunSummary {
    pub run_id: String,
    pub kind: String,
    pub game: Option<String>,
    pub project_id: Option<String>,
    pub experiment_id: Option<String>,
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
    pub game: Option<String>,
    pub project_id: Option<String>,
    pub experiment_id: Option<String>,
    pub experiment_spec: Option<Value>,
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
    /// tuner's own current best config for this run (from its intensifier,
    /// not a naive `MIN(cost)` over `trials` -- see `LogRecord::Incumbent`'s
    /// doc comment for why that distinction matters once multiple baseline
    /// instances are in play). `None` for a non-tuner run, or a tuner run
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

/// A game's tunable strategy search-space metadata, as reported by
/// `GET /api/bench/tuner/kinds` -- the tuner launch form's data-driven
/// counterpart to `BenchGameInfo`.
#[derive(Serialize)]
pub struct TunerGameInfo {
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
    pub(crate) status: StatusCode,
    pub(crate) message: String,
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

impl From<ProjectsError> for BenchError {
    fn from(error: ProjectsError) -> Self {
        attempt_bench_error(error)
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

pub(crate) fn run_command_bench_error(
    error: mcts_bench::run_command_repository::RunCommandRepositoryError,
) -> BenchError {
    use mcts_bench::run_command_repository::RunCommandRepositoryError;
    let status = match error {
        RunCommandRepositoryError::NotFound => StatusCode::NOT_FOUND,
        RunCommandRepositoryError::ContradictoryIdentity | RunCommandRepositoryError::Conflict => {
            StatusCode::CONFLICT
        }
        RunCommandRepositoryError::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    BenchError {
        status,
        message: error.to_string(),
    }
}

pub(crate) fn attempt_bench_error(error: ProjectsError) -> BenchError {
    let status = match &error {
        ProjectsError::Conflict(_) => StatusCode::CONFLICT,
        ProjectsError::NotFound => StatusCode::NOT_FOUND,
        ProjectsError::Corrupt(_) | ProjectsError::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    BenchError {
        status,
        message: error.to_string(),
    }
}

// ---------------------------------------------------------------------------
