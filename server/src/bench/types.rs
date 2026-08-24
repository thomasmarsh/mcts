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
use mcts_bench::project_repository::ProjectRepository;
use mcts_bench::projects_attempt::{CellRequest, ProjectsError, StartRequest};
use mcts_bench::run_command_repository::RunCommandRepository;
use mcts_bench::run_repository::RunRepository;
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::tuning_analysis_repository::TuningAnalysisRepository;
use mcts_bench::tuning_command_repository::TuningCommandRepository;
use mcts_bench::tuning_session_repository::TuningSessionRepository;
use mcts_bench::tuning_trial_repository::TuningTrialRepository;
use mcts_bench::StrategyInfo;

use super::lifecycle;
use super::process;
// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// State shared by all bench routes.  The DuckDB connection is
/// `Mutex`-guarded because `duckdb::Connection` is `Send` but not `Sync`;
/// the ingest loop and API routes all share the same in-process connection.
pub struct BenchState {
    pub db: Arc<Mutex<duckdb::Connection>>,
    pub project_repository: Arc<dyn ProjectRepository + Send + Sync>,
    pub run_repository: Arc<dyn RunRepository + Send + Sync>,
    pub run_command_repository: Arc<dyn RunCommandRepository + Send + Sync>,
    pub tuning_analysis_repository: Arc<dyn TuningAnalysisRepository + Send + Sync>,
    pub tuning_command_repository: Arc<dyn TuningCommandRepository + Send + Sync>,
    pub tuning_session_repository: Arc<dyn TuningSessionRepository + Send + Sync>,
    pub tuning_trial_repository: Arc<dyn TuningTrialRepository + Send + Sync>,
    pub bench_runs_dir: PathBuf,
    pub experiment_validator: ExperimentValidator,
    pub run_launcher: RunLauncher,
    pub process_group_signaller: ProcessGroupSignaller,
    pub runtime: Arc<lifecycle::BenchRuntime>,
}

pub type ExperimentValidator = Arc<
    dyn Fn(&ExperimentSpecV1) -> Result<(), Vec<mcts_bench::experiment::ValidationField>>
        + Send
        + Sync,
>;
pub type RunLauncher = Arc<
    dyn Fn(String, Vec<String>, String, String, Option<String>) -> std::io::Result<LaunchedRun>
        + Send
        + Sync,
>;
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
    /// Modern logical session that owns this physical tuner attempt.
    /// Present as soon as a server-created tuner run has its pinned config,
    /// before lifecycle ingestion projects the attempt row.
    pub tuning_session_id: Option<String>,
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
    /// The modern logical tuning session that owns this physical attempt.
    /// Absent for non-tuner runs and legacy tuner rows.
    pub tuning_session_id: Option<String>,
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

#[derive(Debug)]
pub(crate) struct ValidationError {
    pub(crate) fields: Vec<mcts_bench::experiment::ValidationField>,
}

impl IntoResponse for ValidationError {
    fn into_response(self) -> axum::response::Response {
        let status = if self
            .fields
            .iter()
            .any(|field| field.message.contains("duplicate"))
        {
            StatusCode::CONFLICT
        } else if self.fields.iter().any(|field| field.message == "not found") {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::BAD_REQUEST
        };
        (
            status,
            Json(json!({"error": "validation failed", "fields": self.fields})),
        )
            .into_response()
    }
}

#[derive(Deserialize)]
pub(crate) struct ProjectCreateBody {
    pub(crate) name: String,
    pub(crate) description: String,
}
#[derive(Deserialize)]
pub(crate) struct ProjectPatchBody {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) archived: Option<bool>,
}
#[derive(Deserialize)]
pub(crate) struct ExperimentBody {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) spec: ExperimentSpecV1,
}

#[derive(Serialize)]
pub(crate) struct ProjectResponse {
    pub(crate) project_id: String,
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) archived: bool,
    pub(crate) created_at: String,
    pub(crate) updated_at: String,
}
#[derive(Serialize)]
pub(crate) struct ExperimentResponse {
    pub(crate) experiment_id: String,
    pub(crate) project_id: String,
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) spec: ExperimentSpecV1,
    pub(crate) created_at: String,
    pub(crate) updated_at: String,
}

#[derive(Serialize)]
pub(crate) struct CellResponse {
    pub(crate) cell_id: String,
    pub(crate) cell_seed: Option<u64>,
    pub(crate) game: String,
    pub(crate) game_config: Value,
    pub(crate) variant_id: String,
    pub(crate) variant_label: String,
    pub(crate) candidate_config: Value,
    pub(crate) baseline_id: String,
    pub(crate) baseline_label: String,
    pub(crate) baseline_config: Value,
    pub(crate) budget: Value,
    pub(crate) rounds: i64,
    pub(crate) planned_games: u64,
    pub(crate) completed_games: u64,
    pub(crate) status: String,
    pub(crate) started_at: Option<String>,
    pub(crate) ended_at: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) wins: u64,
    pub(crate) losses: u64,
    pub(crate) draws: u64,
    pub(crate) win_rate: f64,
    pub(crate) ci_lower: f64,
    pub(crate) ci_upper: f64,
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

pub(crate) fn identity_bench_error(error: identity::IdentityError) -> BenchError {
    let status = match &error {
        identity::IdentityError::MissingRun(_) => StatusCode::NOT_FOUND,
        identity::IdentityError::Contradiction(_) => StatusCode::CONFLICT,
        identity::IdentityError::InvalidLinkage(_) => StatusCode::BAD_REQUEST,
        identity::IdentityError::DuckDb(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    BenchError {
        status,
        message: error.to_string(),
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

pub(crate) enum ExperimentRouteError {
    Bench(BenchError),
    Validation(ValidationError),
}

impl From<BenchError> for ExperimentRouteError {
    fn from(error: BenchError) -> Self {
        Self::Bench(error)
    }
}

impl From<duckdb::Error> for ExperimentRouteError {
    fn from(error: duckdb::Error) -> Self {
        Self::Bench(error.into())
    }
}

impl From<ProjectsError> for ExperimentRouteError {
    fn from(error: ProjectsError) -> Self {
        Self::Bench(attempt_bench_error(error))
    }
}

impl From<std::io::Error> for ExperimentRouteError {
    fn from(error: std::io::Error) -> Self {
        Self::Bench(error.into())
    }
}

impl From<serde_json::Error> for ExperimentRouteError {
    fn from(error: serde_json::Error) -> Self {
        Self::Bench(error.into())
    }
}

impl IntoResponse for ExperimentRouteError {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::Bench(error) => error.into_response(),
            Self::Validation(error) => error.into_response(),
        }
    }
}

// ---------------------------------------------------------------------------
