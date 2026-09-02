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
use mcts_bench::run_repository::{
    LeaderboardQuery, RunListQuery, RunRepository, RunRepositoryError,
};
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::StrategyInfo;

use super::{lifecycle, types::*};

pub(crate) async fn stop_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Value>, BenchError> {
    let outcome = lifecycle::stop_run_impl(&state, &run_id, &lifecycle::SystemClock).await?;
    if outcome.prior_status != "running" {
        return Ok(Json(
            json!({"run_id": run_id, "status": outcome.prior_status, "message": "run is not currently running, no signal sent"}),
        ));
    }
    Ok(Json(json!({
        "run_id": run_id,
        "pid": outcome.pid,
        "signal": outcome.signal_sent.then_some("SIGTERM"),
        "message": if outcome.signal_sent { "stop signal sent and run marked as stopped" } else { "run marked as stopped (PID was no longer alive or had no PID)" },
    })))
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

pub(crate) fn iso_timestamp_now() -> String {
    let total_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock set before Unix epoch")
        .as_secs();
    let days = total_secs / 86400;
    let time_secs = total_secs % 86400;
    let (y, m, d) = days_to_ymd(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        time_secs / 3600,
        (time_secs % 3600) / 60,
        time_secs % 60
    )
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
    (if m <= 2 { y + 1 } else { y }, m, d)
}
pub(crate) async fn list_runs(
    AxumState(state): AxumState<Arc<BenchState>>,
    Query(params): Query<ListRunsParams>,
) -> Result<Json<Vec<RunSummary>>, BenchError> {
    let query = RunListQuery {
        game: params.game,
        experiment_id: params.experiment_id,
        project_id: params.project_id,
    };
    let mut runs = state
        .run_repository
        .list_runs(&query)
        .map_err(run_repository_error)?
        .into_iter()
        .map(|run| RunSummary {
            run_id: run.run_id,
            kind: run.kind,
            game: run.game,
            project_id: run.project_id,
            experiment_id: run.experiment_id,
            label: run.label,
            git_sha: run.git_sha,
            git_dirty: run.git_dirty,
            host: run.host,
            pid: run.pid,
            started_at: run.started_at,
            ended_at: run.ended_at,
            status: run.status,
            match_count: run.match_count,
            trial_count: run.trial_count,
        })
        .collect::<Vec<_>>();
    if let Some(ref status) = params.status {
        runs.retain(|run| run.status == *status);
    }
    if let Some(limit) = params.limit {
        runs.truncate(limit.max(0) as usize);
    }

    Ok(Json(runs))
}

/// `GET /api/bench/runs/{run_id}`
pub(crate) async fn get_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<RunDetail>, BenchError> {
    let run = state
        .run_repository
        .load_run(&run_id)
        .map_err(|error| run_repository_error_for_run(error, &run_id))?;
    Ok(Json(RunDetail {
        run_id: run.run_id,
        kind: run.kind,
        game: run.game,
        project_id: run.project_id,
        experiment_id: run.experiment_id,
        experiment_spec: run.experiment_spec,
        label: run.label,
        config: run.config,
        git_sha: run.git_sha,
        git_dirty: run.git_dirty,
        host: run.host,
        pid: run.pid,
        started_at: run.started_at,
        ended_at: run.ended_at,
        status: run.status,
        log_path: run.log_path,
        exit_code: run.exit_code,
        match_count: run.match_count,
        trial_count: run.trial_count,
        incumbent: run.incumbent.map(|incumbent| IncumbentInfo {
            config: incumbent.config,
            cost: incumbent.cost,
        }),
    }))
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
    let log_path = load_run_log_path(state.run_repository.as_ref(), &run_id)?;

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

fn load_run_log_path(repository: &dyn RunRepository, run_id: &str) -> Result<String, BenchError> {
    repository
        .load_log_path(run_id)
        .map_err(|error| run_repository_error_for_run(error, run_id))
}

pub(crate) fn run_repository_error(error: RunRepositoryError) -> BenchError {
    match error {
        RunRepositoryError::NotFound => BenchError {
            status: StatusCode::NOT_FOUND,
            message: "run not found".into(),
        },
        RunRepositoryError::Storage(message) => BenchError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("run storage error: {message}"),
        },
    }
}

pub(crate) fn run_repository_error_for_run(error: RunRepositoryError, run_id: &str) -> BenchError {
    match error {
        RunRepositoryError::NotFound => BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("run '{run_id}' not found"),
        },
        error => run_repository_error(error),
    }
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
    let log_path = load_run_log_path(state.run_repository.as_ref(), &run_id)?;

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

#[cfg(test)]
mod run_repository_tests {
    use super::*;

    struct FakeRunRepository {
        result: Result<String, RunRepositoryError>,
    }

    impl RunRepository for FakeRunRepository {
        fn load_log_path(&self, _: &str) -> Result<String, RunRepositoryError> {
            self.result.clone()
        }

        fn list_runs(
            &self,
            _: &RunListQuery,
        ) -> Result<Vec<mcts_bench::run_repository::RunSummary>, RunRepositoryError> {
            unreachable!()
        }

        fn load_run(
            &self,
            _: &str,
        ) -> Result<mcts_bench::run_repository::RunDetail, RunRepositoryError> {
            unreachable!()
        }

        fn load_leaderboard(
            &self,
            _: &LeaderboardQuery,
        ) -> Result<Vec<mcts_bench::run_repository::LeaderboardRow>, RunRepositoryError> {
            unreachable!()
        }

        fn load_experiment_cells(
            &self,
            _: &str,
        ) -> Result<Vec<mcts_bench::run_repository::ExperimentCell>, RunRepositoryError> {
            unreachable!()
        }

        fn ensure_run_exists(&self, _: &str) -> Result<(), RunRepositoryError> {
            unreachable!()
        }

        fn load_trials(
            &self,
            _: &str,
            _: &mcts_bench::run_repository::RunTrialsQuery,
        ) -> Result<Vec<mcts_bench::run_repository::RunTrial>, RunRepositoryError> {
            unreachable!()
        }

        fn load_games(
            &self,
            _: &str,
            _: &mcts_bench::run_repository::RunGamesQuery,
        ) -> Result<Vec<mcts_bench::run_repository::RunGame>, RunRepositoryError> {
            unreachable!()
        }

        fn load_game_moves(
            &self,
            _: &str,
            _: i64,
            _: Option<i64>,
        ) -> Result<Vec<mcts_bench::run_repository::RunGameMove>, RunRepositoryError> {
            unreachable!()
        }

        fn load_latest_game_seq(&self, _: &str) -> Result<Option<i64>, RunRepositoryError> {
            unreachable!()
        }

        fn load_deletion_info(
            &self,
            _: &str,
        ) -> Result<mcts_bench::run_repository::RunDeletionInfo, RunRepositoryError> {
            unreachable!()
        }

        fn delete_run_records(&self, _: &str, _: &[String]) -> Result<(), RunRepositoryError> {
            unreachable!()
        }
    }

    #[test]
    fn log_path_lookup_uses_logical_not_found_error() {
        let error = load_run_log_path(
            &FakeRunRepository {
                result: Err(RunRepositoryError::NotFound),
            },
            "missing",
        )
        .unwrap_err();

        assert_eq!(error.status, StatusCode::NOT_FOUND);
        assert_eq!(error.message, "run 'missing' not found");
    }

    #[test]
    fn log_path_lookup_uses_logical_storage_error() {
        let error = load_run_log_path(
            &FakeRunRepository {
                result: Err(RunRepositoryError::Storage("unavailable".into())),
            },
            "known",
        )
        .unwrap_err();

        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.message, "run storage error: unavailable");
    }
}
