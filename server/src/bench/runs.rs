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

use super::types::*;
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
            tuning_session_id: run.tuning_session_id,
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
        tuning_session_id: run.tuning_session_id,
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

fn run_repository_error(error: RunRepositoryError) -> BenchError {
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

fn run_repository_error_for_run(error: RunRepositoryError, run_id: &str) -> BenchError {
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

/// `GET /api/bench/leaderboard?game=&git_sha=&since=`
///
/// Aggregated win-rate + Wilson CI over `match_results`.  Computed at query
/// time — no materialized view.
pub(crate) async fn get_leaderboard(
    AxumState(state): AxumState<Arc<BenchState>>,
    Query(params): Query<LeaderboardParams>,
) -> Result<Json<Vec<LeaderboardEntry>>, BenchError> {
    let query = LeaderboardQuery {
        game: params.game,
        git_sha: params.git_sha,
        since: params.since,
    };
    let entries = state
        .run_repository
        .load_leaderboard(&query)
        .map_err(run_repository_error)?
        .into_iter()
        .map(|entry| {
            let total = entry.total as usize;
            let wins = entry.wins as usize;
            let losses = entry.losses as usize;
            let draws = entry.draws as usize;
            let score = wins as f64 + 0.5 * draws as f64;
            let (win_rate, (ci_lower, ci_upper)) = if total > 0 {
                (score / total as f64, wilson_interval(score, total, 1.96))
            } else {
                (0.5, (0.0, 1.0))
            };

            LeaderboardEntry {
                strategy: entry.strategy,
                total,
                wins,
                losses,
                draws,
                win_rate,
                ci_lower,
                ci_upper,
            }
        })
        .collect();

    Ok(Json(entries))
}

pub(crate) async fn get_run_cells(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Vec<CellResponse>>, BenchError> {
    let cells = state
        .run_repository
        .load_experiment_cells(&run_id)
        .map_err(run_repository_error)?;
    let mut result = Vec::with_capacity(cells.len());
    for cell in cells {
        let mut wins = 0_u64;
        let mut losses = 0_u64;
        let mut draws = 0_u64;
        for outcome in cell.match_outcomes.iter().flatten() {
            match outcome.as_str() {
                "candidate_win" => wins += 1,
                "baseline_win" => losses += 1,
                "draw" => draws += 1,
                _ => {}
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
            cell_id: cell.cell_id,
            cell_seed: cell.cell_seed,
            game: cell.game,
            game_config: cell.game_config,
            variant_id: cell.variant_id,
            variant_label: cell.variant_label,
            candidate_config: cell.candidate_config,
            baseline_id: cell.baseline_id,
            baseline_label: cell.baseline_label,
            baseline_config: cell.baseline_config,
            budget: cell.budget,
            rounds: cell.rounds,
            planned_games: cell.planned_games,
            completed_games: cell.completed_games,
            status: cell.status,
            started_at: cell.started_at,
            ended_at: cell.ended_at,
            error: cell.error,
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
