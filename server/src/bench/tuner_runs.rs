use std::sync::Arc;

use std::io::{BufRead, BufReader, Seek, SeekFrom};

use axum::{
    extract::{Path as AxumPath, Query as AxumQuery, State as AxumState},
    http::StatusCode,
    response::Json,
};
use mcts_bench::tuner_launch::{
    self, BudgetExtension, TerminalOutcome, TunerLaunchRecord, TunerLaunchRequest,
};
use serde::Serialize;

use super::{BenchError, BenchState};

#[derive(Serialize)]
pub(crate) struct TunerRunView {
    run_id: String,
    argv: Vec<String>,
    run_dir: String,
    pid: Option<u32>,
    started_at: String,
    terminal_outcome: Option<TerminalOutcome>,
    status: &'static str,
}

fn view(record: TunerLaunchRecord) -> TunerRunView {
    let status = if record.terminal_outcome.is_some() {
        "exited"
    } else if record.pid.is_some_and(tuner_launch::is_alive) {
        "live"
    } else {
        "unknown"
    };
    TunerRunView {
        run_id: record.run_id,
        argv: record.argv,
        run_dir: record.run_dir.to_string_lossy().into_owned(),
        pid: record.pid,
        started_at: record.started_at,
        terminal_outcome: record.terminal_outcome,
        status,
    }
}

fn journal_error(error: std::io::Error) -> BenchError {
    BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("tuner launch journal error: {error}"),
    }
}

pub(crate) async fn launch_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    Json(mut request): Json<TunerLaunchRequest>,
) -> Result<(StatusCode, Json<TunerRunView>), BenchError> {
    request.runs_root = state.bench_runs_dir.clone();
    let record = tuner_launch::launch(&request).map_err(|error| BenchError {
        status: match error.kind() {
            std::io::ErrorKind::InvalidInput => StatusCode::BAD_REQUEST,
            std::io::ErrorKind::AlreadyExists => StatusCode::CONFLICT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        },
        message: format!("failed to launch tuner run: {error}"),
    })?;
    Ok((StatusCode::ACCEPTED, Json(view(record))))
}

pub(crate) async fn list_tuner_runs(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Result<Json<Vec<TunerRunView>>, BenchError> {
    Ok(Json(
        tuner_launch::records(&state.bench_runs_dir)
            .map_err(journal_error)?
            .into_iter()
            .map(view)
            .collect(),
    ))
}

pub(crate) async fn get_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<TunerRunView>, BenchError> {
    let record = tuner_launch::records(&state.bench_runs_dir)
        .map_err(journal_error)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' not found"),
        })?;
    Ok(Json(view(record)))
}

/// `POST /api/bench/tuner/runs/{run_id}/extend`
///
/// Raise one or more of a frozen run's pair budgets and resume it. The tuner
/// records the increase as one append-only `budget_extended` evidence event
/// (`manifest.compute_budget` is never edited) and continues the run; a run
/// that had already completed re-opens at its last cohort boundary.
pub(crate) async fn extend_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Json(extension): Json<BudgetExtension>,
) -> Result<(StatusCode, Json<TunerRunView>), BenchError> {
    let record =
        tuner_launch::extend(&state.bench_runs_dir, &run_id, &extension).map_err(|error| {
            BenchError {
                status: match error.kind() {
                    std::io::ErrorKind::InvalidInput => StatusCode::BAD_REQUEST,
                    std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                },
                message: format!("failed to extend tuner run: {error}"),
            }
        })?;
    Ok((StatusCode::ACCEPTED, Json(view(record))))
}

pub(crate) async fn stop_tuner_run(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<TunerRunView>, BenchError> {
    let record = tuner_launch::records(&state.bench_runs_dir)
        .map_err(journal_error)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' not found"),
        })?;
    if record.terminal_outcome.is_none() {
        if let Some(pid) = record.pid {
            match tuner_launch::interrupt(pid) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(journal_error(error)),
            }
        }
    }
    // A foreground tuner translates SIGINT to exit 130; its reaper writes the
    // terminal record. Until then this response deliberately remains `live`.
    get_tuner_run(AxumState(state), AxumPath(run_id)).await
}

#[derive(serde::Deserialize)]
pub(crate) struct TunerLogParams {
    #[serde(default)]
    since: u64,
}

#[derive(Serialize)]
pub(crate) struct TunerLogResponse {
    /// `launch.out` lines appended since the `since` byte offset.
    lines: Vec<String>,
    /// Byte offset to pass as `since` on the next poll.
    next_offset: u64,
    /// Full contents of `launch.err`, re-sent each poll (it is normally tiny:
    /// a panic backtrace or nothing).
    err_lines: Vec<String>,
}

/// `GET /api/bench/tuner/runs/{run_id}/log?since=<offset>`
///
/// Tail the detached tuner's `launch.out` from a byte offset, plus the full
/// `launch.err`. This reads operational files straight from the run-dir; it is
/// not scientific authority (that is the projection API) — it exists so the UI
/// can show what a run is doing in the seconds before the projection catches
/// up.
pub(crate) async fn get_tuner_run_log(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    AxumQuery(params): AxumQuery<TunerLogParams>,
) -> Result<Json<TunerLogResponse>, BenchError> {
    let record = tuner_launch::records(&state.bench_runs_dir)
        .map_err(journal_error)?
        .into_iter()
        .find(|record| record.run_id == run_id)
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' not found"),
        })?;

    let out_path = record.run_dir.join("launch.out");
    let (lines, next_offset) = tail_from(&out_path, params.since)?;
    let (err_lines, _) = tail_from(&record.run_dir.join("launch.err"), 0)?;
    Ok(Json(TunerLogResponse {
        lines,
        next_offset,
        err_lines,
    }))
}

fn tail_from(path: &std::path::Path, since: u64) -> Result<(Vec<String>, u64), BenchError> {
    let io_error = |error: std::io::Error| BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("failed to read {}: {error}", path.display()),
    };
    if !path.exists() {
        return Ok((vec![], 0));
    }
    let file_len = std::fs::metadata(path).map_err(io_error)?.len();
    if file_len <= since {
        return Ok((vec![], since));
    }
    let mut file = std::fs::File::open(path).map_err(io_error)?;
    file.seek(SeekFrom::Start(since)).map_err(io_error)?;
    let mut lines = Vec::new();
    for line in BufReader::new(file).lines() {
        lines.push(line.map_err(io_error)?);
    }
    Ok((lines, file_len))
}
