use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State as AxumState},
    http::StatusCode,
    response::Json,
};
use mcts_bench::tuner_launch::{self, TerminalOutcome, TunerLaunchRecord, TunerLaunchRequest};
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
