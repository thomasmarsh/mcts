use std::sync::Arc;

use super::{
    BenchError, BenchState, TuningAttemptView, TuningCapabilities, TuningCursorBoundary,
    TuningSessionDetail, TuningSessionSummary, TuningTrialCounts, TuningTrialView,
};
use axum::{
    extract::{Path as AxumPath, State as AxumState},
    response::Json,
};
use duckdb::Connection;

pub(crate) async fn get_tuning_session(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Result<Json<TuningSessionDetail>, BenchError> {
    let db = state.db.lock().unwrap();
    let detail = load_tuning_session_detail(&db, &session_id)?.ok_or_else(|| BenchError {
        status: axum::http::StatusCode::NOT_FOUND,
        message: format!("tuning session '{session_id}' not found"),
    })?;
    Ok(Json(detail))
}

struct TuningSessionRow {
    status: String,
    target_trial_count: Option<i64>,
    manifest: String,
    fingerprint: Option<String>,
    last_sequence: i64,
}

fn load_tuning_session_detail(
    db: &Connection,
    session_id: &str,
) -> Result<Option<TuningSessionDetail>, BenchError> {
    let Some(session) = load_session(db, session_id)? else {
        return Ok(None);
    };
    let counts = load_trial_counts(db, session_id)?;
    let attempts = load_attempts(db, session_id)?;
    let trials = load_trials(db, session_id)?;
    let manifest = decode_manifest(&session.manifest)?;

    Ok(Some(assemble_session_detail(
        session_id, session, counts, attempts, trials, manifest,
    )))
}

fn load_session(
    db: &Connection,
    session_id: &str,
) -> Result<Option<TuningSessionRow>, duckdb::Error> {
    match db.query_row(
        "SELECT status, target_trial_count, CAST(manifest AS TEXT), manifest_fingerprint, last_sequence FROM tuning_sessions WHERE session_id = ?1",
        duckdb::params![&session_id],
        |row| {
            Ok(TuningSessionRow {
                status: row.get(0)?,
                target_trial_count: row.get(1)?,
                manifest: row.get(2)?,
                fingerprint: row.get(3)?,
                last_sequence: row.get(4)?,
            })
        },
    ) {
        Ok(session) => Ok(Some(session)),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error),
    }
}

fn load_trial_counts(
    db: &Connection,
    session_id: &str,
) -> Result<TuningTrialCounts, duckdb::Error> {
    db.query_row(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status IN ('complete', 'failed', 'pruned', 'cancelled') THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'pruned' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END), 0) FROM tuning_trials WHERE session_id = ?1",
        duckdb::params![&session_id],
        |row| {
            Ok(TuningTrialCounts {
                total: row.get(0)?,
                queued: row.get(1)?,
                running: row.get(2)?,
                terminal: row.get(3)?,
                completed: row.get(4)?,
                failed: row.get(5)?,
                pruned: row.get(6)?,
                cancelled: row.get(7)?,
            })
        },
    )
}

fn load_attempts(
    db: &Connection,
    session_id: &str,
) -> Result<Vec<TuningAttemptView>, duckdb::Error> {
    let mut attempts_query = db.prepare("SELECT attempt_id, bench_run_id, status, CAST(started_at AS TEXT), CAST(ended_at AS TEXT), failure FROM tuning_attempts WHERE session_id = ?1 ORDER BY started_at")?;
    attempts_query
        .query_map(duckdb::params![&session_id], |row| {
            Ok(TuningAttemptView {
                attempt_id: row.get(0)?,
                bench_run_id: row.get(1)?,
                status: row.get(2)?,
                started_at: row.get(3)?,
                ended_at: row.get(4)?,
                failure: row.get(5)?,
            })
        })?
        .collect()
}

fn load_trials(db: &Connection, session_id: &str) -> Result<Vec<TuningTrialView>, duckdb::Error> {
    let mut trials_query = db.prepare("SELECT trial_id, trial_number, attempt_id, status, CAST(config AS TEXT), score, mu, sigma, failure FROM tuning_trials WHERE session_id = ?1 ORDER BY trial_number")?;
    trials_query
        .query_map(duckdb::params![&session_id], |row| {
            let config: Option<String> = row.get(4)?;
            Ok(TuningTrialView {
                trial_id: row.get(0)?,
                trial_number: row.get(1)?,
                attempt_id: row.get(2)?,
                status: row.get(3)?,
                config: decode_trial_config(config)?,
                score: row.get(5)?,
                mu: row.get(6)?,
                sigma: row.get(7)?,
                failure: row.get(8)?,
            })
        })?
        .collect()
}

fn decode_trial_config(config: Option<String>) -> Result<Option<serde_json::Value>, duckdb::Error> {
    config
        .map(|json| serde_json::from_str(&json))
        .transpose()
        .map_err(|error| {
            duckdb::Error::FromSqlConversionFailure(4, duckdb::types::Type::Text, Box::new(error))
        })
}

fn decode_manifest(manifest: &str) -> Result<serde_json::Value, BenchError> {
    Ok(serde_json::from_str(manifest)?)
}

fn assemble_session_detail(
    session_id: &str,
    session: TuningSessionRow,
    counts: TuningTrialCounts,
    attempts: Vec<TuningAttemptView>,
    trials: Vec<TuningTrialView>,
    manifest: serde_json::Value,
) -> TuningSessionDetail {
    TuningSessionDetail {
        schema_version: 1,
        summary: TuningSessionSummary {
            session_id: session_id.to_owned(),
            status: session.status,
            target_trial_count: session.target_trial_count,
            counts,
        },
        attempts,
        trials,
        manifest,
        fingerprint: session.fingerprint,
        capabilities: TuningCapabilities {
            has_lifecycle: true,
            has_pairs: false,
            has_renderer_trace: false,
            has_search_reports: false,
        },
        cursor: TuningCursorBoundary {
            session_sequence: session.last_sequence,
        },
    }
}
