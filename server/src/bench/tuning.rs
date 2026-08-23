use std::sync::Arc;

use super::{
    BenchError, BenchState, TuningAttemptView, TuningCapabilities, TuningCursorBoundary,
    TuningGameView, TuningOpponentView, TuningPairView, TuningRatingView, TuningSessionDetail,
    TuningSessionSummary, TuningStrategyMetricsView, TuningTrialCounts, TuningTrialView,
};
use axum::{
    extract::{Path as AxumPath, State as AxumState},
    response::Json,
};
use duckdb::Connection;
use mcts_bench::tuning_lifecycle::{OpponentSnapshot, StrategyMetrics};

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
    let capabilities = load_capabilities(db, session_id)?;
    let manifest = decode_manifest(&session.manifest)?;

    Ok(Some(assemble_session_detail(
        session_id,
        session,
        counts,
        attempts,
        trials,
        manifest,
        capabilities,
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
    let mut query = db.prepare("SELECT trial_id, trial_number, attempt_id, status, CAST(config AS TEXT), score, mu, sigma, failure FROM tuning_trials WHERE session_id = ?1 ORDER BY trial_number")?;
    let rows: Vec<TrialRow> = query
        .query_map(duckdb::params![&session_id], |row| {
            Ok(TrialRow {
                trial_id: row.get(0)?,
                trial_number: row.get(1)?,
                attempt_id: row.get(2)?,
                status: row.get(3)?,
                config: row.get(4)?,
                score: row.get(5)?,
                mu: row.get(6)?,
                sigma: row.get(7)?,
                failure: row.get(8)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    rows.into_iter()
        .map(|row| assemble_trial_view(db, session_id, row))
        .collect()
}

struct TrialRow {
    trial_id: String,
    trial_number: i64,
    attempt_id: String,
    status: String,
    config: Option<String>,
    score: Option<f64>,
    mu: Option<f64>,
    sigma: Option<f64>,
    failure: Option<String>,
}

fn assemble_trial_view(
    db: &Connection,
    session_id: &str,
    row: TrialRow,
) -> Result<TuningTrialView, duckdb::Error> {
    Ok(TuningTrialView {
        trial_id: row.trial_id.clone(),
        trial_number: row.trial_number,
        attempt_id: row.attempt_id,
        status: row.status,
        config: decode_trial_config(row.config)?,
        score: row.score,
        mu: row.mu,
        sigma: row.sigma,
        failure: row.failure,
        pairs: load_pairs_for_trial(db, session_id, &row.trial_id)?,
    })
}

fn load_pairs_for_trial(
    db: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Vec<TuningPairView>, duckdb::Error> {
    let mut query = db.prepare("SELECT pair_id, pair_index, status, seed, round, CAST(opponent AS TEXT), pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, rating_after_mu, rating_after_sigma, score, failure FROM tuning_evaluation_pairs WHERE session_id = ?1 AND trial_id = ?2 ORDER BY pair_index")?;
    query
        .query_map(duckdb::params![session_id, trial_id], |row| {
            assemble_pair_view(db, session_id, PairRow::from_row(row)?)
        })?
        .collect()
}

struct PairRow {
    pair_id: String,
    pair_index: u32,
    status: String,
    seed: u64,
    round: u32,
    opponent: String,
    pool_snapshot_fingerprint: String,
    rating_before_mu: f64,
    rating_before_sigma: f64,
    rating_after_mu: Option<f64>,
    rating_after_sigma: Option<f64>,
    score: Option<f64>,
    failure: Option<String>,
}

impl PairRow {
    fn from_row(row: &duckdb::Row<'_>) -> Result<Self, duckdb::Error> {
        Ok(Self {
            pair_id: row.get(0)?,
            pair_index: row.get(1)?,
            status: row.get(2)?,
            seed: row.get(3)?,
            round: row.get(4)?,
            opponent: row.get(5)?,
            pool_snapshot_fingerprint: row.get(6)?,
            rating_before_mu: row.get(7)?,
            rating_before_sigma: row.get(8)?,
            rating_after_mu: row.get(9)?,
            rating_after_sigma: row.get(10)?,
            score: row.get(11)?,
            failure: row.get(12)?,
        })
    }
}

fn assemble_pair_view(
    db: &Connection,
    session_id: &str,
    row: PairRow,
) -> Result<TuningPairView, duckdb::Error> {
    let opponent = decode_json::<OpponentSnapshot>(&row.opponent, 5)?;
    Ok(TuningPairView {
        pair_id: row.pair_id.clone(),
        pair_index: row.pair_index,
        status: row.status,
        seed: row.seed,
        round: row.round,
        opponent: opponent_view(opponent),
        pool_snapshot_fingerprint: row.pool_snapshot_fingerprint,
        rating_before: rating_view(row.rating_before_mu, row.rating_before_sigma),
        rating_after: row
            .rating_after_mu
            .zip(row.rating_after_sigma)
            .map(|(mu, sigma)| rating_view(mu, sigma)),
        score: row.score,
        failure: row.failure,
        games: load_games_for_pair(db, session_id, &row.pair_id)?,
    })
}

fn load_games_for_pair(
    db: &Connection,
    session_id: &str,
    pair_id: &str,
) -> Result<Vec<TuningGameView>, duckdb::Error> {
    let mut query = db.prepare("SELECT game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, CAST(candidate_metrics AS TEXT), CAST(baseline_metrics AS TEXT) FROM tuning_games WHERE session_id = ?1 AND pair_id = ?2 ORDER BY candidate_side")?;
    query
        .query_map(duckdb::params![session_id, pair_id], |row| {
            let candidate: String = row.get(8)?;
            let baseline: String = row.get(9)?;
            Ok(TuningGameView {
                game_id: row.get(0)?,
                candidate_side: row.get(1)?,
                outcome: row.get(2)?,
                seed: row.get(3)?,
                round: row.get(4)?,
                trace_game_seq: row.get(5)?,
                plies: row.get(6)?,
                elapsed_ms: row.get(7)?,
                candidate: metrics_view(decode_json(&candidate, 8)?),
                baseline: metrics_view(decode_json(&baseline, 9)?),
            })
        })?
        .collect()
}

fn load_capabilities(
    db: &Connection,
    session_id: &str,
) -> Result<TuningCapabilities, duckdb::Error> {
    db.query_row(
        "SELECT COUNT(*), COUNT(trace_game_seq) FROM tuning_evaluation_pairs LEFT JOIN tuning_games USING (session_id, pair_id) WHERE session_id = ?1",
        duckdb::params![session_id],
        |row| Ok(TuningCapabilities { has_lifecycle: true, has_pairs: row.get::<_, i64>(0)? > 0, has_renderer_trace: row.get::<_, i64>(1)? > 0, has_search_reports: false }),
    )
}

fn decode_json<T: serde::de::DeserializeOwned>(
    json: &str,
    column: usize,
) -> Result<T, duckdb::Error> {
    serde_json::from_str(json).map_err(|error| {
        duckdb::Error::FromSqlConversionFailure(column, duckdb::types::Type::Text, Box::new(error))
    })
}

fn opponent_view(value: OpponentSnapshot) -> TuningOpponentView {
    TuningOpponentView {
        anchor_id: value.anchor_id,
        config: value.config,
        mu: value.mu,
        sigma: value.sigma,
        label: value.label,
        provenance: value.provenance,
    }
}

fn rating_view(mu: f64, sigma: f64) -> TuningRatingView {
    TuningRatingView { mu, sigma }
}

fn metrics_view(value: StrategyMetrics) -> TuningStrategyMetricsView {
    TuningStrategyMetricsView {
        iterations_total: value.iterations_total,
        iterations_first_half: value.iterations_first_half,
        move_time_ms: value.move_time_ms,
    }
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
    capabilities: TuningCapabilities,
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
        capabilities,
        cursor: TuningCursorBoundary {
            session_sequence: session.last_sequence,
        },
    }
}
