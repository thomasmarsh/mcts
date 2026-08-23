use std::{collections::HashMap, sync::Arc};

use super::{
    BenchError, BenchState, TuningAttemptSummary, TuningAttemptView, TuningCapabilities,
    TuningCursorBoundary, TuningGameView, TuningOpponentView, TuningPairView, TuningPolicyView,
    TuningPruningPolicyView, TuningRatingPolicyView, TuningRatingView, TuningResourcePolicyView,
    TuningSamplerPolicyView, TuningSessionDetail, TuningSessionList, TuningSessionListItem,
    TuningSessionSummary, TuningStrategyMetricsView, TuningTrialCounts,
    TuningTrialReportDecisionView, TuningTrialReportView, TuningTrialView,
};
use axum::{
    extract::{Path as AxumPath, State as AxumState},
    response::Json,
};
use duckdb::Connection;
use mcts_bench::tuning_lifecycle::{
    OpponentSnapshot, StrategyMetrics, TrialReportOutcome, TrialReportReason,
};
use serde::Deserialize;

pub(crate) async fn get_tuning_sessions(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Result<Json<TuningSessionList>, BenchError> {
    let db = state.db.lock().unwrap();
    Ok(Json(load_tuning_session_list(&db)?))
}

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

fn load_tuning_session_list(db: &Connection) -> Result<TuningSessionList, BenchError> {
    let sessions = load_tuning_session_list_rows(db)?;
    let attempts = load_tuning_session_list_attempts(db)?;
    assemble_tuning_session_list(sessions, attempts)
}

fn load_tuning_session_list_rows(
    db: &Connection,
) -> Result<Vec<TuningSessionListRow>, duckdb::Error> {
    let mut query = db.prepare(
        "WITH trial_counts AS ( \
             SELECT session_id, COUNT(*) AS total, \
                    SUM(status = 'queued') AS queued, SUM(status = 'running') AS running, \
                    SUM(status IN ('complete', 'failed', 'pruned', 'cancelled')) AS terminal, \
                    SUM(status = 'complete') AS completed, SUM(status = 'failed') AS failed, \
                    SUM(status = 'pruned') AS pruned, SUM(status = 'cancelled') AS cancelled \
             FROM tuning_trials GROUP BY session_id \
         ), pair_capabilities AS ( \
             SELECT pairs.session_id, COUNT(*) AS pair_count, \
                    COUNT(DISTINCT renderer_moves.run_id || ':' || renderer_moves.game_seq) AS renderer_trace_count, \
                    COUNT(DISTINCT report_moves.run_id || ':' || report_moves.game_seq) AS search_report_count \
             FROM tuning_evaluation_pairs pairs \
             LEFT JOIN tuning_games games USING (session_id, pair_id) \
             LEFT JOIN tuning_attempts attempts ON attempts.attempt_id = pairs.attempt_id \
             LEFT JOIN (SELECT DISTINCT run_id, game_seq FROM game_moves WHERE trace_schema_version = 1) renderer_moves \
                    ON renderer_moves.run_id = attempts.bench_run_id AND renderer_moves.game_seq = games.trace_game_seq \
             LEFT JOIN (SELECT DISTINCT run_id, game_seq FROM game_moves WHERE search_report IS NOT NULL) report_moves \
                    ON report_moves.run_id = attempts.bench_run_id AND report_moves.game_seq = games.trace_game_seq \
             GROUP BY pairs.session_id \
         ), trial_report_capabilities AS ( \
             SELECT session_id, COUNT(*) AS trial_report_count FROM tuning_trial_reports GROUP BY session_id \
         ), activity AS ( \
             SELECT session_id, created_at AS occurred_at FROM tuning_sessions \
             UNION ALL SELECT session_id, started_at FROM tuning_attempts \
             UNION ALL SELECT session_id, ended_at FROM tuning_attempts WHERE ended_at IS NOT NULL \
             UNION ALL SELECT session_id, created_at FROM tuning_trials \
             UNION ALL SELECT session_id, started_at FROM tuning_trials WHERE started_at IS NOT NULL \
             UNION ALL SELECT session_id, ended_at FROM tuning_trials WHERE ended_at IS NOT NULL \
             UNION ALL SELECT session_id, started_at FROM tuning_evaluation_pairs \
             UNION ALL SELECT session_id, ended_at FROM tuning_evaluation_pairs WHERE ended_at IS NOT NULL \
             UNION ALL SELECT session_id, finished_at FROM tuning_games \
         ) \
         SELECT sessions.session_id, sessions.status, sessions.target_trial_count, \
                CAST(sessions.manifest AS TEXT), CAST(sessions.created_at AS TEXT), \
                CAST(MAX(activity.occurred_at) AS TEXT), \
                COALESCE(trial_counts.total, 0), COALESCE(trial_counts.queued, 0), \
                COALESCE(trial_counts.running, 0), COALESCE(trial_counts.terminal, 0), \
                COALESCE(trial_counts.completed, 0), COALESCE(trial_counts.failed, 0), \
                COALESCE(trial_counts.pruned, 0), COALESCE(trial_counts.cancelled, 0), \
                COALESCE(pair_capabilities.pair_count, 0), COALESCE(pair_capabilities.renderer_trace_count, 0), \
                COALESCE(pair_capabilities.search_report_count, 0), COALESCE(trial_report_capabilities.trial_report_count, 0) \
         FROM tuning_sessions sessions \
         LEFT JOIN trial_counts USING (session_id) \
         LEFT JOIN pair_capabilities USING (session_id) \
         LEFT JOIN trial_report_capabilities USING (session_id) \
         JOIN activity USING (session_id) \
         GROUP BY sessions.session_id, sessions.status, sessions.target_trial_count, sessions.manifest, \
                  sessions.created_at, trial_counts.total, trial_counts.queued, trial_counts.running, \
                  trial_counts.terminal, trial_counts.completed, trial_counts.failed, trial_counts.pruned, \
                  trial_counts.cancelled, pair_capabilities.pair_count, pair_capabilities.renderer_trace_count, \
                  pair_capabilities.search_report_count, trial_report_capabilities.trial_report_count \
         ORDER BY MAX(activity.occurred_at) DESC, sessions.session_id DESC",
    )?;
    query
        .query_map([], TuningSessionListRow::from_row)?
        .collect()
}

fn load_tuning_session_list_attempts(
    db: &Connection,
) -> Result<Vec<TuningSessionAttemptRow>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT session_id, attempt_id, bench_run_id, status, CAST(started_at AS TEXT), \
                CAST(ended_at AS TEXT), failure \
         FROM tuning_attempts ORDER BY session_id, started_at ASC, attempt_id ASC",
    )?;
    query
        .query_map([], TuningSessionAttemptRow::from_row)?
        .collect()
}

fn assemble_tuning_session_list(
    sessions: Vec<TuningSessionListRow>,
    attempts: Vec<TuningSessionAttemptRow>,
) -> Result<TuningSessionList, BenchError> {
    let mut attempts_by_session = group_tuning_session_attempts(attempts);
    let sessions = sessions
        .into_iter()
        .map(|row| {
            let attempts = attempts_by_session
                .remove(&row.session_id)
                .unwrap_or_default();
            assemble_tuning_session_list_item(row, attempts)
        })
        .collect::<Result<_, _>>()?;
    Ok(TuningSessionList {
        schema_version: 1,
        sessions,
    })
}

fn group_tuning_session_attempts(
    rows: Vec<TuningSessionAttemptRow>,
) -> HashMap<String, Vec<TuningAttemptSummary>> {
    let mut grouped = HashMap::new();
    for row in rows {
        let session_id = row.session_id.clone();
        grouped
            .entry(session_id)
            .or_insert_with(Vec::new)
            .push(row.into_summary());
    }
    grouped
}

fn assemble_tuning_session_list_item(
    row: TuningSessionListRow,
    attempts: Vec<TuningAttemptSummary>,
) -> Result<TuningSessionListItem, BenchError> {
    let display = decode_manifest_display(&row.manifest)?;
    Ok(TuningSessionListItem {
        session_id: row.session_id.clone(),
        game: display.game,
        label: display.label,
        status: row.status,
        target_trial_count: row.target_trial_count,
        counts: row.counts,
        created_at: row.created_at,
        last_activity_at: row.last_activity_at,
        attempts,
        capabilities: row.capabilities,
    })
}

struct TuningSessionListRow {
    session_id: String,
    status: String,
    target_trial_count: Option<i64>,
    manifest: String,
    created_at: String,
    last_activity_at: String,
    counts: TuningTrialCounts,
    capabilities: TuningCapabilities,
}

impl TuningSessionListRow {
    fn from_row(row: &duckdb::Row<'_>) -> Result<Self, duckdb::Error> {
        Ok(Self {
            session_id: row.get(0)?,
            status: row.get(1)?,
            target_trial_count: row.get(2)?,
            manifest: row.get(3)?,
            created_at: row.get(4)?,
            last_activity_at: row.get(5)?,
            counts: TuningTrialCounts {
                total: row.get(6)?,
                queued: row.get(7)?,
                running: row.get(8)?,
                terminal: row.get(9)?,
                completed: row.get(10)?,
                failed: row.get(11)?,
                pruned: row.get(12)?,
                cancelled: row.get(13)?,
            },
            capabilities: TuningCapabilities {
                has_lifecycle: true,
                has_pairs: row.get::<_, i64>(14)? > 0,
                has_renderer_trace: row.get::<_, i64>(15)? > 0,
                has_search_reports: row.get::<_, i64>(16)? > 0,
                has_trial_reports: row.get::<_, i64>(17)? > 0,
            },
        })
    }
}

struct TuningSessionAttemptRow {
    session_id: String,
    attempt_id: String,
    bench_run_id: Option<String>,
    status: String,
    started_at: String,
    ended_at: Option<String>,
    failure: Option<String>,
}

impl TuningSessionAttemptRow {
    fn from_row(row: &duckdb::Row<'_>) -> Result<Self, duckdb::Error> {
        Ok(Self {
            session_id: row.get(0)?,
            attempt_id: row.get(1)?,
            bench_run_id: row.get(2)?,
            status: row.get(3)?,
            started_at: row.get(4)?,
            ended_at: row.get(5)?,
            failure: row.get(6)?,
        })
    }

    fn into_summary(self) -> TuningAttemptSummary {
        TuningAttemptSummary {
            attempt_id: self.attempt_id,
            bench_run_id: self.bench_run_id,
            status: self.status,
            started_at: self.started_at,
            ended_at: self.ended_at,
            failure: self.failure,
        }
    }
}

#[derive(Deserialize)]
struct TuningManifestDisplayEnvelope {
    #[serde(default)]
    semantic_inputs: Option<TuningManifestSemanticInputs>,
}

#[derive(Deserialize)]
struct TuningManifestSemanticInputs {
    #[serde(default)]
    game: Option<TuningManifestGame>,
}

#[derive(Deserialize)]
struct TuningManifestGame {
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    label: Option<String>,
}

struct TuningManifestDisplay {
    game: Option<String>,
    label: Option<String>,
}

fn decode_manifest_display(manifest: &str) -> Result<TuningManifestDisplay, BenchError> {
    let manifest: TuningManifestDisplayEnvelope = serde_json::from_str(manifest)?;
    let game = manifest.semantic_inputs.and_then(|inputs| inputs.game);
    Ok(TuningManifestDisplay {
        game: game.as_ref().and_then(|value| value.kind.clone()),
        label: game.and_then(|value| value.label),
    })
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
    let reports = load_trial_reports(db, session_id)?;
    let trials = load_trials(db, session_id, reports)?;
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
    )?))
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

fn load_trials(
    db: &Connection,
    session_id: &str,
    mut reports_by_trial: HashMap<String, Vec<TuningTrialReportView>>,
) -> Result<Vec<TuningTrialView>, duckdb::Error> {
    let mut query = db.prepare("SELECT trial_id, trial_number, attempt_id, status, CAST(config AS TEXT), score, mu, sigma, stop_reason, failure FROM tuning_trials WHERE session_id = ?1 ORDER BY trial_number")?;
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
                stop_reason: decode_optional_report_reason(row.get(8)?, 8)?,
                failure: row.get(9)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    rows.into_iter()
        .map(|row| {
            let reports = reports_by_trial.remove(&row.trial_id).unwrap_or_default();
            assemble_trial_view(db, session_id, row, reports)
        })
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
    stop_reason: Option<TrialReportReason>,
    failure: Option<String>,
}

fn assemble_trial_view(
    db: &Connection,
    session_id: &str,
    row: TrialRow,
    reports: Vec<TuningTrialReportView>,
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
        stop_reason: row.stop_reason,
        failure: row.failure,
        pairs: load_pairs_for_trial(db, session_id, &row.trial_id)?,
        reports,
    })
}

fn load_trial_reports(
    db: &Connection,
    session_id: &str,
) -> Result<HashMap<String, Vec<TuningTrialReportView>>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT trial_id, completed_pairs, CAST(reported_at AS TEXT), mu, sigma, score, \
                score_formula_version, conservative_k, outcome, reason, pruning_exempt, \
                bracket_id, rung_resource \
         FROM tuning_trial_reports WHERE session_id = ?1 \
         ORDER BY trial_number, completed_pairs, event_id",
    )?;
    let reports: Vec<(String, TuningTrialReportView)> = query
        .query_map(duckdb::params![session_id], |row| {
            let outcome: String = row.get(8)?;
            let reason: String = row.get(9)?;
            Ok((
                row.get(0)?,
                TuningTrialReportView {
                    completed_pairs: row.get(1)?,
                    reported_at: row.get(2)?,
                    rating: rating_view(row.get(3)?, row.get(4)?),
                    score: row.get(5)?,
                    score_formula_version: row.get(6)?,
                    conservative_k: row.get(7)?,
                    decision: TuningTrialReportDecisionView {
                        outcome: decode_report_enum(&outcome, 8)?,
                        reason: decode_report_enum(&reason, 9)?,
                        pruning_exempt: row.get(10)?,
                        bracket_id: row.get(11)?,
                        rung_resource: row.get(12)?,
                    },
                },
            ))
        })?
        .collect::<Result<_, _>>()?;

    let mut reports_by_trial = HashMap::new();
    for (trial_id, report) in reports {
        reports_by_trial
            .entry(trial_id)
            .or_insert_with(Vec::new)
            .push(report);
    }
    Ok(reports_by_trial)
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
        "WITH joined_games AS ( \
             SELECT pairs.session_id, pairs.attempt_id, games.trace_game_seq \
             FROM tuning_evaluation_pairs pairs \
             LEFT JOIN tuning_games games USING (session_id, pair_id) \
             WHERE pairs.session_id = ?1 \
         ), renderer_moves AS ( \
             SELECT DISTINCT run_id, game_seq FROM game_moves WHERE trace_schema_version = 1 \
         ), report_moves AS ( \
             SELECT DISTINCT run_id, game_seq FROM game_moves WHERE search_report IS NOT NULL \
         ) \
         SELECT COUNT(*), \
                COUNT(DISTINCT renderer_moves.run_id || ':' || renderer_moves.game_seq), \
                COUNT(DISTINCT report_moves.run_id || ':' || report_moves.game_seq), \
                (SELECT COUNT(*) FROM tuning_trial_reports WHERE session_id = ?1) \
         FROM joined_games \
         LEFT JOIN tuning_attempts attempts ON attempts.attempt_id = joined_games.attempt_id \
         LEFT JOIN renderer_moves ON renderer_moves.run_id = attempts.bench_run_id AND renderer_moves.game_seq = joined_games.trace_game_seq \
         LEFT JOIN report_moves ON report_moves.run_id = attempts.bench_run_id AND report_moves.game_seq = joined_games.trace_game_seq",
        duckdb::params![session_id],
        |row| {
            Ok(TuningCapabilities {
                has_lifecycle: true,
                has_pairs: row.get::<_, i64>(0)? > 0,
                has_renderer_trace: row.get::<_, i64>(1)? > 0,
                has_search_reports: row.get::<_, i64>(2)? > 0,
                has_trial_reports: row.get::<_, i64>(3)? > 0,
            })
        },
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

fn decode_report_enum<T: serde::de::DeserializeOwned>(
    value: &str,
    column: usize,
) -> Result<T, duckdb::Error> {
    serde_json::from_value(serde_json::Value::String(value.to_owned())).map_err(|error| {
        duckdb::Error::FromSqlConversionFailure(column, duckdb::types::Type::Text, Box::new(error))
    })
}

fn decode_optional_report_reason(
    value: Option<String>,
    column: usize,
) -> Result<Option<TrialReportReason>, duckdb::Error> {
    value
        .as_deref()
        .map(|reason| decode_report_enum(reason, column))
        .transpose()
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

#[derive(Deserialize)]
struct TuningManifestPolicyEnvelope {
    #[serde(default)]
    semantic_inputs: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Deserialize)]
struct TuningManifestPolicyInputs {
    optimizer: TuningManifestOptimizerPolicy,
    rating: TuningRatingPolicyView,
}

#[derive(Deserialize)]
struct TuningManifestOptimizerPolicy {
    resource: TuningResourcePolicyView,
    sampler: TuningManifestSamplerPolicy,
    pruning: TuningPruningPolicyView,
}

#[derive(Deserialize)]
struct TuningManifestSamplerPolicy {
    kind: String,
    seed: u64,
    deterministic: bool,
    startup_trials: u64,
}

fn decode_manifest_policy(
    manifest: &serde_json::Value,
) -> Result<Option<TuningPolicyView>, BenchError> {
    let envelope: TuningManifestPolicyEnvelope = serde_json::from_value(manifest.clone())?;
    let Some(inputs) = envelope.semantic_inputs else {
        return Ok(None);
    };
    if !inputs.contains_key("optimizer") && !inputs.contains_key("rating") {
        return Ok(None);
    }
    let inputs: TuningManifestPolicyInputs =
        serde_json::from_value(serde_json::Value::Object(inputs))?;
    Ok(Some(TuningPolicyView {
        resource: inputs.optimizer.resource,
        rating: inputs.rating,
        sampler: TuningSamplerPolicyView {
            kind: inputs.optimizer.sampler.kind,
            seed: inputs.optimizer.sampler.seed,
            deterministic: inputs.optimizer.sampler.deterministic,
            startup_trials: inputs.optimizer.sampler.startup_trials,
        },
        pruning: inputs.optimizer.pruning,
    }))
}

fn assemble_session_detail(
    session_id: &str,
    session: TuningSessionRow,
    counts: TuningTrialCounts,
    attempts: Vec<TuningAttemptView>,
    trials: Vec<TuningTrialView>,
    manifest: serde_json::Value,
    capabilities: TuningCapabilities,
) -> Result<TuningSessionDetail, BenchError> {
    let policy = decode_manifest_policy(&manifest)?;
    Ok(TuningSessionDetail {
        schema_version: 1,
        summary: TuningSessionSummary {
            session_id: session_id.to_owned(),
            status: session.status,
            target_trial_count: session.target_trial_count,
            counts,
        },
        attempts,
        trials,
        policy,
        manifest,
        fingerprint: session.fingerprint,
        capabilities,
        cursor: TuningCursorBoundary {
            session_sequence: session.last_sequence,
        },
    })
}
