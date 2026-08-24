use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path as AxumPath, State as AxumState},
    http::StatusCode,
    response::Json,
};
use mcts_bench::tuning_lifecycle::{OpponentSnapshot, StrategyMetrics};
use mcts_bench::tuning_session_repository::{
    TuningSessionCapabilities as RepositoryCapabilities, TuningSessionDetailData,
    TuningSessionGameRow, TuningSessionListData, TuningSessionListRow, TuningSessionPairRow,
    TuningSessionRepository, TuningSessionRepositoryError, TuningSessionTrialReportRow,
    TuningSessionTrialRow, TuningTrialCountsRow,
};
use serde::Deserialize;

use super::super::{
    BenchError, BenchState, TuningAttemptSummary, TuningAttemptView, TuningCapabilities,
    TuningCursorBoundary, TuningGameView, TuningOpponentView, TuningPairView, TuningPolicyView,
    TuningPruningPolicyView, TuningRatingPolicyView, TuningRatingView, TuningResourcePolicyView,
    TuningSamplerPolicyView, TuningSessionControl, TuningSessionDetail, TuningSessionList,
    TuningSessionListItem, TuningSessionSummary, TuningStrategyMetricsView, TuningTrialCounts,
    TuningTrialReportDecisionView, TuningTrialReportView, TuningTrialView,
};
use super::commands::session_control;

pub(crate) async fn get_tuning_sessions(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Result<Json<TuningSessionList>, BenchError> {
    let data = state
        .tuning_session_repository
        .load_session_list()
        .map_err(tuning_session_repository_error)?;
    let db = state.db.lock().unwrap();
    Ok(Json(load_tuning_session_list(data, &db)?))
}

pub(crate) async fn get_tuning_session(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Result<Json<TuningSessionDetail>, BenchError> {
    let control = {
        let db = state.db.lock().unwrap();
        session_control(&db, &session_id)?
    };
    let detail = state
        .tuning_session_repository
        .load_session_detail(&session_id)
        .map_err(tuning_session_repository_error)?
        .map(|data| assemble_session_detail(data, control))
        .transpose()?
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuning session '{session_id}' not found"),
        })?;
    Ok(Json(detail))
}

fn tuning_session_repository_error(error: TuningSessionRepositoryError) -> BenchError {
    BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("tuning session storage error: {error}"),
    }
}

fn load_tuning_session_list(
    data: TuningSessionListData,
    db: &duckdb::Connection,
) -> Result<TuningSessionList, BenchError> {
    let controls = data
        .sessions
        .iter()
        .map(|session| {
            session_control(db, &session.session_id)
                .map(|control| (session.session_id.clone(), control))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
    assemble_tuning_session_list(data, controls)
}

fn assemble_tuning_session_list(
    data: TuningSessionListData,
    mut controls: HashMap<String, TuningSessionControl>,
) -> Result<TuningSessionList, BenchError> {
    let mut attempts_by_session = HashMap::<String, Vec<TuningAttemptSummary>>::new();
    for attempt in data.attempts {
        attempts_by_session
            .entry(attempt.session_id)
            .or_default()
            .push(TuningAttemptSummary {
                attempt_id: attempt.attempt_id,
                bench_run_id: attempt.bench_run_id,
                status: attempt.status,
                started_at: attempt.started_at,
                ended_at: attempt.ended_at,
                failure: attempt.failure,
            });
    }
    let sessions = data
        .sessions
        .into_iter()
        .map(|row| {
            let control = controls.remove(&row.session_id).ok_or_else(|| BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!(
                    "missing control projection for tuning session '{}'",
                    row.session_id
                ),
            })?;
            let display = decode_manifest_display(&row.manifest)?;
            let capabilities = capabilities_from_list(&row);
            Ok(TuningSessionListItem {
                session_id: row.session_id.clone(),
                game: display.game,
                label: display.label,
                status: row.status,
                target_trial_count: row.target_trial_count,
                counts: trial_counts(row.trial_counts),
                created_at: row.created_at,
                last_activity_at: row.last_activity_at,
                attempts: attempts_by_session
                    .remove(&row.session_id)
                    .unwrap_or_default(),
                capabilities,
                control,
            })
        })
        .collect::<Result<_, BenchError>>()?;
    Ok(TuningSessionList {
        schema_version: 1,
        sessions,
    })
}

fn trial_counts(row: TuningTrialCountsRow) -> TuningTrialCounts {
    TuningTrialCounts {
        total: row.total,
        queued: row.queued,
        running: row.running,
        terminal: row.terminal,
        completed: row.completed,
        failed: row.failed,
        pruned: row.pruned,
        cancelled: row.cancelled,
    }
}

fn capabilities_from_list(row: &TuningSessionListRow) -> TuningCapabilities {
    TuningCapabilities {
        has_lifecycle: true,
        has_pairs: row.pair_count > 0,
        has_renderer_trace: row.renderer_trace_count > 0,
        has_search_reports: row.search_report_count > 0,
        has_trial_reports: row.trial_report_count > 0,
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

fn assemble_session_detail(
    data: TuningSessionDetailData,
    control: TuningSessionControl,
) -> Result<TuningSessionDetail, BenchError> {
    let manifest = decode_manifest(&data.session.manifest)?;
    let mut reports_by_trial = HashMap::<String, Vec<TuningTrialReportView>>::new();
    for report in data.reports {
        reports_by_trial
            .entry(report.trial_id.clone())
            .or_default()
            .push(trial_report_view(report)?);
    }
    let mut games_by_pair = HashMap::<String, Vec<TuningGameView>>::new();
    for game in data.games {
        games_by_pair
            .entry(game.pair_id.clone())
            .or_default()
            .push(game_view(game)?);
    }
    let mut pairs_by_trial = HashMap::<String, Vec<TuningPairView>>::new();
    for pair in data.pairs {
        pairs_by_trial
            .entry(pair.trial_id.clone())
            .or_default()
            .push(pair_view(pair, &mut games_by_pair)?);
    }
    let trials = data
        .trials
        .into_iter()
        .map(|trial| trial_view(trial, &mut pairs_by_trial, &mut reports_by_trial))
        .collect::<Result<_, _>>()?;
    let attempts = data
        .attempts
        .into_iter()
        .map(|attempt| TuningAttemptView {
            attempt_id: attempt.attempt_id,
            bench_run_id: attempt.bench_run_id,
            status: attempt.status,
            started_at: attempt.started_at,
            ended_at: attempt.ended_at,
            failure: attempt.failure,
        })
        .collect();
    Ok(TuningSessionDetail {
        schema_version: 1,
        summary: TuningSessionSummary {
            session_id: data.session.session_id,
            status: data.session.status,
            target_trial_count: data.session.target_trial_count,
            counts: trial_counts(data.trial_counts),
        },
        attempts,
        trials,
        policy: decode_manifest_policy(&manifest)?,
        manifest,
        fingerprint: data.session.fingerprint,
        capabilities: capabilities(data.capabilities),
        control,
        cursor: TuningCursorBoundary {
            session_sequence: data.session.last_sequence,
        },
    })
}

fn trial_view(
    row: TuningSessionTrialRow,
    pairs_by_trial: &mut HashMap<String, Vec<TuningPairView>>,
    reports_by_trial: &mut HashMap<String, Vec<TuningTrialReportView>>,
) -> Result<TuningTrialView, BenchError> {
    Ok(TuningTrialView {
        pairs: pairs_by_trial.remove(&row.trial_id).unwrap_or_default(),
        reports: reports_by_trial.remove(&row.trial_id).unwrap_or_default(),
        trial_id: row.trial_id,
        trial_number: row.trial_number,
        attempt_id: row.attempt_id,
        status: row.status,
        config: decode_trial_config(row.config)?,
        score: row.score,
        mu: row.mu,
        sigma: row.sigma,
        stop_reason: decode_optional_report_reason(row.stop_reason, 8)?,
        failure: row.failure,
    })
}

fn trial_report_view(
    row: TuningSessionTrialReportRow,
) -> Result<TuningTrialReportView, BenchError> {
    Ok(TuningTrialReportView {
        completed_pairs: row.completed_pairs,
        reported_at: row.reported_at,
        rating: rating_view(row.mu, row.sigma),
        score: row.score,
        score_formula_version: row.score_formula_version,
        conservative_k: row.conservative_k,
        decision: TuningTrialReportDecisionView {
            outcome: decode_report_enum(&row.outcome, 8)?,
            reason: decode_report_enum(&row.reason, 9)?,
            pruning_exempt: row.pruning_exempt,
            bracket_id: row.bracket_id,
            rung_resource: row.rung_resource,
        },
    })
}

fn pair_view(
    row: TuningSessionPairRow,
    games_by_pair: &mut HashMap<String, Vec<TuningGameView>>,
) -> Result<TuningPairView, BenchError> {
    Ok(TuningPairView {
        games: games_by_pair.remove(&row.pair_id).unwrap_or_default(),
        pair_id: row.pair_id,
        pair_index: row.pair_index,
        status: row.status,
        seed: row.seed,
        round: row.round,
        opponent: opponent_view(decode_json(&row.opponent, 5)?),
        pool_snapshot_fingerprint: row.pool_snapshot_fingerprint,
        rating_before: rating_view(row.rating_before_mu, row.rating_before_sigma),
        rating_after: row
            .rating_after_mu
            .zip(row.rating_after_sigma)
            .map(|(mu, sigma)| rating_view(mu, sigma)),
        score: row.score,
        failure: row.failure,
    })
}

fn game_view(row: TuningSessionGameRow) -> Result<TuningGameView, BenchError> {
    Ok(TuningGameView {
        game_id: row.game_id,
        candidate_side: row.candidate_side,
        outcome: row.outcome,
        seed: row.seed,
        round: row.round,
        trace_game_seq: row.trace_game_seq,
        plies: row.plies,
        elapsed_ms: row.elapsed_ms,
        candidate: metrics_view(decode_json(&row.candidate_metrics, 8)?),
        baseline: metrics_view(decode_json(&row.baseline_metrics, 9)?),
    })
}

fn capabilities(row: RepositoryCapabilities) -> TuningCapabilities {
    TuningCapabilities {
        has_lifecycle: true,
        has_pairs: row.pair_count > 0,
        has_renderer_trace: row.renderer_trace_count > 0,
        has_search_reports: row.search_report_count > 0,
        has_trial_reports: row.trial_report_count > 0,
    }
}

pub(super) fn decode_json<T: serde::de::DeserializeOwned>(
    json: &str,
    column: usize,
) -> Result<T, duckdb::Error> {
    serde_json::from_str(json).map_err(|error| {
        duckdb::Error::FromSqlConversionFailure(column, duckdb::types::Type::Text, Box::new(error))
    })
}
pub(super) fn decode_report_enum<T: serde::de::DeserializeOwned>(
    value: &str,
    column: usize,
) -> Result<T, duckdb::Error> {
    serde_json::from_value(serde_json::Value::String(value.to_owned())).map_err(|error| {
        duckdb::Error::FromSqlConversionFailure(column, duckdb::types::Type::Text, Box::new(error))
    })
}
pub(super) fn opponent_view(value: OpponentSnapshot) -> TuningOpponentView {
    TuningOpponentView {
        anchor_id: value.anchor_id,
        config: value.config,
        mu: value.mu,
        sigma: value.sigma,
        label: value.label,
        provenance: value.provenance,
    }
}
pub(super) fn rating_view(mu: f64, sigma: f64) -> TuningRatingView {
    TuningRatingView { mu, sigma }
}
pub(super) fn metrics_view(value: StrategyMetrics) -> TuningStrategyMetricsView {
    TuningStrategyMetricsView {
        iterations_total: value.iterations_total,
        iterations_first_half: value.iterations_first_half,
        move_time_ms: value.move_time_ms,
    }
}
pub(super) fn decode_trial_config(
    config: Option<String>,
) -> Result<Option<serde_json::Value>, duckdb::Error> {
    config
        .map(|json| serde_json::from_str(&json))
        .transpose()
        .map_err(|error| {
            duckdb::Error::FromSqlConversionFailure(4, duckdb::types::Type::Text, Box::new(error))
        })
}

pub(super) fn decode_optional_report_reason(
    value: Option<String>,
    column: usize,
) -> Result<Option<mcts_bench::tuning_lifecycle::TrialReportReason>, duckdb::Error> {
    value
        .as_deref()
        .map(|reason| decode_report_enum(reason, column))
        .transpose()
}
pub(super) fn decode_manifest(manifest: &str) -> Result<serde_json::Value, BenchError> {
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

pub(super) fn decode_manifest_policy(
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
