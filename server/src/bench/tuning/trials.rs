use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, Query, State as AxumState},
    response::Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use mcts_bench::tuning_analysis_repository::TuningAnalysisPoolRevision;
use mcts_bench::tuning_lifecycle::{OpponentSnapshot, TrialReportReason};
use mcts_bench::tuning_trial_repository::{
    TuningTrialDetailData, TuningTrialGameRow, TuningTrialPageData,
    TuningTrialPageRow as RepositoryTrialPageRow, TuningTrialPairRow, TuningTrialReportRow,
    TuningTrialRepository, TuningTrialRepositoryError,
};
use serde::{Deserialize, Serialize};

use super::super::{
    BenchError, BenchState, TuningCursorBoundary, TuningPoolRevisionView, TuningReplayReference,
    TuningTrialDetail, TuningTrialDetailGameView, TuningTrialDetailPairView, TuningTrialDetailView,
    TuningTrialPage, TuningTrialReportDecisionView, TuningTrialReportView, TuningTrialSummaryView,
};
use super::analysis::{pool_revision, tuning_analysis_repository_error};
use super::sessions::{
    decode_json, decode_optional_report_reason, decode_report_enum, decode_trial_config,
    metrics_view, opponent_view, rating_view,
};

const DEFAULT_TRIAL_PAGE_LIMIT: u16 = 50;
const MAX_TRIAL_PAGE_LIMIT: u16 = 200;

pub(crate) async fn get_tuning_trials(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
    Query(params): Query<TuningTrialPageParams>,
) -> Result<Json<TuningTrialPage>, BenchError> {
    let query = TrialPageQuery::parse(params)?;
    let page = load_tuning_trial_page(state.tuning_trial_repository.as_ref(), &session_id, query)?;
    Ok(Json(page))
}

pub(crate) async fn get_tuning_trial_detail(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath((session_id, trial_id)): AxumPath<(String, String)>,
) -> Result<Json<TuningTrialDetail>, BenchError> {
    let pool_revisions = state
        .tuning_analysis_repository
        .load_trial_pool_revisions(&session_id, &trial_id)
        .map_err(tuning_analysis_repository_error)?;
    let detail = load_tuning_trial_detail(
        state.tuning_trial_repository.as_ref(),
        &session_id,
        &trial_id,
        &pool_revisions,
    )?
    .ok_or_else(|| BenchError {
        status: axum::http::StatusCode::NOT_FOUND,
        message: format!("tuning trial '{trial_id}' not found in session '{session_id}'"),
    })?;
    Ok(Json(detail))
}

#[derive(Deserialize, Default)]
pub(crate) struct TuningTrialPageParams {
    state: Option<String>,
    bracket: Option<String>,
    reason: Option<String>,
    family: Option<String>,
    q: Option<String>,
    sort: Option<String>,
    direction: Option<String>,
    limit: Option<i64>,
    cursor: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TrialPageQuery {
    state: Option<String>,
    bracket: Option<Option<String>>,
    reason: Option<String>,
    family: Option<String>,
    q: Option<String>,
    sort: TrialSort,
    direction: TrialSortDirection,
    limit: u16,
    cursor: Option<String>,
}

impl TrialPageQuery {
    fn parse(params: TuningTrialPageParams) -> Result<Self, BenchError> {
        let state = validate_choice(params.state, &TRIAL_STATES, "state")?;
        let reason = validate_choice(params.reason, &TRIAL_REASONS, "reason")?;
        let bracket = validate_facet(params.bracket, "bracket")?;
        let family = validate_nonempty(params.family, "family")?;
        let q = params.q.filter(|value| !value.is_empty());
        let sort = params
            .sort
            .as_deref()
            .map(TrialSort::parse)
            .transpose()?
            .unwrap_or(TrialSort::Trial);
        let direction = params
            .direction
            .as_deref()
            .map(TrialSortDirection::parse)
            .transpose()?
            .unwrap_or(TrialSortDirection::Desc);
        let limit = match params.limit {
            None => DEFAULT_TRIAL_PAGE_LIMIT,
            Some(value) if (1..=i64::from(MAX_TRIAL_PAGE_LIMIT)).contains(&value) => value as u16,
            Some(_) => return Err(bad_trial_query("limit must be between 1 and 200")),
        };
        Ok(Self {
            state,
            bracket,
            reason,
            family,
            q,
            sort,
            direction,
            limit,
            cursor: params.cursor,
        })
    }

    fn cursor_query(&self) -> TrialCursorQuery {
        TrialCursorQuery {
            state: self.state.clone(),
            bracket: self.bracket.clone(),
            reason: self.reason.clone(),
            family: self.family.clone(),
            q: self.q.clone(),
            sort: self.sort,
            direction: self.direction,
        }
    }
}

const TRIAL_STATES: [&str; 6] = [
    "queued",
    "running",
    "complete",
    "failed",
    "pruned",
    "cancelled",
];
const TRIAL_REASONS: [&str; 7] = [
    "below_min_pairs",
    "pruning_disabled",
    "startup_exempt",
    "hyperband_keep",
    "confidence",
    "max_pairs",
    "hyperband_prune",
];

fn validate_choice(
    value: Option<String>,
    choices: &[&str],
    field: &str,
) -> Result<Option<String>, BenchError> {
    match value {
        Some(value) if choices.contains(&value.as_str()) => Ok(Some(value)),
        Some(_) => Err(bad_trial_query(&format!("unknown {field}"))),
        None => Ok(None),
    }
}

fn validate_facet(
    value: Option<String>,
    field: &str,
) -> Result<Option<Option<String>>, BenchError> {
    match value.as_deref() {
        Some("unassigned") => Ok(Some(None)),
        Some("") => Err(bad_trial_query(&format!("{field} must not be empty"))),
        Some(_) => Ok(Some(value)),
        None => Ok(None),
    }
}

fn validate_nonempty(value: Option<String>, field: &str) -> Result<Option<String>, BenchError> {
    match value.as_deref() {
        Some("") => Err(bad_trial_query(&format!("{field} must not be empty"))),
        Some(_) => Ok(value),
        None => Ok(None),
    }
}

fn bad_trial_query(message: &str) -> BenchError {
    BenchError {
        status: axum::http::StatusCode::BAD_REQUEST,
        message: message.to_owned(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TrialSort {
    Trial,
    State,
    Score,
    Mu,
    Sigma,
    Resource,
    Family,
}

impl TrialSort {
    fn parse(value: &str) -> Result<Self, BenchError> {
        match value {
            "trial" => Ok(Self::Trial),
            "state" => Ok(Self::State),
            "score" => Ok(Self::Score),
            "mu" => Ok(Self::Mu),
            "sigma" => Ok(Self::Sigma),
            "resource" => Ok(Self::Resource),
            "family" => Ok(Self::Family),
            _ => Err(bad_trial_query("unknown sort")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum TrialSortDirection {
    Asc,
    Desc,
}

impl TrialSortDirection {
    fn parse(value: &str) -> Result<Self, BenchError> {
        match value {
            "asc" => Ok(Self::Asc),
            "desc" => Ok(Self::Desc),
            _ => Err(bad_trial_query("direction must be asc or desc")),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct TrialCursor {
    version: u8,
    query: TrialCursorQuery,
    after_trial_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TrialCursorQuery {
    state: Option<String>,
    bracket: Option<Option<String>>,
    reason: Option<String>,
    family: Option<String>,
    q: Option<String>,
    sort: TrialSort,
    direction: TrialSortDirection,
}

fn decode_trial_cursor(value: &str, query: &TrialPageQuery) -> Result<String, BenchError> {
    let bytes = URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| bad_trial_query("invalid cursor"))?;
    let cursor: TrialCursor =
        serde_json::from_slice(&bytes).map_err(|_| bad_trial_query("invalid cursor"))?;
    if cursor.version != 1 || cursor.query != query.cursor_query() {
        return Err(bad_trial_query("cursor does not match this query"));
    }
    Ok(cursor.after_trial_id)
}

fn encode_trial_cursor(query: &TrialPageQuery, after_trial_id: String) -> String {
    let cursor = TrialCursor {
        version: 1,
        query: query.cursor_query(),
        after_trial_id,
    };
    URL_SAFE_NO_PAD.encode(serde_json::to_vec(&cursor).expect("cursor serialization is infallible"))
}

fn load_tuning_trial_page(
    repository: &dyn TuningTrialRepository,
    session_id: &str,
    query: TrialPageQuery,
) -> Result<TuningTrialPage, BenchError> {
    let Some(TuningTrialPageData {
        session_sequence,
        trials,
    }) = repository
        .load_trial_page(session_id)
        .map_err(tuning_trial_repository_error)?
    else {
        return Err(BenchError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: format!("tuning session '{session_id}' not found"),
        });
    };
    let mut rows = trials
        .into_iter()
        .map(TrialPageRow::from_repository)
        .collect::<Result<Vec<_>, _>>()?;
    rows.retain(|row| trial_matches_query(row, &query));
    rows.sort_by(|left, right| compare_trial_page_rows(left, right, &query));
    let total_count = rows.len() as i64;

    let start = match query.cursor.as_deref() {
        Some(cursor) => {
            let after_trial_id = decode_trial_cursor(cursor, &query)?;
            rows.iter()
                .position(|row| row.trial_id == after_trial_id)
                .map(|index| index + 1)
                .ok_or_else(|| bad_trial_query("cursor position is no longer available"))?
        }
        None => 0,
    };
    let end = (start + usize::from(query.limit)).min(rows.len());
    let trials: Vec<_> = rows[start..end]
        .iter()
        .map(TrialPageRow::summary_view)
        .collect();
    let next_cursor =
        (end < rows.len()).then(|| encode_trial_cursor(&query, rows[end - 1].trial_id.clone()));

    Ok(TuningTrialPage {
        schema_version: 1,
        trials,
        total_count,
        limit: query.limit,
        next_cursor,
        cursor: TuningCursorBoundary { session_sequence },
    })
}

struct TrialPageRow {
    trial_id: String,
    trial_number: i64,
    attempt_id: String,
    state: String,
    config: Option<String>,
    score: Option<f64>,
    mu: Option<f64>,
    sigma: Option<f64>,
    stop_reason: Option<TrialReportReason>,
    last_reason: Option<TrialReportReason>,
    bracket_id: Option<String>,
    resource: Option<u64>,
    pair_count: u64,
    wins: u64,
    losses: u64,
    draws: u64,
    elapsed_ms: u64,
    search_iterations_total: u64,
    search_move_time_ms: u64,
}

impl TrialPageRow {
    fn from_repository(row: RepositoryTrialPageRow) -> Result<Self, duckdb::Error> {
        Ok(Self {
            trial_id: row.trial_id,
            trial_number: row.trial_number,
            attempt_id: row.attempt_id,
            state: row.state,
            config: row.config,
            score: row.score,
            mu: row.mu,
            sigma: row.sigma,
            stop_reason: decode_optional_report_reason(row.stop_reason, 8)?,
            last_reason: decode_optional_report_reason(row.last_reason, 9)?,
            bracket_id: row.bracket_id,
            resource: row.resource,
            pair_count: row.pair_count,
            wins: row.wins,
            losses: row.losses,
            draws: row.draws,
            elapsed_ms: row.elapsed_ms,
            search_iterations_total: row.search_iterations_total,
            search_move_time_ms: row.search_move_time_ms,
        })
    }

    fn reason(&self) -> Option<TrialReportReason> {
        self.stop_reason.or(self.last_reason)
    }

    fn config_display(&self) -> (Option<String>, Option<String>) {
        let value = self
            .config
            .as_deref()
            .and_then(|config| serde_json::from_str::<serde_json::Value>(config).ok());
        let object = value.and_then(|value| value.as_object().cloned());
        let Some(object) = object else {
            return (None, None);
        };
        let family = object
            .get("family")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned);
        let parameters = object.len().saturating_sub(usize::from(family.is_some()));
        let summary = family.as_ref().map(|_| match parameters {
            0 => "default settings".to_owned(),
            1 => "1 setting".to_owned(),
            count => format!("{count} settings"),
        });
        (family, summary)
    }

    fn summary_view(&self) -> TuningTrialSummaryView {
        let (family, config_summary) = self.config_display();
        TuningTrialSummaryView {
            trial_id: self.trial_id.clone(),
            trial_number: self.trial_number,
            attempt_id: self.attempt_id.clone(),
            state: self.state.clone(),
            reason: self.reason(),
            rating: self
                .mu
                .zip(self.sigma)
                .map(|(mu, sigma)| rating_view(mu, sigma)),
            score: self.score,
            family,
            config_summary,
            bracket_id: self.bracket_id.clone(),
            resource: self.resource,
            pair_count: self.pair_count,
            wins: self.wins,
            losses: self.losses,
            draws: self.draws,
            elapsed_ms: self.elapsed_ms,
            search_iterations_total: self.search_iterations_total,
            search_move_time_ms: self.search_move_time_ms,
            has_detail: self.config.is_some() || self.last_reason.is_some() || self.pair_count > 0,
        }
    }
}

fn trial_matches_query(row: &TrialPageRow, query: &TrialPageQuery) -> bool {
    let (family, _) = row.config_display();
    if query
        .state
        .as_deref()
        .is_some_and(|state| state != row.state)
    {
        return false;
    }
    if let Some(bracket) = &query.bracket {
        if bracket.as_ref() != row.bracket_id.as_ref() {
            return false;
        }
    }
    if query.reason.as_deref() != row.reason().map(TrialReportReason::as_str)
        && query.reason.is_some()
    {
        return false;
    }
    if query.family.as_deref() != family.as_deref() && query.family.is_some() {
        return false;
    }
    query.q.as_deref().is_none_or(|needle| {
        let needle = needle.to_lowercase();
        row.trial_id.to_lowercase().contains(&needle)
            || row.trial_number.to_string().contains(&needle)
            || family
                .as_deref()
                .is_some_and(|family| family.to_lowercase().contains(&needle))
    })
}

fn compare_trial_page_rows(
    left: &TrialPageRow,
    right: &TrialPageRow,
    query: &TrialPageQuery,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let primary = match query.sort {
        TrialSort::Trial => left.trial_number.cmp(&right.trial_number),
        TrialSort::State => left.state.cmp(&right.state),
        TrialSort::Score => compare_optional_f64(left.score, right.score),
        TrialSort::Mu => compare_optional_f64(left.mu, right.mu),
        TrialSort::Sigma => compare_optional_f64(left.sigma, right.sigma),
        TrialSort::Resource => compare_optional(left.resource, right.resource),
        TrialSort::Family => compare_optional(left.config_display().0, right.config_display().0),
    };
    let null_is_last = match query.sort {
        TrialSort::Score => left.score.is_none() || right.score.is_none(),
        TrialSort::Mu => left.mu.is_none() || right.mu.is_none(),
        TrialSort::Sigma => left.sigma.is_none() || right.sigma.is_none(),
        TrialSort::Resource => left.resource.is_none() || right.resource.is_none(),
        TrialSort::Family => {
            left.config_display().0.is_none() || right.config_display().0.is_none()
        }
        TrialSort::Trial | TrialSort::State => false,
    };
    let primary = match query.direction {
        TrialSortDirection::Asc => primary,
        TrialSortDirection::Desc if primary.is_ne() && !null_is_last => primary.reverse(),
        TrialSortDirection::Desc => primary,
    };
    if primary.is_eq() {
        left.trial_number
            .cmp(&right.trial_number)
            .then_with(|| left.trial_id.cmp(&right.trial_id))
    } else {
        primary
    }
}

fn compare_optional<T: Ord>(left: Option<T>, right: Option<T>) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn compare_optional_f64(left: Option<f64>, right: Option<f64>) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn load_tuning_trial_detail(
    repository: &dyn TuningTrialRepository,
    session_id: &str,
    trial_id: &str,
    pool_revisions: &[TuningAnalysisPoolRevision],
) -> Result<Option<TuningTrialDetail>, BenchError> {
    let Some(data) = repository
        .load_trial_detail(session_id, trial_id)
        .map_err(tuning_trial_repository_error)?
    else {
        return Ok(None);
    };
    let reports = data
        .reports
        .into_iter()
        .map(report_view)
        .collect::<Result<Vec<_>, _>>()?;
    let reason = decode_optional_report_reason(data.trial.stop_reason, 8)?
        .or_else(|| reports.last().map(|report| report.decision.reason));
    let pairs = data
        .pairs
        .into_iter()
        .map(|pair| pair_view(pair, pool_revisions))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(TuningTrialDetail {
        schema_version: 1,
        trial: TuningTrialDetailView {
            trial_id: data.trial.trial_id,
            trial_number: data.trial.trial_number,
            attempt_id: data.trial.attempt_id,
            state: data.trial.state,
            config: decode_trial_config(data.trial.config)?,
            score: data.trial.score,
            rating: data
                .trial
                .mu
                .zip(data.trial.sigma)
                .map(|(mu, sigma)| rating_view(mu, sigma)),
            reason,
            failure: data.trial.failure,
            reports,
            pairs,
        },
        cursor: TuningCursorBoundary {
            session_sequence: data.session_sequence,
        },
    }))
}

fn report_view(row: TuningTrialReportRow) -> Result<TuningTrialReportView, duckdb::Error> {
    Ok(TuningTrialReportView {
        completed_pairs: row.completed_pairs,
        reported_at: row.reported_at,
        rating: rating_view(row.mu, row.sigma),
        score: row.score,
        score_formula_version: row.score_formula_version,
        conservative_k: row.conservative_k,
        decision: TuningTrialReportDecisionView {
            outcome: decode_report_enum(&row.outcome, 7)?,
            reason: decode_report_enum(&row.reason, 8)?,
            pruning_exempt: row.pruning_exempt,
            bracket_id: row.bracket_id,
            rung_resource: row.rung_resource,
        },
    })
}

fn pair_view(
    row: TuningTrialPairRow,
    pool_revisions: &[TuningAnalysisPoolRevision],
) -> Result<TuningTrialDetailPairView, duckdb::Error> {
    let opponent = decode_json::<OpponentSnapshot>(&row.opponent, 5)?;
    Ok(TuningTrialDetailPairView {
        pair_id: row.pair_id,
        pair_index: row.pair_index,
        state: row.status,
        seed: row.seed,
        round: row.round,
        opponent: opponent_view(opponent),
        pool_revision: load_pool_revision_for_detail(
            pool_revisions,
            &row.pool_snapshot_fingerprint,
        ),
        pool_snapshot_fingerprint: row.pool_snapshot_fingerprint,
        rating_before: rating_view(row.rating_before_mu, row.rating_before_sigma),
        rating_after: row
            .rating_after_mu
            .zip(row.rating_after_sigma)
            .map(|(mu, sigma)| rating_view(mu, sigma)),
        score: row.score,
        failure: row.failure,
        games: row
            .games
            .into_iter()
            .map(game_view)
            .collect::<Result<_, _>>()?,
    })
}

fn load_pool_revision_for_detail(
    pool_revisions: &[TuningAnalysisPoolRevision],
    fingerprint: &str,
) -> Option<TuningPoolRevisionView> {
    pool_revisions
        .iter()
        .find(|revision| revision.pool_snapshot_fingerprint == fingerprint)
        .map(pool_revision)
}

fn game_view(row: TuningTrialGameRow) -> Result<TuningTrialDetailGameView, duckdb::Error> {
    let replay = row
        .run_id
        .zip(row.trace_game_seq)
        .map(|(run_id, game_seq)| TuningReplayReference {
            run_id,
            game_seq,
            has_renderer_trace: row.has_renderer_trace,
            has_search_reports: row.has_search_reports,
        });
    Ok(TuningTrialDetailGameView {
        game_id: row.game_id,
        candidate_side: row.candidate_side,
        outcome: row.outcome,
        seed: row.seed,
        round: row.round,
        plies: row.plies,
        elapsed_ms: row.elapsed_ms,
        candidate: metrics_view(decode_json(&row.candidate_metrics, 8)?),
        baseline: metrics_view(decode_json(&row.baseline_metrics, 9)?),
        replay,
    })
}

fn tuning_trial_repository_error(error: TuningTrialRepositoryError) -> BenchError {
    BenchError {
        status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("tuning trial storage error: {error}"),
    }
}
