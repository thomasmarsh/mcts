use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use super::{
    BenchError, BenchState, TuningAnalysisBest, TuningAnalysisCoverage, TuningAnalysisObjective,
    TuningAnalysisOverview, TuningAnalysisPairCoverage, TuningAnalysisPoint,
    TuningAnalysisPointCoverage, TuningAttemptSummary, TuningAttemptView,
    TuningBracketResourceAggregate, TuningBudgetResult, TuningCapabilities, TuningCursorBoundary,
    TuningDecisionAggregate, TuningGameView, TuningOpponentView, TuningPairView, TuningPolicyView,
    TuningPoolAnchorView, TuningPoolRevisionView, TuningPruningPolicyView, TuningRatingPolicyView,
    TuningRatingView, TuningReplayReference, TuningResourcePolicyView, TuningSamplerPolicyView,
    TuningSessionBudgetBody, TuningSessionCommandBody, TuningSessionCommandResponse,
    TuningSessionControl, TuningSessionDetail, TuningSessionList, TuningSessionListItem,
    TuningSessionSummary, TuningStopSignal, TuningStrategyMetricsView, TuningTrialCounts,
    TuningTrialDetail, TuningTrialDetailGameView, TuningTrialDetailPairView, TuningTrialDetailView,
    TuningTrialPage, TuningTrialReportDecisionView, TuningTrialReportView, TuningTrialSummaryView,
    TuningTrialView,
};
use axum::{
    extract::{Path as AxumPath, Query, State as AxumState},
    http::StatusCode,
    response::Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use duckdb::Connection;
use mcts_bench::tuning_lifecycle::{
    OpponentSnapshot, PoolAnchorInsertionReason, PoolAnchorProvenance, StrategyMetrics,
    TrialReportOutcome, TrialReportReason,
};
use serde::{Deserialize, Serialize};

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
    let control = session_control(&db, &session_id)?;
    let detail =
        load_tuning_session_detail(&db, &session_id, control)?.ok_or_else(|| BenchError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: format!("tuning session '{session_id}' not found"),
        })?;
    Ok(Json(detail))
}

/// Reserve one stop before contacting the process adapter.  The lifecycle
/// journal, rather than this route, remains the authority for terminal state.
pub(crate) async fn stop_tuning_session(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
    Json(body): Json<TuningSessionCommandBody>,
) -> Result<(StatusCode, Json<TuningSessionCommandResponse>), BenchError> {
    validate_command_id(&body.command_id)?;
    let request = command_request(
        &state,
        &session_id,
        &body,
        mcts_bench::tuning_command_store::SessionCommand::Stop,
        None,
        None,
    )?;
    let decision = {
        let db = state.db.lock().unwrap();
        mcts_bench::tuning_command_store::apply_command(&db, &session_id, &request)
            .map_err(command_bench_error)?
    };
    let attempt_id = decision.control.stop_attempt_id.clone();
    let signal = if decision.replay {
        None
    } else {
        let pid = attempt_id
            .as_deref()
            .map(|attempt_id| tuning_attempt_pid(&state, &session_id, attempt_id))
            .transpose()?;
        match pid.flatten() {
            Some(pid) => match super::process::ProcessController::signal_group(state.as_ref(), pid)
            {
                Ok(super::process::SignalOutcome::Sent) => Some(TuningStopSignal::Sent),
                Ok(super::process::SignalOutcome::NotFound) => Some(TuningStopSignal::NotFound),
                Err(super::process::ProcessError::Failed(message)) => {
                    return Err(BenchError {
                        status: StatusCode::INTERNAL_SERVER_ERROR,
                        message: format!(
                            "failed to signal tuning session '{session_id}': {message}"
                        ),
                    });
                }
            },
            None => Some(TuningStopSignal::NotFound),
        }
    };
    Ok((
        StatusCode::ACCEPTED,
        Json(TuningSessionCommandResponse {
            schema_version: 1,
            command_id: decision.command_id,
            replay: decision.replay,
            status: "stopping",
            attempt_id,
            bench_run_id: None,
            signal,
            budget: None,
            launch_error: None,
            control: decision.control.into(),
        }),
    ))
}

/// Reserve a physical continuation before spawning it.  A replay returns the
/// recorded physical identity and never invokes the launcher again.
pub(crate) async fn resume_tuning_session(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
    Json(body): Json<TuningSessionCommandBody>,
) -> Result<(StatusCode, Json<TuningSessionCommandResponse>), BenchError> {
    validate_command_id(&body.command_id)?;
    let mut launch = continuation_launch(&state, &session_id, &body.command_id, None)?;
    let request = command_request(
        &state,
        &session_id,
        &body,
        mcts_bench::tuning_command_store::SessionCommand::Resume,
        Some(mcts_bench::tuning_command_store::LaunchReservation {
            attempt_id: launch.attempt_id.clone(),
            physical_run_id: launch.physical_run_id.clone(),
        }),
        None,
    )?;
    if let Some(reservation) = &request.launch {
        launch.attempt_id.clone_from(&reservation.attempt_id);
        launch
            .physical_run_id
            .clone_from(&reservation.physical_run_id);
    }
    let decision = {
        let db = state.db.lock().unwrap();
        mcts_bench::tuning_command_store::apply_command(&db, &session_id, &request)
            .map_err(command_bench_error)?
    };
    launch.target_trial_count = decision
        .control
        .target_trial_count
        .ok_or_else(|| BenchError {
            status: StatusCode::CONFLICT,
            message: format!("tuning session '{session_id}' has no continuation target"),
        })?;
    let replay_state = if decision.replay {
        Some(resume_replay_state(
            &state,
            &session_id,
            &decision.command_id,
            &launch,
        )?)
    } else {
        None
    };
    if !matches!(replay_state, Some(ResumeReplayState::Launched)) {
        if matches!(replay_state, Some(ResumeReplayState::Failed)) {
            return Err(BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!(
                    "the recorded launch for tuning session '{session_id}' failed before a process was created"
                ),
            });
        }
        super::launch_reserved_tuner_attempt(
            &state,
            &decision.command_id,
            launch.clone(),
            Some(&format!("resume of {session_id}")),
        )?;
    }
    Ok((
        StatusCode::CREATED,
        Json(TuningSessionCommandResponse {
            schema_version: 1,
            command_id: decision.command_id,
            replay: decision.replay,
            status: "resuming",
            attempt_id: Some(launch.attempt_id),
            bench_run_id: Some(launch.physical_run_id),
            signal: None,
            budget: None,
            launch_error: None,
            control: decision.control.into(),
        }),
    ))
}

/// Add positive trial capacity to a logical session, optionally reserving and
/// launching its next physical attempt at the resulting absolute target.
pub(crate) async fn add_tuning_session_budget(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
    Json(body): Json<TuningSessionBudgetBody>,
) -> Result<(StatusCode, Json<TuningSessionCommandResponse>), BenchError> {
    validate_command_id(&body.command_id)?;
    validate_budget_body(&body)?;

    let mut launch = body
        .start
        .then(|| continuation_launch(&state, &session_id, &body.command_id, body.n_workers))
        .transpose()?;
    let request = command_request(
        &state,
        &session_id,
        &TuningSessionCommandBody {
            command_id: body.command_id.clone(),
            expected_version: body.expected_version,
        },
        mcts_bench::tuning_command_store::SessionCommand::AddBudget {
            delta: body.delta,
            start: body.start,
        },
        launch.as_ref().map(
            |launch| mcts_bench::tuning_command_store::LaunchReservation {
                attempt_id: launch.attempt_id.clone(),
                physical_run_id: launch.physical_run_id.clone(),
            },
        ),
        body.n_workers,
    )?;
    if let (Some(launch), Some(reservation)) = (&mut launch, &request.launch) {
        launch.attempt_id.clone_from(&reservation.attempt_id);
        launch
            .physical_run_id
            .clone_from(&reservation.physical_run_id);
    }
    let decision = {
        let db = state.db.lock().unwrap();
        mcts_bench::tuning_command_store::apply_command(&db, &session_id, &request)
            .map_err(command_bench_error)?
    };
    let budget = budget_result(&decision, body.delta)?;

    if !body.start {
        return Ok((
            StatusCode::OK,
            Json(TuningSessionCommandResponse {
                schema_version: 1,
                command_id: decision.command_id,
                replay: decision.replay,
                status: "extended",
                attempt_id: None,
                bench_run_id: None,
                signal: None,
                budget: Some(budget),
                launch_error: None,
                control: decision.control.into(),
            }),
        ));
    }

    let mut launch = launch.expect("a start command has a launch reservation");
    launch.target_trial_count = budget.target_trial_count;
    let replay_state = if decision.replay {
        Some(resume_replay_state(
            &state,
            &session_id,
            &decision.command_id,
            &launch,
        )?)
    } else {
        None
    };
    if matches!(replay_state, Some(ResumeReplayState::Failed)) {
        let control = {
            let db = state.db.lock().unwrap();
            session_control(&db, &session_id)?
        };
        return Ok((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(TuningSessionCommandResponse {
                schema_version: 1,
                command_id: decision.command_id,
                replay: true,
                status: "launch_failed",
                attempt_id: Some(launch.attempt_id),
                bench_run_id: Some(launch.physical_run_id),
                signal: None,
                budget: Some(budget),
                launch_error: Some(
                    "the recorded launch failed before a process was created".into(),
                ),
                control,
            }),
        ));
    }
    if !matches!(replay_state, Some(ResumeReplayState::Launched)) {
        if let Err(error) = super::launch_reserved_tuner_attempt(
            &state,
            &decision.command_id,
            launch.clone(),
            Some(&format!("budget extension of {session_id}")),
        ) {
            let control = {
                let db = state.db.lock().unwrap();
                session_control(&db, &session_id)?
            };
            return Ok((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(TuningSessionCommandResponse {
                    schema_version: 1,
                    command_id: decision.command_id,
                    replay: decision.replay,
                    status: "launch_failed",
                    attempt_id: Some(launch.attempt_id),
                    bench_run_id: Some(launch.physical_run_id),
                    signal: None,
                    budget: Some(budget),
                    launch_error: Some(error.message),
                    control,
                }),
            ));
        }
    }
    Ok((
        StatusCode::CREATED,
        Json(TuningSessionCommandResponse {
            schema_version: 1,
            command_id: decision.command_id,
            replay: decision.replay,
            status: "starting",
            attempt_id: Some(launch.attempt_id),
            bench_run_id: Some(launch.physical_run_id),
            signal: None,
            budget: Some(budget),
            launch_error: None,
            control: decision.control.into(),
        }),
    ))
}

const MAX_BUDGET_DELTA: u64 = 1_000_000;

fn validate_budget_body(body: &TuningSessionBudgetBody) -> Result<(), BenchError> {
    if !(1..=MAX_BUDGET_DELTA).contains(&body.delta) {
        return Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: format!("delta must be between 1 and {MAX_BUDGET_DELTA}"),
        });
    }
    let Some(workers) = body.n_workers else {
        return Ok(());
    };
    if !body.start {
        return Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: "n_workers is allowed only when start is true".into(),
        });
    }
    if !(1..=1_024).contains(&workers) {
        return Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: "n_workers must be between 1 and 1024".into(),
        });
    }
    Ok(())
}

fn budget_result(
    decision: &mcts_bench::tuning_command_store::CommandDecision,
    delta: u64,
) -> Result<TuningBudgetResult, BenchError> {
    let target_trial_count = decision
        .control
        .target_trial_count
        .ok_or_else(|| BenchError {
            status: StatusCode::CONFLICT,
            message: "tuning session has no continuation target".into(),
        })?;
    let previous_target_trial_count =
        target_trial_count
            .checked_sub(delta)
            .ok_or_else(|| BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: "recorded budget command has an invalid target".into(),
            })?;
    Ok(TuningBudgetResult {
        previous_target_trial_count,
        delta,
        target_trial_count,
    })
}

enum ResumeReplayState {
    Launched,
    Reserved,
    Failed,
}

fn resume_replay_state(
    state: &Arc<BenchState>,
    session_id: &str,
    command_id: &str,
    launch: &super::TunerAttemptLaunch,
) -> Result<ResumeReplayState, BenchError> {
    let db = state.db.lock().unwrap();
    let run_exists: bool = db.query_row(
        "SELECT EXISTS (SELECT 1 FROM runs WHERE run_id = ?1)",
        duckdb::params![&launch.physical_run_id],
        |row| row.get(0),
    )?;
    if run_exists {
        return Ok(ResumeReplayState::Launched);
    }
    let reservation_exists: bool = db.query_row(
        "SELECT EXISTS (
             SELECT 1 FROM tuning_launch_reservations
             WHERE session_id = ?1 AND command_id = ?2 AND attempt_id = ?3 AND physical_run_id = ?4
         )",
        duckdb::params![
            session_id,
            command_id,
            &launch.attempt_id,
            &launch.physical_run_id
        ],
        |row| row.get(0),
    )?;
    Ok(if reservation_exists {
        ResumeReplayState::Reserved
    } else {
        ResumeReplayState::Failed
    })
}

fn validate_command_id(command_id: &str) -> Result<(), BenchError> {
    if command_id.is_empty()
        || command_id.len() > 96
        || !command_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(BenchError {
            status: StatusCode::BAD_REQUEST,
            message: "command_id must contain 1..=96 ASCII letters, digits, '-' or '_'".into(),
        });
    }
    Ok(())
}

fn command_request(
    state: &Arc<BenchState>,
    session_id: &str,
    body: &TuningSessionCommandBody,
    command: mcts_bench::tuning_command_store::SessionCommand,
    launch: Option<mcts_bench::tuning_command_store::LaunchReservation>,
    n_workers: Option<u64>,
) -> Result<mcts_bench::tuning_command_store::CommandRequest, BenchError> {
    let existing = {
        let db = state.db.lock().unwrap();
        db.query_row(
            "SELECT session_id, CAST(request AS TEXT) FROM tuning_session_commands WHERE command_id = ?1",
            duckdb::params![&body.command_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .ok()
    };
    if let Some((stored_session, request)) = existing {
        let request: mcts_bench::tuning_command_store::CommandRequest =
            serde_json::from_str(&request)?;
        if stored_session == session_id
            && request.expected_version == body.expected_version
            && request.command == command
            && request.n_workers == n_workers
        {
            return Ok(request);
        }
    }
    Ok(mcts_bench::tuning_command_store::CommandRequest {
        command_id: body.command_id.clone(),
        expected_version: body.expected_version,
        command,
        launch,
        n_workers,
        observed_at: super::iso_timestamp_now(),
    })
}

fn continuation_launch(
    state: &Arc<BenchState>,
    session_id: &str,
    command_id: &str,
    n_workers: Option<u64>,
) -> Result<super::TunerAttemptLaunch, BenchError> {
    let db = state.db.lock().unwrap();
    let (game, config, optimizer_id, lifecycle_path): (String, Option<String>, String, String) = db
        .query_row(
            "SELECT run.game, CAST(run.config AS TEXT), session.optimizer_id, session.lifecycle_path \
             FROM tuning_sessions session \
             JOIN tuning_attempts attempt ON attempt.session_id = session.session_id \
             JOIN runs run ON run.run_id = attempt.bench_run_id \
             WHERE session.session_id = ?1 AND session.optimizer_id IS NOT NULL \
               AND session.lifecycle_path IS NOT NULL \
             ORDER BY attempt.started_at DESC, attempt.attempt_id DESC LIMIT 1",
            duckdb::params![session_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .map_err(|_| BenchError {
            status: StatusCode::CONFLICT,
            message: format!(
                "tuning session '{session_id}' lacks the physical run metadata required for continuation"
            ),
        })?;
    let config = config
        .map(|config| serde_json::from_str(&config))
        .transpose()
        .map_err(|_| BenchError {
            status: StatusCode::CONFLICT,
            message: format!(
                "tuning session '{session_id}' has invalid continuation configuration"
            ),
        })?;
    let physical_run_id = format!(
        "{}-{command_id}",
        mcts_bench::launch::generate_run_id("tuner", &game, crate::BUILD_INFO)
    );
    Ok(super::TunerAttemptLaunch {
        game,
        config,
        session_id: session_id.into(),
        optimizer_id,
        lifecycle_path,
        attempt_id: format!("tuning-attempt-{physical_run_id}"),
        artifact_root: super::tuner_artifact_root(&state.bench_runs_dir, &physical_run_id),
        physical_run_id,
        target_trial_count: 1,
        workers: n_workers,
    })
}

fn tuning_attempt_pid(
    state: &Arc<BenchState>,
    session_id: &str,
    attempt_id: &str,
) -> Result<Option<i64>, BenchError> {
    let db = state.db.lock().unwrap();
    match db.query_row(
        "SELECT run.pid FROM tuning_attempts attempt \
         LEFT JOIN runs run ON run.run_id = attempt.bench_run_id \
         WHERE attempt.session_id = ?1 AND attempt.attempt_id = ?2",
        duckdb::params![session_id, attempt_id],
        |row| row.get(0),
    ) {
        Ok(pid) => Ok(pid),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn command_bench_error(error: mcts_bench::tuning_command_store::CommandStoreError) -> BenchError {
    use mcts_bench::tuning_command_store::CommandStoreError;
    let status = match &error {
        CommandStoreError::SessionNotFound(_) => StatusCode::NOT_FOUND,
        CommandStoreError::DuckDb(_) | CommandStoreError::Serialization(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
        CommandStoreError::CommandIdReuseMismatch { .. }
        | CommandStoreError::ExpectedVersionConflict { .. }
        | CommandStoreError::ActiveAttempt { .. }
        | CommandStoreError::LaunchReserved { .. }
        | CommandStoreError::InvalidDeltaStart { .. }
        | CommandStoreError::ExhaustedResume { .. }
        | CommandStoreError::NoncontinuableLegacy { .. }
        | CommandStoreError::CommandDenied { .. }
        | CommandStoreError::TargetOverflow { .. }
        | CommandStoreError::MissingReservation { .. } => StatusCode::CONFLICT,
    };
    BenchError {
        status,
        message: error.to_string(),
    }
}

fn session_control(
    db: &duckdb::Connection,
    session_id: &str,
) -> Result<TuningSessionControl, BenchError> {
    mcts_bench::tuning_command_store::reconcile(db, session_id)
        .map(TuningSessionControl::from)
        .map_err(command_bench_error)
}

pub(crate) async fn get_tuning_analysis_overview(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Result<Json<TuningAnalysisOverview>, BenchError> {
    let db = state.db.lock().unwrap();
    let control = session_control(&db, &session_id)?;
    let overview =
        load_tuning_analysis_overview(&db, &session_id, control)?.ok_or_else(|| BenchError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: format!("tuning session '{session_id}' not found"),
        })?;
    Ok(Json(overview))
}

const DEFAULT_TRIAL_PAGE_LIMIT: u16 = 50;
const MAX_TRIAL_PAGE_LIMIT: u16 = 200;

pub(crate) async fn get_tuning_trials(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
    Query(params): Query<TuningTrialPageParams>,
) -> Result<Json<TuningTrialPage>, BenchError> {
    let query = TrialPageQuery::parse(params)?;
    let db = state.db.lock().unwrap();
    let page = load_tuning_trial_page(&db, &session_id, query)?;
    Ok(Json(page))
}

pub(crate) async fn get_tuning_trial_detail(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath((session_id, trial_id)): AxumPath<(String, String)>,
) -> Result<Json<TuningTrialDetail>, BenchError> {
    let db = state.db.lock().unwrap();
    let detail =
        load_tuning_trial_detail(&db, &session_id, &trial_id)?.ok_or_else(|| BenchError {
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
    db: &Connection,
    session_id: &str,
    query: TrialPageQuery,
) -> Result<TuningTrialPage, BenchError> {
    let Some(session) = load_session(db, session_id)? else {
        return Err(BenchError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: format!("tuning session '{session_id}' not found"),
        });
    };
    let mut rows = load_trial_page_rows(db, session_id)?;
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
        cursor: TuningCursorBoundary {
            session_sequence: session.last_sequence,
        },
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

fn load_trial_page_rows(
    db: &Connection,
    session_id: &str,
) -> Result<Vec<TrialPageRow>, duckdb::Error> {
    let mut query = db.prepare(
        "WITH ranked_reports AS ( \
             SELECT session_id, trial_id, completed_pairs, mu, sigma, score, reason, bracket_id, event_id, \
                    ROW_NUMBER() OVER (PARTITION BY session_id, trial_id ORDER BY completed_pairs DESC, event_id DESC) AS rank \
             FROM tuning_trial_reports \
         ), last_reports AS ( \
             SELECT session_id, trial_id, completed_pairs, mu, sigma, score, reason, bracket_id \
             FROM ranked_reports WHERE rank = 1 \
         ), game_stats AS ( \
             SELECT pairs.session_id, pairs.trial_id, COUNT(DISTINCT pairs.pair_id) AS pair_count, \
                    COALESCE(SUM(CASE WHEN games.outcome = 'candidate_win' THEN 1 ELSE 0 END), 0) AS wins, \
                    COALESCE(SUM(CASE WHEN games.outcome = 'baseline_win' THEN 1 ELSE 0 END), 0) AS losses, \
                    COALESCE(SUM(CASE WHEN games.outcome = 'draw' THEN 1 ELSE 0 END), 0) AS draws, \
                    COALESCE(SUM(games.elapsed_ms), 0) AS elapsed_ms, \
                    COALESCE(SUM( \
                        COALESCE(TRY_CAST(json_extract_string(games.candidate_metrics, '$.iterations_total') AS UBIGINT), 0) + \
                        COALESCE(TRY_CAST(json_extract_string(games.baseline_metrics, '$.iterations_total') AS UBIGINT), 0) \
                    ), 0) AS search_iterations_total, \
                    COALESCE(SUM( \
                        COALESCE(TRY_CAST(json_extract_string(games.candidate_metrics, '$.move_time_ms') AS UBIGINT), 0) + \
                        COALESCE(TRY_CAST(json_extract_string(games.baseline_metrics, '$.move_time_ms') AS UBIGINT), 0) \
                    ), 0) AS search_move_time_ms \
             FROM tuning_evaluation_pairs pairs \
             LEFT JOIN tuning_games games USING (session_id, pair_id) \
             WHERE pairs.session_id = ?1 \
             GROUP BY pairs.session_id, pairs.trial_id \
         ) \
         SELECT trials.trial_id, trials.trial_number, trials.attempt_id, trials.status, \
                CAST(trials.config AS TEXT), COALESCE(trials.score, reports.score), \
                COALESCE(trials.mu, reports.mu), COALESCE(trials.sigma, reports.sigma), trials.stop_reason, \
                reports.reason, reports.bracket_id, reports.completed_pairs, \
                COALESCE(stats.pair_count, 0), COALESCE(stats.wins, 0), COALESCE(stats.losses, 0), \
                COALESCE(stats.draws, 0), COALESCE(stats.elapsed_ms, 0), \
                COALESCE(stats.search_iterations_total, 0), COALESCE(stats.search_move_time_ms, 0) \
         FROM tuning_trials trials \
         LEFT JOIN last_reports reports USING (session_id, trial_id) \
         LEFT JOIN game_stats stats USING (session_id, trial_id) \
         WHERE trials.session_id = ?1",
    )?;
    query
        .query_map(duckdb::params![session_id], |row| {
            let stop_reason: Option<String> = row.get(8)?;
            let last_reason: Option<String> = row.get(9)?;
            Ok(TrialPageRow {
                trial_id: row.get(0)?,
                trial_number: row.get(1)?,
                attempt_id: row.get(2)?,
                state: row.get(3)?,
                config: row.get(4)?,
                score: row.get(5)?,
                mu: row.get(6)?,
                sigma: row.get(7)?,
                stop_reason: decode_optional_report_reason(stop_reason, 8)?,
                last_reason: decode_optional_report_reason(last_reason, 9)?,
                bracket_id: row.get(10)?,
                resource: row.get(11)?,
                pair_count: row.get(12)?,
                wins: row.get(13)?,
                losses: row.get(14)?,
                draws: row.get(15)?,
                elapsed_ms: row.get(16)?,
                search_iterations_total: row.get(17)?,
                search_move_time_ms: row.get(18)?,
            })
        })?
        .collect()
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
    db: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Option<TuningTrialDetail>, BenchError> {
    let Some(session) = load_session(db, session_id)? else {
        return Ok(None);
    };
    let Some(row) = load_trial_detail_row(db, session_id, trial_id)? else {
        return Ok(None);
    };
    let reports = load_trial_reports_for_trial(db, session_id, trial_id)?;
    let reason = row
        .stop_reason
        .or_else(|| reports.last().map(|report| report.decision.reason));
    let pairs = load_trial_detail_pairs(db, session_id, trial_id)?;
    Ok(Some(TuningTrialDetail {
        schema_version: 1,
        trial: TuningTrialDetailView {
            trial_id: row.trial_id,
            trial_number: row.trial_number,
            attempt_id: row.attempt_id,
            state: row.state,
            config: decode_trial_config(row.config)?,
            score: row.score,
            rating: row
                .mu
                .zip(row.sigma)
                .map(|(mu, sigma)| rating_view(mu, sigma)),
            reason,
            failure: row.failure,
            reports,
            pairs,
        },
        cursor: TuningCursorBoundary {
            session_sequence: session.last_sequence,
        },
    }))
}

struct TrialDetailRow {
    trial_id: String,
    trial_number: i64,
    attempt_id: String,
    state: String,
    config: Option<String>,
    score: Option<f64>,
    mu: Option<f64>,
    sigma: Option<f64>,
    stop_reason: Option<TrialReportReason>,
    failure: Option<String>,
}

fn load_trial_detail_row(
    db: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Option<TrialDetailRow>, duckdb::Error> {
    match db.query_row(
        "SELECT trial_id, trial_number, attempt_id, status, CAST(config AS TEXT), score, mu, sigma, stop_reason, failure \
         FROM tuning_trials WHERE session_id = ?1 AND trial_id = ?2",
        duckdb::params![session_id, trial_id],
        |row| {
            let stop_reason: Option<String> = row.get(8)?;
            Ok(TrialDetailRow {
                trial_id: row.get(0)?,
                trial_number: row.get(1)?,
                attempt_id: row.get(2)?,
                state: row.get(3)?,
                config: row.get(4)?,
                score: row.get(5)?,
                mu: row.get(6)?,
                sigma: row.get(7)?,
                stop_reason: decode_optional_report_reason(stop_reason, 8)?,
                failure: row.get(9)?,
            })
        },
    ) {
        Ok(row) => Ok(Some(row)),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error),
    }
}

fn load_trial_reports_for_trial(
    db: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Vec<TuningTrialReportView>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT completed_pairs, CAST(reported_at AS TEXT), mu, sigma, score, score_formula_version, \
                conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource \
         FROM tuning_trial_reports WHERE session_id = ?1 AND trial_id = ?2 \
         ORDER BY completed_pairs ASC, event_id ASC",
    )?;
    query
        .query_map(duckdb::params![session_id, trial_id], |row| {
            let outcome: String = row.get(7)?;
            let reason: String = row.get(8)?;
            Ok(TuningTrialReportView {
                completed_pairs: row.get(0)?,
                reported_at: row.get(1)?,
                rating: rating_view(row.get(2)?, row.get(3)?),
                score: row.get(4)?,
                score_formula_version: row.get(5)?,
                conservative_k: row.get(6)?,
                decision: TuningTrialReportDecisionView {
                    outcome: decode_report_enum(&outcome, 7)?,
                    reason: decode_report_enum(&reason, 8)?,
                    pruning_exempt: row.get(9)?,
                    bracket_id: row.get(10)?,
                    rung_resource: row.get(11)?,
                },
            })
        })?
        .collect()
}

fn load_trial_detail_pairs(
    db: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Vec<TuningTrialDetailPairView>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT pair_id, pair_index, status, seed, round, CAST(opponent AS TEXT), \
                pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, rating_after_mu, \
                rating_after_sigma, score, failure, attempt_id \
         FROM tuning_evaluation_pairs WHERE session_id = ?1 AND trial_id = ?2 \
         ORDER BY pair_index ASC, pair_id ASC",
    )?;
    query
        .query_map(duckdb::params![session_id, trial_id], |row| {
            let pair = PairRow {
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
            };
            let attempt_id: String = row.get(13)?;
            assemble_trial_detail_pair(db, session_id, pair, &attempt_id)
        })?
        .collect()
}

fn assemble_trial_detail_pair(
    db: &Connection,
    session_id: &str,
    row: PairRow,
    attempt_id: &str,
) -> Result<TuningTrialDetailPairView, duckdb::Error> {
    let opponent = decode_json::<OpponentSnapshot>(&row.opponent, 5)?;
    Ok(TuningTrialDetailPairView {
        pair_id: row.pair_id.clone(),
        pair_index: row.pair_index,
        state: row.status,
        seed: row.seed,
        round: row.round,
        opponent: opponent_view(opponent),
        pool_snapshot_fingerprint: row.pool_snapshot_fingerprint.clone(),
        pool_revision: load_pool_revision_for_detail(
            db,
            session_id,
            &row.pool_snapshot_fingerprint,
        )?,
        rating_before: rating_view(row.rating_before_mu, row.rating_before_sigma),
        rating_after: row
            .rating_after_mu
            .zip(row.rating_after_sigma)
            .map(|(mu, sigma)| rating_view(mu, sigma)),
        score: row.score,
        failure: row.failure,
        games: load_trial_detail_games(db, session_id, &row.pair_id, attempt_id)?,
    })
}

fn load_pool_revision_for_detail(
    db: &Connection,
    session_id: &str,
    fingerprint: &str,
) -> Result<Option<TuningPoolRevisionView>, duckdb::Error> {
    match db.query_row(
        "SELECT display_ordinal, CAST(observed_at AS TEXT) FROM tuning_pool_revisions \
         WHERE session_id = ?1 AND pool_snapshot_fingerprint = ?2",
        duckdb::params![session_id, fingerprint],
        |row| Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?)),
    ) {
        Ok((display_ordinal, observed_at)) => {
            let pair_count = db.query_row(
                "SELECT COUNT(*) FROM tuning_evaluation_pairs \
                 WHERE session_id = ?1 AND pool_snapshot_fingerprint = ?2",
                duckdb::params![session_id, fingerprint],
                |row| row.get(0),
            )?;
            Ok(Some(TuningPoolRevisionView {
                pool_snapshot_fingerprint: fingerprint.to_owned(),
                display_ordinal,
                observed_at,
                pair_count,
                anchors: load_pool_anchors(db, session_id, fingerprint)?,
            }))
        }
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error),
    }
}

fn load_trial_detail_games(
    db: &Connection,
    session_id: &str,
    pair_id: &str,
    attempt_id: &str,
) -> Result<Vec<TuningTrialDetailGameView>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT games.game_id, games.candidate_side, games.outcome, games.seed, games.round, \
                games.trace_game_seq, games.plies, games.elapsed_ms, CAST(games.candidate_metrics AS TEXT), \
                CAST(games.baseline_metrics AS TEXT), attempts.bench_run_id, \
                EXISTS(SELECT 1 FROM game_moves moves WHERE moves.run_id = attempts.bench_run_id \
                       AND moves.game_seq = games.trace_game_seq AND moves.trace_schema_version = 1), \
                EXISTS(SELECT 1 FROM game_moves moves WHERE moves.run_id = attempts.bench_run_id \
                       AND moves.game_seq = games.trace_game_seq AND moves.search_report IS NOT NULL) \
         FROM tuning_games games \
         JOIN tuning_attempts attempts ON attempts.attempt_id = ?3 \
         WHERE games.session_id = ?1 AND games.pair_id = ?2 \
         ORDER BY games.candidate_side ASC, games.game_id ASC",
    )?;
    query
        .query_map(duckdb::params![session_id, pair_id, attempt_id], |row| {
            let candidate: String = row.get(8)?;
            let baseline: String = row.get(9)?;
            let trace_game_seq: Option<u64> = row.get(5)?;
            let run_id: Option<String> = row.get(10)?;
            let replay =
                run_id
                    .zip(trace_game_seq)
                    .map(|(run_id, game_seq)| TuningReplayReference {
                        run_id,
                        game_seq,
                        has_renderer_trace: row.get(11).unwrap_or(false),
                        has_search_reports: row.get(12).unwrap_or(false),
                    });
            Ok(TuningTrialDetailGameView {
                game_id: row.get(0)?,
                candidate_side: row.get(1)?,
                outcome: row.get(2)?,
                seed: row.get(3)?,
                round: row.get(4)?,
                plies: row.get(6)?,
                elapsed_ms: row.get(7)?,
                candidate: metrics_view(decode_json(&candidate, 8)?),
                baseline: metrics_view(decode_json(&baseline, 9)?),
                replay,
            })
        })?
        .collect()
}

const ANALYSIS_POINT_LIMIT: usize = 2_000;

fn load_tuning_analysis_overview(
    db: &Connection,
    session_id: &str,
    control: TuningSessionControl,
) -> Result<Option<TuningAnalysisOverview>, BenchError> {
    let Some(session) = load_session(db, session_id)? else {
        return Ok(None);
    };
    let manifest = decode_manifest(&session.manifest)?;
    let reports = load_analysis_reports(db, session_id)?;
    let counts = load_trial_counts(db, session_id)?;
    let pairs = load_analysis_pair_coverage(db, session_id)?;
    let pool_revisions = load_pool_revisions(db, session_id)?;
    let best = load_analysis_best(db, session_id)?;
    let (bracket_resources, decision_groups) = aggregate_analysis_reports(&reports);
    let points = sample_analysis_points(&reports);
    let returned = points.len() as i64;
    let total = reports.len() as i64;

    Ok(Some(TuningAnalysisOverview {
        schema_version: 1,
        policy: decode_manifest_policy(&manifest)?,
        objective: TuningAnalysisObjective {
            metric: "score",
            direction: "maximize",
            complete_trials_only: true,
        },
        cursor: TuningCursorBoundary {
            session_sequence: session.last_sequence,
        },
        coverage: TuningAnalysisCoverage {
            trials: counts,
            reports: total,
            pairs,
            points: TuningAnalysisPointCoverage {
                total,
                returned,
                sampled: total > returned,
            },
        },
        bracket_resources,
        decision_groups,
        points,
        best,
        pool_revisions,
        control,
    }))
}

struct AnalysisReportRow {
    trial_id: String,
    trial_number: i64,
    trial_status: String,
    resource: u64,
    mu: f64,
    sigma: f64,
    score: f64,
    outcome: TrialReportOutcome,
    reason: TrialReportReason,
    pruning_exempt: bool,
    bracket_id: Option<String>,
    rung_resource: Option<u64>,
}

type AnalysisResourceKey = (Option<String>, u64, Option<u64>);
type AnalysisResourceAggregate = (i64, BTreeSet<String>);
type AnalysisDecisionKey = (u8, u8, bool);
type AnalysisDecisionAggregate = (TrialReportOutcome, TrialReportReason, i64);

fn load_analysis_reports(
    db: &Connection,
    session_id: &str,
) -> Result<Vec<AnalysisReportRow>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT reports.trial_id, reports.trial_number, trials.status, reports.completed_pairs, \
                reports.mu, reports.sigma, reports.score, reports.outcome, reports.reason, \
                reports.pruning_exempt, reports.bracket_id, reports.rung_resource \
         FROM tuning_trial_reports reports \
         JOIN tuning_trials trials USING (session_id, trial_id) \
         WHERE reports.session_id = ?1 \
         ORDER BY reports.bracket_id ASC NULLS FIRST, reports.completed_pairs ASC, \
                  reports.outcome ASC, reports.trial_number ASC, reports.event_id ASC",
    )?;
    query
        .query_map(duckdb::params![session_id], |row| {
            let outcome: String = row.get(7)?;
            let reason: String = row.get(8)?;
            Ok(AnalysisReportRow {
                trial_id: row.get(0)?,
                trial_number: row.get(1)?,
                trial_status: row.get(2)?,
                resource: row.get(3)?,
                mu: row.get(4)?,
                sigma: row.get(5)?,
                score: row.get(6)?,
                outcome: decode_report_enum(&outcome, 7)?,
                reason: decode_report_enum(&reason, 8)?,
                pruning_exempt: row.get(9)?,
                bracket_id: row.get(10)?,
                rung_resource: row.get(11)?,
            })
        })?
        .collect()
}

fn load_analysis_pair_coverage(
    db: &Connection,
    session_id: &str,
) -> Result<TuningAnalysisPairCoverage, duckdb::Error> {
    db.query_row(
        "SELECT COUNT(*), \
                COALESCE(SUM(CASE WHEN pairs.status = 'running' THEN 1 ELSE 0 END), 0), \
                COALESCE(SUM(CASE WHEN pairs.status = 'complete' THEN 1 ELSE 0 END), 0), \
                COALESCE(SUM(CASE WHEN pairs.status = 'failed' THEN 1 ELSE 0 END), 0), \
                COALESCE(SUM(CASE WHEN revisions.pool_snapshot_fingerprint IS NULL THEN 1 ELSE 0 END), 0) \
         FROM tuning_evaluation_pairs pairs \
         LEFT JOIN tuning_pool_revisions revisions \
           ON revisions.session_id = pairs.session_id \
          AND revisions.pool_snapshot_fingerprint = pairs.pool_snapshot_fingerprint \
         WHERE pairs.session_id = ?1",
        duckdb::params![session_id],
        |row| {
            Ok(TuningAnalysisPairCoverage {
                total: row.get(0)?,
                running: row.get(1)?,
                complete: row.get(2)?,
                failed: row.get(3)?,
                unmatched_pool_revisions: row.get(4)?,
            })
        },
    )
}

fn aggregate_analysis_reports(
    reports: &[AnalysisReportRow],
) -> (
    Vec<TuningBracketResourceAggregate>,
    Vec<TuningDecisionAggregate>,
) {
    let mut resources: BTreeMap<AnalysisResourceKey, AnalysisResourceAggregate> = BTreeMap::new();
    let mut decisions: BTreeMap<AnalysisDecisionKey, AnalysisDecisionAggregate> = BTreeMap::new();
    for report in reports {
        let resource = resources
            .entry((
                report.bracket_id.clone(),
                report.resource,
                report.rung_resource,
            ))
            .or_insert_with(|| (0, BTreeSet::new()));
        resource.0 += 1;
        resource.1.insert(report.trial_id.clone());

        let decision = decisions
            .entry((
                report_outcome_rank(report.outcome),
                report_reason_rank(report.reason),
                report.pruning_exempt,
            ))
            .or_insert((report.outcome, report.reason, 0));
        decision.2 += 1;
    }
    (
        resources
            .into_iter()
            .map(
                |((bracket_id, resource, rung_resource), (reports, trials))| {
                    TuningBracketResourceAggregate {
                        bracket_id,
                        resource,
                        rung_resource,
                        reports,
                        trials: trials.len() as i64,
                    }
                },
            )
            .collect(),
        decisions
            .into_iter()
            .map(
                |((_, _, pruning_exempt), (outcome, reason, reports))| TuningDecisionAggregate {
                    outcome,
                    reason,
                    pruning_exempt,
                    reports,
                },
            )
            .collect(),
    )
}

fn sample_analysis_points(reports: &[AnalysisReportRow]) -> Vec<TuningAnalysisPoint> {
    let selected = if reports.len() <= ANALYSIS_POINT_LIMIT {
        vec![true; reports.len()]
    } else {
        let mut strata: BTreeMap<(Option<String>, u64, u8), Vec<usize>> = BTreeMap::new();
        for (index, report) in reports.iter().enumerate() {
            strata
                .entry((
                    report.bracket_id.clone(),
                    report.resource,
                    report_outcome_rank(report.outcome),
                ))
                .or_default()
                .push(index);
        }
        let mut selected = vec![false; reports.len()];
        let mut returned = 0;
        let mut strata_by_coverage: Vec<&Vec<usize>> = strata.values().collect();
        strata_by_coverage.sort_by_key(|indices| indices.len());
        for indices in &strata_by_coverage {
            if returned == ANALYSIS_POINT_LIMIT {
                break;
            }
            selected[indices[0]] = true;
            returned += 1;
        }
        let mut offset = 1;
        while returned < ANALYSIS_POINT_LIMIT {
            let mut added = false;
            for indices in strata.values() {
                if returned == ANALYSIS_POINT_LIMIT {
                    break;
                }
                if let Some(&index) = indices.get(offset) {
                    selected[index] = true;
                    returned += 1;
                    added = true;
                }
            }
            if !added {
                break;
            }
            offset += 1;
        }
        selected
    };
    reports
        .iter()
        .zip(selected)
        .filter(|(_, selected)| *selected)
        .map(|(report, _)| analysis_point(report))
        .collect()
}

fn analysis_point(report: &AnalysisReportRow) -> TuningAnalysisPoint {
    TuningAnalysisPoint {
        trial_id: report.trial_id.clone(),
        trial_number: report.trial_number,
        trial_status: report.trial_status.clone(),
        resource: report.resource,
        rating: rating_view(report.mu, report.sigma),
        score: report.score,
        outcome: report.outcome,
        reason: report.reason,
        pruning_exempt: report.pruning_exempt,
        bracket_id: report.bracket_id.clone(),
        rung_resource: report.rung_resource,
    }
}

fn load_analysis_best(
    db: &Connection,
    session_id: &str,
) -> Result<Option<TuningAnalysisBest>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT trial_id, trial_number, score \
         FROM tuning_trials \
         WHERE session_id = ?1 AND status = 'complete' AND score IS NOT NULL \
         ORDER BY score DESC, trial_number ASC, trial_id ASC",
    )?;
    let trials: Vec<(String, i64, f64)> = query
        .query_map(duckdb::params![session_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .collect::<Result<_, _>>()?;
    let Some((_, _, score)) = trials.first() else {
        return Ok(None);
    };
    let score = *score;
    Ok(Some(TuningAnalysisBest {
        score,
        trial_ids: trials
            .into_iter()
            .take_while(|(_, _, trial_score)| trial_score.total_cmp(&score).is_eq())
            .map(|(trial_id, _, _)| trial_id)
            .collect(),
    }))
}

fn load_pool_revisions(
    db: &Connection,
    session_id: &str,
) -> Result<Vec<TuningPoolRevisionView>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT revisions.pool_snapshot_fingerprint, revisions.display_ordinal, \
                CAST(revisions.observed_at AS TEXT), COUNT(pairs.pair_id) \
         FROM tuning_pool_revisions revisions \
         LEFT JOIN tuning_evaluation_pairs pairs \
           ON pairs.session_id = revisions.session_id \
          AND pairs.pool_snapshot_fingerprint = revisions.pool_snapshot_fingerprint \
         WHERE revisions.session_id = ?1 \
         GROUP BY revisions.pool_snapshot_fingerprint, revisions.display_ordinal, revisions.observed_at \
         ORDER BY revisions.display_ordinal ASC, revisions.pool_snapshot_fingerprint ASC",
    )?;
    query
        .query_map(duckdb::params![session_id], |row| {
            let fingerprint: String = row.get(0)?;
            Ok(TuningPoolRevisionView {
                pool_snapshot_fingerprint: fingerprint.clone(),
                display_ordinal: row.get(1)?,
                observed_at: row.get(2)?,
                pair_count: row.get(3)?,
                anchors: load_pool_anchors(db, session_id, &fingerprint)?,
            })
        })?
        .collect()
}

fn load_pool_anchors(
    db: &Connection,
    session_id: &str,
    fingerprint: &str,
) -> Result<Vec<TuningPoolAnchorView>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT anchor_ordinal, anchor_id, CAST(config AS TEXT), mu, sigma, provenance, \
                insertion_reason, source_trial_id \
         FROM tuning_pool_anchors \
         WHERE session_id = ?1 AND pool_snapshot_fingerprint = ?2 \
         ORDER BY anchor_ordinal ASC, anchor_id ASC",
    )?;
    query
        .query_map(duckdb::params![session_id, fingerprint], |row| {
            let config: String = row.get(2)?;
            let provenance: String = row.get(5)?;
            let insertion_reason: String = row.get(6)?;
            Ok(TuningPoolAnchorView {
                anchor_ordinal: row.get(0)?,
                anchor_id: row.get(1)?,
                config: decode_json(&config, 2)?,
                rating: rating_view(row.get(3)?, row.get(4)?),
                provenance: decode_report_enum(&provenance, 5)?,
                insertion_reason: decode_report_enum(&insertion_reason, 6)?,
                source_trial_id: row.get(7)?,
            })
        })?
        .collect()
}

fn report_outcome_rank(outcome: TrialReportOutcome) -> u8 {
    match outcome {
        TrialReportOutcome::Continue => 0,
        TrialReportOutcome::Complete => 1,
        TrialReportOutcome::Prune => 2,
    }
}

fn report_reason_rank(reason: TrialReportReason) -> u8 {
    match reason {
        TrialReportReason::BelowMinPairs => 0,
        TrialReportReason::PruningDisabled => 1,
        TrialReportReason::StartupExempt => 2,
        TrialReportReason::HyperbandKeep => 3,
        TrialReportReason::Confidence => 4,
        TrialReportReason::MaxPairs => 5,
        TrialReportReason::HyperbandPrune => 6,
    }
}

fn load_tuning_session_list(db: &Connection) -> Result<TuningSessionList, BenchError> {
    let sessions = load_tuning_session_list_rows(db)?;
    let attempts = load_tuning_session_list_attempts(db)?;
    let controls = sessions
        .iter()
        .map(|session| {
            session_control(db, &session.session_id)
                .map(|control| (session.session_id.clone(), control))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
    assemble_tuning_session_list(sessions, attempts, controls)
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
    mut controls: HashMap<String, TuningSessionControl>,
) -> Result<TuningSessionList, BenchError> {
    let mut attempts_by_session = group_tuning_session_attempts(attempts);
    let sessions = sessions
        .into_iter()
        .map(|row| {
            let attempts = attempts_by_session
                .remove(&row.session_id)
                .unwrap_or_default();
            let control = controls.remove(&row.session_id).ok_or_else(|| BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!(
                    "missing control projection for tuning session '{}'",
                    row.session_id
                ),
            })?;
            assemble_tuning_session_list_item(row, attempts, control)
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
    control: TuningSessionControl,
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
        control,
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
    session_id: String,
    status: String,
    target_trial_count: Option<i64>,
    manifest: String,
    fingerprint: Option<String>,
    last_sequence: i64,
}

fn load_tuning_session_detail(
    db: &Connection,
    session_id: &str,
    control: TuningSessionControl,
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
        session,
        counts,
        attempts,
        trials,
        manifest,
        capabilities,
        control,
    )?))
}

fn load_session(
    db: &Connection,
    session_id: &str,
) -> Result<Option<TuningSessionRow>, duckdb::Error> {
    match db.query_row(
        "SELECT session_id, status, target_trial_count, CAST(manifest AS TEXT), manifest_fingerprint, last_sequence FROM tuning_sessions WHERE session_id = ?1",
        duckdb::params![&session_id],
        |row| {
            Ok(TuningSessionRow {
                session_id: row.get(0)?,
                status: row.get(1)?,
                target_trial_count: row.get(2)?,
                manifest: row.get(3)?,
                fingerprint: row.get(4)?,
                last_sequence: row.get(5)?,
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
    session: TuningSessionRow,
    counts: TuningTrialCounts,
    attempts: Vec<TuningAttemptView>,
    trials: Vec<TuningTrialView>,
    manifest: serde_json::Value,
    capabilities: TuningCapabilities,
    control: TuningSessionControl,
) -> Result<TuningSessionDetail, BenchError> {
    let policy = decode_manifest_policy(&manifest)?;
    Ok(TuningSessionDetail {
        schema_version: 1,
        summary: TuningSessionSummary {
            session_id: session.session_id,
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
        control,
        cursor: TuningCursorBoundary {
            session_sequence: session.last_sequence,
        },
    })
}
