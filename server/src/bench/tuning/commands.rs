use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State as AxumState},
    http::StatusCode,
    response::Json,
};

use super::super::{
    BenchError, BenchState, TunerAttemptLaunch, TuningBudgetResult, TuningSessionBudgetBody,
    TuningSessionCommandBody, TuningSessionCommandResponse, TuningSessionControl, TuningStopSignal,
};

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
            Some(pid) => {
                match super::super::process::ProcessController::signal_group(state.as_ref(), pid) {
                    Ok(super::super::process::SignalOutcome::Sent) => Some(TuningStopSignal::Sent),
                    Ok(super::super::process::SignalOutcome::NotFound) => {
                        Some(TuningStopSignal::NotFound)
                    }
                    Err(super::super::process::ProcessError::Failed(message)) => {
                        return Err(BenchError {
                            status: StatusCode::INTERNAL_SERVER_ERROR,
                            message: format!(
                                "failed to signal tuning session '{session_id}': {message}"
                            ),
                        });
                    }
                }
            }
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
        super::super::launch_reserved_tuner_attempt(
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
        if let Err(error) = super::super::launch_reserved_tuner_attempt(
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
    launch: &TunerAttemptLaunch,
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
        observed_at: super::super::iso_timestamp_now(),
    })
}

fn continuation_launch(
    state: &Arc<BenchState>,
    session_id: &str,
    command_id: &str,
    n_workers: Option<u64>,
) -> Result<TunerAttemptLaunch, BenchError> {
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
    Ok(TunerAttemptLaunch {
        game,
        config,
        session_id: session_id.into(),
        optimizer_id,
        lifecycle_path,
        attempt_id: format!("tuning-attempt-{physical_run_id}"),
        artifact_root: super::super::tuner_artifact_root(&state.bench_runs_dir, &physical_run_id),
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

pub(super) fn session_control(
    db: &duckdb::Connection,
    session_id: &str,
) -> Result<TuningSessionControl, BenchError> {
    mcts_bench::tuning_command_store::reconcile(db, session_id)
        .map(TuningSessionControl::from)
        .map_err(command_bench_error)
}
