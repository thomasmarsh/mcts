use duckdb::{params, Transaction};

use crate::tuning_lifecycle::{TuningEventType, TuningLifecycleEvent, TuningPayload};

use super::TuningStoreError;

pub(super) fn apply(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    bench_run_id: &str,
) -> Result<(), TuningStoreError> {
    match event.event_type {
        TuningEventType::SessionStarted => project_session_started(tx, event)?,
        TuningEventType::AttemptStarted => project_attempt_started(tx, event, bench_run_id)?,
        TuningEventType::TrialCreated => project_trial_created(tx, event)?,
        TuningEventType::TrialStarted => {
            mark_trial_started(tx, event)?;
            touch_session(tx, event)?;
        }
        event_type if event_type.is_trial_terminal() => project_trial_terminal(tx, event)?,
        event_type if event_type.is_attempt_terminal() => project_attempt_terminal(tx, event)?,
        _ => unreachable!(),
    }
    Ok(())
}

fn project_session_started(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let TuningPayload::SessionStarted(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    tx.execute(
        "INSERT INTO tuning_sessions (session_id, status, manifest, manifest_fingerprint, target_trial_count, created_at, last_sequence) VALUES (?1, 'active', ?2, ?3, ?4, ?5, ?6)",
        params![event.session_id.as_str(), serde_json::to_string(&payload.manifest)?, payload.manifest_fingerprint, payload.target_trial_count, &event.timestamp, event.session_sequence as i64],
    )?;
    Ok(())
}

fn project_attempt_started(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    bench_run_id: &str,
) -> Result<(), TuningStoreError> {
    let TuningPayload::AttemptStarted(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    tx.execute(
        "INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at) VALUES (?1, ?2, ?3, 'running', ?4)",
        params![event.attempt_id.as_str(), event.session_id.as_str(), bench_run_id, &event.timestamp],
    )?;
    if let Some(target) = payload.target_trial_count {
        tx.execute(
            "UPDATE tuning_sessions SET target_trial_count = CASE WHEN target_trial_count IS NULL OR target_trial_count < ?1 THEN ?1 ELSE target_trial_count END WHERE session_id = ?2",
            params![target, event.session_id.as_str()],
        )?;
    }
    touch_session(tx, event)
}

fn project_trial_created(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let TuningPayload::TrialCreated(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    tx.execute(
        "INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at) VALUES (?1, ?2, ?3, ?4, 'queued', ?5, ?6)",
        params![event.session_id.as_str(), payload.trial_id.as_str(), event.attempt_id.as_str(), payload.trial_number, serde_json::to_string(&payload.config)?, &event.timestamp],
    )?;
    touch_session(tx, event)
}

fn project_trial_terminal(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let status = match event.event_type {
        TuningEventType::TrialCompleted => "complete",
        TuningEventType::TrialPruned => "pruned",
        TuningEventType::TrialFailed => "failed",
        TuningEventType::TrialCancelled => "cancelled",
        _ => unreachable!(),
    };
    let terminal = match event.typed_payload().expect("validated payload") {
        TuningPayload::TrialCompleted(value)
        | TuningPayload::TrialPruned(value)
        | TuningPayload::TrialFailed(value)
        | TuningPayload::TrialCancelled(value) => value,
        _ => unreachable!(),
    };
    tx.execute(
        "UPDATE tuning_trials SET status = ?1, ended_at = ?2, score = ?3, mu = ?4, sigma = ?5, failure = ?6 WHERE session_id = ?7 AND trial_id = ?8",
        params![status, &event.timestamp, terminal.score, terminal.mu, terminal.sigma, terminal.error, event.session_id.as_str(), terminal.trial_id.as_str()],
    )?;
    touch_session(tx, event)
}

fn project_attempt_terminal(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let status = attempt_terminal_status(event);
    let terminal = attempt_terminal_payload(event);
    finish_attempt(tx, event, status, terminal.error.or(terminal.reason))?;
    refresh_session_activity(tx, event)
}

fn attempt_terminal_status(event: &TuningLifecycleEvent) -> &'static str {
    match event.event_type {
        TuningEventType::AttemptCompleted => "completed",
        TuningEventType::AttemptFailed => "failed",
        TuningEventType::AttemptStopped => "stopped",
        _ => unreachable!(),
    }
}

fn attempt_terminal_payload(
    event: &TuningLifecycleEvent,
) -> crate::tuning_lifecycle::AttemptTerminalPayload {
    match event.typed_payload().expect("validated payload") {
        TuningPayload::AttemptCompleted(value)
        | TuningPayload::AttemptFailed(value)
        | TuningPayload::AttemptStopped(value) => value,
        _ => unreachable!(),
    }
}

fn finish_attempt(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    status: &str,
    failure: Option<String>,
) -> Result<(), TuningStoreError> {
    tx.execute(
        "UPDATE tuning_attempts SET status = ?1, ended_at = ?2, failure = ?3 WHERE attempt_id = ?4",
        params![status, &event.timestamp, failure, event.attempt_id.as_str()],
    )?;
    Ok(())
}

fn refresh_session_activity(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    tx.execute(
        "UPDATE tuning_sessions SET status = CASE WHEN EXISTS (SELECT 1 FROM tuning_attempts WHERE session_id = ?2 AND status = 'running') THEN 'active' ELSE 'idle' END, last_sequence = ?1 WHERE session_id = ?2",
        params![event.session_sequence as i64, event.session_id.as_str()],
    )?;
    Ok(())
}

fn mark_trial_started(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let trial_id = event
        .trial_id()
        .expect("validated typed payload")
        .expect("validated trial id");
    tx.execute(
        "UPDATE tuning_trials SET status = 'running', started_at = ?1 WHERE session_id = ?2 AND trial_id = ?3",
        params![
            &event.timestamp,
            event.session_id.as_str(),
            trial_id.as_str()
        ],
    )?;
    Ok(())
}

fn touch_session(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    tx.execute(
        "UPDATE tuning_sessions SET status = 'active', last_sequence = ?1 WHERE session_id = ?2",
        params![event.session_sequence as i64, event.session_id.as_str()],
    )?;
    Ok(())
}
