use duckdb::{params, Transaction};

use crate::tuning_lifecycle::{TuningEventType, TuningLifecycleEvent, TuningPayload};

use super::TuningStoreError;

struct AttemptState {
    session_id: String,
    status: String,
}

struct TrialState {
    attempt_id: String,
    status: String,
}

pub(super) fn validate(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<Option<String>, TuningStoreError> {
    if let Err(error) = event.validate_shape() {
        return Ok(Some(error.to_string()));
    }
    if let Some(reason) = validate_sequence(tx, event)? {
        return Ok(Some(reason));
    }

    let session_exists = session_exists(tx, event)?;
    if let Some(reason) = validate_session_start(session_exists, event) {
        return Ok(Some(reason));
    }
    if event.event_type == TuningEventType::SessionStarted {
        return Ok(None);
    }

    let attempt = load_attempt(tx, event)?;
    if event.event_type == TuningEventType::AttemptStarted {
        return Ok(attempt
            .is_some()
            .then_some("attempt_started may occur only once".into()));
    }
    let Some(attempt) = attempt else {
        return Ok(Some("attempt_started is required before this event".into()));
    };
    if let Some(reason) = validate_attempt(&attempt, event) {
        return Ok(Some(reason));
    }
    if event.event_type.is_trial() {
        return validate_trial(tx, event);
    }
    Ok(None)
}

fn validate_sequence(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<Option<String>, TuningStoreError> {
    let expected_sequence: i64 = tx.query_row(
        "SELECT COALESCE(MAX(session_sequence) + 1, 1) FROM tuning_lifecycle_events WHERE session_id = ?1 AND accepted = true",
        params![event.session_id.as_str()],
        |row| row.get(0),
    )?;
    if event.session_sequence != expected_sequence as u64 {
        return Ok(Some(format!(
            "session sequence {} does not follow {}",
            event.session_sequence, expected_sequence
        )));
    }
    Ok(None)
}

fn session_exists(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<bool, TuningStoreError> {
    let count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM tuning_sessions WHERE session_id = ?1",
        params![event.session_id.as_str()],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn validate_session_start(session_exists: bool, event: &TuningLifecycleEvent) -> Option<String> {
    if !session_exists && event.event_type != TuningEventType::SessionStarted {
        return Some("session_started is required first".into());
    }
    if session_exists && event.event_type == TuningEventType::SessionStarted {
        return Some("session_started may occur only once".into());
    }
    None
}

fn load_attempt(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<Option<AttemptState>, TuningStoreError> {
    Ok(tx
        .query_row(
            "SELECT session_id, status FROM tuning_attempts WHERE attempt_id = ?1",
            params![event.attempt_id.as_str()],
            |row| {
                Ok(AttemptState {
                    session_id: row.get(0)?,
                    status: row.get(1)?,
                })
            },
        )
        .ok())
}

fn validate_attempt(attempt: &AttemptState, event: &TuningLifecycleEvent) -> Option<String> {
    if attempt.session_id != event.session_id.as_str() {
        return Some("attempt belongs to a different session".into());
    }
    if matches!(attempt.status.as_str(), "completed" | "failed" | "stopped") {
        return Some("attempt is already terminal".into());
    }
    None
}

fn validate_trial(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<Option<String>, TuningStoreError> {
    let trial_id = event
        .trial_id()
        .expect("validated typed payload")
        .expect("validated shape includes trial id");
    let trial = load_trial(tx, event, trial_id.as_str())?;
    if let Some(reason) = validate_trial_attempt(trial.as_ref(), event) {
        return Ok(Some(reason));
    }
    let valid = valid_trial_transition(tx, event, trial.as_ref())?;
    Ok((!valid).then_some("invalid trial state transition".into()))
}

fn load_trial(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    trial_id: &str,
) -> Result<Option<TrialState>, TuningStoreError> {
    Ok(tx
        .query_row(
            "SELECT attempt_id, status FROM tuning_trials WHERE session_id = ?1 AND trial_id = ?2",
            params![event.session_id.as_str(), trial_id],
            |row| {
                Ok(TrialState {
                    attempt_id: row.get(0)?,
                    status: row.get(1)?,
                })
            },
        )
        .ok())
}

fn validate_trial_attempt(
    trial: Option<&TrialState>,
    event: &TuningLifecycleEvent,
) -> Option<String> {
    if let Some(trial) = trial {
        if trial.attempt_id != event.attempt_id.as_str() {
            return Some("trial belongs to a different attempt".into());
        }
    }
    None
}

fn valid_trial_transition(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    trial: Option<&TrialState>,
) -> Result<bool, TuningStoreError> {
    let status = trial.map(|trial| trial.status.as_str());
    match event.event_type {
        TuningEventType::TrialCreated => {
            let TuningPayload::TrialCreated(payload) =
                event.typed_payload().expect("validated payload")
            else {
                unreachable!()
            };
            let number_exists: i64 = tx.query_row(
                "SELECT COUNT(*) FROM tuning_trials WHERE session_id = ?1 AND trial_number = ?2",
                params![event.session_id.as_str(), payload.trial_number],
                |row| row.get(0),
            )?;
            Ok(status.is_none() && number_exists == 0)
        }
        TuningEventType::TrialStarted => Ok(status == Some("queued")),
        event_type if event_type.is_trial_terminal() => {
            Ok(matches!(status, Some("queued") | Some("running")))
        }
        _ => Ok(false),
    }
}
