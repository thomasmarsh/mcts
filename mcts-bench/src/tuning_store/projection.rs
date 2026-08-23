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
        TuningEventType::TrialReported => project_trial_reported(tx, event)?,
        TuningEventType::PairStarted => project_pair_started(tx, event)?,
        TuningEventType::GameFinished => project_game_finished(tx, event)?,
        TuningEventType::PairFinished => project_pair_finished(tx, event)?,
        TuningEventType::PairFailed => project_pair_failed(tx, event)?,
        event_type if event_type.is_trial_terminal() => project_trial_terminal(tx, event)?,
        event_type if event_type.is_attempt_terminal() => project_attempt_terminal(tx, event)?,
        _ => unreachable!(),
    }
    Ok(())
}

fn project_trial_reported(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let TuningPayload::TrialReported(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    tx.execute(
        "INSERT INTO tuning_trial_reports (session_id, trial_id, trial_number, completed_pairs, event_id, reported_at, mu, sigma, score, score_formula_version, conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        params![event.session_id.as_str(), payload.trial_id.as_str(), payload.trial_number, payload.completed_pairs, event.event_id.as_str(), &event.timestamp, payload.mu, payload.sigma, payload.score, payload.score_formula_version, payload.conservative_k, payload.outcome.as_str(), payload.reason.as_str(), payload.pruning_exempt, payload.bracket_id, payload.rung_resource],
    )?;
    touch_session(tx, event)
}

fn project_pair_started(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let TuningPayload::PairStarted(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    tx.execute(
        "INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, started_at) VALUES (?1, ?2, ?3, ?4, ?5, 'running', ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![event.session_id.as_str(), payload.pair_id.as_str(), payload.trial_id.as_str(), event.attempt_id.as_str(), payload.pair_index, payload.seed, payload.round, serde_json::to_string(&payload.opponent)?, payload.pool_snapshot_fingerprint, payload.rating_before.mu, payload.rating_before.sigma, &event.timestamp],
    )?;
    touch_session(tx, event)
}

fn project_game_finished(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let TuningPayload::GameFinished(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    tx.execute(
        "INSERT INTO tuning_games (session_id, pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, candidate_metrics, baseline_metrics, finished_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![event.session_id.as_str(), payload.pair_id.as_str(), payload.game_id.as_str(), payload.candidate_side.as_str(), payload.outcome.as_str(), payload.seed, payload.round, payload.trace_game_seq, payload.plies, payload.elapsed_ms, serde_json::to_string(&payload.candidate)?, serde_json::to_string(&payload.baseline)?, &event.timestamp],
    )?;
    touch_session(tx, event)
}

fn project_pair_finished(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let TuningPayload::PairFinished(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    tx.execute(
        "UPDATE tuning_evaluation_pairs SET status = 'complete', ended_at = ?1, rating_after_mu = ?2, rating_after_sigma = ?3, score = ?4 WHERE session_id = ?5 AND pair_id = ?6",
        params![&event.timestamp, payload.rating_after.mu, payload.rating_after.sigma, payload.score, event.session_id.as_str(), payload.pair_id.as_str()],
    )?;
    touch_session(tx, event)
}

fn project_pair_failed(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<(), TuningStoreError> {
    let TuningPayload::PairFailed(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    tx.execute(
        "UPDATE tuning_evaluation_pairs SET status = 'failed', ended_at = ?1, failure = ?2 WHERE session_id = ?3 AND pair_id = ?4",
        params![&event.timestamp, payload.error, event.session_id.as_str(), payload.pair_id.as_str()],
    )?;
    touch_session(tx, event)
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
        "UPDATE tuning_trials SET status = ?1, ended_at = ?2, score = ?3, mu = ?4, sigma = ?5, stop_reason = ?6, failure = ?7 WHERE session_id = ?8 AND trial_id = ?9",
        params![status, &event.timestamp, terminal.score, terminal.mu, terminal.sigma, terminal.stop_reason.map(crate::tuning_lifecycle::TrialReportReason::as_str), terminal.error, event.session_id.as_str(), terminal.trial_id.as_str()],
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
