use duckdb::{params, Transaction};

use crate::tuning_lifecycle::{
    PoolAnchorInsertionReason, PoolAnchorProvenance, PoolAnchorSnapshot, TuningEventType,
    TuningLifecycleEvent, TuningPayload,
};

use super::TuningStoreError;

struct AttemptState {
    session_id: String,
    status: String,
}

struct TrialState {
    attempt_id: String,
    status: String,
}

struct PairState {
    trial_id: String,
    pair_index: u32,
    status: String,
    seed: u64,
    round: u32,
    rating_before_mu: f64,
    rating_before_sigma: f64,
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
    if event.event_type == TuningEventType::PoolRevised {
        return validate_pool_revision(tx, event);
    }
    if event.event_type.is_pair() {
        return validate_pair(tx, event);
    }
    if event.event_type.is_trial() {
        return validate_trial(tx, event);
    }
    Ok(None)
}

fn validate_pool_revision(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<Option<String>, TuningStoreError> {
    let TuningPayload::PoolRevised(payload) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    if payload.pool_snapshot_fingerprint.is_empty()
        || payload.anchors.is_empty()
        || !anchors_are_valid(&payload.anchors)
    {
        return Ok(Some("invalid pool revision snapshot".into()));
    }
    let exists: i64 = tx.query_row(
        "SELECT COUNT(*) FROM tuning_pool_revisions WHERE session_id = ?1 AND pool_snapshot_fingerprint = ?2",
        params![event.session_id.as_str(), &payload.pool_snapshot_fingerprint],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Ok(None);
    }
    let stored = stored_pool_anchors(tx, event, &payload.pool_snapshot_fingerprint)?;
    Ok((stored != payload.anchors)
        .then_some("pool revision fingerprint has conflicting anchor snapshot".into()))
}

fn anchors_are_valid(anchors: &[PoolAnchorSnapshot]) -> bool {
    anchors.iter().enumerate().all(|(index, anchor)| {
        !anchor.anchor_id.is_empty()
            && anchor.mu.is_finite()
            && anchor.sigma.is_finite()
            && anchors[..index]
                .iter()
                .all(|previous| previous.anchor_id != anchor.anchor_id)
            && provenance_matches_reason(anchor)
    })
}

fn provenance_matches_reason(anchor: &PoolAnchorSnapshot) -> bool {
    match (anchor.provenance, anchor.insertion_reason) {
        (
            PoolAnchorProvenance::BootstrapDefault | PoolAnchorProvenance::BootstrapRandom,
            PoolAnchorInsertionReason::Bootstrap,
        )
        | (PoolAnchorProvenance::Configured, PoolAnchorInsertionReason::Configured)
        | (PoolAnchorProvenance::LegacyUnknown, PoolAnchorInsertionReason::LegacyUnknown) => {
            anchor.source_trial_id.is_none()
        }
        (
            PoolAnchorProvenance::Trial,
            PoolAnchorInsertionReason::Champion | PoolAnchorInsertionReason::SkillBand,
        ) => anchor
            .source_trial_id
            .as_ref()
            .is_some_and(|trial_id| !trial_id.is_empty()),
        _ => false,
    }
}

fn stored_pool_anchors(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    fingerprint: &str,
) -> Result<Vec<PoolAnchorSnapshot>, TuningStoreError> {
    let mut statement = tx.prepare(
        "SELECT anchor_id, CAST(config AS TEXT), mu, sigma, provenance, insertion_reason, source_trial_id FROM tuning_pool_anchors WHERE session_id = ?1 AND pool_snapshot_fingerprint = ?2 ORDER BY anchor_ordinal",
    )?;
    let anchors = statement
        .query_map(params![event.session_id.as_str(), fingerprint], |row| {
            let config: String = row.get(1)?;
            Ok((
                row.get::<_, String>(0)?,
                config,
                row.get::<_, f64>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    anchors
        .into_iter()
        .map(
            |(anchor_id, config, mu, sigma, provenance, insertion_reason, source_trial_id)| {
                Ok(PoolAnchorSnapshot {
                    anchor_id,
                    config: serde_json::from_str(&config)?,
                    mu,
                    sigma,
                    provenance: serde_json::from_value(serde_json::Value::String(provenance))?,
                    insertion_reason: serde_json::from_value(serde_json::Value::String(
                        insertion_reason,
                    ))?,
                    source_trial_id,
                })
            },
        )
        .collect::<Result<Vec<_>, TuningStoreError>>()
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
        TuningEventType::TrialReported => validate_trial_report(tx, event, status),
        event_type if event_type.is_trial_terminal() => Ok(terminal_stop_reason_is_valid(event)
            && matches!(status, Some("queued") | Some("running"))
            && !has_running_pairs(tx, event)?),
        _ => Ok(false),
    }
}

fn validate_trial_report(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    status: Option<&str>,
) -> Result<bool, TuningStoreError> {
    if status != Some("running") {
        return Ok(false);
    }
    let TuningPayload::TrialReported(report) = event.typed_payload().expect("validated payload")
    else {
        unreachable!()
    };
    if report.trial_number != trial_number(tx, event, report.trial_id.as_str())?
        || report.completed_pairs == 0
        || report.score_formula_version != 1
        || !report.mu.is_finite()
        || !report.sigma.is_finite()
        || !report.score.is_finite()
        || !report.conservative_k.is_finite()
        || !report.reason.is_valid_for(report.outcome)
    {
        return Ok(false);
    }
    let last_resource: Option<u64> = tx.query_row(
        "SELECT MAX(completed_pairs) FROM tuning_trial_reports WHERE session_id = ?1 AND trial_id = ?2",
        params![event.session_id.as_str(), report.trial_id.as_str()],
        |row| row.get(0),
    )?;
    let expected_resource = last_resource.map_or(Some(1), |last| last.checked_add(1));
    Ok(expected_resource == Some(report.completed_pairs))
}

fn trial_number(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    trial_id: &str,
) -> Result<i64, TuningStoreError> {
    tx.query_row(
        "SELECT trial_number FROM tuning_trials WHERE session_id = ?1 AND trial_id = ?2",
        params![event.session_id.as_str(), trial_id],
        |row| row.get(0),
    )
    .map_err(Into::into)
}

fn terminal_stop_reason_is_valid(event: &TuningLifecycleEvent) -> bool {
    let terminal = match event.typed_payload().expect("validated payload") {
        TuningPayload::TrialCompleted(value)
        | TuningPayload::TrialPruned(value)
        | TuningPayload::TrialFailed(value)
        | TuningPayload::TrialCancelled(value) => value,
        _ => unreachable!(),
    };
    let outcome = match event.event_type {
        TuningEventType::TrialCompleted => {
            Some(crate::tuning_lifecycle::TrialReportOutcome::Complete)
        }
        TuningEventType::TrialPruned => Some(crate::tuning_lifecycle::TrialReportOutcome::Prune),
        TuningEventType::TrialFailed | TuningEventType::TrialCancelled => None,
        _ => unreachable!(),
    };
    terminal
        .stop_reason
        .is_none_or(|reason| outcome.is_some_and(|outcome| reason.is_valid_for(outcome)))
}

fn has_running_pairs(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<bool, TuningStoreError> {
    let count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM tuning_evaluation_pairs WHERE session_id = ?1 AND trial_id = ?2 AND status = 'running'",
        params![event.session_id.as_str(), event.trial_id().expect("validated trial payload").expect("trial event has trial").as_str()],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn validate_pair(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
) -> Result<Option<String>, TuningStoreError> {
    let trial_id = event
        .trial_id()
        .expect("validated pair payload")
        .expect("pair event has trial id");
    let trial = load_trial(tx, event, trial_id.as_str())?;
    if let Some(reason) = validate_trial_attempt(trial.as_ref(), event) {
        return Ok(Some(reason));
    }
    if trial.as_ref().map(|value| value.status.as_str()) != Some("running") {
        return Ok(Some("pair event requires a running trial".into()));
    }
    match event.typed_payload().expect("validated payload") {
        TuningPayload::PairStarted(payload) => validate_pair_start(tx, event, &payload),
        TuningPayload::GameFinished(payload) => validate_game_finished(tx, event, &payload),
        TuningPayload::PairFinished(payload) => validate_pair_finished(tx, event, &payload),
        TuningPayload::PairFailed(payload) => validate_pair_failed(tx, event, &payload),
        _ => unreachable!(),
    }
}

fn validate_pair_start(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    payload: &crate::tuning_lifecycle::PairStartedPayload,
) -> Result<Option<String>, TuningStoreError> {
    let count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM tuning_evaluation_pairs WHERE session_id = ?1 AND (pair_id = ?2 OR (trial_id = ?3 AND pair_index = ?4))",
        params![event.session_id.as_str(), payload.pair_id.as_str(), payload.trial_id.as_str(), payload.pair_index],
        |row| row.get(0),
    )?;
    Ok((count != 0).then_some("pair id or pair index already exists".into()))
}

fn validate_game_finished(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    payload: &crate::tuning_lifecycle::GameFinishedPayload,
) -> Result<Option<String>, TuningStoreError> {
    let Some(pair) = load_pair(tx, event, payload.pair_id.as_str())? else {
        return Ok(Some("game references an unknown pair".into()));
    };
    if pair.trial_id != payload.trial_id.as_str() {
        return Ok(Some("game pair belongs to a different trial".into()));
    }
    if pair.status != "running" {
        return Ok(Some("game follows pair terminal evidence".into()));
    }
    if pair.seed != payload.seed || pair.round != payload.round {
        return Ok(Some("game seed or round differs from its pair".into()));
    }
    let count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM tuning_games WHERE session_id = ?1 AND (game_id = ?2 OR (pair_id = ?3 AND candidate_side = ?4))",
        params![event.session_id.as_str(), payload.game_id.as_str(), payload.pair_id.as_str(), payload.candidate_side.as_str()],
        |row| row.get(0),
    )?;
    Ok((count != 0).then_some("game id or candidate side already exists".into()))
}

fn validate_pair_finished(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    payload: &crate::tuning_lifecycle::PairFinishedPayload,
) -> Result<Option<String>, TuningStoreError> {
    let Some(pair) = load_pair(tx, event, payload.pair_id.as_str())? else {
        return Ok(Some("pair_finished references an unknown pair".into()));
    };
    if pair.trial_id != payload.trial_id.as_str()
        || pair.pair_index != payload.pair_index
        || pair.status != "running"
    {
        return Ok(Some(
            "pair_finished requires a running pair in its trial".into(),
        ));
    }
    if !same_rating(&pair, &payload.rating_before) {
        return Ok(Some(
            "pair_finished rating_before differs from pair_started".into(),
        ));
    }
    let count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM tuning_games WHERE session_id = ?1 AND pair_id = ?2 AND seed = ?3 AND round = ?4",
        params![event.session_id.as_str(), payload.pair_id.as_str(), pair.seed, pair.round],
        |row| row.get(0),
    )?;
    Ok((count != 2).then_some("pair_finished requires two matching games".into()))
}

fn validate_pair_failed(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    payload: &crate::tuning_lifecycle::PairFailedPayload,
) -> Result<Option<String>, TuningStoreError> {
    let Some(pair) = load_pair(tx, event, payload.pair_id.as_str())? else {
        return Ok(Some("pair_failed references an unknown pair".into()));
    };
    Ok((pair.trial_id != payload.trial_id.as_str()
        || pair.pair_index != payload.pair_index
        || pair.status != "running")
        .then_some("pair_failed requires a running pair in its trial".into()))
}

fn load_pair(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    pair_id: &str,
) -> Result<Option<PairState>, TuningStoreError> {
    Ok(tx.query_row(
        "SELECT trial_id, pair_index, status, seed, round, rating_before_mu, rating_before_sigma FROM tuning_evaluation_pairs WHERE session_id = ?1 AND pair_id = ?2",
        params![event.session_id.as_str(), pair_id],
        |row| Ok(PairState { trial_id: row.get(0)?, pair_index: row.get(1)?, status: row.get(2)?, seed: row.get(3)?, round: row.get(4)?, rating_before_mu: row.get(5)?, rating_before_sigma: row.get(6)? }),
    ).ok())
}

fn same_rating(pair: &PairState, rating: &crate::tuning_lifecycle::Rating) -> bool {
    pair.rating_before_mu == rating.mu && pair.rating_before_sigma == rating.sigma
}
