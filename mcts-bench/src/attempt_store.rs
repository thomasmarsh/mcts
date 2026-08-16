//! Durable storage for the physical-attempt lifecycle state machine.
//!
//! The caller owns the transaction.  Actions returned by
//! [`record_attempt_event`] are durable intent, and must not be executed until
//! that transaction has committed successfully.

use crate::orchestration::{
    transition_attempt, AttemptAction, AttemptEvent, AttemptPhase, AttemptState,
    AttemptTransitionError, ExitObservation, StopReason,
};
use duckdb::{params, Transaction};

#[derive(Debug)]
pub enum AttemptStoreError {
    DuckDb(duckdb::Error),
    MissingAttempt(String),
    MissingIdentity(String),
    Uninitialized(String),
    Corrupt {
        attempt_id: String,
        reason: String,
    },
    StaleVersion {
        expected: u64,
        actual: u64,
    },
    EventKeyConflict {
        attempt_id: String,
        event_key: String,
    },
    Transition(AttemptTransitionError),
}

impl std::fmt::Display for AttemptStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuckDb(error) => write!(f, "DuckDB error: {error}"),
            Self::MissingAttempt(id) => write!(f, "attempt '{id}' not found"),
            Self::MissingIdentity(id) => write!(f, "attempt '{id}' has no logical identity"),
            Self::Uninitialized(id) => write!(f, "attempt '{id}' has no typed lifecycle"),
            Self::Corrupt { attempt_id, reason } => {
                write!(f, "corrupt attempt '{attempt_id}' persistence: {reason}")
            }
            Self::StaleVersion { expected, actual } => {
                write!(
                    f,
                    "stale attempt version: expected {expected}, actual {actual}"
                )
            }
            Self::EventKeyConflict {
                attempt_id,
                event_key,
            } => write!(
                f,
                "event key '{event_key}' conflicts for attempt '{attempt_id}'"
            ),
            Self::Transition(error) => write!(f, "attempt transition rejected: {error}"),
        }
    }
}

impl std::error::Error for AttemptStoreError {}

impl From<duckdb::Error> for AttemptStoreError {
    fn from(error: duckdb::Error) -> Self {
        Self::DuckDb(error)
    }
}

impl From<AttemptTransitionError> for AttemptStoreError {
    fn from(error: AttemptTransitionError) -> Self {
        Self::Transition(error)
    }
}

/// Current typed state and its event-history version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttemptSnapshot {
    state: AttemptState,
    version: u64,
}

impl AttemptSnapshot {
    #[must_use]
    pub fn state(&self) -> &AttemptState {
        &self.state
    }

    #[must_use]
    pub const fn version(&self) -> u64 {
        self.version
    }
}

/// Whether an event changed durable state or was an exact event-key replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptTransitionDisposition {
    Applied,
    Replay,
}

/// Result of recording one event.  Actions are empty for a replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedAttemptTransition {
    snapshot: AttemptSnapshot,
    actions: Vec<AttemptAction>,
    disposition: AttemptTransitionDisposition,
}

impl PersistedAttemptTransition {
    #[must_use]
    pub fn state(&self) -> &AttemptState {
        self.snapshot.state()
    }

    #[must_use]
    pub const fn version(&self) -> u64 {
        self.snapshot.version
    }

    #[must_use]
    pub fn actions(&self) -> &[AttemptAction] {
        &self.actions
    }

    #[must_use]
    pub const fn disposition(&self) -> AttemptTransitionDisposition {
        self.disposition
    }

    #[must_use]
    pub const fn is_replay(&self) -> bool {
        matches!(self.disposition, AttemptTransitionDisposition::Replay)
    }
}

#[derive(Clone, Copy)]
struct EncodedEvent {
    event_type: &'static str,
    stop_reason: Option<&'static str>,
    exit_kind: Option<&'static str>,
    exit_code: Option<i32>,
}

fn encode_event(event: AttemptEvent) -> EncodedEvent {
    match event {
        AttemptEvent::StartRequested => EncodedEvent {
            event_type: "start_requested",
            stop_reason: None,
            exit_kind: None,
            exit_code: None,
        },
        AttemptEvent::ProcessObserved => EncodedEvent {
            event_type: "process_observed",
            stop_reason: None,
            exit_kind: None,
            exit_code: None,
        },
        AttemptEvent::SpawnFailed => EncodedEvent {
            event_type: "spawn_failed",
            stop_reason: None,
            exit_kind: None,
            exit_code: None,
        },
        AttemptEvent::StopRequested { reason } => EncodedEvent {
            event_type: "stop_requested",
            stop_reason: Some(stop_reason_token(reason)),
            exit_kind: None,
            exit_code: None,
        },
        AttemptEvent::SignalObserved => EncodedEvent {
            event_type: "signal_observed",
            stop_reason: None,
            exit_kind: None,
            exit_code: None,
        },
        AttemptEvent::ExitObserved { exit } => match exit {
            ExitObservation::Exited { code } => EncodedEvent {
                event_type: "exit_observed",
                stop_reason: None,
                exit_kind: Some("exited"),
                exit_code: code,
            },
            ExitObservation::Lost => EncodedEvent {
                event_type: "exit_observed",
                stop_reason: None,
                exit_kind: Some("lost"),
                exit_code: None,
            },
        },
        AttemptEvent::FinalOutputIngested => EncodedEvent {
            event_type: "final_output_ingested",
            stop_reason: None,
            exit_kind: None,
            exit_code: None,
        },
    }
}

fn stop_reason_token(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Operator => "operator",
        StopReason::BaselinePromotion => "baseline_promotion",
    }
}

fn phase_token(phase: AttemptPhase) -> &'static str {
    match phase {
        AttemptPhase::Planned => "planned",
        AttemptPhase::Starting => "starting",
        AttemptPhase::Running => "running",
        AttemptPhase::StopRequested => "stop_requested",
        AttemptPhase::AwaitingExit => "awaiting_exit",
        AttemptPhase::Finalizing => "finalizing",
        AttemptPhase::Completed => "completed",
        AttemptPhase::Stopped => "stopped",
        AttemptPhase::Crashed => "crashed",
    }
}

fn parse_phase(token: &str) -> Option<AttemptPhase> {
    Some(match token {
        "planned" => AttemptPhase::Planned,
        "starting" => AttemptPhase::Starting,
        "running" => AttemptPhase::Running,
        "stop_requested" => AttemptPhase::StopRequested,
        "awaiting_exit" => AttemptPhase::AwaitingExit,
        "finalizing" => AttemptPhase::Finalizing,
        "completed" => AttemptPhase::Completed,
        "stopped" => AttemptPhase::Stopped,
        "crashed" => AttemptPhase::Crashed,
        _ => return None,
    })
}

fn parse_stop_reason(token: &str) -> Option<StopReason> {
    match token {
        "operator" => Some(StopReason::Operator),
        "baseline_promotion" => Some(StopReason::BaselinePromotion),
        _ => None,
    }
}

fn corrupt(attempt_id: &str, reason: impl Into<String>) -> AttemptStoreError {
    AttemptStoreError::Corrupt {
        attempt_id: attempt_id.to_owned(),
        reason: reason.into(),
    }
}

fn identity_exists(tx: &Transaction<'_>, attempt_id: &str) -> Result<(), AttemptStoreError> {
    let linkage: Option<String> = tx
        .query_row(
            "SELECT logical_run_id FROM runs WHERE run_id = ?1",
            params![attempt_id],
            |row| row.get(0),
        )
        .map_err(|error| match error {
            duckdb::Error::QueryReturnedNoRows => {
                AttemptStoreError::MissingAttempt(attempt_id.to_owned())
            }
            other => AttemptStoreError::DuckDb(other),
        })?;
    let logical =
        linkage.ok_or_else(|| AttemptStoreError::MissingIdentity(attempt_id.to_owned()))?;
    let count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM logical_runs WHERE logical_run_id = ?1",
        params![logical],
        |row| row.get(0),
    )?;
    if count == 0 {
        return Err(AttemptStoreError::MissingIdentity(attempt_id.to_owned()));
    }
    Ok(())
}

#[derive(Clone)]
struct Projection {
    phase: Option<String>,
    stop_reason: Option<String>,
    process_observed: Option<bool>,
    signal_observed: Option<bool>,
    exit_kind: Option<String>,
    exit_code: Option<i32>,
    version: Option<u64>,
}

fn read_projection(
    tx: &Transaction<'_>,
    attempt_id: &str,
) -> Result<Projection, AttemptStoreError> {
    tx.query_row(
        "SELECT attempt_phase, attempt_stop_reason, attempt_process_observed, attempt_signal_observed, attempt_exit_kind, attempt_exit_code, attempt_version FROM runs WHERE run_id = ?1",
        params![attempt_id],
        |row| {
            Ok(Projection {
                phase: row.get(0)?,
                stop_reason: row.get(1)?,
                process_observed: row.get(2)?,
                signal_observed: row.get(3)?,
                exit_kind: row.get(4)?,
                exit_code: row.get(5)?,
                version: row.get(6)?,
            })
        },
    )
    .map_err(AttemptStoreError::DuckDb)
}

fn projection_is_empty(projection: &Projection) -> bool {
    projection.phase.is_none()
        && projection.stop_reason.is_none()
        && projection.process_observed.is_none()
        && projection.signal_observed.is_none()
        && projection.exit_kind.is_none()
        && projection.exit_code.is_none()
        && projection.version.is_none()
}

fn validate_projection(
    attempt_id: &str,
    projection: &Projection,
) -> Result<u64, AttemptStoreError> {
    let phase_token = projection
        .phase
        .as_deref()
        .ok_or_else(|| corrupt(attempt_id, "attempt phase is null"))?;
    let phase = parse_phase(phase_token)
        .ok_or_else(|| corrupt(attempt_id, format!("unknown attempt phase '{phase_token}'")))?;
    let version = projection
        .version
        .ok_or_else(|| corrupt(attempt_id, "attempt version is null"))?;
    let _stop_reason = match projection.stop_reason.as_deref() {
        None => None,
        Some(token) => Some(
            parse_stop_reason(token)
                .ok_or_else(|| corrupt(attempt_id, format!("unknown stop reason '{token}'")))?,
        ),
    };
    let _process_observed = projection
        .process_observed
        .ok_or_else(|| corrupt(attempt_id, "process observation is null"))?;
    let _signal_observed = projection
        .signal_observed
        .ok_or_else(|| corrupt(attempt_id, "signal observation is null"))?;
    let _exit = match projection.exit_kind.as_deref() {
        None if projection.exit_code.is_none() => None,
        Some("lost") if projection.exit_code.is_none() => Some(ExitObservation::Lost),
        Some("exited") => Some(ExitObservation::Exited {
            code: projection.exit_code,
        }),
        Some(token) => return Err(corrupt(attempt_id, format!("unknown exit kind '{token}'"))),
        None => return Err(corrupt(attempt_id, "exit code exists without exit kind")),
    };
    let _ = (
        phase,
        _stop_reason,
        _process_observed,
        _signal_observed,
        _exit,
    );
    Ok(version)
}

fn parse_event(
    attempt_id: &str,
    event_type: String,
    stop_reason: Option<String>,
    exit_kind: Option<String>,
    exit_code: Option<i32>,
) -> Result<AttemptEvent, AttemptStoreError> {
    let no_payload = || stop_reason.is_none() && exit_kind.is_none() && exit_code.is_none();
    match event_type.as_str() {
        "start_requested" if no_payload() => Ok(AttemptEvent::StartRequested),
        "process_observed" if no_payload() => Ok(AttemptEvent::ProcessObserved),
        "spawn_failed" if no_payload() => Ok(AttemptEvent::SpawnFailed),
        "signal_observed" if no_payload() => Ok(AttemptEvent::SignalObserved),
        "final_output_ingested" if no_payload() => Ok(AttemptEvent::FinalOutputIngested),
        "stop_requested" if exit_kind.is_none() && exit_code.is_none() => {
            let token =
                stop_reason.ok_or_else(|| corrupt(attempt_id, "stop event has no reason"))?;
            let reason = parse_stop_reason(&token)
                .ok_or_else(|| corrupt(attempt_id, format!("unknown stop reason '{token}'")))?;
            Ok(AttemptEvent::StopRequested { reason })
        }
        "exit_observed" if stop_reason.is_none() => match exit_kind.as_deref() {
            Some("lost") if exit_code.is_none() => Ok(AttemptEvent::ExitObserved {
                exit: ExitObservation::Lost,
            }),
            Some("exited") => Ok(AttemptEvent::ExitObserved {
                exit: ExitObservation::Exited { code: exit_code },
            }),
            Some(token) => Err(corrupt(attempt_id, format!("unknown exit kind '{token}'"))),
            None => Err(corrupt(attempt_id, "exit event has no kind")),
        },
        _ => Err(corrupt(
            attempt_id,
            format!("unknown or inconsistent event type '{event_type}'"),
        )),
    }
}

fn read_event_history(
    tx: &Transaction<'_>,
    attempt_id: &str,
    projection_version: u64,
) -> Result<Vec<(String, AttemptEvent)>, AttemptStoreError> {
    let mut statement = tx.prepare(
        "SELECT attempt_version, event_key, event_type, stop_reason, exit_kind, exit_code FROM attempt_events WHERE attempt_id = ?1 ORDER BY attempt_version",
    )?;
    let rows = statement.query_map(params![attempt_id], |row| {
        Ok((
            row.get::<_, u64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<i32>>(5)?,
        ))
    })?;
    let mut history = Vec::new();
    for row in rows {
        let (version, key, event_type, stop_reason, exit_kind, exit_code) = row?;
        let expected = history.len() as u64 + 1;
        if version != expected {
            return Err(corrupt(
                attempt_id,
                format!("event versions are not contiguous at {version}, expected {expected}"),
            ));
        }
        let event = parse_event(attempt_id, event_type, stop_reason, exit_kind, exit_code)?;
        history.push((key, event));
    }
    if history.len() as u64 != projection_version {
        return Err(corrupt(
            attempt_id,
            format!(
                "projection version {} disagrees with {} event rows",
                projection_version,
                history.len()
            ),
        ));
    }
    Ok(history)
}

fn state_matches_projection(
    attempt_id: &str,
    state: AttemptState,
    projection: &Projection,
) -> Result<(), AttemptStoreError> {
    if projection.phase.as_deref() != Some(phase_token(state.phase()))
        || projection.stop_reason.as_deref() != state.stop_reason().map(stop_reason_token)
        || projection.process_observed != Some(state.process_observed())
        || projection.signal_observed != Some(state.signal_observed())
    {
        return Err(corrupt(
            attempt_id,
            "projection facts disagree with event history",
        ));
    }
    match (
        state.exit_observation(),
        projection.exit_kind.as_deref(),
        projection.exit_code,
    ) {
        (None, None, None) => Ok(()),
        (Some(ExitObservation::Lost), Some("lost"), None) => Ok(()),
        (Some(ExitObservation::Exited { code }), Some("exited"), projected)
            if code == projected =>
        {
            Ok(())
        }
        _ => Err(corrupt(
            attempt_id,
            "exit projection disagrees with event history",
        )),
    }
}

/// Load and validate the complete typed history and materialized projection.
pub fn load_attempt(
    tx: &Transaction<'_>,
    attempt_id: &str,
) -> Result<AttemptSnapshot, AttemptStoreError> {
    identity_exists(tx, attempt_id)?;
    let projection = read_projection(tx, attempt_id)?;
    if projection_is_empty(&projection) {
        let events: i64 = tx.query_row(
            "SELECT COUNT(*) FROM attempt_events WHERE attempt_id = ?1",
            params![attempt_id],
            |row| row.get(0),
        )?;
        if events != 0 {
            return Err(corrupt(
                attempt_id,
                "uninitialized projection has event history",
            ));
        }
        return Err(AttemptStoreError::Uninitialized(attempt_id.to_owned()));
    }
    let projected_version = validate_projection(attempt_id, &projection)?;
    let history = read_event_history(tx, attempt_id, projected_version)?;
    let mut state = AttemptState::planned();
    for (_, event) in history {
        state = *transition_attempt(&state, event)
            .map_err(|error| corrupt(attempt_id, format!("event replay failed: {error}")))?
            .state();
    }
    state_matches_projection(attempt_id, state, &projection)?;
    Ok(AttemptSnapshot {
        state,
        version: projected_version,
    })
}

/// Initialize an identity-linked attempt in the caller's transaction.
pub fn initialize_attempt(
    tx: &Transaction<'_>,
    attempt_id: &str,
) -> Result<AttemptSnapshot, AttemptStoreError> {
    identity_exists(tx, attempt_id)?;
    let projection = read_projection(tx, attempt_id)?;
    if projection_is_empty(&projection) {
        let events: i64 = tx.query_row(
            "SELECT COUNT(*) FROM attempt_events WHERE attempt_id = ?1",
            params![attempt_id],
            |row| row.get(0),
        )?;
        if events != 0 {
            return Err(corrupt(
                attempt_id,
                "uninitialized projection has event history",
            ));
        }
        tx.execute(
            "UPDATE runs SET attempt_phase = 'planned', attempt_stop_reason = NULL, attempt_process_observed = FALSE, attempt_signal_observed = FALSE, attempt_exit_kind = NULL, attempt_exit_code = NULL, attempt_version = 0 WHERE run_id = ?1",
            params![attempt_id],
        )?;
        return Ok(AttemptSnapshot {
            state: AttemptState::planned(),
            version: 0,
        });
    }
    validate_projection(attempt_id, &projection)?;
    if projection.phase.as_deref() != Some("planned")
        || projection.stop_reason.is_some()
        || projection.process_observed != Some(false)
        || projection.signal_observed != Some(false)
        || projection.exit_kind.is_some()
        || projection.exit_code.is_some()
        || projection.version != Some(0)
    {
        return Err(corrupt(
            attempt_id,
            "attempt is partially initialized or not planned",
        ));
    }
    let count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM attempt_events WHERE attempt_id = ?1",
        params![attempt_id],
        |row| row.get(0),
    )?;
    if count != 0 {
        return Err(corrupt(attempt_id, "planned projection has event history"));
    }
    Ok(AttemptSnapshot {
        state: AttemptState::planned(),
        version: 0,
    })
}

fn same_event(a: AttemptEvent, b: AttemptEvent) -> bool {
    encode_event(a).event_type == encode_event(b).event_type
        && encode_event(a).stop_reason == encode_event(b).stop_reason
        && encode_event(a).exit_kind == encode_event(b).exit_kind
        && encode_event(a).exit_code == encode_event(b).exit_code
}

/// Append one typed event and update its projection atomically in `tx`.
pub fn record_attempt_event(
    tx: &Transaction<'_>,
    attempt_id: &str,
    expected_version: u64,
    event_key: &str,
    event: AttemptEvent,
    observed_at: &str,
) -> Result<PersistedAttemptTransition, AttemptStoreError> {
    let current = load_attempt(tx, attempt_id)?;
    if current.version != expected_version {
        return Err(AttemptStoreError::StaleVersion {
            expected: expected_version,
            actual: current.version,
        });
    }
    let existing = tx
        .query_row(
            "SELECT event_type, stop_reason, exit_kind, exit_code FROM attempt_events WHERE attempt_id = ?1 AND event_key = ?2",
            params![attempt_id, event_key],
            |row| Ok((row.get::<_, String>(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .map(Some)
        .or_else(|error| match error {
            duckdb::Error::QueryReturnedNoRows => Ok(None),
            other => Err(AttemptStoreError::DuckDb(other)),
        })?;
    if let Some((event_type, stop_reason, exit_kind, exit_code)) = existing {
        let stored = parse_event(attempt_id, event_type, stop_reason, exit_kind, exit_code)?;
        if !same_event(stored, event) {
            return Err(AttemptStoreError::EventKeyConflict {
                attempt_id: attempt_id.to_owned(),
                event_key: event_key.to_owned(),
            });
        }
        return Ok(PersistedAttemptTransition {
            snapshot: current,
            actions: Vec::new(),
            disposition: AttemptTransitionDisposition::Replay,
        });
    }

    let transition = transition_attempt(&current.state, event)?;
    let next_version = current
        .version
        .checked_add(1)
        .ok_or_else(|| corrupt(attempt_id, "attempt version overflow"))?;
    let encoded = encode_event(event);
    tx.execute(
        "INSERT INTO attempt_events (attempt_id, attempt_version, event_key, event_type, stop_reason, exit_kind, exit_code, observed_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            attempt_id,
            next_version,
            event_key,
            encoded.event_type,
            encoded.stop_reason,
            encoded.exit_kind,
            encoded.exit_code,
            observed_at
        ],
    )?;
    let next = *transition.state();
    let next_exit = next.exit_observation();
    let (exit_kind, exit_code) = match next_exit {
        None => (None, None),
        Some(ExitObservation::Lost) => (Some("lost"), None),
        Some(ExitObservation::Exited { code }) => (Some("exited"), code),
    };
    let changed = match tx.execute(
        "UPDATE runs SET attempt_phase = ?1, attempt_stop_reason = ?2, attempt_process_observed = ?3, attempt_signal_observed = ?4, attempt_exit_kind = ?5, attempt_exit_code = ?6, attempt_version = ?7 WHERE run_id = ?8 AND attempt_version = ?9",
        params![
            phase_token(next.phase()),
            next.stop_reason().map(stop_reason_token),
            next.process_observed(),
            next.signal_observed(),
            exit_kind,
            exit_code,
            next_version,
            attempt_id,
            current.version
        ],
    ) {
        Ok(changed) => changed,
        Err(error) => {
            // The event insert is the only preceding write.  Remove it before
            // returning so a caller that commits an otherwise-valid
            // transaction cannot persist a journal row without its projection.
            let _ = tx.execute(
                "DELETE FROM attempt_events WHERE attempt_id = ?1 AND event_key = ?2 AND attempt_version = ?3",
                params![attempt_id, event_key, next_version],
            );
            return Err(AttemptStoreError::DuckDb(error));
        }
    };
    if changed != 1 {
        let _ = tx.execute(
            "DELETE FROM attempt_events WHERE attempt_id = ?1 AND event_key = ?2 AND attempt_version = ?3",
            params![attempt_id, event_key, next_version],
        );
        return Err(AttemptStoreError::StaleVersion {
            expected: expected_version,
            actual: current.version,
        });
    }
    Ok(PersistedAttemptTransition {
        snapshot: AttemptSnapshot {
            state: next,
            version: next_version,
        },
        actions: transition.actions().to_vec(),
        disposition: AttemptTransitionDisposition::Applied,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ensure_schema;
    use duckdb::Connection;

    type ProjectionRow = (
        Option<String>,
        Option<String>,
        Option<bool>,
        Option<bool>,
        Option<String>,
        Option<i32>,
        Option<u64>,
    );

    fn database() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO logical_runs (logical_run_id, kind, created_at, current_attempt_id) VALUES ('logical', 'smac3', CURRENT_TIMESTAMP, 'attempt');
             INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path, logical_run_id, attempt_ordinal) VALUES ('attempt', 'smac3', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'running', '/tmp/log', 'logical', 1);",
        )
        .unwrap();
        conn
    }

    fn event(
        conn: &mut Connection,
        version: u64,
        key: &str,
        value: AttemptEvent,
    ) -> PersistedAttemptTransition {
        let tx = conn.transaction().unwrap();
        let result =
            record_attempt_event(&tx, "attempt", version, key, value, "2026-01-01T00:00:00Z")
                .unwrap();
        tx.commit().unwrap();
        result
    }

    #[test]
    fn initialize_and_replay_are_durable_and_action_free() {
        let mut conn = database();
        {
            let tx = conn.transaction().unwrap();
            let snapshot = initialize_attempt(&tx, "attempt").unwrap();
            assert_eq!(snapshot.state().phase(), AttemptPhase::Planned);
            assert_eq!(snapshot.version(), 0);
            tx.commit().unwrap();
        }
        let first = event(&mut conn, 0, "start", AttemptEvent::StartRequested);
        assert_eq!(first.disposition(), AttemptTransitionDisposition::Applied);
        assert_eq!(first.version(), 1);
        assert_eq!(first.actions(), &[AttemptAction::SpawnProcess]);
        let replay = event(&mut conn, 1, "start", AttemptEvent::StartRequested);
        assert!(replay.is_replay());
        assert!(replay.actions().is_empty());
        assert_eq!(replay.version(), 1);
        let tx = conn.transaction().unwrap();
        let loaded = load_attempt(&tx, "attempt").unwrap();
        assert_eq!(loaded.state().phase(), AttemptPhase::Starting);
        assert_eq!(loaded.version(), 1);
    }

    #[test]
    fn complete_path_replays_and_projects_each_fact() {
        let mut conn = database();
        {
            let tx = conn.transaction().unwrap();
            initialize_attempt(&tx, "attempt").unwrap();
            tx.commit().unwrap();
        }
        event(&mut conn, 0, "start", AttemptEvent::StartRequested);
        {
            let tx = conn.transaction().unwrap();
            assert_eq!(
                projection(&tx, "attempt"),
                (
                    Some("starting".into()),
                    None,
                    Some(false),
                    Some(false),
                    None,
                    None,
                    Some(1)
                )
            );
            assert_eq!(history_count(&tx, "attempt"), 1);
            tx.commit().unwrap();
        }
        event(&mut conn, 1, "process", AttemptEvent::ProcessObserved);
        {
            let tx = conn.transaction().unwrap();
            assert_eq!(
                projection(&tx, "attempt"),
                (
                    Some("running".into()),
                    None,
                    Some(true),
                    Some(false),
                    None,
                    None,
                    Some(2)
                )
            );
            assert_eq!(history_count(&tx, "attempt"), 2);
            tx.commit().unwrap();
        }
        event(
            &mut conn,
            2,
            "exit",
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Exited { code: Some(0) },
            },
        );
        {
            let tx = conn.transaction().unwrap();
            assert_eq!(
                projection(&tx, "attempt"),
                (
                    Some("finalizing".into()),
                    None,
                    Some(true),
                    Some(false),
                    Some("exited".into()),
                    Some(0),
                    Some(3)
                )
            );
            assert_eq!(history_count(&tx, "attempt"), 3);
            tx.commit().unwrap();
        }
        let finalizing = event(&mut conn, 3, "output", AttemptEvent::FinalOutputIngested);
        assert_eq!(finalizing.state().phase(), AttemptPhase::Completed);
        let tx = conn.transaction().unwrap();
        let row: (String, bool, bool, String, Option<i32>, u64) = tx
            .query_row(
                "SELECT attempt_phase, attempt_process_observed, attempt_signal_observed, attempt_exit_kind, attempt_exit_code, attempt_version FROM runs WHERE run_id = 'attempt'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
            )
            .unwrap();
        assert_eq!(
            row,
            ("completed".into(), true, false, "exited".into(), Some(0), 4)
        );
        assert_eq!(
            tx.query_row("SELECT COUNT(*) FROM attempt_events", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            4
        );
    }

    #[test]
    fn conflict_stale_and_illegal_event_do_not_change_state() {
        let mut conn = database();
        {
            let tx = conn.transaction().unwrap();
            initialize_attempt(&tx, "attempt").unwrap();
            tx.commit().unwrap();
        }
        event(&mut conn, 0, "start", AttemptEvent::StartRequested);
        let tx = conn.transaction().unwrap();
        let before = projection(&tx, "attempt");
        let replay = record_attempt_event(
            &tx,
            "attempt",
            1,
            "start",
            AttemptEvent::StartRequested,
            "2030-01-01T00:00:00Z",
        )
        .unwrap();
        assert!(replay.is_replay());
        assert!(replay.actions().is_empty());
        assert_eq!(projection(&tx, "attempt"), before);
        assert_eq!(history_count(&tx, "attempt"), 1);
        tx.commit().unwrap();

        let tx = conn.transaction().unwrap();
        let before = projection(&tx, "attempt");
        let conflict =
            record_attempt_event(&tx, "attempt", 1, "start", AttemptEvent::SpawnFailed, "now");
        assert!(matches!(
            conflict,
            Err(AttemptStoreError::EventKeyConflict { .. })
        ));
        assert_eq!(projection(&tx, "attempt"), before);
        assert_eq!(history_count(&tx, "attempt"), 1);
        tx.commit().unwrap();

        let tx = conn.transaction().unwrap();
        let before = projection(&tx, "attempt");
        let stale = record_attempt_event(
            &tx,
            "attempt",
            0,
            "other",
            AttemptEvent::ProcessObserved,
            "now",
        );
        assert!(matches!(
            stale,
            Err(AttemptStoreError::StaleVersion {
                expected: 0,
                actual: 1
            })
        ));
        assert_eq!(projection(&tx, "attempt"), before);
        assert_eq!(history_count(&tx, "attempt"), 1);
        tx.commit().unwrap();

        let tx = conn.transaction().unwrap();
        let before = projection(&tx, "attempt");
        let illegal = record_attempt_event(
            &tx,
            "attempt",
            1,
            "bad",
            AttemptEvent::FinalOutputIngested,
            "now",
        );
        assert!(matches!(illegal, Err(AttemptStoreError::Transition(_))));
        assert_eq!(projection(&tx, "attempt"), before);
        assert_eq!(history_count(&tx, "attempt"), 1);
        tx.commit().unwrap();
    }

    fn initialize(conn: &mut Connection, attempt_id: &str) {
        let tx = conn.transaction().unwrap();
        initialize_attempt(&tx, attempt_id).unwrap();
        tx.commit().unwrap();
    }

    fn projection(tx: &Transaction<'_>, attempt_id: &str) -> ProjectionRow {
        tx.query_row(
            "SELECT attempt_phase, attempt_stop_reason, attempt_process_observed, attempt_signal_observed, attempt_exit_kind, attempt_exit_code, attempt_version FROM runs WHERE run_id = ?1",
            [attempt_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?)),
        )
        .unwrap()
    }

    fn history_count(tx: &Transaction<'_>, attempt_id: &str) -> i64 {
        tx.query_row(
            "SELECT COUNT(*) FROM attempt_events WHERE attempt_id = ?1",
            [attempt_id],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn initialization_rejects_missing_identity_partial_state_and_history() {
        let mut conn = database();
        let tx = conn.transaction().unwrap();
        assert!(matches!(
            initialize_attempt(&tx, "missing"),
            Err(AttemptStoreError::MissingAttempt(_))
        ));
        tx.commit().unwrap();

        conn.execute(
            "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('no-identity', 'smac3', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'running', '/tmp/log')",
            [],
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        assert!(matches!(
            initialize_attempt(&tx, "no-identity"),
            Err(AttemptStoreError::MissingIdentity(_))
        ));
        tx.commit().unwrap();

        conn.execute(
            "UPDATE runs SET logical_run_id = 'dangling' WHERE run_id = 'no-identity'",
            [],
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        assert!(matches!(
            initialize_attempt(&tx, "no-identity"),
            Err(AttemptStoreError::MissingIdentity(_))
        ));
        tx.commit().unwrap();

        let tx = conn.transaction().unwrap();
        tx.execute(
            "UPDATE runs SET attempt_phase = 'planned' WHERE run_id = 'attempt'",
            [],
        )
        .unwrap();
        let before = projection(&tx, "attempt");
        assert!(matches!(
            initialize_attempt(&tx, "attempt"),
            Err(AttemptStoreError::Corrupt { .. })
        ));
        assert_eq!(projection(&tx, "attempt"), before);
        tx.commit().unwrap();

        conn.execute(
            "UPDATE runs SET attempt_phase = NULL WHERE run_id = 'attempt'; INSERT INTO attempt_events (attempt_id, attempt_version, event_key, event_type, observed_at) VALUES ('attempt', 1, 'orphan', 'start_requested', CURRENT_TIMESTAMP)",
            [],
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        assert!(matches!(
            initialize_attempt(&tx, "attempt"),
            Err(AttemptStoreError::Corrupt { .. })
        ));
        assert!(matches!(
            load_attempt(&tx, "attempt"),
            Err(AttemptStoreError::Corrupt { .. })
        ));
        assert_eq!(
            projection(&tx, "attempt"),
            (None, None, None, None, None, None, None)
        );
        assert_eq!(history_count(&tx, "attempt"), 1);
        tx.commit().unwrap();

        let mut fresh = database();
        initialize(&mut fresh, "attempt");
        let tx = fresh.transaction().unwrap();
        let repeated = initialize_attempt(&tx, "attempt").unwrap();
        assert_eq!(repeated.state().phase(), AttemptPhase::Planned);
        assert_eq!(repeated.version(), 0);
        assert_eq!(history_count(&tx, "attempt"), 0);
        tx.commit().unwrap();
    }

    #[test]
    fn stop_path_persists_reason_observations_exit_and_actions() {
        let mut conn = database();
        initialize(&mut conn, "attempt");
        let start = event(&mut conn, 0, "start", AttemptEvent::StartRequested);
        assert_eq!(start.actions(), &[AttemptAction::SpawnProcess]);
        event(&mut conn, 1, "process", AttemptEvent::ProcessObserved);
        let stop = event(
            &mut conn,
            2,
            "stop",
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        assert_eq!(stop.actions(), &[AttemptAction::SignalProcessGroup]);
        event(&mut conn, 3, "signal", AttemptEvent::SignalObserved);
        let exit = event(
            &mut conn,
            4,
            "exit",
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Exited { code: Some(0) },
            },
        );
        assert_eq!(exit.actions(), &[AttemptAction::FinalizeOutput]);
        let completed = event(&mut conn, 5, "output", AttemptEvent::FinalOutputIngested);
        assert_eq!(completed.state().phase(), AttemptPhase::Stopped);
        assert_eq!(completed.version(), 6);
        let tx = conn.transaction().unwrap();
        assert_eq!(
            projection(&tx, "attempt"),
            (
                Some("stopped".into()),
                Some("operator".into()),
                Some(true),
                Some(true),
                Some("exited".into()),
                Some(0),
                Some(6)
            )
        );
        assert_eq!(history_count(&tx, "attempt"), 6);
    }

    #[test]
    fn abnormal_exits_remain_finalizing_until_output_then_crash() {
        for (suffix, exit, expected_kind, expected_code) in [
            (
                "nonzero",
                ExitObservation::Exited { code: Some(7) },
                "exited",
                Some(7),
            ),
            (
                "unknown",
                ExitObservation::Exited { code: None },
                "exited",
                None,
            ),
            ("lost", ExitObservation::Lost, "lost", None),
        ] {
            let mut conn = database();
            initialize(&mut conn, "attempt");
            event(
                &mut conn,
                0,
                &format!("start-{suffix}"),
                AttemptEvent::StartRequested,
            );
            event(
                &mut conn,
                1,
                &format!("process-{suffix}"),
                AttemptEvent::ProcessObserved,
            );
            let finalizing = event(
                &mut conn,
                2,
                &format!("exit-{suffix}"),
                AttemptEvent::ExitObserved { exit },
            );
            assert_eq!(finalizing.state().phase(), AttemptPhase::Finalizing);
            let tx = conn.transaction().unwrap();
            assert_eq!(
                projection(&tx, "attempt"),
                (
                    Some("finalizing".into()),
                    None,
                    Some(true),
                    Some(false),
                    Some(expected_kind.into()),
                    expected_code,
                    Some(3)
                )
            );
            tx.commit().unwrap();
            let crashed = event(
                &mut conn,
                3,
                &format!("output-{suffix}"),
                AttemptEvent::FinalOutputIngested,
            );
            assert_eq!(crashed.state().phase(), AttemptPhase::Crashed);
            assert_eq!(crashed.version(), 4);
        }
    }

    #[test]
    fn caller_rollback_removes_event_and_projection_update() {
        let mut conn = database();
        initialize(&mut conn, "attempt");
        let tx = conn.transaction().unwrap();
        let applied = record_attempt_event(
            &tx,
            "attempt",
            0,
            "start",
            AttemptEvent::StartRequested,
            "2026-01-01T00:00:00Z",
        )
        .unwrap();
        assert_eq!(applied.version(), 1);
        assert_eq!(history_count(&tx, "attempt"), 1);
        assert_eq!(projection(&tx, "attempt").0, Some("starting".into()));
        tx.rollback().unwrap();
        let tx = conn.transaction().unwrap();
        assert_eq!(history_count(&tx, "attempt"), 0);
        assert_eq!(
            projection(&tx, "attempt"),
            (
                Some("planned".into()),
                None,
                Some(false),
                Some(false),
                None,
                None,
                Some(0)
            )
        );
        tx.commit().unwrap();
    }

    #[test]
    fn load_rejects_unknown_tokens_gaps_and_projection_disagreement() {
        let mut conn = database();
        initialize(&mut conn, "attempt");
        event(&mut conn, 0, "start", AttemptEvent::StartRequested);
        conn.execute(
            "UPDATE attempt_events SET event_type = 'unknown' WHERE attempt_id = 'attempt' AND attempt_version = 1",
            [],
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        assert!(matches!(
            load_attempt(&tx, "attempt"),
            Err(AttemptStoreError::Corrupt { .. })
        ));
        tx.commit().unwrap();

        conn.execute(
            "UPDATE attempt_events SET event_type = 'start_requested' WHERE attempt_id = 'attempt' AND attempt_version = 1; UPDATE attempt_events SET attempt_version = 3 WHERE attempt_id = 'attempt' AND attempt_version = 1",
            [],
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        assert!(matches!(
            load_attempt(&tx, "attempt"),
            Err(AttemptStoreError::Corrupt { .. })
        ));
        tx.commit().unwrap();

        conn.execute(
            "UPDATE attempt_events SET attempt_version = 1 WHERE attempt_id = 'attempt' AND attempt_version = 3; UPDATE runs SET attempt_phase = 'running' WHERE run_id = 'attempt'",
            [],
        )
        .unwrap();
        let tx = conn.transaction().unwrap();
        assert!(matches!(
            load_attempt(&tx, "attempt"),
            Err(AttemptStoreError::Corrupt { .. })
        ));
        tx.commit().unwrap();
    }

    #[test]
    fn initialized_attempts_are_isolated() {
        let mut conn = database();
        conn.execute_batch(
            "INSERT INTO logical_runs (logical_run_id, kind, created_at, current_attempt_id) VALUES ('logical-two', 'smac3', CURRENT_TIMESTAMP, 'attempt-two');
             INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path, logical_run_id, attempt_ordinal) VALUES ('attempt-two', 'smac3', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'running', '/tmp/log', 'logical-two', 1);",
        )
        .unwrap();
        initialize(&mut conn, "attempt");
        initialize(&mut conn, "attempt-two");
        event(&mut conn, 0, "start", AttemptEvent::StartRequested);
        let tx = conn.transaction().unwrap();
        assert_eq!(
            projection(&tx, "attempt-two"),
            (
                Some("planned".into()),
                None,
                Some(false),
                Some(false),
                None,
                None,
                Some(0)
            )
        );
        assert_eq!(history_count(&tx, "attempt-two"), 0);
        assert_eq!(history_count(&tx, "attempt"), 1);
        tx.commit().unwrap();
    }
}
