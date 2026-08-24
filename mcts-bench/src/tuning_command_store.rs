//! Transactional control decisions for persisted tuning sessions.
//!
//! This module only reserves work and records the decision that made the
//! reservation. A launcher owns process creation and reports its durable
//! outcome back through [`record_launch_outcome`].

use duckdb::{params, Connection, OptionalExt, Transaction};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionCommand {
    Stop,
    Resume,
    AddBudget { delta: u64, start: bool },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchReservation {
    pub attempt_id: String,
    pub physical_run_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandRequest {
    pub command_id: String,
    pub expected_version: u64,
    pub command: SessionCommand,
    pub launch: Option<LaunchReservation>,
    pub observed_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandKind {
    Stop,
    Resume,
    AddBudget,
}

impl From<&SessionCommand> for CommandKind {
    fn from(command: &SessionCommand) -> Self {
        match command {
            SessionCommand::Stop => Self::Stop,
            SessionCommand::Resume => Self::Resume,
            SessionCommand::AddBudget { .. } => Self::AddBudget,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenialReason {
    NoActiveAttempt,
    StopAlreadyReserved,
    ActiveAttempt,
    LaunchReserved,
    Exhausted,
    NoncontinuableLegacy,
    RecoveryRequired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllowedCommand {
    pub command: CommandKind,
    pub allowed: bool,
    pub denial_reason: Option<DenialReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionControl {
    pub control_version: u64,
    pub target_trial_count: Option<u64>,
    pub consumed_trial_count: u64,
    pub active_attempt_id: Option<String>,
    pub launch_reservation: Option<LaunchReservation>,
    pub stop_attempt_id: Option<String>,
    pub recovery_required: bool,
    pub allowed_commands: Vec<AllowedCommand>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandDecision {
    pub command_id: String,
    pub command: SessionCommand,
    pub replay: bool,
    pub control: SessionControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaunchOutcome {
    Spawned,
    SpawnFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum StoredOutcome {
    Accepted { decision: CommandDecision },
    Rejected { error: StoredError },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum StoredError {
    ExpectedVersionConflict {
        expected: u64,
        control: SessionControl,
    },
    ActiveAttempt {
        attempt_id: String,
        control: SessionControl,
    },
    LaunchReserved {
        attempt_id: String,
        control: SessionControl,
    },
    InvalidDeltaStart {
        control: SessionControl,
    },
    ExhaustedResume {
        control: SessionControl,
    },
    NoncontinuableLegacy {
        control: SessionControl,
    },
    CommandDenied {
        reason: DenialReason,
        control: SessionControl,
    },
    TargetOverflow {
        control: SessionControl,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandStoreError {
    DuckDb(String),
    Serialization(String),
    SessionNotFound(String),
    CommandIdReuseMismatch {
        command_id: String,
    },
    ExpectedVersionConflict {
        expected: u64,
        control: Box<SessionControl>,
    },
    ActiveAttempt {
        attempt_id: String,
        control: Box<SessionControl>,
    },
    LaunchReserved {
        attempt_id: String,
        control: Box<SessionControl>,
    },
    InvalidDeltaStart {
        control: Box<SessionControl>,
    },
    ExhaustedResume {
        control: Box<SessionControl>,
    },
    NoncontinuableLegacy {
        control: Box<SessionControl>,
    },
    CommandDenied {
        reason: DenialReason,
        control: Box<SessionControl>,
    },
    TargetOverflow {
        control: Box<SessionControl>,
    },
    MissingReservation {
        command_id: String,
    },
}

impl std::fmt::Display for CommandStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuckDb(error) => write!(f, "DuckDB error: {error}"),
            Self::Serialization(error) => write!(f, "command serialization error: {error}"),
            Self::SessionNotFound(session_id) => write!(f, "session {session_id} was not found"),
            Self::CommandIdReuseMismatch { command_id } => {
                write!(f, "command id {command_id} was reused with different input")
            }
            Self::ExpectedVersionConflict { expected, control } => write!(
                f,
                "expected control version {expected}, found {}",
                control.control_version
            ),
            Self::ActiveAttempt { attempt_id, .. } => write!(f, "attempt {attempt_id} is active"),
            Self::LaunchReserved { attempt_id, .. } => {
                write!(f, "attempt {attempt_id} is already reserved for launch")
            }
            Self::InvalidDeltaStart { .. } => write!(f, "invalid budget delta/start combination"),
            Self::ExhaustedResume { .. } => write!(f, "session budget is exhausted"),
            Self::NoncontinuableLegacy { .. } => write!(f, "legacy session cannot be continued"),
            Self::CommandDenied { reason, .. } => write!(f, "command denied: {reason:?}"),
            Self::TargetOverflow { .. } => write!(f, "target trial count overflow"),
            Self::MissingReservation { command_id } => {
                write!(
                    f,
                    "launch reservation for command {command_id} was not found"
                )
            }
        }
    }
}

impl std::error::Error for CommandStoreError {}

impl From<duckdb::Error> for CommandStoreError {
    fn from(error: duckdb::Error) -> Self {
        Self::DuckDb(error.to_string())
    }
}

impl From<serde_json::Error> for CommandStoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error.to_string())
    }
}

/// Execute one idempotent session command. IDs and timestamps are supplied by
/// the caller so this state machine stays deterministic and process-free.
pub fn apply_command(
    conn: &Connection,
    session_id: &str,
    request: &CommandRequest,
) -> Result<CommandDecision, CommandStoreError> {
    let tx = conn.unchecked_transaction()?;
    let result = apply_in_tx(&tx, session_id, request);
    match result {
        Ok(decision) => {
            tx.commit()?;
            Ok(decision)
        }
        Err(error) => {
            // Typed denials and version conflicts are durable command outcomes;
            // storage faults must leave the transaction untouched.
            if is_durable_error(&error) {
                tx.commit()?;
            } else {
                tx.rollback()?;
            }
            Err(error)
        }
    }
}

/// Record the result of the separate spawn step. A spawn failure only releases
/// the reservation; any accepted budget extension remains in the session.
pub fn record_launch_outcome(
    conn: &Connection,
    session_id: &str,
    command_id: &str,
    outcome: LaunchOutcome,
) -> Result<SessionControl, CommandStoreError> {
    let tx = conn.unchecked_transaction()?;
    let exists: Option<String> = tx
        .query_row(
            "SELECT command_id FROM tuning_launch_reservations WHERE session_id = ?1 AND command_id = ?2",
            params![session_id, command_id],
            |row| row.get(0),
        )
        .optional()?;
    if exists.is_none() {
        tx.rollback()?;
        return Err(CommandStoreError::MissingReservation {
            command_id: command_id.into(),
        });
    }
    if outcome == LaunchOutcome::SpawnFailed {
        tx.execute(
            "DELETE FROM tuning_launch_reservations WHERE session_id = ?1 AND command_id = ?2",
            params![session_id, command_id],
        )?;
    }
    let control = synchronize_control(&tx, session_id)?;
    tx.commit()?;
    Ok(control)
}

/// Reconcile reservations using only persisted run, launch, and lifecycle
/// projections. It deliberately does not inspect a process or reconstruct a
/// tuner configuration.
pub fn reconcile(conn: &Connection, session_id: &str) -> Result<SessionControl, CommandStoreError> {
    let tx = conn.unchecked_transaction()?;
    release_reservations_from_durable_facts(&tx, session_id)?;
    let control = synchronize_control(&tx, session_id)?;
    tx.commit()?;
    Ok(control)
}

fn apply_in_tx(
    tx: &Transaction<'_>,
    session_id: &str,
    request: &CommandRequest,
) -> Result<CommandDecision, CommandStoreError> {
    release_reservations_from_durable_facts(tx, session_id)?;
    let control = synchronize_control(tx, session_id)?;
    let request_fingerprint = serde_json::to_string(request)?;
    if let Some((stored_session, stored_fingerprint, outcome)) = tx
        .query_row(
            "SELECT session_id, request_fingerprint, CAST(outcome AS TEXT) FROM tuning_session_commands WHERE command_id = ?1",
            params![&request.command_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)),
        )
        .optional()?
    {
        if stored_session != session_id || stored_fingerprint != request_fingerprint {
            return Err(CommandStoreError::CommandIdReuseMismatch {
                command_id: request.command_id.clone(),
            });
        }
        return replay_outcome(&outcome);
    }

    let outcome = if request.expected_version != control.control_version {
        StoredOutcome::Rejected {
            error: StoredError::ExpectedVersionConflict {
                expected: request.expected_version,
                control: control.clone(),
            },
        }
    } else {
        decide(tx, session_id, request, control.clone())?
    };
    store_outcome(tx, session_id, request, &request_fingerprint, &outcome)?;
    outcome_result(outcome)
}

fn decide(
    tx: &Transaction<'_>,
    session_id: &str,
    request: &CommandRequest,
    control: SessionControl,
) -> Result<StoredOutcome, CommandStoreError> {
    let reject = |error| Ok(StoredOutcome::Rejected { error });
    match &request.command {
        SessionCommand::Stop => {
            if request.launch.is_some() {
                return reject(StoredError::InvalidDeltaStart { control });
            }
            let Some(attempt_id) = control.active_attempt_id.clone() else {
                return reject(StoredError::CommandDenied {
                    reason: if control.recovery_required {
                        DenialReason::RecoveryRequired
                    } else if control.launch_reservation.is_some() {
                        DenialReason::LaunchReserved
                    } else {
                        DenialReason::NoActiveAttempt
                    },
                    control,
                });
            };
            if control.stop_attempt_id.is_some() {
                return reject(StoredError::CommandDenied {
                    reason: DenialReason::StopAlreadyReserved,
                    control,
                });
            }
            tx.execute(
                "INSERT INTO tuning_stop_reservations (session_id, command_id, attempt_id, reserved_at) VALUES (?1, ?2, ?3, ?4)",
                params![session_id, &request.command_id, &attempt_id, &request.observed_at],
            )?;
        }
        SessionCommand::Resume => {
            if !has_valid_launch(request) {
                return reject(StoredError::InvalidDeltaStart { control });
            }
            if let Some(attempt_id) = &control.active_attempt_id {
                if !control.recovery_required {
                    return reject(StoredError::ActiveAttempt {
                        attempt_id: attempt_id.clone(),
                        control,
                    });
                }
            }
            if let Some(reservation) = &control.launch_reservation {
                return reject(StoredError::LaunchReserved {
                    attempt_id: reservation.attempt_id.clone(),
                    control,
                });
            }
            if legacy_session(tx, session_id)? {
                return reject(StoredError::NoncontinuableLegacy { control });
            }
            if control
                .target_trial_count
                .is_none_or(|target| target <= control.consumed_trial_count)
            {
                return reject(StoredError::ExhaustedResume { control });
            }
            insert_launch_reservation(
                tx,
                session_id,
                request,
                control.target_trial_count.expect("checked"),
            )?;
        }
        SessionCommand::AddBudget { delta, start } => {
            if *delta == 0
                || (*start != request.launch.is_some())
                || (*start && !has_valid_launch(request))
            {
                return reject(StoredError::InvalidDeltaStart { control });
            }
            if *start {
                if let Some(attempt_id) = &control.active_attempt_id {
                    if !control.recovery_required {
                        return reject(StoredError::ActiveAttempt {
                            attempt_id: attempt_id.clone(),
                            control,
                        });
                    }
                }
                if let Some(reservation) = &control.launch_reservation {
                    return reject(StoredError::LaunchReserved {
                        attempt_id: reservation.attempt_id.clone(),
                        control,
                    });
                }
                if legacy_session(tx, session_id)? {
                    return reject(StoredError::NoncontinuableLegacy { control });
                }
            }
            if !*start && legacy_session(tx, session_id)? {
                return reject(StoredError::NoncontinuableLegacy { control });
            }
            let Some(target) = control.target_trial_count else {
                return reject(StoredError::NoncontinuableLegacy { control });
            };
            let Some(next_target) = target.checked_add(*delta) else {
                return reject(StoredError::TargetOverflow { control });
            };
            tx.execute(
                "UPDATE tuning_sessions SET target_trial_count = ?1 WHERE session_id = ?2",
                params![
                    i64::try_from(next_target).map_err(|_| CommandStoreError::TargetOverflow {
                        control: Box::new(control.clone())
                    })?,
                    session_id
                ],
            )?;
            if *start {
                insert_launch_reservation(tx, session_id, request, next_target)?;
            }
        }
    }
    let next = synchronize_control(tx, session_id)?;
    Ok(StoredOutcome::Accepted {
        decision: CommandDecision {
            command_id: request.command_id.clone(),
            command: request.command.clone(),
            replay: false,
            control: next,
        },
    })
}

fn has_valid_launch(request: &CommandRequest) -> bool {
    match &request.launch {
        Some(ids) => !ids.attempt_id.is_empty() && !ids.physical_run_id.is_empty(),
        None => false,
    }
}

fn insert_launch_reservation(
    tx: &Transaction<'_>,
    session_id: &str,
    request: &CommandRequest,
    target_trial_count: u64,
) -> Result<(), CommandStoreError> {
    let reservation = request.launch.as_ref().expect("validated launch");
    tx.execute(
        "INSERT INTO tuning_launch_reservations (session_id, command_id, attempt_id, physical_run_id, target_trial_count, reserved_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![session_id, &request.command_id, &reservation.attempt_id, &reservation.physical_run_id, i64::try_from(target_trial_count).map_err(|_| CommandStoreError::DuckDb("target trial count exceeds DuckDB BIGINT".into()))?, &request.observed_at],
    )?;
    Ok(())
}

fn store_outcome(
    tx: &Transaction<'_>,
    session_id: &str,
    request: &CommandRequest,
    fingerprint: &str,
    outcome: &StoredOutcome,
) -> Result<(), CommandStoreError> {
    tx.execute(
        "INSERT INTO tuning_session_commands (command_id, session_id, request, request_fingerprint, outcome, recorded_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![&request.command_id, session_id, serde_json::to_string(request)?, fingerprint, serde_json::to_string(outcome)?, &request.observed_at],
    )?;
    Ok(())
}

fn replay_outcome(raw: &str) -> Result<CommandDecision, CommandStoreError> {
    let outcome: StoredOutcome = serde_json::from_str(raw)?;
    match outcome {
        StoredOutcome::Accepted { mut decision } => {
            decision.replay = true;
            Ok(decision)
        }
        StoredOutcome::Rejected { error } => Err(stored_error(error)),
    }
}

fn outcome_result(outcome: StoredOutcome) -> Result<CommandDecision, CommandStoreError> {
    match outcome {
        StoredOutcome::Accepted { decision } => Ok(decision),
        StoredOutcome::Rejected { error } => Err(stored_error(error)),
    }
}

fn stored_error(error: StoredError) -> CommandStoreError {
    match error {
        StoredError::ExpectedVersionConflict { expected, control } => {
            CommandStoreError::ExpectedVersionConflict {
                expected,
                control: Box::new(control),
            }
        }
        StoredError::ActiveAttempt {
            attempt_id,
            control,
        } => CommandStoreError::ActiveAttempt {
            attempt_id,
            control: Box::new(control),
        },
        StoredError::LaunchReserved {
            attempt_id,
            control,
        } => CommandStoreError::LaunchReserved {
            attempt_id,
            control: Box::new(control),
        },
        StoredError::InvalidDeltaStart { control } => CommandStoreError::InvalidDeltaStart {
            control: Box::new(control),
        },
        StoredError::ExhaustedResume { control } => CommandStoreError::ExhaustedResume {
            control: Box::new(control),
        },
        StoredError::NoncontinuableLegacy { control } => CommandStoreError::NoncontinuableLegacy {
            control: Box::new(control),
        },
        StoredError::CommandDenied { reason, control } => CommandStoreError::CommandDenied {
            reason,
            control: Box::new(control),
        },
        StoredError::TargetOverflow { control } => CommandStoreError::TargetOverflow {
            control: Box::new(control),
        },
    }
}

fn is_durable_error(error: &CommandStoreError) -> bool {
    matches!(
        error,
        CommandStoreError::ExpectedVersionConflict { .. }
            | CommandStoreError::ActiveAttempt { .. }
            | CommandStoreError::LaunchReserved { .. }
            | CommandStoreError::InvalidDeltaStart { .. }
            | CommandStoreError::ExhaustedResume { .. }
            | CommandStoreError::NoncontinuableLegacy { .. }
            | CommandStoreError::CommandDenied { .. }
            | CommandStoreError::TargetOverflow { .. }
    )
}

fn release_reservations_from_durable_facts(
    tx: &Transaction<'_>,
    session_id: &str,
) -> Result<(), CommandStoreError> {
    // A projected lifecycle attempt is definitive proof that a reserved launch
    // happened. A recorded spawn failure or terminal lifecycle state releases
    // the reservation without consulting the operating system.
    tx.execute(
        "DELETE FROM tuning_launch_reservations reservation
         WHERE reservation.session_id = ?1 AND (
             EXISTS (SELECT 1 FROM tuning_attempts attempt WHERE attempt.session_id = reservation.session_id AND attempt.attempt_id = reservation.attempt_id)
             OR EXISTS (SELECT 1 FROM projects_launches launch JOIN runs run ON run.run_id = launch.attempt_id WHERE launch.attempt_id = reservation.physical_run_id AND launch.launch_result = 'spawn_failed')
         )",
        params![session_id],
    )?;
    tx.execute(
        "DELETE FROM tuning_stop_reservations reservation
         WHERE reservation.session_id = ?1 AND NOT EXISTS (
             SELECT 1 FROM tuning_attempts attempt WHERE attempt.session_id = reservation.session_id AND attempt.attempt_id = reservation.attempt_id AND attempt.status = 'running'
         )",
        params![session_id],
    )?;
    Ok(())
}

fn synchronize_control(
    tx: &Transaction<'_>,
    session_id: &str,
) -> Result<SessionControl, CommandStoreError> {
    let snapshot = load_snapshot(tx, session_id)?;
    let mut control = control_from_snapshot(&snapshot)?;
    let signature = serde_json::to_string(&control.allowed_commands)?;
    match snapshot.control_signature {
        None => {
            tx.execute(
                "UPDATE tuning_sessions SET control_signature = ?1 WHERE session_id = ?2",
                params![signature, session_id],
            )?;
        }
        Some(previous) if previous != signature => {
            tx.execute(
                "UPDATE tuning_sessions SET control_version = control_version + 1, control_signature = ?1 WHERE session_id = ?2",
                params![signature, session_id],
            )?;
            control.control_version = control
                .control_version
                .checked_add(1)
                .ok_or_else(|| CommandStoreError::DuckDb("control version overflow".into()))?;
        }
        Some(_) => {}
    }
    Ok(control)
}

struct Snapshot {
    control_version: u64,
    control_signature: Option<String>,
    target_trial_count: Option<u64>,
    optimizer_id: Option<String>,
    lifecycle_path: Option<String>,
    consumed_trial_count: u64,
    active_attempt_id: Option<String>,
    physical_dead_lifecycle_active: bool,
    launch_reservation: Option<LaunchReservation>,
    stop_attempt_id: Option<String>,
}

fn load_snapshot(tx: &Transaction<'_>, session_id: &str) -> Result<Snapshot, CommandStoreError> {
    let (control_version, control_signature, target_trial_count, optimizer_id, lifecycle_path): (
        u64,
        Option<String>,
        Option<i64>,
        Option<String>,
        Option<String>,
    ) = tx
        .query_row(
            "SELECT control_version, control_signature, target_trial_count, optimizer_id, lifecycle_path FROM tuning_sessions WHERE session_id = ?1",
            params![session_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )
        .optional()?
        .ok_or_else(|| CommandStoreError::SessionNotFound(session_id.into()))?;
    let consumed_trial_count: i64 = tx.query_row(
        "SELECT COUNT(DISTINCT trial_number) FROM tuning_trials WHERE session_id = ?1",
        params![session_id],
        |row| row.get(0),
    )?;
    let active_attempt_id: Option<String> = tx
        .query_row(
            "SELECT attempt_id FROM tuning_attempts WHERE session_id = ?1 AND status = 'running' ORDER BY started_at DESC, attempt_id DESC LIMIT 1",
            params![session_id],
            |row| row.get(0),
        )
        .optional()?;
    let physical_dead_lifecycle_active: bool = tx.query_row(
        "SELECT EXISTS (
            SELECT 1 FROM tuning_attempts attempt JOIN runs run ON run.run_id = attempt.bench_run_id
            WHERE attempt.session_id = ?1 AND attempt.status = 'running' AND run.status <> 'running'
         )",
        params![session_id],
        |row| row.get(0),
    )?;
    let launch_reservation = tx
        .query_row(
            "SELECT attempt_id, physical_run_id FROM tuning_launch_reservations WHERE session_id = ?1",
            params![session_id],
            |row| {
                Ok(LaunchReservation {
                    attempt_id: row.get(0)?,
                    physical_run_id: row.get(1)?,
                })
            },
        )
        .optional()?;
    let stop_attempt_id = tx
        .query_row(
            "SELECT attempt_id FROM tuning_stop_reservations WHERE session_id = ?1",
            params![session_id],
            |row| row.get(0),
        )
        .optional()?;
    Ok(Snapshot {
        control_version,
        control_signature,
        target_trial_count: target_trial_count.map(|value| value as u64),
        optimizer_id,
        lifecycle_path,
        consumed_trial_count: consumed_trial_count as u64,
        active_attempt_id,
        physical_dead_lifecycle_active,
        launch_reservation,
        stop_attempt_id,
    })
}

fn control_from_snapshot(snapshot: &Snapshot) -> Result<SessionControl, CommandStoreError> {
    let recovery_required = snapshot.physical_dead_lifecycle_active;
    let continuation_denial = if snapshot.active_attempt_id.is_some() && !recovery_required {
        Some(DenialReason::ActiveAttempt)
    } else if snapshot.launch_reservation.is_some() {
        Some(DenialReason::LaunchReserved)
    } else if snapshot.optimizer_id.is_none() || snapshot.lifecycle_path.is_none() {
        Some(DenialReason::NoncontinuableLegacy)
    } else if snapshot
        .target_trial_count
        .is_none_or(|target| target <= snapshot.consumed_trial_count)
    {
        Some(DenialReason::Exhausted)
    } else {
        None
    };
    let stop_denial = if recovery_required {
        Some(DenialReason::RecoveryRequired)
    } else if snapshot.active_attempt_id.is_none() {
        Some(if snapshot.launch_reservation.is_some() {
            DenialReason::LaunchReserved
        } else {
            DenialReason::NoActiveAttempt
        })
    } else if snapshot.stop_attempt_id.is_some() {
        Some(DenialReason::StopAlreadyReserved)
    } else {
        None
    };
    let add_budget_denial = if recovery_required {
        Some(DenialReason::RecoveryRequired)
    } else if snapshot.optimizer_id.is_none() || snapshot.lifecycle_path.is_none() {
        Some(DenialReason::NoncontinuableLegacy)
    } else {
        None
    };
    Ok(SessionControl {
        control_version: snapshot.control_version,
        target_trial_count: snapshot.target_trial_count,
        consumed_trial_count: snapshot.consumed_trial_count,
        active_attempt_id: snapshot.active_attempt_id.clone(),
        launch_reservation: snapshot.launch_reservation.clone(),
        stop_attempt_id: snapshot.stop_attempt_id.clone(),
        recovery_required,
        allowed_commands: vec![
            allowed(CommandKind::Stop, stop_denial),
            allowed(CommandKind::Resume, continuation_denial),
            allowed(CommandKind::AddBudget, add_budget_denial),
        ],
    })
}

fn allowed(command: CommandKind, denial_reason: Option<DenialReason>) -> AllowedCommand {
    AllowedCommand {
        command,
        allowed: denial_reason.is_none(),
        denial_reason,
    }
}

fn legacy_session(tx: &Transaction<'_>, session_id: &str) -> Result<bool, CommandStoreError> {
    let value: bool = tx.query_row(
        "SELECT optimizer_id IS NULL OR lifecycle_path IS NULL FROM tuning_sessions WHERE session_id = ?1",
        params![session_id],
        |row| row.get(0),
    )?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ensure_schema;

    fn db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();
        conn.execute(
            "INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence, optimizer_id, lifecycle_path) VALUES ('s', 'idle', '{}', 3, '2026-01-01T00:00:00Z', 0, 'optimizer', '/tmp/lifecycle.jsonl')",
            [],
        )
        .unwrap();
        conn
    }

    fn request(id: &str, version: u64, command: SessionCommand) -> CommandRequest {
        let launch = match command {
            SessionCommand::Resume | SessionCommand::AddBudget { start: true, .. } => {
                Some(LaunchReservation {
                    attempt_id: format!("attempt-{id}"),
                    physical_run_id: format!("run-{id}"),
                })
            }
            _ => None,
        };
        CommandRequest {
            command_id: id.into(),
            expected_version: version,
            command,
            launch,
            observed_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    fn denied(control: &SessionControl, kind: CommandKind, reason: DenialReason) {
        assert!(control
            .allowed_commands
            .iter()
            .any(|value| value.command == kind
                && !value.allowed
                && value.denial_reason == Some(reason)));
    }

    #[test]
    fn allowed_and_denied_states_are_projected_with_reasons() {
        let conn = db();
        let initial = reconcile(&conn, "s").unwrap();
        assert!(initial
            .allowed_commands
            .iter()
            .any(|value| value.command == CommandKind::Resume && value.allowed));
        denied(&initial, CommandKind::Stop, DenialReason::NoActiveAttempt);

        let decision =
            apply_command(&conn, "s", &request("resume", 0, SessionCommand::Resume)).unwrap();
        denied(
            &decision.control,
            CommandKind::Resume,
            DenialReason::LaunchReserved,
        );
        denied(
            &decision.control,
            CommandKind::Stop,
            DenialReason::LaunchReserved,
        );
    }

    #[test]
    fn budget_arithmetic_overflow_and_idempotency_are_exact() {
        let conn = db();
        let add = request(
            "add",
            0,
            SessionCommand::AddBudget {
                delta: 2,
                start: false,
            },
        );
        let first = apply_command(&conn, "s", &add).unwrap();
        assert_eq!(first.control.target_trial_count, Some(5));
        let replay = apply_command(&conn, "s", &add).unwrap();
        assert!(replay.replay);
        assert_eq!(replay.control.target_trial_count, Some(5));
        let mismatch = CommandRequest {
            command: SessionCommand::AddBudget {
                delta: 3,
                start: false,
            },
            ..add.clone()
        };
        assert!(matches!(
            apply_command(&conn, "s", &mismatch),
            Err(CommandStoreError::CommandIdReuseMismatch { .. })
        ));
        conn.execute("UPDATE tuning_sessions SET target_trial_count = 9223372036854775807 WHERE session_id = 's'", []).unwrap();
        let error = apply_command(
            &conn,
            "s",
            &request(
                "overflow",
                0,
                SessionCommand::AddBudget {
                    delta: 1,
                    start: false,
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(error, CommandStoreError::TargetOverflow { .. }));
    }

    #[test]
    fn expected_version_admits_only_one_concurrent_form() {
        let conn = db();
        let first =
            apply_command(&conn, "s", &request("resume", 0, SessionCommand::Resume)).unwrap();
        assert_eq!(first.control.control_version, 1);
        let error =
            apply_command(&conn, "s", &request("other", 0, SessionCommand::Resume)).unwrap_err();
        assert!(matches!(
            error,
            CommandStoreError::ExpectedVersionConflict { .. }
        ));
    }

    #[test]
    fn launch_and_stop_reservations_gate_commands_and_spawn_failure_keeps_budget() {
        let conn = db();
        let launch = request(
            "extend",
            0,
            SessionCommand::AddBudget {
                delta: 2,
                start: true,
            },
        );
        let decision = apply_command(&conn, "s", &launch).unwrap();
        assert_eq!(decision.control.target_trial_count, Some(5));
        assert!(decision.control.launch_reservation.is_some());
        assert!(matches!(
            apply_command(
                &conn,
                "s",
                &request(
                    "second",
                    decision.control.control_version,
                    SessionCommand::Resume
                )
            ),
            Err(CommandStoreError::LaunchReserved { .. })
        ));
        let after_failure =
            record_launch_outcome(&conn, "s", "extend", LaunchOutcome::SpawnFailed).unwrap();
        assert_eq!(after_failure.target_trial_count, Some(5));
        assert!(after_failure.launch_reservation.is_none());

        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at) VALUES ('live', 's', 'running', '2026-01-01T00:00:01Z')", []).unwrap();
        let active = reconcile(&conn, "s").unwrap();
        let stop = apply_command(
            &conn,
            "s",
            &request("stop", active.control_version, SessionCommand::Stop),
        )
        .unwrap();
        assert_eq!(stop.control.stop_attempt_id.as_deref(), Some("live"));
        denied(
            &stop.control,
            CommandKind::Stop,
            DenialReason::StopAlreadyReserved,
        );
    }

    #[test]
    fn restart_reconciliation_uses_durable_lifecycle_and_detects_recovery() {
        let conn = db();
        let launch = request("resume", 0, SessionCommand::Resume);
        apply_command(&conn, "s", &launch).unwrap();
        let spawned = record_launch_outcome(&conn, "s", "resume", LaunchOutcome::Spawned).unwrap();
        assert!(spawned.launch_reservation.is_some());
        conn.execute("INSERT INTO runs (run_id, kind, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('run-resume', 'tuning', 'sha', false, 'host', '2026-01-01T00:00:01Z', 'running', '/tmp/log')", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at) VALUES ('attempt-resume', 's', 'run-resume', 'running', '2026-01-01T00:00:01Z')", []).unwrap();
        let active = reconcile(&conn, "s").unwrap();
        assert!(active.launch_reservation.is_none());
        assert_eq!(active.active_attempt_id.as_deref(), Some("attempt-resume"));

        conn.execute(
            "UPDATE runs SET status = 'crashed' WHERE run_id = 'run-resume'",
            [],
        )
        .unwrap();
        let recovery = reconcile(&conn, "s").unwrap();
        assert!(recovery.recovery_required);
        assert!(recovery
            .allowed_commands
            .iter()
            .any(|command| command.command == CommandKind::Resume && command.allowed));
        let resumed = apply_command(
            &conn,
            "s",
            &request(
                "recovery-resume",
                recovery.control_version,
                SessionCommand::Resume,
            ),
        )
        .unwrap();
        assert!(resumed.control.launch_reservation.is_some());
    }

    #[test]
    fn consumed_slots_are_distinct_across_all_trial_states_and_legacy_cannot_resume() {
        let conn = db();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at) VALUES ('a', 's', 'completed', '2026-01-01T00:00:00Z')", []).unwrap();
        for (id, number, status) in [
            ("one", 1, "queued"),
            ("one-copy", 1, "failed"),
            ("two", 2, "cancelled"),
        ] {
            conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, created_at) VALUES ('s', ?1, 'a', ?2, ?3, '2026-01-01T00:00:00Z')", params![id, number, status]).unwrap();
        }
        let control = reconcile(&conn, "s").unwrap();
        assert_eq!(control.consumed_trial_count, 2);
        conn.execute(
            "UPDATE tuning_sessions SET optimizer_id = NULL WHERE session_id = 's'",
            [],
        )
        .unwrap();
        let legacy = reconcile(&conn, "s").unwrap();
        let error = apply_command(
            &conn,
            "s",
            &request("legacy", legacy.control_version, SessionCommand::Resume),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CommandStoreError::NoncontinuableLegacy { .. }
        ));
    }

    #[test]
    fn control_version_ignores_lifecycle_sequence_without_a_validity_change() {
        let conn = db();
        let before = reconcile(&conn, "s").unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at) VALUES ('finished', 's', 'completed', '2026-01-01T00:00:00Z')", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, created_at) VALUES ('s', 'trial', 'finished', 1, 'queued', '2026-01-01T00:00:00Z')", []).unwrap();
        let after_trial = reconcile(&conn, "s").unwrap();
        conn.execute("UPDATE tuning_trials SET status = 'complete' WHERE session_id = 's' AND trial_id = 'trial'", []).unwrap();
        let after_report = reconcile(&conn, "s").unwrap();
        assert_eq!(after_trial.control_version, before.control_version);
        assert_eq!(after_report.control_version, before.control_version);
    }

    #[test]
    fn invalid_combinations_and_transaction_rollback_leave_no_command() {
        let conn = db();
        let invalid = CommandRequest {
            launch: Some(LaunchReservation {
                attempt_id: "a".into(),
                physical_run_id: "r".into(),
            }),
            ..request("bad", 0, SessionCommand::Stop)
        };
        assert!(matches!(
            apply_command(&conn, "s", &invalid),
            Err(CommandStoreError::InvalidDeltaStart { .. })
        ));
        let rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM tuning_session_commands WHERE command_id = 'bad'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(rows, 1);
        // A duplicate physical id makes the reservation insert fail, and the
        // target update performed just before it is rolled back with the tx.
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence, optimizer_id, lifecycle_path) VALUES ('other', 'idle', '{}', 3, '2026-01-01T00:00:00Z', 0, 'optimizer', '/tmp/lifecycle')", []).unwrap();
        conn.execute("INSERT INTO tuning_launch_reservations (session_id, command_id, attempt_id, physical_run_id, target_trial_count, reserved_at) VALUES ('other', 'other-command', 'other-attempt', 'run-collision', 3, '2026-01-01T00:00:00Z')", []).unwrap();
        let collision = CommandRequest {
            launch: Some(LaunchReservation {
                attempt_id: "collision".into(),
                physical_run_id: "run-collision".into(),
            }),
            ..request(
                "collision",
                0,
                SessionCommand::AddBudget {
                    delta: 1,
                    start: true,
                },
            )
        };
        assert!(matches!(
            apply_command(&conn, "s", &collision),
            Err(CommandStoreError::DuckDb(_))
        ));
        let target: i64 = conn
            .query_row(
                "SELECT target_trial_count FROM tuning_sessions WHERE session_id = 's'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(target, 3);
    }

    #[test]
    fn responses_have_deterministic_serialization() {
        let conn = db();
        let decision =
            apply_command(&conn, "s", &request("resume", 0, SessionCommand::Resume)).unwrap();
        let first = serde_json::to_string(&decision).unwrap();
        let replay =
            apply_command(&conn, "s", &request("resume", 0, SessionCommand::Resume)).unwrap();
        let mut replay_without_marker = replay;
        replay_without_marker.replay = false;
        assert_eq!(
            first,
            serde_json::to_string(&replay_without_marker).unwrap()
        );
    }
}
