//! Idempotent raw-evidence storage for tuning lifecycle events.

mod projection;
mod transitions;

use crate::tuning_lifecycle::TuningLifecycleEvent;
use duckdb::{params, Transaction};

#[derive(Debug)]
pub enum TuningStoreError {
    DuckDb(duckdb::Error),
    Serialization(serde_json::Error),
}

impl From<duckdb::Error> for TuningStoreError {
    fn from(error: duckdb::Error) -> Self {
        Self::DuckDb(error)
    }
}

impl From<serde_json::Error> for TuningStoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

impl std::fmt::Display for TuningStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuckDb(error) => write!(f, "DuckDB error: {error}"),
            Self::Serialization(error) => write!(f, "tuning event serialization error: {error}"),
        }
    }
}

impl std::error::Error for TuningStoreError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyDisposition {
    Applied,
    Rejected,
    Replay,
    Conflict,
}

enum RawEventDisposition {
    Inserted,
    Existing(ApplyDisposition),
}

pub fn apply_event(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    bench_run_id: &str,
    source_path: &str,
    source_offset: u64,
) -> Result<ApplyDisposition, TuningStoreError> {
    match store_raw_event(tx, event, source_path, source_offset)? {
        RawEventDisposition::Inserted => {}
        RawEventDisposition::Existing(disposition) => return Ok(disposition),
    }

    if let Some(reason) = transitions::validate(tx, event)? {
        mark_rejected(tx, event, &reason)?;
        return Ok(ApplyDisposition::Rejected);
    }

    projection::apply(tx, event, bench_run_id)?;
    mark_accepted(tx, event)?;
    Ok(ApplyDisposition::Applied)
}

fn store_raw_event(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    source_path: &str,
    source_offset: u64,
) -> Result<RawEventDisposition, TuningStoreError> {
    let raw = serde_json::to_string(event)?;
    let inserted = tx.execute(
        "INSERT INTO tuning_lifecycle_events \
         (event_id, session_id, attempt_id, session_sequence, timestamp, event_type, payload, raw, source_path, source_offset, accepted, rejection_reason) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL, NULL) \
         ON CONFLICT (event_id) DO NOTHING",
        params![
            event.event_id.as_str(),
            event.session_id.as_str(),
            event.attempt_id.as_str(),
            event.session_sequence as i64,
            &event.timestamp,
            event.event_type.as_str(),
            serde_json::to_string(&event.payload)?,
            raw,
            source_path,
            source_offset as i64,
        ],
    )?;
    if inserted != 0 {
        return Ok(RawEventDisposition::Inserted);
    }

    let existing_raw: String = tx.query_row(
        "SELECT CAST(raw AS TEXT) FROM tuning_lifecycle_events WHERE event_id = ?1",
        params![event.event_id.as_str()],
        |row| row.get(0),
    )?;
    let disposition = if existing_raw == raw {
        ApplyDisposition::Replay
    } else {
        ApplyDisposition::Conflict
    };
    Ok(RawEventDisposition::Existing(disposition))
}

fn mark_rejected(
    tx: &Transaction<'_>,
    event: &TuningLifecycleEvent,
    reason: &str,
) -> Result<(), duckdb::Error> {
    tx.execute(
        "UPDATE tuning_lifecycle_events SET accepted = false, rejection_reason = ?1 WHERE event_id = ?2",
        params![reason, event.event_id.as_str()],
    )?;
    Ok(())
}

fn mark_accepted(tx: &Transaction<'_>, event: &TuningLifecycleEvent) -> Result<(), duckdb::Error> {
    tx.execute(
        "UPDATE tuning_lifecycle_events SET accepted = true WHERE event_id = ?1",
        params![event.event_id.as_str()],
    )?;
    Ok(())
}
