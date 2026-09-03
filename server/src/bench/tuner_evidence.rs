//! Live view of a tuner run's own `evidence.jsonl` journal.
//!
//! `report.json` is written once, at run end, so every projection-backed
//! science surface is blind for the hours-to-days a run is live. The tuner's
//! `evidence.jsonl` is append-only and `fsync`'d after every line, one
//! `{schema_version, sequence, type, payload}` envelope per line with a
//! contiguous `sequence` from 1 -- these two routes stream it as it grows:
//!
//! - `GET .../evidence?since_seq=N&limit=M` -- a forward tail: the
//!   decoded-but-verbatim envelopes with `sequence > N`, capped at `M`.
//! - `GET .../evidence/stream` (SSE) -- one event per appended line, modelled
//!   on `traces::live_run_moves`: a spawned task polls the file every 500 ms
//!   and pushes appended lines through an mpsc channel. It sends a final
//!   `event: end` and closes once the run is no longer `live` and no new line
//!   has landed for 3 s, and ends immediately on client disconnect.
//!
//! The payload is passed through untouched; only `sequence` and `type` are
//! read here. No evidence decode schema lives on the server side of this
//! module -- that is the projector's job.

use std::convert::Infallible;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{Path as AxumPath, Query as AxumQuery, State as AxumState},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        Json,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};

use mcts_bench::tuner_launch;

use super::{tuner_runs, BenchError, BenchState};

/// One evidence line, reshaped to the three fields the UI needs. `payload`
/// is the tuner's own object, untouched.
fn envelope(line: &str) -> Option<(u64, Value)> {
    let value: Value = serde_json::from_str(line).ok()?;
    let sequence = value.get("sequence")?.as_u64()?;
    let event_type = value.get("type")?.clone();
    let payload = value.get("payload").cloned().unwrap_or(Value::Null);
    Some((
        sequence,
        json!({ "sequence": sequence, "type": event_type, "payload": payload }),
    ))
}

/// The prefix of `text` up to and including its last newline -- i.e. every
/// complete line, with a torn trailing line (writer mid-append) excluded.
fn complete_prefix(text: &str) -> &str {
    match text.rfind('\n') {
        Some(index) => &text[..=index],
        None => "",
    }
}

// ---------------------------------------------------------------------------
// GET .../evidence  (forward tail)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct EvidenceTailParams {
    #[serde(default)]
    since_seq: u64,
    limit: Option<usize>,
}

#[derive(Serialize)]
pub(crate) struct EvidenceTailResponse {
    events: Vec<Value>,
    /// The sequence to pass as `since_seq` on the next poll: the last event
    /// returned, or -- when the caller is already caught up -- the highest
    /// sequence in the log.
    next_seq: u64,
    /// `live | exited | failed | unknown` for the run, so the UI knows
    /// whether to keep polling.
    run_status: &'static str,
}

/// `GET /api/bench/tuner/runs/{run_id}/evidence?since_seq=N&limit=M`
pub(crate) async fn evidence_tail(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    AxumQuery(params): AxumQuery<EvidenceTailParams>,
) -> Result<Json<EvidenceTailResponse>, BenchError> {
    let record = tuner_runs::find_record(&state, &run_id)?;
    let run_status = tuner_runs::liveness(&record);
    let path = record.run_dir.join("evidence.jsonl");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("failed to read {}: {error}", path.display()),
            })
        }
    };

    let limit = params.limit.unwrap_or(500).clamp(1, 5_000);
    let mut events: Vec<Value> = Vec::new();
    let mut next_seq = params.since_seq;
    for line in complete_prefix(&text).lines() {
        let Some((sequence, env)) = envelope(line) else {
            continue;
        };
        if sequence > params.since_seq && events.len() < limit {
            events.push(env);
            next_seq = sequence;
        } else if events.is_empty() {
            next_seq = next_seq.max(sequence);
        }
    }

    Ok(Json(EvidenceTailResponse {
        events,
        next_seq,
        run_status,
    }))
}

// ---------------------------------------------------------------------------
// GET .../evidence/stream  (SSE)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct EvidenceStreamParams {
    #[serde(default)]
    since_seq: u64,
}

/// Poll cadence and quiet-close window for `pump_evidence`. A field so the
/// tests can drive the loop in milliseconds without a real process or clock.
pub(crate) struct StreamTiming {
    pub poll: Duration,
    pub quiet_close: Duration,
}

impl Default for StreamTiming {
    fn default() -> Self {
        Self {
            poll: Duration::from_millis(500),
            quiet_close: Duration::from_secs(3),
        }
    }
}

async fn send_envelope(tx: &tokio::sync::mpsc::Sender<Event>, env: &Value) -> Result<(), ()> {
    match Event::default().json_data(env) {
        Ok(event) => tx.send(event).await.map_err(|_| ()),
        // An envelope that will not serialise is dropped rather than killing
        // the stream -- it cannot happen for a value we just parsed.
        Err(_) => Ok(()),
    }
}

/// Tail `path` from the end of the last complete line, pushing one `Event`
/// per newly appended evidence line into `tx`. Returns when the client hangs
/// up (a `tx.send` error) or when `is_live()` is false and no line has landed
/// for `timing.quiet_close` (after emitting `event: end`).
pub(crate) async fn pump_evidence(
    path: PathBuf,
    since_seq: u64,
    is_live: Arc<dyn Fn() -> bool + Send + Sync>,
    tx: tokio::sync::mpsc::Sender<Event>,
    timing: StreamTiming,
) {
    let mut offset: u64 = 0;
    let mut last_seq = since_seq;
    let mut last_line_at = Instant::now();

    // Catch-up pass: everything already past `since_seq`, then park the byte
    // cursor at the end of the last complete line.
    if let Ok(text) = std::fs::read_to_string(&path) {
        let prefix = complete_prefix(&text);
        offset = prefix.len() as u64;
        for line in prefix.lines() {
            if let Some((sequence, env)) = envelope(line) {
                if sequence > last_seq {
                    last_seq = sequence;
                    if send_envelope(&tx, &env).await.is_err() {
                        return;
                    }
                }
            }
        }
    }

    let mut interval = tokio::time::interval(timing.poll);
    loop {
        interval.tick().await;

        let mut appended = String::new();
        if let Ok(mut file) = std::fs::File::open(&path) {
            if file.seek(SeekFrom::Start(offset)).is_ok() {
                let _ = file.read_to_string(&mut appended);
            }
        }
        let prefix = complete_prefix(&appended);
        let mut sent_any = false;
        for line in prefix.lines() {
            if let Some((sequence, env)) = envelope(line) {
                if sequence > last_seq {
                    last_seq = sequence;
                    sent_any = true;
                    if send_envelope(&tx, &env).await.is_err() {
                        return;
                    }
                }
            }
        }
        offset += prefix.len() as u64;

        if sent_any {
            last_line_at = Instant::now();
        } else if !is_live() && last_line_at.elapsed() >= timing.quiet_close {
            let _ = tx.send(Event::default().event("end").data("")).await;
            return;
        }
    }
}

/// `GET /api/bench/tuner/runs/{run_id}/evidence/stream`
pub(crate) async fn evidence_stream(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    AxumQuery(params): AxumQuery<EvidenceStreamParams>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, BenchError> {
    let record = tuner_runs::find_record(&state, &run_id)?;
    let path = record.run_dir.join("evidence.jsonl");

    let runs_root = state.bench_runs_dir.clone();
    let is_live: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
        run_is_live(&runs_root, &run_id)
    });

    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(pump_evidence(
        path,
        params.since_seq,
        is_live,
        tx,
        StreamTiming::default(),
    ));

    Ok(Sse::new(ReceiverStream::new(rx).map(Ok)).keep_alive(KeepAlive::default()))
}

fn run_is_live(runs_root: &Path, run_id: &str) -> bool {
    tuner_launch::records(runs_root)
        .ok()
        .and_then(|records| records.into_iter().find(|record| record.run_id == run_id))
        .is_some_and(|record| tuner_runs::liveness(&record) == "live")
}
