use super::*;
use serde_json::{Map, Value};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn read_journal(path: impl AsRef<Path>) -> Result<JournalRead, LifecycleError> {
    let path = path.as_ref();
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(JournalRead::Missing)
        }
        Err(source) => {
            return Err(LifecycleError::Io {
                path: path.to_path_buf(),
                operation: "open",
                source,
            })
        }
    };
    let mut records = Vec::new();
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    let mut line = 0;
    loop {
        bytes.clear();
        let count = reader
            .read_until(b'\n', &mut bytes)
            .map_err(|source| LifecycleError::Io {
                path: path.to_path_buf(),
                operation: "read",
                source,
            })?;
        if count == 0 {
            break;
        }
        line += 1;
        if bytes.last() != Some(&b'\n') {
            return Err(invalid(
                path,
                Some(line),
                None,
                InvalidReason::UnterminatedRecord,
            ));
        }
        bytes.pop();
        if bytes.iter().all(u8::is_ascii_whitespace) {
            return Err(invalid(path, Some(line), None, InvalidReason::BlankRecord));
        }
        let record = parse_record(path, line, &bytes)?;
        validate_record(
            &record,
            path,
            line,
            records.len() as u64,
            records.is_empty(),
        )?;
        records.push(record);
    }
    if records.is_empty() {
        return Err(invalid(path, None, None, InvalidReason::EmptyJournal));
    }
    build_snapshot(path, records)
}

fn parse_record(path: &Path, line: usize, bytes: &[u8]) -> Result<LifecycleRecord, LifecycleError> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|_| invalid(path, Some(line), None, InvalidReason::JsonSyntax))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid(path, Some(line), None, InvalidReason::ClosedSchemaViolation))?;
    let sequence = object.get("sequence").and_then(Value::as_u64);
    exact_keys(
        object,
        &[
            "schema_version",
            "sequence",
            "attempt_id",
            "launch_nonce",
            "timestamp",
            "payload",
        ],
        InvalidReason::ClosedSchemaViolation,
    )
    .map_err(|reason| invalid(path, Some(line), sequence, reason))?;
    if let Some(version) = object.get("schema_version").and_then(Value::as_u64) {
        if version != 1 {
            return Err(invalid(
                path,
                Some(line),
                sequence,
                InvalidReason::UnsupportedSchemaVersion,
            ));
        }
    }
    validate_payload_shape(object.get("payload").unwrap())
        .map_err(|reason| invalid(path, Some(line), sequence, reason))?;
    serde_json::from_value(value).map_err(|_| {
        invalid(
            path,
            Some(line),
            sequence,
            InvalidReason::ClosedSchemaViolation,
        )
    })
}

fn exact_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    reason: InvalidReason,
) -> Result<(), InvalidReason> {
    (object.len() == expected.len() && expected.iter().all(|key| object.contains_key(*key)))
        .then_some(())
        .ok_or(reason)
}

fn validate_payload_shape(value: &Value) -> Result<(), InvalidReason> {
    let payload = value
        .as_object()
        .ok_or(InvalidReason::ClosedSchemaViolation)?;
    exact_keys(
        payload,
        &["type", "value"],
        InvalidReason::ClosedSchemaViolation,
    )?;
    let kind = payload
        .get("type")
        .and_then(Value::as_str)
        .ok_or(InvalidReason::ClosedSchemaViolation)?;
    let body = payload
        .get("value")
        .and_then(Value::as_object)
        .ok_or(InvalidReason::ClosedSchemaViolation)?;
    match kind {
        "wrapper_started" => exact_keys(
            body,
            &[
                "logical_run_id",
                "attempt_id",
                "parent_attempt_id",
                "argv",
                "wrapper_pid",
                "process_group_id",
                "hostname",
                "boot_id",
                "process_start_id",
            ],
            InvalidReason::ClosedSchemaViolation,
        ),
        "child_started" => exact_keys(body, &["child_pid"], InvalidReason::ClosedSchemaViolation),
        "child_spawn_failed" => exact_keys(
            body,
            &["stage", "error"],
            InvalidReason::ClosedSchemaViolation,
        ),
        "child_exited" => {
            exact_keys(body, &["outcome"], InvalidReason::ClosedSchemaViolation)?;
            validate_exit_shape(body.get("outcome").unwrap())
        }
        "outputs_closed" => {
            exact_keys(body, &["outputs"], InvalidReason::ClosedSchemaViolation)?;
            let outputs = body
                .get("outputs")
                .and_then(Value::as_array)
                .ok_or(InvalidReason::ClosedSchemaViolation)?;
            outputs.iter().try_for_each(|output| {
                exact_keys(
                    output
                        .as_object()
                        .ok_or(InvalidReason::ClosedSchemaViolation)?,
                    &["path", "byte_length"],
                    InvalidReason::ClosedSchemaViolation,
                )
            })
        }
        _ => Err(InvalidReason::UnsupportedRecordType),
    }
}

fn validate_exit_shape(value: &Value) -> Result<(), InvalidReason> {
    let exit = value.as_object().ok_or(InvalidReason::InvalidExitVariant)?;
    exact_keys(exit, &["kind", "value"], InvalidReason::InvalidExitVariant)?;
    let kind = exit
        .get("kind")
        .and_then(Value::as_str)
        .ok_or(InvalidReason::InvalidExitVariant)?;
    let body = exit
        .get("value")
        .and_then(Value::as_object)
        .ok_or(InvalidReason::InvalidExitVariant)?;
    match kind {
        "code" => exact_keys(body, &["code"], InvalidReason::InvalidExitVariant),
        "signal" => exact_keys(body, &["signal"], InvalidReason::InvalidExitVariant),
        "wait_failed" => exact_keys(body, &["error"], InvalidReason::InvalidExitVariant),
        _ => Err(InvalidReason::InvalidExitVariant),
    }
}

fn validate_record(
    record: &LifecycleRecord,
    path: &Path,
    line: usize,
    expected: u64,
    first: bool,
) -> Result<(), LifecycleError> {
    if record.sequence != expected {
        return Err(invalid(
            path,
            Some(line),
            Some(record.sequence),
            InvalidReason::SequenceMismatch,
        ));
    }
    named(&record.attempt_id, "attempt_id")
        .and_then(|_| named(&record.launch_nonce, "launch_nonce"))
        .and_then(|_| named(&record.timestamp, "timestamp"))
        .and_then(|_| validate_payload(&record.payload))
        .map_err(|reason| invalid(path, Some(line), Some(record.sequence), reason))?;
    if first && !matches!(record.payload, LifecyclePayload::WrapperStarted(_)) {
        return Err(invalid(
            path,
            Some(line),
            Some(record.sequence),
            InvalidReason::FirstRecordNotWrapper,
        ));
    }
    Ok(())
}

fn build_snapshot(
    path: &Path,
    records: Vec<LifecycleRecord>,
) -> Result<JournalRead, LifecycleError> {
    let manifest = match &records[0].payload {
        LifecyclePayload::WrapperStarted(manifest) => manifest.clone(),
        _ => unreachable!(),
    };
    if records[0].attempt_id != manifest.attempt_id {
        return Err(invalid(
            path,
            Some(1),
            Some(0),
            InvalidReason::AttemptIdDrift,
        ));
    }
    let attempt_id = records[0].attempt_id.clone();
    let nonce = records[0].launch_nonce.clone();
    let mut child = None;
    let mut terminal = None;
    let mut outputs = None;
    for (index, record) in records.iter().enumerate() {
        let line = index + 1;
        if record.attempt_id != attempt_id {
            return Err(invalid(
                path,
                Some(line),
                Some(record.sequence),
                InvalidReason::AttemptIdDrift,
            ));
        }
        if record.launch_nonce != nonce {
            return Err(invalid(
                path,
                Some(line),
                Some(record.sequence),
                InvalidReason::LaunchNonceDrift,
            ));
        }
        if outputs.is_some() {
            return Err(invalid(
                path,
                Some(line),
                Some(record.sequence),
                InvalidReason::RecordsAfterClose,
            ));
        }
        match &record.payload {
            LifecyclePayload::WrapperStarted(_) if index > 0 => {
                return Err(invalid(
                    path,
                    Some(line),
                    Some(record.sequence),
                    InvalidReason::DuplicateWrapper,
                ))
            }
            LifecyclePayload::WrapperStarted(_) => {}
            LifecyclePayload::ChildStarted { child_pid } => {
                if child.replace(*child_pid).is_some() || terminal.is_some() {
                    return Err(invalid(
                        path,
                        Some(line),
                        Some(record.sequence),
                        InvalidReason::InvalidTypedRecordOrdering,
                    ));
                }
            }
            LifecyclePayload::ChildSpawnFailed { stage, error } => {
                if child.is_some() || terminal.is_some() {
                    return Err(invalid(
                        path,
                        Some(line),
                        Some(record.sequence),
                        InvalidReason::InvalidTypedRecordOrdering,
                    ));
                }
                terminal = Some(TerminalEvidence::SpawnFailed {
                    stage: stage.clone(),
                    error: error.clone(),
                });
            }
            LifecyclePayload::ChildExited { outcome } => {
                if child.is_none() || terminal.is_some() {
                    return Err(invalid(
                        path,
                        Some(line),
                        Some(record.sequence),
                        InvalidReason::InvalidTypedRecordOrdering,
                    ));
                }
                terminal = Some(TerminalEvidence::Exited(outcome.clone()));
            }
            LifecyclePayload::OutputsClosed { outputs: values } => {
                if terminal.is_none() {
                    return Err(invalid(
                        path,
                        Some(line),
                        Some(record.sequence),
                        InvalidReason::InvalidTypedRecordOrdering,
                    ));
                }
                outputs = Some(values.clone());
            }
        }
    }
    let snapshot = JournalSnapshot {
        manifest,
        launch_nonce: nonce,
        child,
        terminal,
        outputs,
        last_sequence: records.last().unwrap().sequence,
    };
    Ok(if snapshot.outputs.is_some() {
        JournalRead::Complete(snapshot)
    } else {
        JournalRead::Incomplete(snapshot)
    })
}
