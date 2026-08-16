use super::*;
use std::fs;

struct TempDir(std::path::PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
impl TempDir {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
fn tempdir() -> TempDir {
    let path = std::env::temp_dir().join(format!(
        "mcts-lifecycle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    TempDir(path)
}
fn path(dir: &TempDir) -> std::path::PathBuf {
    dir.path().join("lifecycle.jsonl")
}
fn manifest() -> WrapperManifest {
    WrapperManifest {
        logical_run_id: "run".into(),
        attempt_id: "attempt".into(),
        parent_attempt_id: None,
        argv: vec!["bench".into()],
        wrapper_pid: 7,
        process_group_id: 7,
        hostname: "host".into(),
        boot_id: Some("boot".into()),
        process_start_id: Some("start".into()),
    }
}
fn writer(dir: &TempDir) -> LifecycleWriter {
    LifecycleWriter::create(path(dir), manifest(), "nonce", "t0").unwrap()
}
fn line(sequence: u64, payload: &str) -> String {
    format!(
        r#"{{"schema_version":1,"sequence":{sequence},"attempt_id":"attempt","launch_nonce":"nonce","timestamp":"t","payload":{payload}}}"#
    )
}
fn wrapper(sequence: u64) -> String {
    line(
        sequence,
        r#"{"type":"wrapper_started","value":{"logical_run_id":"run","attempt_id":"attempt","parent_attempt_id":null,"argv":["bench"],"wrapper_pid":7,"process_group_id":7,"hostname":"host","boot_id":null,"process_start_id":null}}"#,
    )
}
fn started(sequence: u64) -> String {
    line(
        sequence,
        r#"{"type":"child_started","value":{"child_pid":8}}"#,
    )
}
fn exited(sequence: u64) -> String {
    line(
        sequence,
        r#"{"type":"child_exited","value":{"outcome":{"kind":"code","value":{"code":0}}}}"#,
    )
}
fn spawn_failed(sequence: u64) -> String {
    line(
        sequence,
        r#"{"type":"child_spawn_failed","value":{"stage":"spawn","error":"no"}}"#,
    )
}
fn closed(sequence: u64) -> String {
    line(
        sequence,
        r#"{"type":"outputs_closed","value":{"outputs":[{"path":"out","byte_length":2}]}}"#,
    )
}
fn journal(lines: &[String]) -> String {
    format!("{}\n", lines.join("\n"))
}
fn assert_reason(
    text: String,
    expected: InvalidReason,
    line: Option<usize>,
    sequence: Option<u64>,
) {
    let dir = tempdir();
    let file = path(&dir);
    fs::write(&file, text).unwrap();
    let Err(LifecycleError::Invalid {
        reason,
        line: found_line,
        sequence: found_sequence,
        ..
    }) = read_journal(file)
    else {
        panic!("expected invalid journal")
    };
    assert_eq!(reason, expected);
    assert_eq!(found_line, line);
    assert_eq!(found_sequence, sequence);
}

#[test]
fn every_payload_and_exit_variant_round_trips() {
    for payload in [
        LifecyclePayload::WrapperStarted(manifest()),
        LifecyclePayload::ChildStarted { child_pid: 8 },
        LifecyclePayload::ChildSpawnFailed {
            stage: "spawn".into(),
            error: "no".into(),
        },
        LifecyclePayload::ChildExited {
            outcome: ExitEvidence::Code { code: -1 },
        },
        LifecyclePayload::ChildExited {
            outcome: ExitEvidence::Signal { signal: 9 },
        },
        LifecyclePayload::ChildExited {
            outcome: ExitEvidence::WaitFailed {
                error: "wait".into(),
            },
        },
        LifecyclePayload::OutputsClosed {
            outputs: vec![OutputClosure {
                path: "out".into(),
                byte_length: Some(0),
            }],
        },
    ] {
        assert_eq!(
            payload,
            serde_json::from_str(&serde_json::to_string(&payload).unwrap()).unwrap()
        );
    }
}

#[test]
fn complete_histories_preserve_exact_terminal_evidence() {
    for outcome in [
        ExitEvidence::Code { code: 0 },
        ExitEvidence::Code { code: 4 },
        ExitEvidence::Signal { signal: 15 },
        ExitEvidence::WaitFailed {
            error: "wait".into(),
        },
    ] {
        let dir = tempdir();
        let mut w = writer(&dir);
        w.child_started(8, "t1").unwrap();
        w.child_exited(outcome.clone(), "t2").unwrap();
        w.outputs_closed(vec![], "t3").unwrap();
        let JournalRead::Complete(snapshot) = read_journal(path(&dir)).unwrap() else {
            panic!()
        };
        assert_eq!(snapshot.terminal, Some(TerminalEvidence::Exited(outcome)));
    }
    let dir = tempdir();
    let mut w = writer(&dir);
    w.child_spawn_failed("spawn", "no", "t1").unwrap();
    w.outputs_closed(vec![], "t2").unwrap();
    let JournalRead::Complete(snapshot) = read_journal(path(&dir)).unwrap() else {
        panic!()
    };
    assert!(matches!(
        snapshot.terminal,
        Some(TerminalEvidence::SpawnFailed { .. })
    ));
}

#[test]
fn valid_prefixes_are_incomplete() {
    let dir = tempdir();
    let mut w = writer(&dir);
    assert!(matches!(
        read_journal(path(&dir)).unwrap(),
        JournalRead::Incomplete(_)
    ));
    w.child_started(8, "t1").unwrap();
    assert!(matches!(
        read_journal(path(&dir)).unwrap(),
        JournalRead::Incomplete(_)
    ));
    w.child_exited(ExitEvidence::Code { code: 0 }, "t2")
        .unwrap();
    assert!(matches!(
        read_journal(path(&dir)).unwrap(),
        JournalRead::Incomplete(_)
    ));

    let failed_dir = tempdir();
    let mut failed = writer(&failed_dir);
    failed.child_spawn_failed("spawn", "missing", "t1").unwrap();
    assert!(matches!(
        read_journal(path(&failed_dir)).unwrap(),
        JournalRead::Incomplete(_)
    ));
}

#[test]
fn reader_rejection_matrix_has_exact_reason_line_and_sequence() {
    let unknown_version = String::from(
        r#"{"schema_version":2,"sequence":0,"attempt_id":"attempt","launch_nonce":"nonce","timestamp":"t","payload":{"type":"wrapper_started","value":{"logical_run_id":"run","attempt_id":"attempt","parent_attempt_id":null,"argv":["bench"],"wrapper_pid":7,"process_group_id":7,"hostname":"host","boot_id":null,"process_start_id":null}}}"#,
    ) + "\n";
    let cases = vec![
        ("{\n".into(), InvalidReason::JsonSyntax, Some(1), None),
        (
            line(0, r#"{"type":"wrapper_started","value":{}}"#) + "\n",
            InvalidReason::ClosedSchemaViolation,
            Some(1),
            Some(0),
        ),
        (
            unknown_version,
            InvalidReason::UnsupportedSchemaVersion,
            Some(1),
            Some(0),
        ),
        (
            line(0, r#"{"type":"unknown","value":{}}"#) + "\n",
            InvalidReason::UnsupportedRecordType,
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0), started(2)]),
            InvalidReason::SequenceMismatch,
            Some(2),
            Some(2),
        ),
        (
            journal(&[wrapper(0), started(1), exited(1)]),
            InvalidReason::SequenceMismatch,
            Some(3),
            Some(1),
        ),
        (
            journal(&[started(0)]),
            InvalidReason::FirstRecordNotWrapper,
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0), wrapper(1)]),
            InvalidReason::DuplicateWrapper,
            Some(2),
            Some(1),
        ),
        (
            journal(&[wrapper(0), exited(1)]),
            InvalidReason::InvalidTypedRecordOrdering,
            Some(2),
            Some(1),
        ),
        (
            journal(&[wrapper(0), started(1), spawn_failed(2)]),
            InvalidReason::InvalidTypedRecordOrdering,
            Some(3),
            Some(2),
        ),
        (
            journal(&[wrapper(0), spawn_failed(1), started(2)]),
            InvalidReason::InvalidTypedRecordOrdering,
            Some(3),
            Some(2),
        ),
        (
            journal(&[wrapper(0), closed(1)]),
            InvalidReason::InvalidTypedRecordOrdering,
            Some(2),
            Some(1),
        ),
        (
            journal(&[wrapper(0), started(1), exited(2), exited(3)]),
            InvalidReason::InvalidTypedRecordOrdering,
            Some(4),
            Some(3),
        ),
        (
            journal(&[wrapper(0), started(1), exited(2), closed(3), started(4)]),
            InvalidReason::RecordsAfterClose,
            Some(5),
            Some(4),
        ),
        ("".into(), InvalidReason::EmptyJournal, None, None),
        ("\n".into(), InvalidReason::BlankRecord, Some(1), None),
        (wrapper(0), InvalidReason::UnterminatedRecord, Some(1), None),
    ];
    for (text, reason, line, sequence) in cases {
        assert_reason(text, reason, line, sequence);
    }
}

#[test]
fn reader_rejects_identity_content_exit_and_closed_schema_violations() {
    let cases = vec![
        (
            journal(&[wrapper(0).replace("\"attempt_id\":\"attempt\"", "\"attempt_id\":\"\"")]),
            InvalidReason::InvalidNamedField {
                field: "attempt_id",
            },
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0), started(1).replace("\"attempt\"", "\"other\"")]),
            InvalidReason::AttemptIdDrift,
            Some(2),
            Some(1),
        ),
        (
            journal(&[wrapper(0), started(1).replace("\"nonce\"", "\"other\"")]),
            InvalidReason::LaunchNonceDrift,
            Some(2),
            Some(1),
        ),
        (
            journal(&[wrapper(0).replace("\"launch_nonce\":\"nonce\"", "\"launch_nonce\":\"\"")]),
            InvalidReason::InvalidNamedField {
                field: "launch_nonce",
            },
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0).replace("\"timestamp\":\"t\"", "\"timestamp\":\"\"")]),
            InvalidReason::InvalidNamedField { field: "timestamp" },
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0).replace("\"argv\":[\"bench\"]", "\"argv\":[]")]),
            InvalidReason::InvalidNamedField { field: "argv" },
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0).replace("\"logical_run_id\":\"run\"", "\"logical_run_id\":\"\"")]),
            InvalidReason::InvalidNamedField {
                field: "logical_run_id",
            },
            Some(1),
            Some(0),
        ),
        (
            journal(&[
                wrapper(0).replace("\"parent_attempt_id\":null", "\"parent_attempt_id\":\"\"")
            ]),
            InvalidReason::InvalidNamedField {
                field: "identity field",
            },
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0).replace("\"wrapper_pid\":7", "\"wrapper_pid\":0")]),
            InvalidReason::InvalidNamedField {
                field: "wrapper_pid",
            },
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0).replace("\"process_group_id\":7", "\"process_group_id\":0")]),
            InvalidReason::InvalidNamedField {
                field: "process_group_id",
            },
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0).replace("\"hostname\":\"host\"", "\"hostname\":\"\"")]),
            InvalidReason::InvalidNamedField { field: "hostname" },
            Some(1),
            Some(0),
        ),
        (
            journal(&[
                wrapper(0),
                started(1).replace("\"child_pid\":8", "\"child_pid\":0"),
            ]),
            InvalidReason::InvalidNamedField { field: "child_pid" },
            Some(2),
            Some(1),
        ),
        (
            journal(&[
                wrapper(0),
                line(
                    1,
                    r#"{"type":"child_spawn_failed","value":{"stage":"","error":""}}"#,
                ),
            ]),
            InvalidReason::InvalidNamedField { field: "stage" },
            Some(2),
            Some(1),
        ),
        (
            journal(&[
                wrapper(0),
                started(1),
                line(
                    2,
                    r#"{"type":"child_exited","value":{"outcome":{"kind":"wait_failed","value":{"error":""}}}}"#,
                ),
            ]),
            InvalidReason::InvalidNamedField {
                field: "wait failure",
            },
            Some(3),
            Some(2),
        ),
        (
            journal(&[
                wrapper(0),
                line(
                    1,
                    r#"{"type":"child_spawn_failed","value":{"stage":"spawn","error":""}}"#,
                ),
            ]),
            InvalidReason::InvalidNamedField { field: "error" },
            Some(2),
            Some(1),
        ),
        (
            journal(&[
                wrapper(0),
                line(1, r#"{"type":"child_exited","value":{"outcome":{}}}"#),
            ]),
            InvalidReason::InvalidExitVariant,
            Some(2),
            Some(1),
        ),
        (
            journal(&[
                wrapper(0),
                line(
                    1,
                    r#"{"type":"child_exited","value":{"outcome":{"kind":"code","value":{"code":0,"signal":9}}}}"#,
                ),
            ]),
            InvalidReason::InvalidExitVariant,
            Some(2),
            Some(1),
        ),
        (
            journal(&[
                wrapper(0),
                line(
                    1,
                    r#"{"type":"child_exited","value":{"outcome":{"kind":"wat","value":{}}}}"#,
                ),
            ]),
            InvalidReason::InvalidExitVariant,
            Some(2),
            Some(1),
        ),
        (
            journal(&[wrapper(0).replace(
                "\"process_start_id\":null",
                "\"process_start_id\":null,\"extra\":true",
            )]),
            InvalidReason::ClosedSchemaViolation,
            Some(1),
            Some(0),
        ),
        (
            journal(&[wrapper(0).replace(
                "\"timestamp\":\"t\",",
                "\"timestamp\":\"t\",\"extra\":true,",
            )]),
            InvalidReason::ClosedSchemaViolation,
            Some(1),
            Some(0),
        ),
        (
            journal(&[
                wrapper(0),
                started(1),
                exited(2),
                line(
                    3,
                    r#"{"type":"outputs_closed","value":{"outputs":[{"path":"","byte_length":2}]}}"#,
                ),
            ]),
            InvalidReason::InvalidNamedField {
                field: "output path",
            },
            Some(4),
            Some(3),
        ),
    ];
    for (text, reason, line, sequence) in cases {
        assert_reason(text, reason, line, sequence);
    }
}

#[test]
fn writer_is_exclusive_stable_and_reader_accepts_every_prefix() {
    for contents in ["", "old\n"] {
        let dir = tempdir();
        let file = path(&dir);
        fs::write(&file, contents).unwrap();
        assert!(matches!(
            LifecycleWriter::create(&file, manifest(), "nonce", "t"),
            Err(LifecycleError::Conflict { .. })
        ));
    }
    let dir = tempdir();
    let file = path(&dir);
    let mut w = writer(&dir);
    assert!(
        matches!(read_journal(&file).unwrap(), JournalRead::Incomplete(snapshot) if snapshot.last_sequence == 0)
    );
    w.child_started(8, "t1").unwrap();
    assert!(
        matches!(read_journal(&file).unwrap(), JournalRead::Incomplete(snapshot) if snapshot.child == Some(8) && snapshot.last_sequence == 1)
    );
    w.child_exited(ExitEvidence::Code { code: 3 }, "t2")
        .unwrap();
    assert!(
        matches!(read_journal(&file).unwrap(), JournalRead::Incomplete(snapshot) if snapshot.last_sequence == 2)
    );
    let outputs = vec![
        OutputClosure {
            path: "first".into(),
            byte_length: Some(0),
        },
        OutputClosure {
            path: "second".into(),
            byte_length: None,
        },
    ];
    w.outputs_closed(outputs.clone(), "t3").unwrap();
    let JournalRead::Complete(snapshot) = read_journal(&file).unwrap() else {
        panic!()
    };
    assert_eq!(snapshot.outputs, Some(outputs));
    assert_eq!(snapshot.last_sequence, 3);
    let records: Vec<LifecycleRecord> = fs::read_to_string(&file)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 4);
    assert_eq!(
        records
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert!(records
        .iter()
        .all(|record| record.attempt_id == "attempt" && record.launch_nonce == "nonce"));
}

#[test]
fn writer_rejects_invalid_state_without_changing_valid_prefix_and_poison_is_sticky() {
    let dir = tempdir();
    let file = path(&dir);
    let mut w = writer(&dir);
    let initial = fs::read_to_string(&file).unwrap();
    assert!(matches!(
        w.child_exited(ExitEvidence::Code { code: 0 }, "t"),
        Err(LifecycleError::Invalid {
            reason: InvalidReason::InvalidTypedRecordOrdering,
            ..
        })
    ));
    assert_eq!(fs::read_to_string(&file).unwrap(), initial);
    w.child_started(8, "t1").unwrap();
    let started_prefix = fs::read_to_string(&file).unwrap();
    assert!(matches!(
        w.child_spawn_failed("spawn", "no", "t2"),
        Err(LifecycleError::Invalid {
            reason: InvalidReason::InvalidTypedRecordOrdering,
            ..
        })
    ));
    assert_eq!(fs::read_to_string(&file).unwrap(), started_prefix);
    assert!(matches!(
        w.outputs_closed(vec![], "t2"),
        Err(LifecycleError::Invalid {
            reason: InvalidReason::InvalidTypedRecordOrdering,
            ..
        })
    ));
    assert_eq!(fs::read_to_string(&file).unwrap(), started_prefix);
    w.fail_next_append_for_test();
    assert!(matches!(
        w.child_exited(ExitEvidence::Code { code: 0 }, "t2"),
        Err(LifecycleError::Io { .. })
    ));
    assert!(matches!(
        w.child_exited(ExitEvidence::Code { code: 0 }, "t3"),
        Err(LifecycleError::Poisoned { .. })
    ));
    assert!(matches!(
        read_journal(&file).unwrap(),
        JournalRead::Incomplete(_)
    ));
}

#[test]
fn writer_rejects_duplicate_terminal_and_every_append_after_close() {
    let dir = tempdir();
    let file = path(&dir);
    let mut w = writer(&dir);
    w.child_started(8, "t1").unwrap();
    w.child_exited(ExitEvidence::Code { code: 0 }, "t2")
        .unwrap();
    let terminal_prefix = fs::read_to_string(&file).unwrap();
    assert!(w
        .child_exited(ExitEvidence::Code { code: 1 }, "t3")
        .is_err());
    assert_eq!(fs::read_to_string(&file).unwrap(), terminal_prefix);

    w.outputs_closed(vec![], "t3").unwrap();
    let closed_prefix = fs::read_to_string(&file).unwrap();
    assert!(w.outputs_closed(vec![], "t4").is_err());
    assert!(w.child_started(9, "t4").is_err());
    assert!(w.child_spawn_failed("spawn", "late", "t4").is_err());
    assert_eq!(fs::read_to_string(&file).unwrap(), closed_prefix);
    assert!(matches!(
        read_journal(&file).unwrap(),
        JournalRead::Complete(_)
    ));
}

#[test]
fn missing_file_is_distinct() {
    let dir = tempdir();
    assert!(matches!(
        read_journal(path(&dir)).unwrap(),
        JournalRead::Missing
    ));
}
