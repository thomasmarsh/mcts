use super::*;
use crate::lifecycle::{
    InvalidReason, JournalRead, JournalSnapshot, LifecycleError, TerminalEvidence, WrapperManifest,
};
use std::ffi::OsString;

fn d() -> LaunchDescriptor {
    LaunchDescriptor {
        supervisor: "bench".into(),
        logical_run_id: "run".into(),
        attempt_id: "attempt".into(),
        parent_attempt_id: Some("parent".into()),
        launch_nonce: "nonce".into(),
        workload_argv: vec!["worker".into(), "--literal".into()],
        journal_path: "journal".into(),
        stdout_path: "out".into(),
        stderr_path: "err".into(),
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    Spawn(SupervisorCommand),
    Readiness {
        attempt: String,
        nonce: String,
        wrapper: WrapperIdentity,
    },
}
struct Fake {
    events: Vec<Event>,
    spawn: Result<WrapperIdentity, SpawnFailure>,
    ready: Result<ReadinessEvidence, ReadinessFailure>,
}
impl DetachedSupervisorPort for Fake {
    fn spawn_detached(&mut self, c: &SupervisorCommand) -> Result<WrapperIdentity, SpawnFailure> {
        self.events.push(Event::Spawn(c.clone()));
        self.spawn.clone()
    }
    fn observe_readiness(
        &mut self,
        d: &LaunchDescriptor,
        w: WrapperIdentity,
    ) -> Result<ReadinessEvidence, ReadinessFailure> {
        self.events.push(Event::Readiness {
            attempt: d.attempt_id.clone(),
            nonce: d.launch_nonce.clone(),
            wrapper: w,
        });
        self.ready.clone()
    }
}
fn fake(ready: Result<ReadinessEvidence, ReadinessFailure>) -> Fake {
    Fake {
        events: vec![],
        spawn: Ok(WrapperIdentity {
            pid: 4,
            process_group_id: 9,
        }),
        ready,
    }
}
fn evidence(a: &str, n: &str, pid: u64, pgid: u64) -> ReadinessEvidence {
    ReadinessEvidence {
        attempt_id: a.into(),
        launch_nonce: n.into(),
        wrapper: WrapperIdentity {
            pid,
            process_group_id: pgid,
        },
    }
}
fn manifest(x: &LaunchDescriptor) -> WrapperManifest {
    WrapperManifest {
        logical_run_id: x.logical_run_id.clone(),
        attempt_id: x.attempt_id.clone(),
        parent_attempt_id: x.parent_attempt_id.clone(),
        argv: x.workload_argv.clone(),
        wrapper_pid: 4,
        process_group_id: 9,
        hostname: "host".into(),
        boot_id: None,
        process_start_id: None,
    }
}
fn journal(
    m: WrapperManifest,
    nonce: &str,
    child: Option<u64>,
    terminal: Option<TerminalEvidence>,
) -> JournalRead {
    JournalRead::Incomplete(JournalSnapshot {
        manifest: m,
        launch_nonce: nonce.into(),
        child,
        terminal,
        outputs: None,
        last_sequence: 0,
    })
}
fn complete(m: WrapperManifest, nonce: &str, child: u64) -> JournalRead {
    JournalRead::Complete(JournalSnapshot {
        manifest: m,
        launch_nonce: nonce.into(),
        child: Some(child),
        terminal: Some(TerminalEvidence::Exited(
            crate::lifecycle::ExitEvidence::Code { code: 0 },
        )),
        outputs: Some(vec![]),
        last_sequence: 3,
    })
}

#[test]
fn exact_command_and_non_utf8_values() {
    let mut x = d();
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        x.supervisor = OsString::from_vec(vec![b'b', 0xff]);
        x.journal_path = OsString::from_vec(vec![b'j', 0xfd]).into();
    }
    let c = supervisor_command(&x).unwrap();
    let mut expected: Vec<OsString> = [
        "supervise",
        "--logical-run-id",
        "run",
        "--attempt-id",
        "attempt",
        "--launch-nonce",
        "nonce",
        "--journal-path",
        "journal",
        "--stdout-path",
        "out",
        "--stderr-path",
        "err",
        "--parent-attempt-id",
        "parent",
        "--",
        "worker",
        "--literal",
    ]
    .into_iter()
    .map(OsString::from)
    .collect();
    expected[8] = x.journal_path.clone().into_os_string();
    assert_eq!(c.executable, x.supervisor);
    assert_eq!(c.arguments, expected);
}

#[test]
fn classifier_decides_pending_ready_startup_and_correlation() {
    let x = d();
    let w = WrapperIdentity {
        pid: 4,
        process_group_id: 9,
    };
    assert_eq!(
        classify_readiness(&x, w, Ok(JournalRead::Missing)),
        ReadinessDecision::Pending
    );
    assert_eq!(
        classify_readiness(&x, w, Ok(journal(manifest(&x), "nonce", None, None))),
        ReadinessDecision::Pending
    );
    assert_eq!(
        classify_readiness(&x, w, Ok(journal(manifest(&x), "nonce", Some(8), None))),
        ReadinessDecision::Ready(ReadinessEvidence {
            attempt_id: "attempt".into(),
            launch_nonce: "nonce".into(),
            wrapper: w
        })
    );
    assert_eq!(
        classify_readiness(&x, w, Ok(complete(manifest(&x), "nonce", 8))),
        ReadinessDecision::Ready(ReadinessEvidence {
            attempt_id: "attempt".into(),
            launch_nonce: "nonce".into(),
            wrapper: w
        })
    );
    assert_eq!(
        classify_readiness(
            &x,
            w,
            Ok(journal(
                manifest(&x),
                "nonce",
                None,
                Some(TerminalEvidence::SpawnFailed {
                    stage: "spawn".into(),
                    error: "no".into()
                })
            ))
        ),
        ReadinessDecision::StartupFailed {
            stage: "spawn".into(),
            error: "no".into()
        }
    );
}
#[test]
fn ordered_spawn_readiness_and_ready_facts() {
    let x = d();
    let mut f = fake(Ok(evidence("attempt", "nonce", 4, 9)));
    assert_eq!(
        launch(&x, &mut f),
        LaunchOutcome::Ready(ReadySupervisor {
            wrapper: WrapperIdentity {
                pid: 4,
                process_group_id: 9,
            },
            launch_nonce: x.launch_nonce.clone(),
            journal_path: x.journal_path.clone(),
            workload_argv: x.workload_argv.clone(),
        })
    );
    assert!(matches!(f.events[0], Event::Spawn(_)));
    assert_eq!(
        f.events[1],
        Event::Readiness {
            attempt: "attempt".into(),
            nonce: "nonce".into(),
            wrapper: WrapperIdentity {
                pid: 4,
                process_group_id: 9
            }
        }
    );
}
#[test]
fn invalid_and_spawn_failure_effects() {
    let mut x = d();
    x.workload_argv.clear();
    let mut f = fake(Ok(evidence("attempt", "nonce", 4, 9)));
    assert!(matches!(launch(&x, &mut f), LaunchOutcome::Invalid(_)));
    assert!(f.events.is_empty());
    let x = d();
    f.spawn = Err(SpawnFailure::Failed {
        message: "denied".into(),
    });
    assert!(matches!(
        launch(&x, &mut f),
        LaunchOutcome::SpawnFailed(SpawnFailure::Failed { .. })
    ));
    assert_eq!(f.events.len(), 1);
}
#[test]
fn coordinator_classifies_mismatches() {
    let cases = [
        (
            evidence("other", "nonce", 4, 9),
            ReadinessFailure::AttemptMismatch {
                expected: "attempt".into(),
                observed: "other".into(),
            },
        ),
        (
            evidence("attempt", "other", 4, 9),
            ReadinessFailure::NonceMismatch {
                expected: "nonce".into(),
                observed: "other".into(),
            },
        ),
        (
            evidence("attempt", "nonce", 8, 9),
            ReadinessFailure::WrapperPidMismatch {
                expected: 4,
                observed: 8,
            },
        ),
        (
            evidence("attempt", "nonce", 4, 8),
            ReadinessFailure::ProcessGroupMismatch {
                expected: 9,
                observed: 8,
            },
        ),
    ];
    for (e, failure) in cases {
        let x = d();
        let mut f = fake(Ok(e));
        assert_eq!(
            launch(&x, &mut f),
            LaunchOutcome::Unready {
                wrapper: WrapperIdentity {
                    pid: 4,
                    process_group_id: 9
                },
                failure
            }
        );
    }
}
#[test]
fn every_readiness_failure_retains_wrapper() {
    for failure in [
        ReadinessFailure::StartupFailure,
        ReadinessFailure::EarlyExit,
        ReadinessFailure::Timeout,
        ReadinessFailure::UnavailableEvidence,
        ReadinessFailure::Malformed(InvalidReason::JsonSyntax),
        ReadinessFailure::Conflict,
    ] {
        let x = d();
        let mut f = fake(Err(failure.clone()));
        assert_eq!(
            launch(&x, &mut f),
            LaunchOutcome::Unready {
                wrapper: WrapperIdentity {
                    pid: 4,
                    process_group_id: 9
                },
                failure
            }
        );
    }
}

#[test]
fn classifier_reports_each_exact_mismatch_and_error_kind() {
    let x = d();
    let w = WrapperIdentity {
        pid: 4,
        process_group_id: 9,
    };
    let mut cases = Vec::new();
    let mut m = manifest(&x);
    m.logical_run_id = "other".into();
    cases.push((
        journal(m, "nonce", None, None),
        ReadinessFailure::LogicalRunIdMismatch {
            expected: "run".into(),
            observed: "other".into(),
        },
    ));
    let mut m = manifest(&x);
    m.attempt_id = "other".into();
    cases.push((
        journal(m, "nonce", None, None),
        ReadinessFailure::AttemptMismatch {
            expected: "attempt".into(),
            observed: "other".into(),
        },
    ));
    let mut m = manifest(&x);
    m.parent_attempt_id = None;
    cases.push((
        journal(m, "nonce", None, None),
        ReadinessFailure::ParentAttemptIdMismatch {
            expected: Some("parent".into()),
            observed: None,
        },
    ));
    cases.push((
        journal(manifest(&x), "other", None, None),
        ReadinessFailure::NonceMismatch {
            expected: "nonce".into(),
            observed: "other".into(),
        },
    ));
    let mut m = manifest(&x);
    m.argv = vec!["other".into(); 1];
    cases.push((
        journal(m, "nonce", None, None),
        ReadinessFailure::WorkloadMismatch {
            expected: vec!["worker".into(), "--literal".into()],
            observed: vec!["other".into()],
        },
    ));
    let mut m = manifest(&x);
    m.wrapper_pid = 8;
    cases.push((
        journal(m, "nonce", None, None),
        ReadinessFailure::WrapperPidMismatch {
            expected: 4,
            observed: 8,
        },
    ));
    let mut m = manifest(&x);
    m.process_group_id = 8;
    cases.push((
        journal(m, "nonce", None, None),
        ReadinessFailure::ProcessGroupMismatch {
            expected: 9,
            observed: 8,
        },
    ));
    for (read, failure) in cases {
        assert_eq!(
            classify_readiness(&x, w, Ok(read)),
            ReadinessDecision::Invalid(failure)
        );
    }
    assert!(matches!(
        classify_readiness(
            &x,
            w,
            Err(LifecycleError::Invalid {
                path: "x".into(),
                line: None,
                sequence: None,
                reason: InvalidReason::BlankRecord
            })
        ),
        ReadinessDecision::Invalid(ReadinessFailure::Malformed(InvalidReason::BlankRecord))
    ));
    assert_eq!(
        classify_readiness(&x, w, Err(LifecycleError::Conflict { path: "x".into() })),
        ReadinessDecision::Invalid(ReadinessFailure::Conflict)
    );
    let io = std::io::Error::new(std::io::ErrorKind::Other, "io");
    assert_eq!(
        classify_readiness(
            &x,
            w,
            Err(LifecycleError::Io {
                path: "x".into(),
                operation: "read",
                source: io
            })
        ),
        ReadinessDecision::Invalid(ReadinessFailure::Unreadable)
    );
    assert_eq!(
        classify_readiness(&x, w, Err(LifecycleError::Poisoned { path: "x".into() })),
        ReadinessDecision::Invalid(ReadinessFailure::Unreadable)
    );
}
