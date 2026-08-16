use super::*;
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
        ReadinessFailure::MalformedEvidence,
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
