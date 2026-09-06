use super::*;
use crate::lifecycle::{JournalRead, JournalSnapshot, TerminalEvidence, WrapperManifest};

fn descriptor() -> LaunchDescriptor {
    LaunchDescriptor {
        supervisor: "bench".into(),
        logical_run_id: "run".into(),
        attempt_id: "attempt".into(),
        parent_attempt_id: Some("parent".into()),
        launch_nonce: "nonce".into(),
        workload_argv: vec!["worker".into()],
        journal_path: "journal".into(),
        stdout_path: "out".into(),
        stderr_path: "err".into(),
    }
}

fn manifest(descriptor: &LaunchDescriptor) -> WrapperManifest {
    WrapperManifest {
        logical_run_id: descriptor.logical_run_id.clone(),
        attempt_id: descriptor.attempt_id.clone(),
        parent_attempt_id: descriptor.parent_attempt_id.clone(),
        argv: descriptor.workload_argv.clone(),
        wrapper_pid: 4,
        process_group_id: 9,
        hostname: "host".into(),
        boot_id: None,
        process_start_id: None,
    }
}

#[test]
fn command_preserves_exact_descriptor_arguments() {
    let descriptor = descriptor();
    let command = supervisor_command(&descriptor).unwrap();
    assert_eq!(command.executable, "bench");
    assert!(command.arguments.ends_with(&["--".into(), "worker".into()]));
}

#[test]
fn classifier_requires_exact_correlated_child_start() {
    let descriptor = descriptor();
    let wrapper = WrapperIdentity {
        pid: 4,
        process_group_id: 9,
    };
    assert_eq!(
        classify_readiness(&descriptor, wrapper, Ok(JournalRead::Missing)),
        ReadinessDecision::Pending
    );
    let snapshot = JournalSnapshot {
        manifest: manifest(&descriptor),
        launch_nonce: "nonce".into(),
        child: Some(8),
        terminal: Some(TerminalEvidence::Exited(
            crate::lifecycle::ExitEvidence::Code { code: 0 },
        )),
        outputs: None,
        last_sequence: 2,
    };
    assert!(matches!(
        classify_readiness(&descriptor, wrapper, Ok(JournalRead::Incomplete(snapshot))),
        ReadinessDecision::Ready(_)
    ));
}
