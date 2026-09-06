//! Typed launch and lifecycle-readiness decisions for one detached supervisor.
use crate::lifecycle::{ExitEvidence, InvalidReason, JournalRead, TerminalEvidence};
use std::ffi::OsString;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchDescriptor {
    pub supervisor: OsString,
    pub logical_run_id: String,
    pub attempt_id: String,
    pub parent_attempt_id: Option<String>,
    pub launch_nonce: String,
    pub workload_argv: Vec<String>,
    pub journal_path: PathBuf,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupervisorCommand {
    pub executable: OsString,
    pub arguments: Vec<OsString>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrapperIdentity {
    pub pid: u64,
    pub process_group_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationTarget {
    pub logical_run_id: String,
    pub attempt_id: String,
    pub parent_attempt_id: Option<String>,
    pub launch_nonce: String,
    pub workload_argv: Vec<String>,
    pub journal_path: PathBuf,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
    pub wrapper: WrapperIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidInput {
    Empty { field: &'static str },
    EmptyWorkload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadinessFailure {
    Conflict,
    AttemptMismatch {
        expected: String,
        observed: String,
    },
    NonceMismatch {
        expected: String,
        observed: String,
    },
    WrapperPidMismatch {
        expected: u64,
        observed: u64,
    },
    ProcessGroupMismatch {
        expected: u64,
        observed: u64,
    },
    Malformed(InvalidReason),
    Unreadable,
    LogicalRunIdMismatch {
        expected: String,
        observed: String,
    },
    ParentAttemptIdMismatch {
        expected: Option<String>,
        observed: Option<String>,
    },
    WorkloadMismatch {
        expected: Vec<String>,
        observed: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadinessEvidence {
    pub attempt_id: String,
    pub launch_nonce: String,
    pub wrapper: WrapperIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadinessDecision {
    Pending,
    Ready(ReadinessEvidence),
    StartupFailed { stage: String, error: String },
    Invalid(ReadinessFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationDecision {
    Pending,
    Terminal(ExitEvidence),
    Invalid(ReadinessFailure),
}

pub fn classify_observation(
    target: &ObservationTarget,
    journal: Result<JournalRead, crate::lifecycle::LifecycleError>,
) -> ObservationDecision {
    let read = match journal {
        Ok(read) => read,
        Err(crate::lifecycle::LifecycleError::Invalid { reason, .. }) => {
            return ObservationDecision::Invalid(ReadinessFailure::Malformed(reason))
        }
        Err(crate::lifecycle::LifecycleError::Conflict { .. }) => {
            return ObservationDecision::Invalid(ReadinessFailure::Conflict)
        }
        Err(
            crate::lifecycle::LifecycleError::Io { .. }
            | crate::lifecycle::LifecycleError::Poisoned { .. },
        ) => return ObservationDecision::Invalid(ReadinessFailure::Unreadable),
    };
    let JournalRead::Complete(snapshot) = read else {
        return ObservationDecision::Pending;
    };
    if let Err(error) = validate_correlation(target, &snapshot) {
        return ObservationDecision::Invalid(error);
    }
    let outputs_match = snapshot.outputs.as_ref().is_some_and(|outputs| {
        outputs.len() == 2
            && outputs
                .iter()
                .any(|output| output.path == target.stdout_path.to_string_lossy())
            && outputs
                .iter()
                .any(|output| output.path == target.stderr_path.to_string_lossy())
    });
    if !outputs_match {
        return ObservationDecision::Invalid(ReadinessFailure::Conflict);
    }
    match snapshot.terminal {
        Some(TerminalEvidence::Exited(exit)) => ObservationDecision::Terminal(exit),
        Some(TerminalEvidence::SpawnFailed { .. }) => {
            ObservationDecision::Terminal(ExitEvidence::WaitFailed {
                error: "child spawn failed".into(),
            })
        }
        None => ObservationDecision::Pending,
    }
}

fn validate_correlation(
    target: &ObservationTarget,
    snapshot: &crate::lifecycle::JournalSnapshot,
) -> Result<(), ReadinessFailure> {
    let manifest = &snapshot.manifest;
    if manifest.logical_run_id != target.logical_run_id {
        return Err(ReadinessFailure::LogicalRunIdMismatch {
            expected: target.logical_run_id.clone(),
            observed: manifest.logical_run_id.clone(),
        });
    }
    if manifest.attempt_id != target.attempt_id {
        return Err(ReadinessFailure::AttemptMismatch {
            expected: target.attempt_id.clone(),
            observed: manifest.attempt_id.clone(),
        });
    }
    if manifest.parent_attempt_id != target.parent_attempt_id {
        return Err(ReadinessFailure::ParentAttemptIdMismatch {
            expected: target.parent_attempt_id.clone(),
            observed: manifest.parent_attempt_id.clone(),
        });
    }
    if snapshot.launch_nonce != target.launch_nonce {
        return Err(ReadinessFailure::NonceMismatch {
            expected: target.launch_nonce.clone(),
            observed: snapshot.launch_nonce.clone(),
        });
    }
    if manifest.argv != target.workload_argv {
        return Err(ReadinessFailure::WorkloadMismatch {
            expected: target.workload_argv.clone(),
            observed: manifest.argv.clone(),
        });
    }
    if manifest.wrapper_pid != target.wrapper.pid {
        return Err(ReadinessFailure::WrapperPidMismatch {
            expected: target.wrapper.pid,
            observed: manifest.wrapper_pid,
        });
    }
    if manifest.process_group_id != target.wrapper.process_group_id {
        return Err(ReadinessFailure::ProcessGroupMismatch {
            expected: target.wrapper.process_group_id,
            observed: manifest.process_group_id,
        });
    }
    Ok(())
}

pub fn classify_readiness(
    descriptor: &LaunchDescriptor,
    wrapper: WrapperIdentity,
    journal: Result<JournalRead, crate::lifecycle::LifecycleError>,
) -> ReadinessDecision {
    let read = match journal {
        Ok(read) => read,
        Err(crate::lifecycle::LifecycleError::Invalid { reason, .. }) => {
            return ReadinessDecision::Invalid(ReadinessFailure::Malformed(reason))
        }
        Err(crate::lifecycle::LifecycleError::Conflict { .. }) => {
            return ReadinessDecision::Invalid(ReadinessFailure::Conflict)
        }
        Err(
            crate::lifecycle::LifecycleError::Io { .. }
            | crate::lifecycle::LifecycleError::Poisoned { .. },
        ) => return ReadinessDecision::Invalid(ReadinessFailure::Unreadable),
    };
    let snapshot = match read {
        JournalRead::Missing => return ReadinessDecision::Pending,
        JournalRead::Incomplete(snapshot) | JournalRead::Complete(snapshot) => snapshot,
    };
    let target = ObservationTarget {
        logical_run_id: descriptor.logical_run_id.clone(),
        attempt_id: descriptor.attempt_id.clone(),
        parent_attempt_id: descriptor.parent_attempt_id.clone(),
        launch_nonce: descriptor.launch_nonce.clone(),
        workload_argv: descriptor.workload_argv.clone(),
        journal_path: descriptor.journal_path.clone(),
        stdout_path: descriptor.stdout_path.clone(),
        stderr_path: descriptor.stderr_path.clone(),
        wrapper,
    };
    if let Err(error) = validate_correlation(&target, &snapshot) {
        return ReadinessDecision::Invalid(error);
    }
    match snapshot.terminal {
        Some(TerminalEvidence::SpawnFailed { stage, error }) => {
            ReadinessDecision::StartupFailed { stage, error }
        }
        Some(TerminalEvidence::Exited(_)) if snapshot.child.is_some() => {
            ReadinessDecision::Ready(ReadinessEvidence {
                attempt_id: descriptor.attempt_id.clone(),
                launch_nonce: descriptor.launch_nonce.clone(),
                wrapper,
            })
        }
        None if snapshot.child.is_some() => ReadinessDecision::Ready(ReadinessEvidence {
            attempt_id: descriptor.attempt_id.clone(),
            launch_nonce: descriptor.launch_nonce.clone(),
            wrapper,
        }),
        Some(TerminalEvidence::Exited(_)) => ReadinessDecision::Invalid(ReadinessFailure::Conflict),
        None => ReadinessDecision::Pending,
    }
}

fn non_empty_os(value: &OsString, field: &'static str) -> Result<(), InvalidInput> {
    if value.is_empty() {
        Err(InvalidInput::Empty { field })
    } else {
        Ok(())
    }
}
fn non_empty(value: &str, field: &'static str) -> Result<(), InvalidInput> {
    if value.is_empty() {
        Err(InvalidInput::Empty { field })
    } else {
        Ok(())
    }
}

pub fn supervisor_command(
    descriptor: &LaunchDescriptor,
) -> Result<SupervisorCommand, InvalidInput> {
    non_empty_os(&descriptor.supervisor, "supervisor")?;
    non_empty(&descriptor.logical_run_id, "logical_run_id")?;
    non_empty(&descriptor.attempt_id, "attempt_id")?;
    non_empty(&descriptor.launch_nonce, "launch_nonce")?;
    if descriptor.workload_argv.is_empty() {
        return Err(InvalidInput::EmptyWorkload);
    }
    for (field, path) in [
        ("journal_path", &descriptor.journal_path),
        ("stdout_path", &descriptor.stdout_path),
        ("stderr_path", &descriptor.stderr_path),
    ] {
        if path.as_os_str().is_empty() {
            return Err(InvalidInput::Empty { field });
        }
    }
    if descriptor
        .workload_argv
        .iter()
        .any(|argument| argument.is_empty())
    {
        return Err(InvalidInput::Empty {
            field: "workload_argv",
        });
    }
    let mut arguments: Vec<OsString> = vec![
        "supervise".into(),
        "--logical-run-id".into(),
        descriptor.logical_run_id.clone().into(),
        "--attempt-id".into(),
        descriptor.attempt_id.clone().into(),
        "--launch-nonce".into(),
        descriptor.launch_nonce.clone().into(),
        "--journal-path".into(),
        descriptor.journal_path.clone().into_os_string(),
        "--stdout-path".into(),
        descriptor.stdout_path.clone().into_os_string(),
        "--stderr-path".into(),
        descriptor.stderr_path.clone().into_os_string(),
    ];
    if let Some(parent) = &descriptor.parent_attempt_id {
        non_empty(parent, "parent_attempt_id")?;
        arguments.extend(["--parent-attempt-id".into(), parent.clone().into()]);
    }
    arguments.push("--".into());
    arguments.extend(descriptor.workload_argv.iter().cloned().map(OsString::from));
    Ok(SupervisorCommand {
        executable: descriptor.supervisor.clone(),
        arguments,
    })
}

#[cfg(test)]
mod tests;
