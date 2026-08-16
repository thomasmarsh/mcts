//! Typed launch and lifecycle-readiness decisions for one detached supervisor.
use crate::lifecycle::{InvalidReason, JournalRead, TerminalEvidence};
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
pub enum InvalidInput {
    Empty { field: &'static str },
    EmptyWorkload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnFailure {
    Unsupported,
    Failed { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadinessFailure {
    StartupFailure,
    EarlyExit,
    Timeout,
    UnavailableEvidence,
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
    let manifest = &snapshot.manifest;
    if manifest.logical_run_id != descriptor.logical_run_id {
        return ReadinessDecision::Invalid(ReadinessFailure::LogicalRunIdMismatch {
            expected: descriptor.logical_run_id.clone(),
            observed: manifest.logical_run_id.clone(),
        });
    }
    if manifest.attempt_id != descriptor.attempt_id {
        return ReadinessDecision::Invalid(ReadinessFailure::AttemptMismatch {
            expected: descriptor.attempt_id.clone(),
            observed: manifest.attempt_id.clone(),
        });
    }
    if manifest.parent_attempt_id != descriptor.parent_attempt_id {
        return ReadinessDecision::Invalid(ReadinessFailure::ParentAttemptIdMismatch {
            expected: descriptor.parent_attempt_id.clone(),
            observed: manifest.parent_attempt_id.clone(),
        });
    }
    if snapshot.launch_nonce != descriptor.launch_nonce {
        return ReadinessDecision::Invalid(ReadinessFailure::NonceMismatch {
            expected: descriptor.launch_nonce.clone(),
            observed: snapshot.launch_nonce,
        });
    }
    if manifest.argv != descriptor.workload_argv {
        return ReadinessDecision::Invalid(ReadinessFailure::WorkloadMismatch {
            expected: descriptor.workload_argv.clone(),
            observed: manifest.argv.clone(),
        });
    }
    if manifest.wrapper_pid != wrapper.pid {
        return ReadinessDecision::Invalid(ReadinessFailure::WrapperPidMismatch {
            expected: wrapper.pid,
            observed: manifest.wrapper_pid,
        });
    }
    if manifest.process_group_id != wrapper.process_group_id {
        return ReadinessDecision::Invalid(ReadinessFailure::ProcessGroupMismatch {
            expected: wrapper.process_group_id,
            observed: manifest.process_group_id,
        });
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadySupervisor {
    pub wrapper: WrapperIdentity,
    pub launch_nonce: String,
    pub journal_path: PathBuf,
    pub workload_argv: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchOutcome {
    Invalid(InvalidInput),
    SpawnFailed(SpawnFailure),
    Ready(ReadySupervisor),
    Unready {
        wrapper: WrapperIdentity,
        failure: ReadinessFailure,
    },
}

pub trait DetachedSupervisorPort {
    fn spawn_detached(
        &mut self,
        command: &SupervisorCommand,
    ) -> Result<WrapperIdentity, SpawnFailure>;
    fn observe_readiness(
        &mut self,
        descriptor: &LaunchDescriptor,
        wrapper: WrapperIdentity,
    ) -> Result<ReadinessEvidence, ReadinessFailure>;
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

pub fn launch<P: DetachedSupervisorPort>(
    descriptor: &LaunchDescriptor,
    port: &mut P,
) -> LaunchOutcome {
    let command = match supervisor_command(descriptor) {
        Ok(command) => command,
        Err(error) => return LaunchOutcome::Invalid(error),
    };
    let wrapper = match port.spawn_detached(&command) {
        Ok(wrapper) => wrapper,
        Err(error) => return LaunchOutcome::SpawnFailed(error),
    };
    match port.observe_readiness(descriptor, wrapper) {
        Ok(evidence) if evidence.attempt_id != descriptor.attempt_id => LaunchOutcome::Unready {
            wrapper,
            failure: ReadinessFailure::AttemptMismatch {
                expected: descriptor.attempt_id.clone(),
                observed: evidence.attempt_id,
            },
        },
        Ok(evidence) if evidence.launch_nonce != descriptor.launch_nonce => {
            LaunchOutcome::Unready {
                wrapper,
                failure: ReadinessFailure::NonceMismatch {
                    expected: descriptor.launch_nonce.clone(),
                    observed: evidence.launch_nonce,
                },
            }
        }
        Ok(evidence) if evidence.wrapper.pid != wrapper.pid => LaunchOutcome::Unready {
            wrapper,
            failure: ReadinessFailure::WrapperPidMismatch {
                expected: wrapper.pid,
                observed: evidence.wrapper.pid,
            },
        },
        Ok(evidence) if evidence.wrapper.process_group_id != wrapper.process_group_id => {
            LaunchOutcome::Unready {
                wrapper,
                failure: ReadinessFailure::ProcessGroupMismatch {
                    expected: wrapper.process_group_id,
                    observed: evidence.wrapper.process_group_id,
                },
            }
        }
        Ok(_) => LaunchOutcome::Ready(ReadySupervisor {
            wrapper,
            launch_nonce: descriptor.launch_nonce.clone(),
            journal_path: descriptor.journal_path.clone(),
            workload_argv: descriptor.workload_argv.clone(),
        }),
        Err(failure) => LaunchOutcome::Unready { wrapper, failure },
    }
}

#[cfg(test)]
mod tests;
