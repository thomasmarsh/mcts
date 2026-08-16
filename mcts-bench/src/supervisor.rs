//! One-workload lifecycle journal coordinator.
use crate::lifecycle::{named, validate_manifest, ExitEvidence, OutputClosure, WrapperManifest};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupervisorInput {
    pub manifest: WrapperManifest,
    pub launch_nonce: String,
    pub journal_path: PathBuf,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkloadRequest {
    pub argv: Vec<String>,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnFailure {
    pub stage: String,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaitFailure {
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupervisorOutcome {
    ChildExited(ExitEvidence),
    SpawnFailed(SpawnFailure),
    InvalidInput(InvalidInputReason),
    JournalFailed(JournalFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalFailure {
    Conflict,
    Persistence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidInputReason {
    Manifest,
    LaunchNonce,
    JournalPath,
    StdoutPath,
    StderrPath,
}

pub trait JournalPort {
    fn child_started(&mut self, child: ChildId, timestamp: String) -> Result<(), JournalFailure>;
    fn spawn_failed(
        &mut self,
        failure: &SpawnFailure,
        timestamp: String,
    ) -> Result<(), JournalFailure>;
    fn child_exited(&mut self, exit: ExitEvidence, timestamp: String)
        -> Result<(), JournalFailure>;
    fn outputs_closed(
        &mut self,
        outputs: Vec<OutputClosure>,
        timestamp: String,
    ) -> Result<(), JournalFailure>;
}

pub trait JournalFactory {
    type Journal: JournalPort;

    fn create(
        &mut self,
        input: &SupervisorInput,
        timestamp: String,
    ) -> Result<Self::Journal, JournalFailure>;
}

pub trait WorkloadPort {
    fn spawn(&mut self, request: &WorkloadRequest) -> Result<ChildId, SpawnFailure>;
    fn wait(&mut self, child: ChildId) -> Result<ExitEvidence, WaitFailure>;
    fn close_outputs(&mut self) -> Vec<OutputClosure>;
}

pub trait Clock {
    fn now(&mut self) -> String;
}

pub fn supervise<J, W, C>(
    input: &SupervisorInput,
    journals: &mut J,
    workload: &mut W,
    clock: &mut C,
) -> SupervisorOutcome
where
    J: JournalFactory,
    W: WorkloadPort,
    C: Clock,
{
    if let Err(reason) = input_is_valid(input) {
        return SupervisorOutcome::InvalidInput(reason);
    }
    let mut journal = match journals.create(input, clock.now()) {
        Ok(journal) => journal,
        Err(error) => return SupervisorOutcome::JournalFailed(error),
    };
    let request = WorkloadRequest {
        argv: input.manifest.argv.clone(),
        stdout_path: input.stdout_path.clone(),
        stderr_path: input.stderr_path.clone(),
    };
    let child = match workload.spawn(&request) {
        Ok(child) => child,
        Err(failure) => return finish_spawn_failure(&mut journal, workload, clock, failure),
    };
    if let Err(error) = journal.child_started(child, clock.now()) {
        return SupervisorOutcome::JournalFailed(error);
    }
    let exit = match workload.wait(child) {
        Ok(exit) => exit,
        Err(failure) => ExitEvidence::WaitFailed {
            error: failure.error,
        },
    };
    if let Err(error) = journal.child_exited(exit.clone(), clock.now()) {
        return SupervisorOutcome::JournalFailed(error);
    }
    let outputs = workload.close_outputs();
    if let Err(error) = journal.outputs_closed(outputs, clock.now()) {
        return SupervisorOutcome::JournalFailed(error);
    }
    SupervisorOutcome::ChildExited(exit)
}

fn finish_spawn_failure<J, W, C>(
    journal: &mut J,
    workload: &mut W,
    clock: &mut C,
    failure: SpawnFailure,
) -> SupervisorOutcome
where
    J: JournalPort,
    W: WorkloadPort,
    C: Clock,
{
    if let Err(error) = journal.spawn_failed(&failure, clock.now()) {
        return SupervisorOutcome::JournalFailed(error);
    }
    let outputs = workload.close_outputs();
    if let Err(error) = journal.outputs_closed(outputs, clock.now()) {
        return SupervisorOutcome::JournalFailed(error);
    }
    SupervisorOutcome::SpawnFailed(failure)
}

pub const WRAPPER_FAILURE_EXIT_CODE: i32 = 70;

pub fn exit_code(outcome: &SupervisorOutcome) -> i32 {
    match outcome {
        SupervisorOutcome::ChildExited(ExitEvidence::Code { code: 0 }) => 0,
        SupervisorOutcome::ChildExited(ExitEvidence::Code { code }) if (1..=255).contains(code) => {
            *code
        }
        SupervisorOutcome::ChildExited(_)
        | SupervisorOutcome::SpawnFailed(_)
        | SupervisorOutcome::InvalidInput(_)
        | SupervisorOutcome::JournalFailed(_) => WRAPPER_FAILURE_EXIT_CODE,
    }
}

fn input_is_valid(input: &SupervisorInput) -> Result<(), InvalidInputReason> {
    validate_manifest(&input.manifest).map_err(|_| InvalidInputReason::Manifest)?;
    named(&input.launch_nonce, "launch_nonce").map_err(|_| InvalidInputReason::LaunchNonce)?;
    (!input.journal_path.as_os_str().is_empty())
        .then_some(())
        .ok_or(InvalidInputReason::JournalPath)?;
    (!input.stdout_path.as_os_str().is_empty())
        .then_some(())
        .ok_or(InvalidInputReason::StdoutPath)?;
    (!input.stderr_path.as_os_str().is_empty())
        .then_some(())
        .ok_or(InvalidInputReason::StderrPath)
}

#[cfg(test)]
mod tests;
