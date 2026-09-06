use mcts_bench::lifecycle::{
    ExitEvidence, LifecycleError, LifecycleWriter, OutputClosure, WrapperManifest,
};
use mcts_bench::supervisor::{
    ChildId, Clock, JournalFactory, JournalFailure, JournalPort, SpawnFailure, SupervisorInput,
    WaitFailure, WorkloadPort, WorkloadRequest,
};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

#[derive(clap::Args)]
pub struct Args {
    #[arg(long)]
    pub logical_run_id: String,
    #[arg(long)]
    pub attempt_id: String,
    #[arg(long)]
    pub parent_attempt_id: Option<String>,
    #[arg(long)]
    pub launch_nonce: String,
    #[arg(long)]
    pub journal_path: String,
    #[arg(long)]
    pub stdout_path: String,
    #[arg(long)]
    pub stderr_path: String,
    #[arg(trailing_var_arg = true, required = true)]
    pub workload: Vec<String>,
}

pub fn run(args: Args) {
    use mcts_bench::supervisor::{exit_code, supervise, SupervisorInput};
    let input = SupervisorInput {
        manifest: manifest(
            args.logical_run_id,
            args.attempt_id,
            args.parent_attempt_id,
            args.workload,
        ),
        launch_nonce: args.launch_nonce,
        journal_path: args.journal_path.into(),
        stdout_path: args.stdout_path.into(),
        stderr_path: args.stderr_path.into(),
    };
    let outcome = supervise(
        &input,
        &mut LifecycleJournals,
        &mut StdWorkload::new(),
        &mut SystemClock,
    );
    std::process::exit(exit_code(&outcome));
}

pub struct SystemClock;
impl Clock for SystemClock {
    fn now(&mut self) -> String {
        format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        )
    }
}

pub struct LifecycleJournals;
pub struct WriterJournal(LifecycleWriter);
impl JournalFactory for LifecycleJournals {
    type Journal = WriterJournal;
    fn create(
        &mut self,
        input: &SupervisorInput,
        timestamp: String,
    ) -> Result<Self::Journal, JournalFailure> {
        LifecycleWriter::create(
            &input.journal_path,
            input.manifest.clone(),
            input.launch_nonce.clone(),
            timestamp,
        )
        .map(WriterJournal)
        .map_err(journal_failure)
    }
}
impl JournalPort for WriterJournal {
    fn child_started(&mut self, child: ChildId, timestamp: String) -> Result<(), JournalFailure> {
        self.0
            .child_started(child.0, timestamp)
            .map_err(journal_failure)
    }
    fn spawn_failed(
        &mut self,
        failure: &SpawnFailure,
        timestamp: String,
    ) -> Result<(), JournalFailure> {
        self.0
            .child_spawn_failed(failure.stage.clone(), failure.error.clone(), timestamp)
            .map_err(journal_failure)
    }
    fn child_exited(
        &mut self,
        exit: ExitEvidence,
        timestamp: String,
    ) -> Result<(), JournalFailure> {
        self.0
            .child_exited(exit, timestamp)
            .map_err(journal_failure)
    }
    fn outputs_closed(
        &mut self,
        outputs: Vec<OutputClosure>,
        timestamp: String,
    ) -> Result<(), JournalFailure> {
        self.0
            .outputs_closed(outputs, timestamp)
            .map_err(journal_failure)
    }
}
fn journal_failure(error: LifecycleError) -> JournalFailure {
    if matches!(error, LifecycleError::Conflict { .. }) {
        JournalFailure::Conflict
    } else {
        JournalFailure::Persistence
    }
}

pub struct StdWorkload {
    child: Option<Child>,
    outputs: Option<(PathBuf, Option<File>, PathBuf, Option<File>)>,
}
impl StdWorkload {
    pub fn new() -> Self {
        Self {
            child: None,
            outputs: None,
        }
    }
}
impl WorkloadPort for StdWorkload {
    fn spawn(&mut self, request: &WorkloadRequest) -> Result<ChildId, SpawnFailure> {
        self.outputs = Some((
            request.stdout_path.clone(),
            None,
            request.stderr_path.clone(),
            None,
        ));
        let stdout = open_output(&request.stdout_path)?;
        self.outputs.as_mut().unwrap().1 = Some(stdout);
        let stderr = open_output(&request.stderr_path)?;
        self.outputs.as_mut().unwrap().3 = Some(stderr);
        let (stdout, stderr) = match self.outputs.as_mut().unwrap() {
            (_, Some(stdout), _, Some(stderr)) => (stdout, stderr),
            _ => unreachable!(),
        };
        let stdout_child = stdout.try_clone().map_err(|error| SpawnFailure {
            stage: "open_stdout".into(),
            error: error.to_string(),
        })?;
        let stderr_child = stderr.try_clone().map_err(|error| SpawnFailure {
            stage: "open_stderr".into(),
            error: error.to_string(),
        })?;
        let mut command = Command::new(&request.argv[0]);
        command
            .args(&request.argv[1..])
            .stdout(Stdio::from(stdout_child))
            .stderr(Stdio::from(stderr_child));
        let child = command.spawn().map_err(|error| SpawnFailure {
            stage: "spawn".into(),
            error: error.to_string(),
        })?;
        let id = ChildId(u64::from(child.id()));
        self.child = Some(child);
        Ok(id)
    }
    fn wait(&mut self, _: ChildId) -> Result<ExitEvidence, WaitFailure> {
        let child = self.child.as_mut().ok_or_else(|| WaitFailure {
            error: "child unavailable".into(),
        })?;
        child
            .wait()
            .map(exit_evidence)
            .map_err(|error| WaitFailure {
                error: error.to_string(),
            })
    }
    fn close_outputs(&mut self) -> Vec<OutputClosure> {
        let Some((stdout_path, stdout, stderr_path, stderr)) = self.outputs.take() else {
            return Vec::new();
        };
        drop(stdout);
        drop(stderr);
        vec![
            OutputClosure {
                byte_length: std::fs::metadata(&stdout_path).ok().map(|meta| meta.len()),
                path: stdout_path.to_string_lossy().into_owned(),
            },
            OutputClosure {
                byte_length: std::fs::metadata(&stderr_path).ok().map(|meta| meta.len()),
                path: stderr_path.to_string_lossy().into_owned(),
            },
        ]
    }
}
fn open_output(path: &PathBuf) -> Result<File, SpawnFailure> {
    OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .map_err(|error| SpawnFailure {
            stage: "open_output".into(),
            error: error.to_string(),
        })
}
fn exit_evidence(status: std::process::ExitStatus) -> ExitEvidence {
    if let Some(code) = status.code() {
        return ExitEvidence::Code { code };
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return ExitEvidence::Signal { signal };
        }
    }
    ExitEvidence::WaitFailed {
        error: "process ended without exit code".into(),
    }
}

pub fn manifest(
    logical_run_id: String,
    attempt_id: String,
    parent_attempt_id: Option<String>,
    argv: Vec<String>,
) -> WrapperManifest {
    WrapperManifest {
        logical_run_id,
        attempt_id,
        parent_attempt_id,
        argv,
        wrapper_pid: u64::from(std::process::id()),
        process_group_id: process_group_id(),
        hostname: hostname().unwrap_or_default(),
        boot_id: read_identity("/proc/sys/kernel/random/boot_id"),
        process_start_id: process_start_identity(),
    }
}
fn hostname() -> Option<String> {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|v| v.trim_end_matches(['\r', '\n']).to_owned())
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok().filter(|v| !v.is_empty()))
        .or_else(command_hostname)
}
fn read_identity(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}
fn process_start_identity() -> Option<String> {
    read_identity(&format!("/proc/{}/stat", std::process::id()))
        .and_then(|v| {
            v.rsplit_once(')')
                .and_then(|(_, fields)| fields.split_whitespace().nth(19).map(str::to_owned))
        })
        .or_else(process_start_from_ps)
}
fn command_hostname() -> Option<String> {
    Command::new("hostname")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}
#[cfg(unix)]
fn process_start_from_ps() -> Option<String> {
    Command::new("ps")
        .args(["-o", "lstart=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}
#[cfg(not(unix))]
fn process_start_from_ps() -> Option<String> {
    None
}
#[cfg(unix)]
fn process_group_id() -> u64 {
    unsafe extern "C" {
        fn getpgrp() -> i32;
    }
    unsafe { u64::try_from(getpgrp()).unwrap_or(u64::from(std::process::id())) }
}
#[cfg(not(unix))]
fn process_group_id() -> u64 {
    u64::from(std::process::id())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use mcts_bench::lifecycle::{read_journal, JournalRead, TerminalEvidence};
    use mcts_bench::supervisor::{supervise, SupervisorInput};

    #[test]
    fn unix_adapter_preserves_exit_and_signal_evidence() {
        for (script, expected) in [
            ("exit 0", ExitEvidence::Code { code: 0 }),
            ("exit 7", ExitEvidence::Code { code: 7 }),
            ("kill -TERM $$", ExitEvidence::Signal { signal: 15 }),
        ] {
            let root = std::env::temp_dir().join(format!(
                "mcts-supervise-{}-{}",
                std::process::id(),
                script.len()
            ));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir(&root).unwrap();
            let input = SupervisorInput {
                manifest: manifest(
                    "run".into(),
                    "attempt".into(),
                    None,
                    vec!["sh".into(), "-c".into(), script.into()],
                ),
                launch_nonce: "nonce".into(),
                journal_path: root.join("lifecycle.jsonl"),
                stdout_path: root.join("out"),
                stderr_path: root.join("err"),
            };
            let outcome = supervise(
                &input,
                &mut LifecycleJournals,
                &mut StdWorkload::new(),
                &mut SystemClock,
            );
            assert_eq!(
                outcome,
                mcts_bench::supervisor::SupervisorOutcome::ChildExited(expected.clone())
            );
            let JournalRead::Complete(snapshot) = read_journal(&input.journal_path).unwrap() else {
                panic!()
            };
            assert_eq!(snapshot.terminal, Some(TerminalEvidence::Exited(expected)));
            assert_eq!(
                snapshot.outputs,
                Some(vec![
                    OutputClosure {
                        path: input.stdout_path.to_string_lossy().into_owned(),
                        byte_length: Some(0)
                    },
                    OutputClosure {
                        path: input.stderr_path.to_string_lossy().into_owned(),
                        byte_length: Some(0)
                    }
                ])
            );
            let _ = std::fs::remove_dir_all(root);
        }
    }
}
