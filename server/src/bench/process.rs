use std::io;
use std::path::PathBuf;

use mcts_bench::launch;

use super::BenchState;

#[derive(Debug)]
pub enum ProcessError {
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalOutcome {
    Sent,
    NotFound,
}

pub struct SpawnRequest {
    pub run_id: String,
    pub command: Vec<String>,
    pub kind: String,
    pub game: String,
    pub label: Option<String>,
}

pub struct SpawnedProcess {
    pub pid: u32,
    pub log_path: PathBuf,
}

pub fn signal_process_group(pid: i64) -> std::io::Result<()> {
    let mut command = process_group_signal_command(pid);
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else if !launch::is_alive(pid as u32) {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "process group no longer exists",
        ))
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("kill exited with status {status}"),
        ))
    }
}

pub(super) fn process_group_signal_command(pid: i64) -> std::process::Command {
    let mut command = std::process::Command::new("kill");
    command.arg("-TERM").arg(format!("-{pid}"));
    command
}

pub(super) trait ProcessController {
    fn spawn(&self, request: SpawnRequest) -> Result<SpawnedProcess, ProcessError>;
    fn signal_group(&self, pid: i64) -> Result<SignalOutcome, ProcessError>;
}

impl ProcessController for BenchState {
    fn spawn(&self, request: SpawnRequest) -> Result<SpawnedProcess, ProcessError> {
        (self.run_launcher)(
            request.run_id,
            request.command,
            request.kind,
            request.game,
            request.label,
        )
        .map(|run| SpawnedProcess {
            pid: run.pid,
            log_path: run.log_path,
        })
        .map_err(|error| ProcessError::Failed(error.to_string()))
    }

    fn signal_group(&self, pid: i64) -> Result<SignalOutcome, ProcessError> {
        #[cfg(unix)]
        {
            match (self.process_group_signaller)(pid) {
                Ok(()) => Ok(SignalOutcome::Sent),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    Ok(SignalOutcome::NotFound)
                }
                Err(error) => Err(ProcessError::Failed(error.to_string())),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = pid;
            Err(ProcessError::Failed(
                "process signalling is not supported on this platform".into(),
            ))
        }
    }
}
