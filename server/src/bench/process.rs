use mcts_bench::launch;
use std::io;

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
    fn signal_group(&self, pid: i64) -> Result<SignalOutcome, ProcessError>;
}

impl ProcessController for BenchState {
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
