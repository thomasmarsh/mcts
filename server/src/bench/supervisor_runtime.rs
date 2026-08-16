use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use mcts_bench::projects_attempt::LaunchResult;
use mcts_bench::supervised_launch::{
    classify_readiness, supervisor_command, LaunchDescriptor, ReadinessDecision, WrapperIdentity,
};

pub(crate) trait SupervisorPort: Send + Sync {
    fn launch(&self, descriptor: &LaunchDescriptor) -> LaunchResult;
}

impl<F> SupervisorPort for F
where
    F: Fn(&LaunchDescriptor) -> LaunchResult + Send + Sync,
{
    fn launch(&self, descriptor: &LaunchDescriptor) -> LaunchResult {
        self(descriptor)
    }
}

pub(crate) struct SupervisorRuntime {
    observation_window: Duration,
    registry_path: PathBuf,
}

impl SupervisorRuntime {
    pub(crate) fn new(registry_path: PathBuf) -> Self {
        Self {
            observation_window: Duration::from_millis(500),
            registry_path,
        }
    }

    fn outer_paths(descriptor: &LaunchDescriptor) -> (PathBuf, PathBuf) {
        let parent = descriptor
            .journal_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        (
            parent.join("supervisor.stdout.log"),
            parent.join("supervisor.stderr.log"),
        )
    }

    fn observe(&self, descriptor: &LaunchDescriptor, wrapper: WrapperIdentity) -> LaunchResult {
        let deadline = Instant::now() + self.observation_window;
        loop {
            match classify_readiness(
                descriptor,
                wrapper,
                mcts_bench::lifecycle::read_journal(&descriptor.journal_path),
            ) {
                ReadinessDecision::Ready(_) => return LaunchResult::Ready(wrapper),
                ReadinessDecision::StartupFailed { stage, error } => {
                    return LaunchResult::SpawnFailed(format!("{stage}: {error}"));
                }
                ReadinessDecision::Invalid(diagnostic) => {
                    return LaunchResult::Conflict {
                        wrapper,
                        diagnostic,
                    };
                }
                ReadinessDecision::Pending if Instant::now() >= deadline => {
                    return LaunchResult::Pending {
                        wrapper,
                        diagnostic: "supervisor readiness observation timed out".into(),
                    };
                }
                ReadinessDecision::Pending => thread::sleep(Duration::from_millis(10)),
            }
        }
    }
}

impl SupervisorPort for SupervisorRuntime {
    fn launch(&self, descriptor: &LaunchDescriptor) -> LaunchResult {
        let command = match supervisor_command(descriptor) {
            Ok(command) => command,
            Err(error) => return LaunchResult::SpawnFailed(format!("{error:?}")),
        };
        let (stdout, stderr) = Self::outer_paths(descriptor);
        match mcts_bench::launch::launch_supervisor(
            &descriptor.logical_run_id,
            &command,
            &stdout,
            &stderr,
            &descriptor.stdout_path,
            &self.registry_path,
            crate::BUILD_INFO,
        ) {
            Ok(wrapper) => self.observe(descriptor, wrapper),
            Err(error) => LaunchResult::SpawnFailed(error.to_string()),
        }
    }
}
