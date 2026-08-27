use std::io::Write;
use std::sync::Arc;

#[cfg(test)]
use mcts_bench::projects_attempt::LaunchResult;
use mcts_bench::projects_attempt::{
    LaunchToken, ProjectsError, ProjectsRepository, StartAuthorization, StartRequest,
};
use mcts_bench::supervised_launch::LaunchDescriptor;

use super::{process, BenchError, BenchState};

pub(super) struct StopOutcome {
    pub(super) pid: Option<i64>,
    pub(super) prior_status: String,
    pub(super) signal_sent: bool,
}
pub(crate) trait Clock: Send + Sync {
    fn now(&self) -> String;
}
pub(crate) struct SystemClock;
impl Clock for SystemClock {
    fn now(&self) -> String {
        super::iso_timestamp_now()
    }
}
#[derive(Debug)]
pub(super) enum StopError {
    Attempt(ProjectsError),
    Process(String),
    NotTyped,
}

pub(super) fn stop_projects(
    repo: &dyn ProjectsRepository,
    process: &dyn process::ProcessController,
    run_id: &str,
    clock: &dyn Clock,
) -> Result<StopOutcome, StopError> {
    let target = repo.load_stop_target(run_id).map_err(StopError::Attempt)?;
    if !target.typed {
        return Err(StopError::NotTyped);
    }
    if target.status != "running" {
        repo.load_if_initialized(run_id)
            .map_err(StopError::Attempt)?;
        return Ok(StopOutcome {
            pid: target.pid,
            prior_status: target.status,
            signal_sent: false,
        });
    }
    let should_signal = repo
        .request_operator_stop(run_id, &clock.now())
        .map_err(StopError::Attempt)?
        .signal_process_group;
    let signal_sent = if should_signal {
        match target.pid {
            Some(pid) => match process.signal_group(pid) {
                Ok(process::SignalOutcome::Sent) => {
                    repo.observe_signal(run_id, &clock.now())
                        .map_err(StopError::Attempt)?;
                    true
                }
                Ok(process::SignalOutcome::NotFound) => false,
                Err(process::ProcessError::Failed(message)) => {
                    return Err(StopError::Process(message))
                }
            },
            None => false,
        }
    } else {
        false
    };
    repo.project_stop(run_id, &clock.now())
        .map_err(StopError::Attempt)?;
    Ok(StopOutcome {
        pid: target.pid,
        prior_status: target.status,
        signal_sent,
    })
}

pub(super) async fn stop_run_impl(
    state: &Arc<BenchState>,
    run_id: &str,
    clock: &dyn Clock,
) -> Result<StopOutcome, BenchError> {
    let target = state
        .projects_repository
        .load_stop_target(run_id)
        .map_err(super::attempt_bench_error)?;
    if target.typed {
        return stop_projects(
            state.projects_repository.as_ref(),
            state.as_ref(),
            run_id,
            clock,
        )
        .map_err(|error| stop_error(run_id, error));
    }
    if target.status != "running" {
        return Ok(StopOutcome {
            pid: target.pid,
            prior_status: target.status,
            signal_sent: false,
        });
    }
    let signal_sent = match target.pid {
        Some(pid) => match process::ProcessController::signal_group(state.as_ref(), pid) {
            Ok(process::SignalOutcome::Sent) => true,
            Ok(process::SignalOutcome::NotFound) => false,
            Err(error) => return Err(process_error(run_id, error)),
        },
        None => false,
    };
    let ended_at = super::project_legacy_stop(state, run_id, &target.kind)?;
    append_legacy_stop(state, run_id, ended_at);
    Ok(StopOutcome {
        pid: target.pid,
        prior_status: target.status,
        signal_sent,
    })
}
fn process_error(run_id: &str, error: process::ProcessError) -> BenchError {
    BenchError {
        status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        message: match error {
            process::ProcessError::Failed(message) => {
                format!("failed to signal run '{run_id}': {message}")
            }
        },
    }
}
fn stop_error(run_id: &str, error: StopError) -> BenchError {
    match error {
        StopError::Attempt(error) => super::attempt_bench_error(error),
        StopError::Process(message) => BenchError {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("failed to signal run '{run_id}': {message}"),
        },
        StopError::NotTyped => BenchError {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            message: "typed stop requested for a legacy run".into(),
        },
    }
}
fn append_legacy_stop(state: &Arc<BenchState>, run_id: &str, ended_at: String) {
    let path = state.bench_runs_dir.join("registry.log");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let mut line = mcts_bench::log::RegistryEvent::Stop {
            run_id: run_id.to_owned(),
            exit_code: None,
            ended_at,
        }
        .to_json_line();
        line.push('\n');
        let _ = file.write_all(line.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts_bench::orchestration::{AttemptState, ExitObservation};
    use mcts_bench::projects_attempt::{
        ExitAuthorization, Receipt, StartAuthorization, StopAuthorization, StopTarget,
    };
    use mcts_bench::supervised_launch::WrapperIdentity;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    struct FakeStopRepo {
        events: Arc<Mutex<Vec<&'static str>>>,
        status: Mutex<String>,
        timestamps: Mutex<Vec<(&'static str, String)>>,
    }

    impl FakeStopRepo {
        fn new(events: Arc<Mutex<Vec<&'static str>>>) -> Self {
            Self {
                events,
                status: Mutex::new("running".into()),
                timestamps: Mutex::new(Vec::new()),
            }
        }
    }

    impl ProjectsRepository for FakeStopRepo {
        fn authorize_start(
            &self,
            _: &StartRequest,
            _: &LaunchDescriptor,
        ) -> Result<StartAuthorization, ProjectsError> {
            unreachable!()
        }

        fn record_launch(
            &self,
            _: &str,
            _: &LaunchResult,
            _: &str,
        ) -> Result<mcts_bench::projects_attempt::LaunchRecord, ProjectsError> {
            unreachable!()
        }

        fn load_stop_target(&self, _: &str) -> Result<StopTarget, ProjectsError> {
            self.events.lock().unwrap().push("load");
            Ok(StopTarget {
                pid: Some(42),
                status: self.status.lock().unwrap().clone(),
                kind: "experiment".into(),
                typed: true,
            })
        }

        fn load_if_initialized(&self, _: &str) -> Result<Option<Receipt>, ProjectsError> {
            self.events.lock().unwrap().push("initialized");
            Ok(Some(receipt()))
        }

        fn observation_targets(
            &self,
        ) -> Result<Vec<mcts_bench::supervised_launch::ObservationTarget>, ProjectsError> {
            unreachable!()
        }

        fn request_operator_stop(
            &self,
            _: &str,
            at: &str,
        ) -> Result<StopAuthorization, ProjectsError> {
            self.events.lock().unwrap().push("stop");
            self.timestamps.lock().unwrap().push(("request", at.into()));
            Ok(StopAuthorization {
                signal_process_group: true,
            })
        }

        fn observe_signal(&self, _: &str, at: &str) -> Result<Receipt, ProjectsError> {
            self.events.lock().unwrap().push("observed");
            self.timestamps.lock().unwrap().push(("signal", at.into()));
            Ok(receipt())
        }

        fn observe_exit(
            &self,
            _: &str,
            _: ExitObservation,
            _: &str,
        ) -> Result<ExitAuthorization, ProjectsError> {
            unreachable!()
        }

        fn finalize_output(&self, _: &str, _: &str) -> Result<Receipt, ProjectsError> {
            unreachable!()
        }

        fn project_stop(&self, _: &str, at: &str) -> Result<(), ProjectsError> {
            self.events.lock().unwrap().push("project");
            self.timestamps
                .lock()
                .unwrap()
                .push(("projection", at.into()));
            *self.status.lock().unwrap() = "stopped".into();
            Ok(())
        }
    }

    fn receipt() -> Receipt {
        Receipt {
            state: AttemptState::planned(),
            version: 1,
            replay: false,
        }
    }

    struct SequenceClock {
        events: Arc<Mutex<Vec<&'static str>>>,
        values: Mutex<VecDeque<String>>,
    }

    impl SequenceClock {
        fn new(events: Arc<Mutex<Vec<&'static str>>>, values: &[&str]) -> Self {
            Self {
                events,
                values: Mutex::new(values.iter().map(|value| (*value).into()).collect()),
            }
        }
    }

    impl Clock for SequenceClock {
        fn now(&self) -> String {
            self.events.lock().unwrap().push("clock");
            self.values.lock().unwrap().pop_front().unwrap()
        }
    }

    enum FakeSignal {
        Sent,
        NotFound,
        Failed,
    }

    struct FakeProcess {
        events: Arc<Mutex<Vec<&'static str>>>,
        result: FakeSignal,
    }

    impl process::ProcessController for FakeProcess {
        fn signal_group(&self, pid: i64) -> Result<process::SignalOutcome, process::ProcessError> {
            assert_eq!(pid, 42);
            self.events.lock().unwrap().push("signal");
            match self.result {
                FakeSignal::Sent => Ok(process::SignalOutcome::Sent),
                FakeSignal::NotFound => Ok(process::SignalOutcome::NotFound),
                FakeSignal::Failed => Err(process::ProcessError::Failed("denied".into())),
            }
        }
    }

    #[test]
    fn stop_signals_once_and_preserves_not_found_and_error_semantics() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let repo = FakeStopRepo::new(events.clone());
        let process = FakeProcess {
            events: events.clone(),
            result: FakeSignal::Sent,
        };
        let clock = SequenceClock::new(events.clone(), &["request", "signal", "projection"]);
        let outcome = stop_projects(&repo, &process, "run", &clock).unwrap();
        assert!(outcome.signal_sent);
        assert_eq!(
            *events.lock().unwrap(),
            ["load", "clock", "stop", "signal", "clock", "observed", "clock", "project"]
        );
        assert_eq!(
            *repo.timestamps.lock().unwrap(),
            [
                ("request", "request".into()),
                ("signal", "signal".into()),
                ("projection", "projection".into())
            ]
        );
        assert!(
            !stop_projects(&repo, &process, "run", &clock)
                .unwrap()
                .signal_sent
        );
        assert_eq!(
            events
                .lock()
                .unwrap()
                .iter()
                .filter(|event| **event == "signal")
                .count(),
            1
        );

        let events = Arc::new(Mutex::new(Vec::new()));
        let repo = FakeStopRepo::new(events.clone());
        let process = FakeProcess {
            events: events.clone(),
            result: FakeSignal::NotFound,
        };
        let clock = SequenceClock::new(events.clone(), &["request", "projection"]);
        let outcome = stop_projects(&repo, &process, "run", &clock).unwrap();
        assert!(!outcome.signal_sent);
        assert_eq!(
            *events.lock().unwrap(),
            ["load", "clock", "stop", "signal", "clock", "project"]
        );
        assert_eq!(
            *repo.timestamps.lock().unwrap(),
            [
                ("request", "request".into()),
                ("projection", "projection".into())
            ]
        );

        let events = Arc::new(Mutex::new(Vec::new()));
        let repo = FakeStopRepo::new(events.clone());
        let process = FakeProcess {
            events: events.clone(),
            result: FakeSignal::Failed,
        };
        let clock = SequenceClock::new(events.clone(), &["request"]);
        assert!(matches!(
            stop_projects(&repo, &process, "run", &clock),
            Err(StopError::Process(message)) if message == "denied"
        ));
        assert_eq!(*events.lock().unwrap(), ["load", "clock", "stop", "signal"]);
        assert_eq!(
            *repo.timestamps.lock().unwrap(),
            [("request", "request".into())]
        );
    }
}
