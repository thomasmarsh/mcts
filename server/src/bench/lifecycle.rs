use std::io::Write;
use std::sync::Arc;

#[cfg(test)]
use mcts_bench::projects_attempt::LaunchResult;
use mcts_bench::projects_attempt::{
    LaunchToken, ProjectsError, ProjectsRepository, StartAuthorization, StartRequest,
};
use mcts_bench::supervised_launch::LaunchDescriptor;

use super::{process, supervisor_runtime::SupervisorPort, BenchError, BenchState};

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

pub(crate) struct BenchRuntime {
    repository: Arc<dyn ProjectsRepository + Send + Sync>,
    supervisor: Arc<dyn SupervisorPort>,
    clock: Arc<dyn Clock>,
}
pub(super) struct RuntimeLaunch {
    pub(super) pid: u32,
    pub(super) diagnostic: Option<String>,
}
impl BenchRuntime {
    pub(crate) fn new(
        repository: Arc<dyn ProjectsRepository + Send + Sync>,
        supervisor: Arc<dyn SupervisorPort>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            repository,
            supervisor,
            clock,
        }
    }
    pub(super) fn start_projects(
        &self,
        request: StartRequest,
        descriptor: LaunchDescriptor,
    ) -> Result<RuntimeLaunch, ProjectsError> {
        match self.repository.authorize_start(&request, &descriptor)? {
            StartAuthorization::New => {}
            StartAuthorization::Replay(previous) => return recorded_launch(previous.result),
        }
        let result = self.supervisor.launch(&descriptor);
        let observed_at = self.clock.now();
        let recorded = self
            .repository
            .record_launch(&request.run_id, &result, &observed_at)?;
        recorded_launch(Some(recorded))
    }
}

fn recorded_launch(
    previous: Option<mcts_bench::projects_attempt::LaunchRecord>,
) -> Result<RuntimeLaunch, ProjectsError> {
    let Some(previous) = previous else {
        return Ok(RuntimeLaunch {
            pid: 0,
            diagnostic: Some("launch outcome remains pending".into()),
        });
    };
    match previous.token {
        LaunchToken::SpawnFailed => Err(ProjectsError::Storage(
            previous
                .diagnostic
                .unwrap_or_else(|| "supervisor spawn failed".into()),
        )),
        LaunchToken::Ready | LaunchToken::Pending | LaunchToken::Conflict => Ok(RuntimeLaunch {
            pid: previous.wrapper.map_or(0, |wrapper| wrapper.pid as u32),
            diagnostic: previous.diagnostic,
        }),
    }
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

    struct FakeRepo {
        events: Arc<Mutex<Vec<&'static str>>>,
        replay: bool,
        result: Mutex<Option<LaunchResult>>,
        timestamps: Arc<Mutex<Vec<String>>>,
    }
    impl ProjectsRepository for FakeRepo {
        fn authorize_start(
            &self,
            _: &StartRequest,
            _: &LaunchDescriptor,
        ) -> Result<StartAuthorization, ProjectsError> {
            self.events.lock().unwrap().push("authorize");
            if self.replay {
                Ok(StartAuthorization::Replay(
                    mcts_bench::projects_attempt::PreviousLaunch {
                        result: Some(mcts_bench::projects_attempt::LaunchRecord {
                            token: LaunchToken::Ready,
                            wrapper: Some(WrapperIdentity {
                                pid: 7,
                                process_group_id: 7,
                            }),
                            diagnostic: Some("prior".into()),
                        }),
                    },
                ))
            } else {
                Ok(StartAuthorization::New)
            }
        }
        fn record_launch(
            &self,
            _: &str,
            result: &LaunchResult,
            at: &str,
        ) -> Result<mcts_bench::projects_attempt::LaunchRecord, ProjectsError> {
            self.events.lock().unwrap().push("record");
            *self.result.lock().unwrap() = Some(result.clone());
            self.timestamps.lock().unwrap().push(at.into());
            Ok(match result {
                LaunchResult::Ready(wrapper) => mcts_bench::projects_attempt::LaunchRecord {
                    token: LaunchToken::Ready,
                    wrapper: Some(*wrapper),
                    diagnostic: None,
                },
                LaunchResult::SpawnFailed(message) => mcts_bench::projects_attempt::LaunchRecord {
                    token: LaunchToken::SpawnFailed,
                    wrapper: None,
                    diagnostic: Some(message.clone()),
                },
                LaunchResult::Pending {
                    wrapper,
                    diagnostic,
                } => mcts_bench::projects_attempt::LaunchRecord {
                    token: LaunchToken::Pending,
                    wrapper: Some(*wrapper),
                    diagnostic: Some(diagnostic.clone()),
                },
                LaunchResult::Conflict {
                    wrapper,
                    diagnostic,
                } => mcts_bench::projects_attempt::LaunchRecord {
                    token: LaunchToken::Conflict,
                    wrapper: Some(*wrapper),
                    diagnostic: Some(format!("{diagnostic:?}")),
                },
            })
        }
        fn load_stop_target(
            &self,
            _: &str,
        ) -> Result<mcts_bench::projects_attempt::StopTarget, ProjectsError> {
            unreachable!()
        }
        fn load_if_initialized(&self, _: &str) -> Result<Option<Receipt>, ProjectsError> {
            unreachable!()
        }
        fn observation_targets(
            &self,
        ) -> Result<Vec<mcts_bench::supervised_launch::ObservationTarget>, ProjectsError> {
            unreachable!()
        }
        fn request_operator_stop(
            &self,
            _: &str,
            _: &str,
        ) -> Result<mcts_bench::projects_attempt::StopAuthorization, ProjectsError> {
            unreachable!()
        }
        fn observe_signal(&self, _: &str, _: &str) -> Result<Receipt, ProjectsError> {
            unreachable!()
        }
        fn observe_exit(
            &self,
            _: &str,
            _: mcts_bench::orchestration::ExitObservation,
            _: &str,
        ) -> Result<mcts_bench::projects_attempt::ExitAuthorization, ProjectsError> {
            unreachable!()
        }
        fn finalize_output(&self, _: &str, _: &str) -> Result<Receipt, ProjectsError> {
            unreachable!()
        }
        fn project_stop(&self, _: &str, _: &str) -> Result<(), ProjectsError> {
            unreachable!()
        }
    }
    fn request() -> StartRequest {
        StartRequest {
            run_id: "run".into(),
            game: None,
            project_id: "p".into(),
            experiment_id: "e".into(),
            spec_json: "{}".into(),
            label: "x".into(),
            git_sha: "s".into(),
            git_dirty: false,
            host: "h".into(),
            started_at: "start".into(),
            log_path: "log".into(),
            cells: vec![],
        }
    }
    fn descriptor() -> LaunchDescriptor {
        LaunchDescriptor {
            supervisor: "bench".into(),
            logical_run_id: "run".into(),
            attempt_id: "run".into(),
            parent_attempt_id: None,
            launch_nonce: "n".into(),
            workload_argv: vec!["work".into()],
            journal_path: "journal".into(),
            stdout_path: "out".into(),
            stderr_path: "err".into(),
        }
    }
    #[test]
    fn start_orders_effect_clock_and_each_result() {
        for result in [
            LaunchResult::Ready(WrapperIdentity {
                pid: 4,
                process_group_id: 4,
            }),
            LaunchResult::SpawnFailed("no".into()),
            LaunchResult::Pending {
                wrapper: WrapperIdentity {
                    pid: 4,
                    process_group_id: 4,
                },
                diagnostic: "wait".into(),
            },
            LaunchResult::Conflict {
                wrapper: WrapperIdentity {
                    pid: 4,
                    process_group_id: 4,
                },
                diagnostic: mcts_bench::supervised_launch::ReadinessFailure::Conflict,
            },
        ] {
            let events = Arc::new(Mutex::new(Vec::new()));
            let timestamps = Arc::new(Mutex::new(Vec::new()));
            let repo = Arc::new(FakeRepo {
                events: events.clone(),
                replay: false,
                result: Mutex::new(None),
                timestamps: timestamps.clone(),
            });
            let launch_result = result.clone();
            let launch_events = events.clone();
            let runtime = BenchRuntime::new(
                repo.clone(),
                Arc::new(move |_: &LaunchDescriptor| {
                    launch_events.lock().unwrap().push("launch");
                    launch_result.clone()
                }),
                Arc::new(FakeClock {
                    events: events.clone(),
                }),
            );
            let returned = runtime.start_projects(request(), descriptor());
            assert_eq!(
                *events.lock().unwrap(),
                ["authorize", "launch", "clock", "record"]
            );
            assert_eq!(*timestamps.lock().unwrap(), ["after"]);
            assert_eq!(*repo.result.lock().unwrap(), Some(result.clone()));
            assert!(matches!(
                (result, returned),
                (LaunchResult::SpawnFailed(_), Err(_)) | (_, Ok(_))
            ));
        }
    }
    #[test]
    fn replay_returns_persisted_identity_without_launch() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let repo = Arc::new(FakeRepo {
            events: events.clone(),
            replay: true,
            result: Mutex::new(None),
            timestamps: Arc::new(Mutex::new(Vec::new())),
        });
        let runtime = BenchRuntime::new(
            repo,
            Arc::new(|_: &LaunchDescriptor| panic!("must not launch")),
            Arc::new(FakeClock {
                events: events.clone(),
            }),
        );
        assert_eq!(
            runtime.start_projects(request(), descriptor()).unwrap().pid,
            7
        );
        assert_eq!(*events.lock().unwrap(), ["authorize"]);
    }
    struct FakeClock {
        events: Arc<Mutex<Vec<&'static str>>>,
    }
    impl Clock for FakeClock {
        fn now(&self) -> String {
            self.events.lock().unwrap().push("clock");
            "after".into()
        }
    }

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
