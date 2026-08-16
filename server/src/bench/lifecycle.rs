use std::io::Write;
use std::sync::Arc;

use mcts_bench::projects_attempt::{ProjectsError, ProjectsRepository};

use super::{iso_timestamp_now, process, BenchError, BenchState};

pub(super) struct StopOutcome {
    pub(super) pid: Option<i64>,
    pub(super) prior_status: String,
    pub(super) signal_sent: bool,
}

#[derive(Debug)]
pub(super) enum StopError {
    Attempt(ProjectsError),
    Process(String),
    NotTyped,
}

#[derive(Debug)]
pub(super) enum LaunchError {
    Attempt(ProjectsError),
    Spawn(String),
}

pub(super) fn launch_projects(
    repo: &dyn ProjectsRepository,
    process: &dyn process::ProcessController,
    request: &mcts_bench::projects_attempt::StartRequest,
    spawn: process::SpawnRequest,
    failure_at: &str,
    observed_at: &str,
) -> Result<process::SpawnedProcess, LaunchError> {
    repo.create_and_request_start(request)
        .map_err(LaunchError::Attempt)?;
    let launched = match process.spawn(spawn) {
        Ok(launched) => launched,
        Err(process::ProcessError::Failed(message)) => {
            repo.observe_spawn_failure(&request.run_id, &message, failure_at)
                .map_err(LaunchError::Attempt)?;
            return Err(LaunchError::Spawn(message));
        }
    };
    repo.observe_process(
        &request.run_id,
        launched.pid as i64,
        &launched.log_path.to_string_lossy(),
        observed_at,
    )
    .map_err(LaunchError::Attempt)?;
    Ok(launched)
}

pub(super) fn stop_projects(
    repo: &dyn ProjectsRepository,
    process: &dyn process::ProcessController,
    run_id: &str,
    requested_at: &str,
    signal_at: &str,
    ended_at: &str,
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
        .request_operator_stop(run_id, requested_at)
        .map_err(StopError::Attempt)?
        .signal_process_group;
    let signal_sent = if should_signal {
        match target.pid {
            None => false,
            Some(pid) => match process.signal_group(pid) {
                Ok(process::SignalOutcome::Sent) => {
                    repo.observe_signal(run_id, signal_at)
                        .map_err(StopError::Attempt)?;
                    true
                }
                Ok(process::SignalOutcome::NotFound) => false,
                Err(process::ProcessError::Failed(message)) => {
                    return Err(StopError::Process(message));
                }
            },
        }
    } else {
        false
    };
    repo.project_stop(run_id, ended_at)
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
) -> Result<StopOutcome, BenchError> {
    let target = state
        .db
        .load_stop_target(run_id)
        .map_err(super::attempt_bench_error)?;
    if target.typed {
        let now = iso_timestamp_now();
        return stop_projects(&state.db, state.as_ref(), run_id, &now, &now, &now)
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
    let event = mcts_bench::log::RegistryEvent::Stop {
        run_id: run_id.to_owned(),
        exit_code: None,
        ended_at,
    };
    let path = state.bench_runs_dir.join("registry.log");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let mut line = event.to_json_line();
        line.push('\n');
        let _ = file.write_all(line.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use mcts_bench::orchestration::{AttemptState, ExitObservation};
    use mcts_bench::projects_attempt::{
        ExitAuthorization, LivenessTarget, ProjectsError, ProjectsRepository, Receipt,
        StartRequest, StopAuthorization, StopTarget,
    };

    use super::*;

    struct FakeRepository {
        events: Arc<Mutex<Vec<&'static str>>>,
        status: Mutex<String>,
        authorize_signal: Mutex<bool>,
    }

    impl FakeRepository {
        fn new(events: Arc<Mutex<Vec<&'static str>>>) -> Self {
            Self {
                events,
                status: Mutex::new("running".into()),
                authorize_signal: Mutex::new(true),
            }
        }

        fn mark(&self, event: &'static str) {
            self.events.lock().unwrap().push(event);
        }

        fn receipt() -> Receipt {
            Receipt {
                state: AttemptState::planned(),
                version: 0,
                replay: false,
            }
        }
    }

    impl ProjectsRepository for FakeRepository {
        fn load_stop_target(&self, _run_id: &str) -> Result<StopTarget, ProjectsError> {
            self.mark("load");
            Ok(StopTarget {
                pid: Some(42),
                status: self.status.lock().unwrap().clone(),
                kind: "experiment".into(),
                typed: true,
            })
        }

        fn load_if_initialized(&self, _run_id: &str) -> Result<Option<Receipt>, ProjectsError> {
            self.mark("validate");
            Ok(Some(Self::receipt()))
        }

        fn typed_liveness_targets(&self) -> Result<Vec<LivenessTarget>, ProjectsError> {
            Ok(Vec::new())
        }

        fn create_and_request_start(&self, _request: &StartRequest) -> Result<(), ProjectsError> {
            self.mark("create");
            Ok(())
        }

        fn observe_process(
            &self,
            _run_id: &str,
            _pid: i64,
            _log_path: &str,
            _observed_at: &str,
        ) -> Result<Receipt, ProjectsError> {
            self.mark("process");
            Ok(Self::receipt())
        }

        fn observe_spawn_failure(
            &self,
            _run_id: &str,
            _message: &str,
            _observed_at: &str,
        ) -> Result<Receipt, ProjectsError> {
            self.mark("spawn-failure");
            Ok(Self::receipt())
        }

        fn request_operator_stop(
            &self,
            _run_id: &str,
            _observed_at: &str,
        ) -> Result<StopAuthorization, ProjectsError> {
            self.mark("stop");
            let mut authorized = self.authorize_signal.lock().unwrap();
            let signal_process_group = *authorized;
            *authorized = false;
            Ok(StopAuthorization {
                signal_process_group,
            })
        }

        fn observe_signal(
            &self,
            _run_id: &str,
            _observed_at: &str,
        ) -> Result<Receipt, ProjectsError> {
            self.mark("observed");
            Ok(Self::receipt())
        }

        fn observe_exit(
            &self,
            _run_id: &str,
            _exit: ExitObservation,
            _ended_at: &str,
        ) -> Result<ExitAuthorization, ProjectsError> {
            Ok(ExitAuthorization {
                finalize_output: true,
                state: AttemptState::planned(),
            })
        }

        fn finalize_output(
            &self,
            _run_id: &str,
            _observed_at: &str,
        ) -> Result<Receipt, ProjectsError> {
            Ok(Self::receipt())
        }

        fn project_stop(&self, _run_id: &str, _ended_at: &str) -> Result<(), ProjectsError> {
            self.mark("project");
            *self.status.lock().unwrap() = "stopped".into();
            Ok(())
        }
    }

    struct FakeProcess {
        events: Arc<Mutex<Vec<&'static str>>>,
        signal: Result<process::SignalOutcome, String>,
        spawn: Result<(), String>,
    }

    impl process::ProcessController for FakeProcess {
        fn spawn(
            &self,
            _request: process::SpawnRequest,
        ) -> Result<process::SpawnedProcess, process::ProcessError> {
            self.events.lock().unwrap().push("spawn");
            self.spawn
                .clone()
                .map(|()| process::SpawnedProcess {
                    pid: 42,
                    log_path: "log".into(),
                })
                .map_err(process::ProcessError::Failed)
        }

        fn signal_group(&self, _pid: i64) -> Result<process::SignalOutcome, process::ProcessError> {
            self.events.lock().unwrap().push("signal");
            self.signal.clone().map_err(process::ProcessError::Failed)
        }
    }

    fn start_request() -> StartRequest {
        StartRequest {
            run_id: "run".into(),
            game: Some("nim".into()),
            project_id: "project".into(),
            experiment_id: "experiment".into(),
            spec_json: "{}".into(),
            label: "Run".into(),
            git_sha: "sha".into(),
            git_dirty: false,
            host: "host".into(),
            started_at: "start".into(),
            log_path: "log".into(),
            cells: Vec::new(),
        }
    }

    #[test]
    fn launch_commits_before_spawn_and_records_process() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let repo = FakeRepository::new(events.clone());
        let process = FakeProcess {
            events: events.clone(),
            signal: Ok(process::SignalOutcome::Sent),
            spawn: Ok(()),
        };
        launch_projects(
            &repo,
            &process,
            &start_request(),
            process::SpawnRequest {
                run_id: "run".into(),
                command: vec!["coordinator".into()],
                kind: "experiment".into(),
                game: "nim".into(),
                label: None,
            },
            "failure",
            "observed",
        )
        .unwrap();
        assert_eq!(*events.lock().unwrap(), ["create", "spawn", "process"]);
    }

    #[test]
    fn launch_failure_records_spawn_failure_after_effect() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let repo = FakeRepository::new(events.clone());
        let process = FakeProcess {
            events: events.clone(),
            signal: Ok(process::SignalOutcome::Sent),
            spawn: Err("spawn failed".into()),
        };
        assert!(matches!(
            launch_projects(
                &repo,
                &process,
                &start_request(),
                process::SpawnRequest {
                    run_id: "run".into(),
                    command: vec![],
                    kind: "experiment".into(),
                    game: "nim".into(),
                    label: None,
                },
                "failure",
                "observed",
            ),
            Err(LaunchError::Spawn(message)) if message == "spawn failed"
        ));
        assert_eq!(
            *events.lock().unwrap(),
            ["create", "spawn", "spawn-failure"]
        );
    }

    #[test]
    fn stop_signals_once_and_preserves_not_found_and_error_semantics() {
        for signal in [
            Ok(process::SignalOutcome::Sent),
            Ok(process::SignalOutcome::NotFound),
            Err("signal failed".into()),
        ] {
            let events = Arc::new(Mutex::new(Vec::new()));
            let repo = FakeRepository::new(events.clone());
            let process = FakeProcess {
                events: events.clone(),
                signal: signal.clone(),
                spawn: Ok(()),
            };
            let result = stop_projects(&repo, &process, "run", "requested", "signal", "ended");
            match signal {
                Ok(process::SignalOutcome::Sent) => {
                    assert!(result.is_ok());
                    assert_eq!(
                        *events.lock().unwrap(),
                        ["load", "stop", "signal", "observed", "project"]
                    );
                    let second =
                        stop_projects(&repo, &process, "run", "requested", "signal", "ended")
                            .unwrap();
                    assert!(!second.signal_sent);
                    assert_eq!(
                        events
                            .lock()
                            .unwrap()
                            .iter()
                            .filter(|event| **event == "signal")
                            .count(),
                        1
                    );
                }
                Ok(process::SignalOutcome::NotFound) => {
                    assert!(result.is_ok());
                    assert_eq!(
                        *events.lock().unwrap(),
                        ["load", "stop", "signal", "project"]
                    );
                }
                Err(_) => {
                    assert!(result.is_err());
                    assert_eq!(*events.lock().unwrap(), ["load", "stop", "signal"]);
                }
            }
        }
    }
}
