use super::*;
use crate::lifecycle::{OutputClosure, WrapperManifest};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    Clock(String),
    Manifest,
    Spawn(WorkloadRequest),
    Started(ChildId, String),
    SpawnFailed(SpawnFailure, String),
    Wait(ChildId),
    Exited(ExitEvidence, String),
    Close,
    Outputs(Vec<OutputClosure>, String),
}
type Trace = Rc<RefCell<Vec<Event>>>;
#[derive(Clone, Copy)]
enum FailAt {
    Start,
    Terminal,
    Close,
}

struct Scenario {
    trace: Trace,
    spawn: Result<ChildId, SpawnFailure>,
    wait: Result<ExitEvidence, WaitFailure>,
    outputs: Vec<OutputClosure>,
    factory_failure: Option<JournalFailure>,
    journal_failure: Option<FailAt>,
    clocks: VecDeque<String>,
}
impl Scenario {
    fn success(exit: Result<ExitEvidence, WaitFailure>) -> Self {
        Self {
            trace: Rc::new(RefCell::new(vec![])),
            spawn: Ok(ChildId(4)),
            wait: exit,
            outputs: outputs(),
            factory_failure: None,
            journal_failure: None,
            clocks: VecDeque::from([
                "manifest".into(),
                "started".into(),
                "terminal".into(),
                "closed".into(),
            ]),
        }
    }
    fn events(&self) -> Vec<Event> {
        self.trace.borrow().clone()
    }
}
impl Clock for Scenario {
    fn now(&mut self) -> String {
        let value = self.clocks.pop_front().unwrap();
        self.trace.borrow_mut().push(Event::Clock(value.clone()));
        value
    }
}
struct Journal {
    trace: Trace,
    failure: Option<FailAt>,
}
impl JournalPort for Journal {
    fn child_started(&mut self, child: ChildId, timestamp: String) -> Result<(), JournalFailure> {
        self.trace
            .borrow_mut()
            .push(Event::Started(child, timestamp));
        if matches!(self.failure, Some(FailAt::Start)) {
            Err(JournalFailure::Persistence)
        } else {
            Ok(())
        }
    }
    fn spawn_failed(
        &mut self,
        failure: &SpawnFailure,
        timestamp: String,
    ) -> Result<(), JournalFailure> {
        self.trace
            .borrow_mut()
            .push(Event::SpawnFailed(failure.clone(), timestamp));
        if matches!(self.failure, Some(FailAt::Terminal)) {
            Err(JournalFailure::Persistence)
        } else {
            Ok(())
        }
    }
    fn child_exited(
        &mut self,
        exit: ExitEvidence,
        timestamp: String,
    ) -> Result<(), JournalFailure> {
        self.trace.borrow_mut().push(Event::Exited(exit, timestamp));
        if matches!(self.failure, Some(FailAt::Terminal)) {
            Err(JournalFailure::Persistence)
        } else {
            Ok(())
        }
    }
    fn outputs_closed(
        &mut self,
        outputs: Vec<OutputClosure>,
        timestamp: String,
    ) -> Result<(), JournalFailure> {
        self.trace
            .borrow_mut()
            .push(Event::Outputs(outputs, timestamp));
        if matches!(self.failure, Some(FailAt::Close)) {
            Err(JournalFailure::Persistence)
        } else {
            Ok(())
        }
    }
}
fn input() -> SupervisorInput {
    SupervisorInput {
        manifest: WrapperManifest {
            logical_run_id: "run".into(),
            attempt_id: "attempt".into(),
            parent_attempt_id: None,
            argv: vec!["worker".into(), "--literal space".into()],
            wrapper_pid: 1,
            process_group_id: 1,
            hostname: "host".into(),
            boot_id: None,
            process_start_id: None,
        },
        launch_nonce: "nonce".into(),
        journal_path: "journal".into(),
        stdout_path: "out path".into(),
        stderr_path: "err path".into(),
    }
}
fn outputs() -> Vec<OutputClosure> {
    vec![
        OutputClosure {
            path: "out path".into(),
            byte_length: Some(2),
        },
        OutputClosure {
            path: "err path".into(),
            byte_length: None,
        },
    ]
}
struct Journals {
    trace: Trace,
    factory_failure: Option<JournalFailure>,
    journal_failure: Option<FailAt>,
}
impl JournalFactory for Journals {
    type Journal = Journal;
    fn create(&mut self, _: &SupervisorInput, _: String) -> Result<Journal, JournalFailure> {
        self.trace.borrow_mut().push(Event::Manifest);
        match self.factory_failure {
            Some(error) => Err(error),
            None => Ok(Journal {
                trace: self.trace.clone(),
                failure: self.journal_failure,
            }),
        }
    }
}
struct Workload {
    trace: Trace,
    spawn: Result<ChildId, SpawnFailure>,
    wait: Result<ExitEvidence, WaitFailure>,
    outputs: Vec<OutputClosure>,
}
impl WorkloadPort for Workload {
    fn spawn(&mut self, request: &WorkloadRequest) -> Result<ChildId, SpawnFailure> {
        self.trace.borrow_mut().push(Event::Spawn(request.clone()));
        self.spawn.clone()
    }
    fn wait(&mut self, child: ChildId) -> Result<ExitEvidence, WaitFailure> {
        self.trace.borrow_mut().push(Event::Wait(child));
        self.wait.clone()
    }
    fn close_outputs(&mut self) -> Vec<OutputClosure> {
        self.trace.borrow_mut().push(Event::Close);
        self.outputs.clone()
    }
}
fn run(scenario: &mut Scenario) -> SupervisorOutcome {
    let input = input();
    let mut clock = Scenario {
        trace: scenario.trace.clone(),
        spawn: Ok(ChildId(0)),
        wait: Ok(ExitEvidence::Code { code: 0 }),
        outputs: vec![],
        factory_failure: None,
        journal_failure: None,
        clocks: std::mem::take(&mut scenario.clocks),
    };
    let mut journals = Journals {
        trace: scenario.trace.clone(),
        factory_failure: scenario.factory_failure,
        journal_failure: scenario.journal_failure,
    };
    let mut workload = Workload {
        trace: scenario.trace.clone(),
        spawn: scenario.spawn.clone(),
        wait: scenario.wait.clone(),
        outputs: scenario.outputs.clone(),
    };
    let outcome = supervise(&input, &mut journals, &mut workload, &mut clock);
    scenario.clocks = clock.clocks;
    outcome
}
fn journal(events: &[Event]) -> Vec<Event> {
    events
        .iter()
        .filter(|event| {
            matches!(
                event,
                Event::Started(..)
                    | Event::SpawnFailed(..)
                    | Event::Exited(..)
                    | Event::Outputs(..)
            )
        })
        .cloned()
        .collect()
}

#[test]
fn branches_write_exact_terminal_and_close_evidence() {
    let cases = [
        (
            Ok(ExitEvidence::Code { code: 0 }),
            SupervisorOutcome::ChildExited(ExitEvidence::Code { code: 0 }),
            ExitEvidence::Code { code: 0 },
        ),
        (
            Ok(ExitEvidence::Code { code: 7 }),
            SupervisorOutcome::ChildExited(ExitEvidence::Code { code: 7 }),
            ExitEvidence::Code { code: 7 },
        ),
        (
            Ok(ExitEvidence::Signal { signal: 9 }),
            SupervisorOutcome::ChildExited(ExitEvidence::Signal { signal: 9 }),
            ExitEvidence::Signal { signal: 9 },
        ),
        (
            Err(WaitFailure {
                error: "wait failed".into(),
            }),
            SupervisorOutcome::ChildExited(ExitEvidence::WaitFailed {
                error: "wait failed".into(),
            }),
            ExitEvidence::WaitFailed {
                error: "wait failed".into(),
            },
        ),
    ];
    for (wait, expected_outcome, expected_exit) in cases {
        let mut scenario = Scenario::success(wait);
        assert_eq!(run(&mut scenario), expected_outcome);
        let events = scenario.events();
        assert_eq!(
            events,
            vec![
                Event::Clock("manifest".into()),
                Event::Manifest,
                Event::Spawn(WorkloadRequest {
                    argv: vec!["worker".into(), "--literal space".into()],
                    stdout_path: "out path".into(),
                    stderr_path: "err path".into()
                }),
                Event::Clock("started".into()),
                Event::Started(ChildId(4), "started".into()),
                Event::Wait(ChildId(4)),
                Event::Clock("terminal".into()),
                Event::Exited(expected_exit, "terminal".into()),
                Event::Close,
                Event::Clock("closed".into()),
                Event::Outputs(outputs(), "closed".into())
            ]
        );
    }
}

#[test]
fn spawn_failure_never_claims_a_child_and_closes_declared_outputs() {
    let mut scenario = Scenario::success(Ok(ExitEvidence::Code { code: 0 }));
    scenario.spawn = Err(SpawnFailure {
        stage: "spawn".into(),
        error: "missing".into(),
    });
    assert_eq!(
        run(&mut scenario),
        SupervisorOutcome::SpawnFailed(SpawnFailure {
            stage: "spawn".into(),
            error: "missing".into()
        })
    );
    let evidence = journal(&scenario.events());
    assert_eq!(
        evidence,
        vec![
            Event::SpawnFailed(
                SpawnFailure {
                    stage: "spawn".into(),
                    error: "missing".into()
                },
                "started".into()
            ),
            Event::Outputs(outputs(), "terminal".into())
        ]
    );
    assert!(!scenario.events().iter().any(|event| matches!(
        event,
        Event::Started(..) | Event::Exited(..) | Event::Wait(_)
    )));
}

#[test]
fn journal_failure_stops_at_the_failed_record_without_repeat_effects() {
    for (failure, expected, waits) in [
        (FailAt::Start, 1, 0),
        (FailAt::Terminal, 2, 1),
        (FailAt::Close, 3, 1),
    ] {
        let mut scenario = Scenario::success(Ok(ExitEvidence::Code { code: 0 }));
        scenario.journal_failure = Some(failure);
        assert_eq!(
            run(&mut scenario),
            SupervisorOutcome::JournalFailed(JournalFailure::Persistence)
        );
        assert_eq!(journal(&scenario.events()).len(), expected);
        assert_eq!(
            scenario
                .events()
                .iter()
                .filter(|event| matches!(event, Event::Spawn(_)))
                .count(),
            1
        );
        assert_eq!(
            scenario
                .events()
                .iter()
                .filter(|event| matches!(event, Event::Wait(_)))
                .count(),
            waits
        );
    }
}

#[test]
fn conflict_and_invalid_input_prevent_spawn() {
    let mut conflict = Scenario::success(Ok(ExitEvidence::Code { code: 0 }));
    conflict.factory_failure = Some(JournalFailure::Conflict);
    assert_eq!(
        run(&mut conflict),
        SupervisorOutcome::JournalFailed(JournalFailure::Conflict)
    );
    assert_eq!(
        conflict.events(),
        vec![Event::Clock("manifest".into()), Event::Manifest]
    );
    let invalid = Scenario::success(Ok(ExitEvidence::Code { code: 0 }));
    let mut bad = input();
    bad.manifest.argv.clear();
    let mut clock = Scenario::success(Ok(ExitEvidence::Code { code: 0 }));
    let mut journals = Journals {
        trace: invalid.trace.clone(),
        factory_failure: None,
        journal_failure: None,
    };
    let mut workload = Workload {
        trace: invalid.trace.clone(),
        spawn: Ok(ChildId(4)),
        wait: Ok(ExitEvidence::Code { code: 0 }),
        outputs: outputs(),
    };
    assert!(matches!(
        supervise(&bad, &mut journals, &mut workload, &mut clock),
        SupervisorOutcome::InvalidInput(_)
    ));
    assert!(invalid.events().is_empty());
}

#[test]
fn exit_code_mapping_is_exhaustive() {
    assert_eq!(
        exit_code(&SupervisorOutcome::ChildExited(ExitEvidence::Code {
            code: 0
        })),
        0
    );
    assert_eq!(
        exit_code(&SupervisorOutcome::ChildExited(ExitEvidence::Code {
            code: 23
        })),
        23
    );
    for outcome in [
        SupervisorOutcome::ChildExited(ExitEvidence::Code { code: -1 }),
        SupervisorOutcome::ChildExited(ExitEvidence::Signal { signal: 9 }),
        SupervisorOutcome::ChildExited(ExitEvidence::WaitFailed { error: "x".into() }),
        SupervisorOutcome::SpawnFailed(SpawnFailure {
            stage: "x".into(),
            error: "x".into(),
        }),
        SupervisorOutcome::InvalidInput(InvalidInputReason::Manifest),
        SupervisorOutcome::JournalFailed(JournalFailure::Persistence),
    ] {
        assert_eq!(exit_code(&outcome), WRAPPER_FAILURE_EXIT_CODE);
    }
}
