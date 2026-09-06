//! Pure lifecycle state machine for one detached benchmark process attempt.

use std::fmt;

/// Lifecycle phase of one physical process attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptPhase {
    Planned,
    Starting,
    Running,
    StopRequested,
    AwaitingExit,
    Finalizing,
    Completed,
    Stopped,
    Crashed,
}

impl AttemptPhase {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Stopped | Self::Crashed)
    }
}

/// Durable reason for requesting an attempt stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Operator,
    BaselinePromotion,
}

/// Authoritative observation of how an attempt process ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitObservation {
    Exited { code: Option<i32> },
    Signaled { signal: i32 },
    Unavailable,
}

/// Domain fact delivered to the attempt transition kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptEvent {
    StartRequested,
    ProcessObserved,
    SpawnFailed,
    StopRequested { reason: StopReason },
    SignalObserved,
    ExitObserved { exit: ExitObservation },
    FinalOutputIngested,
}

/// Effect proposed by the attempt transition kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptAction {
    SpawnProcess,
    SignalProcessGroup,
    FinalizeOutput,
}

/// Replayable evidence and phase for one physical process attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttemptState {
    phase: AttemptPhase,
    stop_reason: Option<StopReason>,
    process_observed: bool,
    signal_observed: bool,
    exit: Option<ExitObservation>,
}

impl AttemptState {
    #[must_use]
    /// Create a new attempt before its start authorization is recorded.
    pub const fn planned() -> Self {
        Self {
            phase: AttemptPhase::Planned,
            stop_reason: None,
            process_observed: false,
            signal_observed: false,
            exit: None,
        }
    }

    #[must_use]
    /// Return the current lifecycle phase.
    pub const fn phase(self) -> AttemptPhase {
        self.phase
    }

    #[must_use]
    /// Return the durable stop reason, when one was recorded.
    pub const fn stop_reason(self) -> Option<StopReason> {
        self.stop_reason
    }

    #[must_use]
    /// Return whether the matching launched process was observed.
    pub const fn process_observed(self) -> bool {
        self.process_observed
    }

    #[must_use]
    /// Return whether a stop-signal attempt was durably observed.
    pub const fn signal_observed(self) -> bool {
        self.signal_observed
    }

    #[must_use]
    /// Return the recorded exit or loss observation, when present.
    pub const fn exit_observation(self) -> Option<ExitObservation> {
        self.exit
    }
}

#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Resulting state and effects from applying one attempt event.
pub struct AttemptTransition {
    state: AttemptState,
    actions: Vec<AttemptAction>,
}

impl AttemptTransition {
    #[must_use]
    /// Borrow the resulting attempt state.
    pub fn state(&self) -> &AttemptState {
        &self.state
    }

    #[must_use]
    /// Borrow the effects proposed by the transition.
    pub fn actions(&self) -> &[AttemptAction] {
        &self.actions
    }
}

/// Rejection explaining why an attempt event cannot be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptTransitionError {
    IllegalEvent {
        phase: AttemptPhase,
        event: AttemptEvent,
    },
    ConflictingStopReason {
        recorded: StopReason,
        incoming: StopReason,
    },
    ConflictingExitObservation {
        recorded: ExitObservation,
        incoming: ExitObservation,
    },
}

impl fmt::Display for AttemptTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IllegalEvent { phase, event } => {
                write!(f, "event {event:?} is illegal in phase {phase:?}")
            }
            Self::ConflictingStopReason { recorded, incoming } => {
                write!(
                    f,
                    "stop reason conflict: recorded {recorded:?}, incoming {incoming:?}"
                )
            }
            Self::ConflictingExitObservation { recorded, incoming } => {
                write!(
                    f,
                    "exit observation conflict: recorded {recorded:?}, incoming {incoming:?}"
                )
            }
        }
    }
}

impl std::error::Error for AttemptTransitionError {}

fn illegal(state: AttemptState, event: AttemptEvent) -> AttemptTransitionError {
    AttemptTransitionError::IllegalEvent {
        phase: state.phase,
        event,
    }
}

fn transition(state: AttemptState, actions: Vec<AttemptAction>) -> AttemptTransition {
    AttemptTransition { state, actions }
}

fn same_stop_or_conflict(
    state: AttemptState,
    reason: StopReason,
) -> Result<AttemptTransition, AttemptTransitionError> {
    match state.stop_reason {
        Some(recorded) if recorded != reason => {
            Err(AttemptTransitionError::ConflictingStopReason {
                recorded,
                incoming: reason,
            })
        }
        Some(_) => Ok(transition(state, vec![])),
        None => Err(illegal(state, AttemptEvent::StopRequested { reason })),
    }
}

/// Apply one domain fact and return the resulting state plus proposed effects.
pub fn transition_attempt(
    state: &AttemptState,
    event: AttemptEvent,
) -> Result<AttemptTransition, AttemptTransitionError> {
    let mut next = *state;
    match (state.phase, event) {
        (AttemptPhase::Planned, AttemptEvent::StartRequested) => {
            next.phase = AttemptPhase::Starting;
            Ok(transition(next, vec![AttemptAction::SpawnProcess]))
        }
        (AttemptPhase::Starting, AttemptEvent::StartRequested) => Ok(transition(next, vec![])),
        (AttemptPhase::Planned, AttemptEvent::StopRequested { reason }) => {
            next.stop_reason = Some(reason);
            next.phase = AttemptPhase::Stopped;
            Ok(transition(next, vec![]))
        }
        (AttemptPhase::Starting, AttemptEvent::ProcessObserved) => {
            next.process_observed = true;
            next.phase = AttemptPhase::Running;
            Ok(transition(next, vec![]))
        }
        (AttemptPhase::Starting, AttemptEvent::SpawnFailed) => {
            next.phase = AttemptPhase::Crashed;
            Ok(transition(next, vec![]))
        }
        (AttemptPhase::Starting, AttemptEvent::StopRequested { reason }) => {
            next.stop_reason = Some(reason);
            next.phase = AttemptPhase::StopRequested;
            Ok(transition(next, vec![]))
        }
        (AttemptPhase::Running, AttemptEvent::StopRequested { reason }) => {
            next.stop_reason = Some(reason);
            next.phase = AttemptPhase::StopRequested;
            Ok(transition(next, vec![AttemptAction::SignalProcessGroup]))
        }
        (AttemptPhase::StopRequested, AttemptEvent::StopRequested { reason })
        | (AttemptPhase::AwaitingExit, AttemptEvent::StopRequested { reason })
        | (AttemptPhase::Finalizing, AttemptEvent::StopRequested { reason })
        | (AttemptPhase::Stopped, AttemptEvent::StopRequested { reason }) => {
            same_stop_or_conflict(next, reason)
        }
        (AttemptPhase::StopRequested, AttemptEvent::SpawnFailed) if !state.process_observed => {
            next.phase = AttemptPhase::Crashed;
            Ok(transition(next, vec![]))
        }
        (AttemptPhase::StopRequested, AttemptEvent::ProcessObserved) if !state.process_observed => {
            next.process_observed = true;
            Ok(transition(next, vec![AttemptAction::SignalProcessGroup]))
        }
        (AttemptPhase::StopRequested, AttemptEvent::ProcessObserved) => {
            Ok(transition(next, vec![]))
        }
        (
            AttemptPhase::Starting
            | AttemptPhase::Running
            | AttemptPhase::StopRequested
            | AttemptPhase::AwaitingExit,
            AttemptEvent::ExitObserved { exit },
        ) => record_exit(next, exit),
        (AttemptPhase::StopRequested, AttemptEvent::SignalObserved) => {
            next.signal_observed = true;
            next.phase = AttemptPhase::AwaitingExit;
            Ok(transition(next, vec![]))
        }
        (
            AttemptPhase::AwaitingExit | AttemptPhase::Finalizing | AttemptPhase::Stopped,
            AttemptEvent::SignalObserved,
        ) if state.signal_observed => Ok(transition(next, vec![])),
        (AttemptPhase::Finalizing, AttemptEvent::FinalOutputIngested) => {
            next.phase = terminal_phase(next);
            Ok(transition(next, vec![]))
        }
        (
            AttemptPhase::Completed | AttemptPhase::Stopped | AttemptPhase::Crashed,
            AttemptEvent::FinalOutputIngested,
        ) if state.exit.is_some() => Ok(transition(next, vec![])),
        (AttemptPhase::Completed | AttemptPhase::Crashed, AttemptEvent::ExitObserved { exit })
            if state.exit == Some(exit) =>
        {
            Ok(transition(next, vec![]))
        }
        (AttemptPhase::Finalizing, AttemptEvent::ExitObserved { exit })
            if state.exit == Some(exit) =>
        {
            Ok(transition(next, vec![]))
        }
        (AttemptPhase::Stopped, AttemptEvent::ExitObserved { exit })
            if state.exit == Some(exit) =>
        {
            Ok(transition(next, vec![]))
        }
        (phase, AttemptEvent::ProcessObserved)
            if state.process_observed
                && matches!(
                    phase,
                    AttemptPhase::Running
                        | AttemptPhase::StopRequested
                        | AttemptPhase::AwaitingExit
                        | AttemptPhase::Finalizing
                ) =>
        {
            Ok(transition(next, vec![]))
        }
        (phase, AttemptEvent::ExitObserved { exit })
            if state.exit.is_some()
                && matches!(
                    phase,
                    AttemptPhase::Finalizing
                        | AttemptPhase::Completed
                        | AttemptPhase::Stopped
                        | AttemptPhase::Crashed
                ) =>
        {
            if state.exit == Some(exit) {
                Ok(transition(next, vec![]))
            } else {
                Err(AttemptTransitionError::ConflictingExitObservation {
                    recorded: state.exit.unwrap(),
                    incoming: exit,
                })
            }
        }
        (phase, AttemptEvent::StopRequested { reason })
            if state.stop_reason.is_some()
                && matches!(
                    phase,
                    AttemptPhase::StopRequested
                        | AttemptPhase::AwaitingExit
                        | AttemptPhase::Finalizing
                        | AttemptPhase::Stopped
                ) =>
        {
            same_stop_or_conflict(next, reason)
        }
        (_, AttemptEvent::ExitObserved { exit }) if state.exit.is_some() => {
            Err(AttemptTransitionError::ConflictingExitObservation {
                recorded: state.exit.unwrap(),
                incoming: exit,
            })
        }
        _ => Err(illegal(next, event)),
    }
}

fn record_exit(
    mut state: AttemptState,
    exit: ExitObservation,
) -> Result<AttemptTransition, AttemptTransitionError> {
    state.exit = Some(exit);
    state.phase = AttemptPhase::Finalizing;
    Ok(transition(state, vec![AttemptAction::FinalizeOutput]))
}

fn terminal_phase(state: AttemptState) -> AttemptPhase {
    if state.stop_reason.is_some() && state.signal_observed {
        AttemptPhase::Stopped
    } else if state.exit == Some(ExitObservation::Exited { code: Some(0) }) {
        AttemptPhase::Completed
    } else {
        AttemptPhase::Crashed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(state: AttemptState, event: AttemptEvent) -> AttemptState {
        transition_attempt(&state, event)
            .unwrap()
            .state()
            .to_owned()
    }

    fn natural(exit: ExitObservation) -> AttemptState {
        let state = apply(AttemptState::planned(), AttemptEvent::StartRequested);
        let state = apply(state, AttemptEvent::ProcessObserved);
        apply(state, AttemptEvent::ExitObserved { exit })
    }

    fn assert_replay(state: AttemptState, event: AttemptEvent) {
        let replay = transition_attempt(&state, event).unwrap();
        assert_eq!(*replay.state(), state);
        assert!(replay.actions().is_empty());
    }

    #[test]
    fn complete_paths_and_actions() {
        let planned = AttemptState::planned();
        let started = transition_attempt(&planned, AttemptEvent::StartRequested).unwrap();
        assert_eq!(started.state().phase(), AttemptPhase::Starting);
        assert_eq!(started.actions(), &[AttemptAction::SpawnProcess]);
        let running = transition_attempt(started.state(), AttemptEvent::ProcessObserved).unwrap();
        assert_eq!(running.state().phase(), AttemptPhase::Running);
        let finalizing = transition_attempt(
            running.state(),
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Exited { code: Some(0) },
            },
        )
        .unwrap();
        assert_eq!(finalizing.state().phase(), AttemptPhase::Finalizing);
        assert_eq!(finalizing.actions(), &[AttemptAction::FinalizeOutput]);
        let completed =
            transition_attempt(finalizing.state(), AttemptEvent::FinalOutputIngested).unwrap();
        assert_eq!(completed.state().phase(), AttemptPhase::Completed);

        for exit in [
            ExitObservation::Exited { code: Some(1) },
            ExitObservation::Exited { code: None },
            ExitObservation::Unavailable,
        ] {
            assert_eq!(
                apply(natural(exit), AttemptEvent::FinalOutputIngested).phase(),
                AttemptPhase::Crashed
            );
        }

        let crashed = apply(AttemptState::planned(), AttemptEvent::StartRequested);
        let crashed = apply(crashed, AttemptEvent::SpawnFailed);
        assert_eq!(crashed.phase(), AttemptPhase::Crashed);
        assert!(!crashed.process_observed());
        assert_eq!(crashed.exit_observation(), None);

        let stopped = apply(
            AttemptState::planned(),
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        assert_eq!(stopped.phase(), AttemptPhase::Stopped);
        assert_eq!(stopped.stop_reason(), Some(StopReason::Operator));

        let starting = apply(AttemptState::planned(), AttemptEvent::StartRequested);
        let starting = apply(
            starting,
            AttemptEvent::StopRequested {
                reason: StopReason::BaselinePromotion,
            },
        );
        assert_eq!(starting.phase(), AttemptPhase::StopRequested);
        let signalled = transition_attempt(&starting, AttemptEvent::ProcessObserved).unwrap();
        assert_eq!(signalled.actions(), &[AttemptAction::SignalProcessGroup]);
        let awaiting = apply(*signalled.state(), AttemptEvent::SignalObserved);
        let finalizing = transition_attempt(
            &awaiting,
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Exited { code: Some(0) },
            },
        )
        .unwrap();
        assert_eq!(
            apply(*finalizing.state(), AttemptEvent::FinalOutputIngested).phase(),
            AttemptPhase::Stopped
        );

        let running = apply(
            apply(AttemptState::planned(), AttemptEvent::StartRequested),
            AttemptEvent::ProcessObserved,
        );
        let stopping = transition_attempt(
            &running,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        )
        .unwrap();
        assert_eq!(stopping.actions(), &[AttemptAction::SignalProcessGroup]);
        assert_eq!(
            apply(*stopping.state(), AttemptEvent::SignalObserved).phase(),
            AttemptPhase::AwaitingExit
        );

        let intent = apply(
            running,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        let natural = apply(
            intent,
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Exited { code: Some(0) },
            },
        );
        assert_eq!(
            apply(natural, AttemptEvent::FinalOutputIngested).phase(),
            AttemptPhase::Completed
        );

        let failed = apply(
            apply(AttemptState::planned(), AttemptEvent::StartRequested),
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        let failed = apply(failed, AttemptEvent::SpawnFailed);
        assert_eq!(failed.phase(), AttemptPhase::Crashed);
        assert_eq!(failed.exit_observation(), None);
    }

    #[test]
    fn replays_are_exact_and_action_free() {
        let start = AttemptState::planned();
        let starting = transition_attempt(&start, AttemptEvent::StartRequested)
            .unwrap()
            .state()
            .to_owned();
        assert_replay(starting, AttemptEvent::StartRequested);

        let running = apply(starting, AttemptEvent::ProcessObserved);
        let stopping = apply(
            running,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        assert_replay(stopping, AttemptEvent::ProcessObserved);
        assert_replay(
            stopping,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        let awaiting = apply(stopping, AttemptEvent::SignalObserved);
        assert_replay(awaiting, AttemptEvent::SignalObserved);
        let finalizing = apply(
            awaiting,
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Unavailable,
            },
        );
        assert_replay(
            finalizing,
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Unavailable,
            },
        );
        let stopped = apply(finalizing, AttemptEvent::FinalOutputIngested);
        assert_replay(stopped, AttemptEvent::FinalOutputIngested);
        assert_replay(
            stopped,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        assert_replay(stopped, AttemptEvent::SignalObserved);

        for exit in [
            ExitObservation::Exited { code: Some(0) },
            ExitObservation::Exited { code: Some(1) },
        ] {
            let finalizing = natural(exit);
            let terminal = apply(finalizing, AttemptEvent::FinalOutputIngested);
            assert_replay(terminal, AttemptEvent::ExitObserved { exit });
            assert_replay(terminal, AttemptEvent::FinalOutputIngested);
        }
    }

    #[test]
    fn conflicts_are_typed() {
        let stopping = apply(
            apply(AttemptState::planned(), AttemptEvent::StartRequested),
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        assert!(matches!(
            transition_attempt(
                &stopping,
                AttemptEvent::StopRequested {
                    reason: StopReason::BaselinePromotion
                }
            ),
            Err(AttemptTransitionError::ConflictingStopReason { .. })
        ));
        let finalizing = apply(
            stopping,
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Unavailable,
            },
        );
        assert!(matches!(
            transition_attempt(
                &finalizing,
                AttemptEvent::ExitObserved {
                    exit: ExitObservation::Exited { code: Some(1) }
                }
            ),
            Err(AttemptTransitionError::ConflictingExitObservation { .. })
        ));
    }

    #[test]
    fn spawn_failure_after_stop_request_depends_on_process_observation() {
        let starting = apply(AttemptState::planned(), AttemptEvent::StartRequested);
        let pending_stop = apply(
            starting,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        assert_eq!(
            apply(pending_stop, AttemptEvent::SpawnFailed).phase(),
            AttemptPhase::Crashed
        );

        let running = apply(starting, AttemptEvent::ProcessObserved);
        let observed_stop = apply(
            running,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        assert!(matches!(
            transition_attempt(&observed_stop, AttemptEvent::SpawnFailed),
            Err(AttemptTransitionError::IllegalEvent {
                phase: AttemptPhase::StopRequested,
                event: AttemptEvent::SpawnFailed,
            })
        ));
    }

    #[test]
    fn every_phase_event_pair_has_an_explicit_decision() {
        let planned = AttemptState::planned();
        let starting = apply(planned, AttemptEvent::StartRequested);
        let running = apply(starting, AttemptEvent::ProcessObserved);
        let stopping = apply(
            running,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
        );
        let awaiting = apply(stopping, AttemptEvent::SignalObserved);
        let finalizing = apply(
            awaiting,
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Exited { code: Some(0) },
            },
        );
        let completed = apply(
            natural(ExitObservation::Exited { code: Some(0) }),
            AttemptEvent::FinalOutputIngested,
        );
        let stopped = apply(finalizing, AttemptEvent::FinalOutputIngested);
        let crashed = apply(
            apply(planned, AttemptEvent::StartRequested),
            AttemptEvent::SpawnFailed,
        );
        let states = [
            planned, starting, running, stopping, awaiting, finalizing, completed, stopped, crashed,
        ];
        let events = [
            AttemptEvent::StartRequested,
            AttemptEvent::ProcessObserved,
            AttemptEvent::SpawnFailed,
            AttemptEvent::StopRequested {
                reason: StopReason::Operator,
            },
            AttemptEvent::SignalObserved,
            AttemptEvent::ExitObserved {
                exit: ExitObservation::Exited { code: Some(0) },
            },
            AttemptEvent::FinalOutputIngested,
        ];
        let expected = [
            [true, false, false, true, false, false, false],
            [true, true, true, true, false, true, false],
            [false, true, false, true, false, true, false],
            [false, true, false, true, true, true, false],
            [false, true, false, true, true, true, false],
            [false, true, false, true, true, true, true],
            [false, false, false, false, false, true, true],
            [false, false, false, true, true, true, true],
            [false, false, false, false, false, false, false],
        ];
        for (row, state) in states.into_iter().enumerate() {
            for (column, event) in events.into_iter().enumerate() {
                let result = transition_attempt(&state, event);
                if expected[row][column] {
                    assert!(
                        result.is_ok(),
                        "expected transition for row {row}, column {column}: {result:?}"
                    );
                } else {
                    assert!(
                        matches!(result, Err(AttemptTransitionError::IllegalEvent { .. })),
                        "expected illegal event for row {row}, column {column}: {result:?}"
                    );
                }
            }
        }
        for phase in [
            AttemptPhase::Planned,
            AttemptPhase::Starting,
            AttemptPhase::Running,
            AttemptPhase::StopRequested,
            AttemptPhase::AwaitingExit,
            AttemptPhase::Finalizing,
            AttemptPhase::Completed,
            AttemptPhase::Stopped,
            AttemptPhase::Crashed,
        ] {
            assert_eq!(
                phase.is_terminal(),
                matches!(
                    phase,
                    AttemptPhase::Completed | AttemptPhase::Stopped | AttemptPhase::Crashed
                )
            );
        }
        assert!(transition_attempt(&completed, AttemptEvent::StartRequested).is_err());
        assert!(transition_attempt(&stopped, AttemptEvent::StartRequested).is_err());
        assert!(transition_attempt(&crashed, AttemptEvent::StartRequested).is_err());
        assert!(transition_attempt(&running, AttemptEvent::SignalObserved).is_err());
        assert!(transition_attempt(&running, AttemptEvent::SpawnFailed).is_err());
        assert!(transition_attempt(&running, AttemptEvent::FinalOutputIngested).is_err());
    }
}
