"""The single pure decision point for foreground run scheduling.

``decide_allocation`` is the sole source of every "what to do next" choice in a
foreground run. It is a total, pure function of the manifest and the replayed
state: no writer, target, filesystem, clock, or RNG. ``continuation.advance_one``
matches on its result and performs exactly one effect per branch.
"""

from __future__ import annotations

from .artifacts import Manifest
from .cohort import accepted_candidates, pending_proposal
from .domain import (
    AllocationDecision,
    Candidate,
    CompleteCohort,
    CompleteRun,
    EmitObservation,
    ExecutePair,
    IntroduceProposal,
    NoDecision,
    PairResult,
    PairTask,
    Phase,
    Proposal,
    ReplayState,
    ResolveProposal,
    SelectFinalists,
)
from .identity import pair_task

# Identifies the "same order as the pre-allocator continuation" contract. Not
# written anywhere yet; a later slice stamps it onto an allocation-decision
# evidence event.
ALLOCATION_POLICY_VERSION = "fixed-cohort-order-v1"


def decide_allocation(manifest: Manifest, state: ReplayState) -> AllocationDecision:
    if proposal := pending_proposal(state):
        return ResolveProposal(proposal.proposal_index)
    if task := pending_pair(manifest, state):
        return ExecutePair(task)
    if (due := observation_due(manifest, state)) is not None:
        candidate, phase = due
        return EmitObservation(candidate.candidate_id, phase)
    if cohort_due(manifest, state):
        return CompleteCohort()
    if finalists_due(state):
        return SelectFinalists()
    if run_due(state):
        return CompleteRun()
    if len(accepted_candidates(state)) < manifest.cohort_size:
        return IntroduceProposal()
    return NoDecision()


def proposal_at(state: ReplayState, proposal_index: int) -> Proposal:
    return state.proposals[proposal_index]


def pair_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    if state.finalists is not None:
        return state.finalists
    return accepted_candidates(state)


def pair_phase(state: ReplayState) -> Phase:
    return "validation" if state.finalists is not None else "tuning"


def pending_pair(manifest: Manifest, state: ReplayState) -> PairTask | None:
    if state.next_pair_id is None:
        return None
    for candidate in pair_candidates(state):
        for case in manifest.prefix_cases(pair_phase(state)):
            effort = manifest.efforts[pair_phase(state)]
            task = pair_task(candidate, case, effort)
            if task.pair_id == state.next_pair_id:
                return task
    raise ValueError("replay pending pair is not part of the frozen task plan")


def observation_candidate(manifest: Manifest, state: ReplayState) -> tuple[Candidate | None, Phase]:
    if state.finalists is not None:
        observed = {item.candidate_id for item in state.observations if item.phase == "validation"}
        return next(
            (item for item in state.finalists if item.candidate_id not in observed), None
        ), "validation"
    if len(accepted_candidates(state)) < manifest.bootstrap_candidates:
        return None, "tuning"
    observed = {item.candidate_id for item in state.observations if item.phase == "tuning"}
    return next(
        (item for item in accepted_candidates(state) if item.candidate_id not in observed), None
    ), "tuning"


def matching_pairs(state: ReplayState, candidate: Candidate, phase: Phase) -> list[PairResult]:
    return [
        pair
        for pair in state.completed_pairs
        if pair.task.candidate_id == candidate.candidate_id and pair.task.task_case.phase == phase
    ]


def observation_due(manifest: Manifest, state: ReplayState) -> tuple[Candidate, Phase] | None:
    candidate, phase = observation_candidate(manifest, state)
    if candidate is None:
        return None
    prefix = manifest.tuning_prefix if phase == "tuning" else manifest.validation_prefix
    if len(matching_pairs(state, candidate, phase)) != prefix.length:
        return None
    return candidate, phase


def cohort_due(manifest: Manifest, state: ReplayState) -> bool:
    accepted = accepted_candidates(state)
    tuning = tuple(item for item in state.observations if item.phase == "tuning")
    return (
        state.cohort is None
        and len(accepted) == manifest.cohort_size
        and len(tuning) == len(accepted)
    )


def finalists_due(state: ReplayState) -> bool:
    return state.cohort is not None and state.finalists is None


def run_due(state: ReplayState) -> bool:
    if state.finalists is None or state.terminal_status != "open":
        return False
    validation = [item for item in state.observations if item.phase == "validation"]
    return len(validation) == len(state.finalists)
