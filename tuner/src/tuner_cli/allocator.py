"""The pure policy that orders foreground tuning work."""

from __future__ import annotations

from .artifacts import Manifest
from .cohort import accepted_candidates, pending_proposal
from .domain import (
    AllocationDecision,
    BeginValidation,
    Candidate,
    CompleteCohort,
    CompleteRun,
    DeepenCohort,
    DeepenCohortAllocation,
    EmitObservation,
    ExecutePair,
    IntroduceCandidate,
    IntroduceProposal,
    NoDecision,
    PairResult,
    PairTask,
    Phase,
    Proposal,
    ReplayState,
    ResolveProposal,
    ResourceAllocation,
    SelectFinalists,
    TaskPrefix,
)
from .identity import pair_task

ALLOCATION_POLICY_VERSION = "iterated-common-blocks-v1"


def decide_allocation(manifest: Manifest, state: ReplayState) -> AllocationDecision:
    if proposal := pending_proposal(state):
        return ResolveProposal(proposal.proposal_index)
    accepted = accepted_candidates(state)
    if len(accepted) < manifest.bootstrap_candidates:
        return IntroduceProposal()
    if task := pending_pair(manifest, state):
        return ExecutePair(task)
    if candidate := observation_due(manifest, state):
        return EmitObservation(candidate.candidate_id, pair_phase(state))
    if len(accepted) < manifest.cohort_size and _frontier(
        state, accepted, manifest.tuning_blocks[0].prefix_id
    ):
        return IntroduceProposal()
    if (
        state.cohort is None
        and len(accepted) == manifest.cohort_size
        and _frontier(state, accepted, active_prefix(manifest, state).prefix_id)
    ):
        if state.tuning_block_index + 1 < len(manifest.tuning_blocks):
            prefix = manifest.tuning_blocks[state.tuning_block_index + 1]
            return DeepenCohort(state.tuning_block_index + 1, prefix.prefix_id)
        return CompleteCohort()
    if state.cohort is not None and state.finalists is None:
        return SelectFinalists()
    if state.finalists is not None and _validation_complete(manifest, state):
        return CompleteRun()
    return NoDecision()


def resource_allocation(
    decision: AllocationDecision, manifest: Manifest, state: ReplayState
) -> ResourceAllocation | None:
    match decision:
        case IntroduceProposal():
            slot = len(accepted_candidates(state))
            return IntroduceCandidate(slot, manifest.source_schedule[slot])
        case DeepenCohort(block_index, prefix_id):
            return DeepenCohortAllocation(block_index, prefix_id)
        case SelectFinalists():
            return BeginValidation(manifest.tuning_prefix.prefix_id)
        case _:
            return None


def proposal_at(state: ReplayState, proposal_index: int) -> Proposal:
    return state.proposals[proposal_index]


def pair_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    return state.finalists if state.finalists is not None else accepted_candidates(state)


def pair_phase(state: ReplayState) -> Phase:
    return "validation" if state.finalists is not None else "tuning"


def active_prefix(manifest: Manifest, state: ReplayState) -> TaskPrefix:
    return (
        manifest.validation_prefix
        if pair_phase(state) == "validation"
        else manifest.tuning_blocks[state.tuning_block_index]
    )


def pending_pair(manifest: Manifest, state: ReplayState) -> PairTask | None:
    phase, prefix = pair_phase(state), active_prefix(manifest, state)
    start = (
        0
        if phase == "validation" or state.tuning_block_index == 0
        else manifest.tuning_blocks[state.tuning_block_index - 1].length
    )
    completed = {pair.task.pair_id for pair in state.completed_pairs}
    for case in manifest.prefix_cases(phase)[start : prefix.length]:
        for candidate in pair_candidates(state):
            task = pair_task(candidate, case, manifest.efforts[phase])
            if task.pair_id not in completed:
                return task
    return None


def matching_pairs(state: ReplayState, candidate: Candidate, phase: Phase) -> list[PairResult]:
    return [
        pair
        for pair in state.completed_pairs
        if pair.task.candidate_id == candidate.candidate_id and pair.task.task_case.phase == phase
    ]


def observation_due(manifest: Manifest, state: ReplayState) -> Candidate | None:
    phase, prefix = pair_phase(state), active_prefix(manifest, state)
    for candidate in pair_candidates(state):
        if (
            not _observed(state, candidate, phase, prefix.prefix_id)
            and len(matching_pairs(state, candidate, phase)) >= prefix.length
        ):
            return candidate
    return None


def _observed(state: ReplayState, candidate: Candidate, phase: Phase, prefix_id: str) -> bool:
    return any(
        item.candidate_id == candidate.candidate_id
        and item.phase == phase
        and item.context.task_prefix.prefix_id == prefix_id
        for item in state.observations
    )


def _frontier(state: ReplayState, candidates: tuple[Candidate, ...], prefix_id: str) -> bool:
    return bool(candidates) and all(
        _observed(state, candidate, "tuning", prefix_id) for candidate in candidates
    )


def _validation_complete(manifest: Manifest, state: ReplayState) -> bool:
    return all(
        _observed(state, candidate, "validation", manifest.validation_prefix.prefix_id)
        for candidate in state.finalists or ()
    )
