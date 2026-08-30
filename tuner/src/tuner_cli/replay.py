"""Strict replay of evidence into factual foreground-run state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .allocator import (
    ALLOCATION_POLICY_VERSION,
    decide_allocation,
    pending_pair,
    resource_allocation,
)
from .artifacts import Manifest, production_claim
from .cohort import accepted_candidates
from .domain import (
    Candidate,
    DeepenCohortAllocation,
    ExecutePair,
    IntroduceCandidate,
    Observation,
    ObservationContext,
    ObservationFrontier,
    PairResult,
    Phase,
    Proposal,
    ProposalProvenance,
    ReplayState,
    ResourceAllocation,
)
from .event_payloads import (
    AllocationDecidedPayload,
    CohortCompletedPayload,
    FinalistsSelectedPayload,
    ObservationCompletedPayload,
    PairCompletedPayload,
    PairFailedPayload,
    PairStartedPayload,
    ProposalAcceptedPayload,
    ProposalCreatedPayload,
    ProposalRejectedPayload,
    RunCompletedPayload,
    RunFailedPayload,
    RunInterruptedPayload,
)
from .evidence import EvidenceEvent, decode_pair_payload
from .identity import candidate_from_canonical_config
from .observations import comparable_prefix_observations, contextual_observation
from .proposer import POLICY_VERSION, empty_frontier, tuning_frontier
from .selection import select_finalists

Disposition = Literal["accepted", "rejected"]
Terminal = Literal["open", "configuration_failed", "complete"]


def _context(
    manifest: Manifest, phase: Phase, state: ReplayState | None = None
) -> ObservationContext:
    prefix = (
        manifest.validation_prefix
        if phase == "validation"
        else manifest.tuning_blocks[0 if state is None else state.tuning_block_index]
    )
    return ObservationContext(manifest.epoch.epoch_id, phase, prefix, manifest.efforts[phase])


def observation_payload(value: Observation, opponent_count: int) -> ObservationCompletedPayload:
    context, estimate = value.context, value.estimate
    return ObservationCompletedPayload(
        value.observation_id,
        value.candidate_id,
        context.phase,
        context.objective_epoch_id,
        context.task_prefix.corpus_id,
        context.task_prefix.prefix_id,
        context.task_prefix.task_ids,
        context.task_prefix.length,
        context.search_effort.max_iterations,
        value.pair_utilities,
        {"mean": estimate.mean, "lower": estimate.lower, "upper": estimate.upper},
        {
            "pairs": context.task_prefix.length,
            "games": context.task_prefix.length * 2,
            "opponents": opponent_count,
        },
    )


@dataclass(slots=True)
class _Replay:
    manifest: Manifest
    proposals: list[Proposal] = field(default_factory=lambda: list[Proposal]())
    dispositions: dict[int, Disposition] = field(default_factory=lambda: dict[int, Disposition]())
    completed: list[PairResult] = field(default_factory=lambda: list[PairResult]())
    observations: list[Observation] = field(default_factory=lambda: list[Observation]())
    cohort: tuple[Candidate, ...] | None = None
    finalists: tuple[Candidate, ...] | None = None
    terminal: Terminal = "open"
    tuning_block_index: int = 0
    pending: ResourceAllocation | None = None
    allocations: int = 0

    def state(self) -> ReplayState:
        return ReplayState(
            tuple(self.proposals),
            tuple(sorted(self.dispositions.items())),
            self.cohort,
            tuple(self.completed),
            tuple(self.observations),
            self.finalists,
            self.terminal,
            self.tuning_block_index,
            self.pending,
        )

    def accepted(self) -> tuple[Candidate, ...]:
        return accepted_candidates(self.state())

    def frontier(self) -> ObservationFrontier:
        accepted = self.accepted()
        if not accepted or any(
            not any(
                item.phase == "tuning"
                and item.candidate_id == candidate.candidate_id
                and item.context.task_prefix == self.manifest.tuning_blocks[0]
                for item in self.observations
            )
            for candidate in accepted
        ):
            return empty_frontier(_context(self.manifest, "tuning"))
        values = comparable_prefix_observations(
            tuple(self.observations), accepted, self.manifest.tuning_blocks[0]
        )
        return tuning_frontier(values)


def _proposal(state: _Replay, payload: ProposalCreatedPayload) -> Proposal:
    identity = payload.identity
    candidate = candidate_from_canonical_config(identity.canonical_config)
    if (
        candidate.candidate_id != identity.candidate_id
        or candidate.fingerprint != identity.fingerprint
    ):
        raise ValueError("proposal candidate identity is invalid")
    frontier = state.frontier()
    if (
        payload.frontier_id != frontier.frontier_id
        or payload.frontier_observation_ids != frontier.observation_ids
    ):
        raise ValueError("proposal does not bind the visible observation frontier")
    return Proposal(
        identity.proposal_index,
        identity.cohort_slot,
        candidate,
        frontier,
        ProposalProvenance(
            identity.source,
            payload.proposer_version,
            identity.source_attempt,
            payload.origin,
            payload.acquisition,
            payload.prediction,
            payload.uncertainty,
            payload.parent_candidate_id,
        ),
    )


def _apply_proposal_created(state: _Replay, payload: ProposalCreatedPayload) -> None:
    pending = state.pending
    if (
        not isinstance(pending, IntroduceCandidate)
        or payload.identity.cohort_slot != pending.cohort_slot
        or payload.identity.source != pending.source
    ):
        raise ValueError("proposal does not match pending allocation")
    if state.proposals and state.proposals[-1].proposal_index not in state.dispositions:
        raise ValueError("proposal follows an undisposed proposal")
    proposal = _proposal(state, payload)
    slot = len(state.accepted())
    if (
        proposal.proposal_index != len(state.proposals)
        or proposal.cohort_slot != slot
        or proposal.source != state.manifest.source_schedule[slot]
    ):
        raise ValueError("proposal does not match frozen schedule")
    attempt = 1 + sum(item.source == proposal.source for item in state.proposals)
    if proposal.provenance.source_attempt != attempt:
        raise ValueError("proposal source attempt is not contiguous")
    if slot < state.manifest.bootstrap_candidates and proposal.frontier.observation_ids:
        raise ValueError("bootstrap proposal has a nonempty frontier")
    if (
        slot >= state.manifest.bootstrap_candidates
        and len(proposal.frontier.observation_ids) != slot
    ):
        raise ValueError("guided proposal lacks a complete comparable frontier")
    state.proposals.append(proposal)
    state.pending = None


def _apply_disposition(
    state: _Replay, payload: ProposalAcceptedPayload | ProposalRejectedPayload
) -> None:
    identity, index = payload.identity, payload.identity.proposal_index
    if index not in range(len(state.proposals)) or index in state.dispositions:
        raise ValueError("invalid or repeated proposal disposition")
    proposal = state.proposals[index]
    if (
        identity.candidate_id != proposal.candidate.candidate_id
        or identity.cohort_slot != proposal.cohort_slot
        or identity.source != proposal.source
    ):
        raise ValueError("proposal disposition does not reference its proposal")
    state.dispositions[index] = (
        "accepted" if isinstance(payload, ProposalAcceptedPayload) else "rejected"
    )
    if len({item.fingerprint for item in state.accepted()}) != len(state.accepted()):
        raise ValueError("accepted cohort contains a duplicate")


def _apply_pair(state: _Replay, payload: PairCompletedPayload) -> None:
    decision = decide_allocation(state.manifest, state.state())
    if not isinstance(decision, ExecutePair):
        raise ValueError("pair completion is not expected")
    state.completed.append(decode_pair_payload(payload, decision.task))


def _apply_observation(state: _Replay, payload: ObservationCompletedPayload) -> None:
    current = state.state()
    context = _context(state.manifest, payload.phase, current)
    candidates = state.finalists if payload.phase == "validation" else state.accepted()
    candidate = next(
        (item for item in candidates or () if item.candidate_id == payload.candidate_id), None
    )
    if candidate is None or any(
        item.phase == payload.phase
        and item.candidate_id == candidate.candidate_id
        and item.context.task_prefix.prefix_id == context.task_prefix.prefix_id
        for item in state.observations
    ):
        raise ValueError("invalid or repeated observation")
    pairs = [
        item
        for item in state.completed
        if item.task.candidate_id == candidate.candidate_id
        and item.task.task_case.phase == payload.phase
    ]
    value = contextual_observation(candidate, context, pairs)
    if payload != observation_payload(
        value, len({pair.task.task_case.opponent_id for pair in pairs})
    ):
        raise ValueError("observation does not match completed raw pairs")
    state.observations.append(value)


def _apply_allocation(state: _Replay, payload: AllocationDecidedPayload) -> None:
    if state.pending is not None:
        raise ValueError("resource allocation is already pending")
    expected = resource_allocation(
        decide_allocation(state.manifest, state.state()), state.manifest, state.state()
    )
    if payload.policy_version != ALLOCATION_POLICY_VERSION or payload.allocation != expected:
        raise ValueError("allocation decision does not match policy")
    state.pending = payload.allocation
    if isinstance(payload.allocation, DeepenCohortAllocation):
        state.tuning_block_index = payload.allocation.block_index
        state.pending = None
    state.allocations += 1


def _apply_cohort(state: _Replay, payload: CohortCompletedPayload) -> None:
    accepted = state.accepted()
    tuning = comparable_prefix_observations(
        tuple(state.observations), accepted, state.manifest.tuning_prefix
    )
    expected = CohortCompletedPayload(
        tuple(item.candidate_id for item in accepted),
        tuple(state.manifest.source_schedule),
        POLICY_VERSION,
        tuning_frontier(tuning).frontier_id,
    )
    if (
        state.cohort is not None
        or len(accepted) != state.manifest.cohort_size
        or payload != expected
    ):
        raise ValueError("cohort completion does not bind final tuning observations")
    state.cohort = accepted


def _apply_finalists(state: _Replay, payload: FinalistsSelectedPayload) -> None:
    if (
        state.pending is None
        or getattr(state.pending, "tuning_prefix_id", None)
        != state.manifest.tuning_prefix.prefix_id
    ):
        raise ValueError("finalist selection does not match pending allocation")
    if state.cohort is None or state.finalists is not None:
        raise ValueError("finalist selection is premature")
    tuning = comparable_prefix_observations(
        tuple(state.observations), state.cohort, state.manifest.tuning_prefix
    )
    finalists = select_finalists(state.cohort, tuning, state.manifest.finalists)
    context = _context(state.manifest, "tuning", state.state())
    expected = FinalistsSelectedPayload(
        tuple(item.candidate_id for item in finalists),
        {item.candidate_id: item.estimate.mean for item in tuning},
        context.objective_epoch_id,
        context.task_prefix.corpus_id,
        context.task_prefix.prefix_id,
        context.task_prefix.task_ids,
        context.search_effort.max_iterations,
        "tuning_point_estimate_fingerprint_v1",
    )
    if payload != expected:
        raise ValueError("finalist selection does not match tuning evidence")
    state.finalists, state.pending = finalists, None


def _apply_completion(state: _Replay, payload: RunCompletedPayload) -> None:
    if (
        state.cohort is None
        or state.finalists is None
        or pending_pair(state.manifest, state.state()) is not None
    ):
        raise ValueError("run completion is premature")
    claim, missing = production_claim(
        state.manifest.validation_prefix,
        state.manifest.production_validation_corpus,
        state.manifest.efforts["validation"],
        state.manifest.efforts["production"],
    )
    tuning = comparable_prefix_observations(
        tuple(state.observations), state.cohort, state.manifest.tuning_prefix
    )
    expected = RunCompletedPayload(
        state.manifest.fingerprint,
        tuple(item.candidate_id for item in state.cohort),
        tuple(item.candidate_id for item in state.finalists),
        {"events": _scientific_count(state)},
        claim,
        state.manifest.epoch.epoch_id,
        state.manifest.validation_prefix.prefix_id,
        state.manifest.efforts["validation"].max_iterations,
        tuple(missing),
        tuning_frontier(tuning).frontier_id,
    )
    if payload != expected:
        raise ValueError("run completion does not bind replay state")
    state.terminal = "complete"


def _scientific_count(state: _Replay) -> int:
    return (
        len(state.proposals)
        + len(state.dispositions)
        + len(state.completed)
        + len(state.observations)
        + state.allocations
        + 3
    )


def _operational_pair(state: _Replay, payload: PairStartedPayload | PairFailedPayload) -> None:
    task = pending_pair(state.manifest, state.state())
    if task is None or payload.identity.pair_id != task.pair_id:
        raise ValueError("operational pair record does not match pending pair")


def _apply(state: _Replay, event: EvidenceEvent) -> None:
    if state.terminal != "open":
        raise ValueError("event follows terminal run state")
    match event.payload:
        case AllocationDecidedPayload() as payload:
            _apply_allocation(state, payload)
        case ProposalCreatedPayload() as payload:
            _apply_proposal_created(state, payload)
        case ProposalAcceptedPayload() | ProposalRejectedPayload() as payload:
            _apply_disposition(state, payload)
        case PairCompletedPayload() as payload:
            _apply_pair(state, payload)
        case ObservationCompletedPayload() as payload:
            _apply_observation(state, payload)
        case CohortCompletedPayload() as payload:
            _apply_cohort(state, payload)
        case FinalistsSelectedPayload() as payload:
            _apply_finalists(state, payload)
        case RunCompletedPayload() as payload:
            _apply_completion(state, payload)
        case RunFailedPayload():
            state.terminal = "configuration_failed"
        case PairStartedPayload() | PairFailedPayload() as payload:
            _operational_pair(state, payload)
        case RunInterruptedPayload():
            return


def fold_events(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    state = _Replay(manifest)
    for event in events:
        _apply(state, event)
    return state.state()


def replay(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    return fold_events(manifest, events)
