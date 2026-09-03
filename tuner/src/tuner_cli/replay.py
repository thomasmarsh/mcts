"""Strict replay of evidence into factual foreground-run state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from .allocator import (
    allocation_policy_version,
    decide_allocation,
    pending_pair,
    ready_pairs,
    resource_allocation,
)
from .artifacts import Manifest, production_claim
from .codec import is_json_object
from .cohort import (
    accepted_proposal_candidates,
    accepted_proposal_candidates_for_cohort,
    current_active_candidates,
    globally_accepted_block0_candidates,
    latest_completed_cohort,
)
from .compute import LedgerBuilder
from .constraints import require_candidate_allowed
from .diagnostic_graph import build_diagnostic_graph
from .domain import (
    ApplyElimination,
    Candidate,
    CandidateFailure,
    CohortRecord,
    ComputeBudget,
    DeepenCohortAllocation,
    DiagnosticPairResult,
    EmitShadowRace,
    EvaluateDiagnosticPair,
    IntroduceCandidate,
    Observation,
    ObservationContext,
    ObservationFrontier,
    PairAttemptFacts,
    PairResult,
    PairTask,
    Phase,
    Proposal,
    ProposalProvenance,
    RefillCandidate,
    ReplayState,
    ResourceAllocation,
    RetainElites,
    ShadowRaceDecision,
    SuspendActiveElimination,
)
from .event_payloads import (
    AllocationDecidedPayload,
    BudgetExtendedPayload,
    CandidateFailedPayload,
    CohortCompletedPayload,
    DiagnosticPairCompletedPayload,
    DiagnosticPairFailedPayload,
    DiagnosticPairStartedPayload,
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
    ShadowRaceDecidedPayload,
)
from .evidence import EvidenceEvent, decode_pair_payload
from .identity import candidate_from_canonical_config
from .observations import comparable_prefix_observations, contextual_observation
from .proposer import POLICY_VERSION, empty_frontier, tuning_frontier
from .race_policy import decide_shadow_race
from .selection import select_top_candidates, select_validation_shortlist

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
        context.search_effort,
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
    completed_cohorts: list[CohortRecord] = field(default_factory=lambda: list[CohortRecord]())
    active_elites: tuple[Candidate, ...] = ()
    finalists: tuple[Candidate, ...] | None = None
    terminal: Terminal = "open"
    tuning_block_index: int = 0
    pending: ResourceAllocation | None = None
    allocations: int = 0
    ledger: LedgerBuilder = field(default_factory=LedgerBuilder)
    shadow_races: list[ShadowRaceDecision] = field(default_factory=lambda: [])
    candidate_failures: list[CandidateFailure] = field(default_factory=lambda: [])
    pair_attempts: dict[str, PairAttemptFacts] = field(default_factory=lambda: {})
    refill_attempts: dict[int, str] = field(default_factory=lambda: {})
    elimination_allocations: list[ApplyElimination] = field(default_factory=lambda: [])
    active_elimination_suspension: SuspendActiveElimination | None = None
    diagnostic_pairs: list[DiagnosticPairResult] = field(default_factory=lambda: [])
    diagnostic_attempts: dict[str, PairAttemptFacts] = field(default_factory=lambda: {})
    budget_extensions: list[BudgetExtendedPayload] = field(default_factory=lambda: [])
    superseded_finalists: list[tuple[Candidate, ...]] = field(default_factory=lambda: [])
    superseded_pairs: list[PairResult] = field(default_factory=lambda: [])
    superseded_observations: list[Observation] = field(default_factory=lambda: [])

    def effective_budget(self) -> ComputeBudget:
        base = self.manifest.compute_budget
        return ComputeBudget(
            base.tuning_pair_attempts
            + sum(item.tuning_pair_attempts_delta for item in self.budget_extensions),
            base.validation_pair_attempts
            + sum(item.validation_pair_attempts_delta for item in self.budget_extensions),
            base.diagnostic_pair_attempts
            + sum(item.diagnostic_pair_attempts_delta for item in self.budget_extensions),
        )

    def state(self) -> ReplayState:
        attempts = tuple(
            sorted(
                (
                    pair_id,
                    PairAttemptFacts(
                        facts.started_attempts,
                        facts.failed_attempts,
                        facts.started_attempts - facts.failed_attempts - facts.completed_attempts,
                        facts.completed_attempts,
                    ),
                )
                for pair_id, facts in self.pair_attempts.items()
            )
        )
        return ReplayState(
            tuple(self.proposals),
            tuple(sorted(self.dispositions.items())),
            tuple(self.completed_cohorts),
            self.active_elites,
            tuple(self.completed),
            tuple(self.observations),
            self.finalists,
            self.terminal,
            self.tuning_block_index,
            self.pending,
            self.ledger.ledger(),
            tuple(self.shadow_races),
            tuple(self.candidate_failures),
            attempts,
            tuple(sorted(self.refill_attempts.items())),
            tuple(self.elimination_allocations),
            self.active_elimination_suspension,
            tuple(self.diagnostic_pairs),
            tuple(sorted(self.diagnostic_attempts.items())),
            self.effective_budget(),
            tuple(self.superseded_finalists),
        )

    def active(self) -> tuple[Candidate, ...]:
        return current_active_candidates(self.state())

    def frontier(self) -> ObservationFrontier:
        block0 = globally_accepted_block0_candidates(self.state())
        if not block0:
            block0 = self.active()
        if not block0 or any(
            not any(
                item.phase == "tuning"
                and item.candidate_id == candidate.candidate_id
                and item.context.task_prefix == self.manifest.tuning_blocks[0]
                for item in self.observations
            )
            for candidate in block0
        ):
            return empty_frontier(_context(self.manifest, "tuning"))
        values = comparable_prefix_observations(
            tuple(self.observations), block0, self.manifest.tuning_blocks[0]
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
        identity.cohort_index,
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
        not isinstance(pending, (IntroduceCandidate, RefillCandidate))
        or payload.identity.cohort_slot != pending.cohort_slot
        or payload.identity.source != pending.source
    ):
        raise ValueError("proposal does not match pending allocation")
    if state.proposals and state.proposals[-1].proposal_index not in state.dispositions:
        raise ValueError("proposal follows an undisposed proposal")
    proposal = _proposal(state, payload)
    require_candidate_allowed(proposal.candidate, state.manifest.constraints)
    cohort_index = len(state.completed_cohorts)
    slot = len(accepted_proposal_candidates_for_cohort(state.state(), cohort_index))
    from .cohort import proposal_source

    if (
        proposal.proposal_index != len(state.proposals)
        or proposal.cohort_index != cohort_index
        or proposal.cohort_slot != slot
        or proposal.source != proposal_source(state.manifest, cohort_index, slot)
    ):
        raise ValueError("proposal does not match frozen schedule")
    attempt = 1 + sum(item.source == proposal.source for item in state.proposals)
    if proposal.provenance.source_attempt != attempt:
        raise ValueError("proposal source attempt is not contiguous")
    if (
        cohort_index == 0
        and slot < state.manifest.bootstrap_candidates
        and proposal.frontier.observation_ids
    ):
        raise ValueError("bootstrap proposal has a nonempty frontier")
    # Every non-bootstrap proposal binds a complete comparable frontier from
    # globally accepted block-0 candidates.
    if (cohort_index == 0 and slot >= state.manifest.bootstrap_candidates) or cohort_index > 0:
        expected_count = len(globally_accepted_block0_candidates(state.state()) or state.active())
        if len(proposal.frontier.observation_ids) != expected_count:
            raise ValueError("guided proposal lacks a complete comparable frontier")
    state.proposals.append(proposal)
    if isinstance(pending, RefillCandidate):
        state.refill_attempts[proposal.proposal_index] = pending.failed_candidate_id
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
        or identity.cohort_index != proposal.cohort_index
        or identity.cohort_slot != proposal.cohort_slot
        or identity.source != proposal.source
    ):
        raise ValueError("proposal disposition does not reference its proposal")
    state.dispositions[index] = (
        "accepted" if isinstance(payload, ProposalAcceptedPayload) else "rejected"
    )
    accepted = accepted_proposal_candidates(state.state())
    if len({item.fingerprint for item in accepted}) != len(accepted):
        raise ValueError("accepted cohort contains a duplicate")


def _apply_pair(state: _Replay, payload: PairCompletedPayload) -> None:
    task = next(
        (
            item
            for item in ready_pairs(state.manifest, state.state())
            if item.pair_id == payload.identity.pair_id
        ),
        None,
    )
    if task is None:
        raise ValueError("pair completion is not expected")
    state.completed.append(decode_pair_payload(payload, task))


def _candidate_failure(state: _Replay, payload: CandidateFailedPayload) -> CandidateFailure:
    from .allocator import candidate_failure_due

    expected = candidate_failure_due(state.manifest, state.state())
    if expected is None:
        raise ValueError("candidate failure is not due")
    identity = payload.triggering_pair
    task = next(
        item
        for item in ready_pairs(state.manifest, state.state())
        if item.pair_id == expected.triggering_pair_id
    )
    if (
        payload.policy_version
        != state.manifest.candidate_failure_policy.encoded()["policy_version"]
        or payload.reason != "pair_attempts_exhausted"
        or payload.cohort_index != expected.cohort_index
        or payload.candidate_id != expected.candidate_id
        or identity.phase != task.task_case.phase
        or identity.candidate_id != task.candidate_id
        or identity.task_id != task.task_case.task_id
        or identity.pair_id != task.pair_id
        or identity.opponent_id != task.task_case.opponent_id
        or identity.search_effort != task.budget
        or payload.started_attempts != expected.started_attempts
        or payload.failed_attempts != expected.failed_attempts
        or payload.censored_attempts != expected.censored_attempts
        or payload.completed_tuning_pair_ids != expected.completed_tuning_pair_ids
    ):
        raise ValueError("candidate failure does not match attempt evidence")
    return expected


def _apply_candidate_failure(state: _Replay, payload: CandidateFailedPayload) -> None:
    failure = _candidate_failure(state, payload)
    state.candidate_failures.append(failure)
    state.active_elites = tuple(
        item for item in state.active_elites if item.candidate_id != failure.candidate_id
    )
    state.tuning_block_index = 0


def _apply_observation(state: _Replay, payload: ObservationCompletedPayload) -> None:
    current = state.state()
    context = _context(state.manifest, payload.phase, current)
    candidates = state.finalists if payload.phase == "validation" else state.active()
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


def _apply_shadow_race(state: _Replay, payload: ShadowRaceDecidedPayload) -> None:
    decision = decide_allocation(state.manifest, state.state())
    if not isinstance(decision, EmitShadowRace):
        raise ValueError("shadow race is not expected")
    prefix = state.manifest.tuning_blocks[state.tuning_block_index]
    expected = decide_shadow_race(state.manifest, state.state(), decision.cohort_index, prefix)
    if payload.decision != expected:
        raise ValueError("shadow race does not match policy")
    state.shadow_races.append(payload.decision)


def _apply_allocation(state: _Replay, payload: AllocationDecidedPayload) -> None:
    if state.pending is not None:
        raise ValueError("resource allocation is already pending")
    expected = resource_allocation(
        decide_allocation(state.manifest, state.state()), state.manifest, state.state()
    )
    if (
        payload.policy_version != allocation_policy_version(state.manifest)
        or payload.allocation != expected
    ):
        raise ValueError("allocation decision does not match policy")
    state.pending = payload.allocation
    if isinstance(payload.allocation, DeepenCohortAllocation):
        state.tuning_block_index = payload.allocation.block_index
        state.pending = None
    if isinstance(payload.allocation, RetainElites):
        cohort = latest_completed_cohort(state.state())
        if cohort is None:
            raise ValueError("elite retention lacks a completed cohort")
        by_id = {candidate.candidate_id: candidate for candidate in cohort.candidates}
        state.active_elites = tuple(
            by_id[candidate_id] for candidate_id in payload.allocation.candidate_ids
        )
        state.tuning_block_index = 0
        state.pending = None
    if isinstance(payload.allocation, ApplyElimination):
        state.elimination_allocations.append(payload.allocation)
        state.pending = None
    if isinstance(payload.allocation, SuspendActiveElimination):
        if state.active_elimination_suspension is not None:
            raise ValueError("active elimination is already suspended")
        state.active_elimination_suspension = payload.allocation
        state.pending = None
    state.allocations += 1


def _decode_diagnostic_completion(payload: DiagnosticPairCompletedPayload) -> DiagnosticPairResult:
    from .target import parse_pair_output

    raw: list[str] = []
    for game in payload.games:
        record = game.get("raw_record")
        if not isinstance(record, str):
            raise ValueError("diagnostic game lacks raw record")
        raw.append(record)
    outcomes: list[str] = []
    for item in raw:
        record = json.loads(item)
        if not is_json_object(record):
            raise ValueError("diagnostic game raw record is invalid")
        outcome = record.get("outcome")
        if not isinstance(outcome, str):
            raise ValueError("diagnostic game raw record is invalid")
        outcomes.append(outcome)
    summary = {
        "type": "configured_comparison_summary",
        "games": 2,
        "wins": outcomes.count("candidate_win"),
        "losses": outcomes.count("baseline_win"),
        "draws": outcomes.count("draw"),
    }
    result = parse_pair_output(
        "\n".join((*raw, json.dumps(summary, sort_keys=True, separators=(",", ":")))),
        payload.task,
    )
    if not isinstance(result, DiagnosticPairResult):
        raise ValueError("diagnostic completion decoded as objective pair")
    return result


def _apply_diagnostic(
    state: _Replay,
    payload: DiagnosticPairStartedPayload
    | DiagnosticPairCompletedPayload
    | DiagnosticPairFailedPayload,
) -> None:
    pending = state.pending
    if not isinstance(pending, EvaluateDiagnosticPair) or pending.task != payload.task:
        raise ValueError("diagnostic event does not match pending allocation")
    facts = state.diagnostic_attempts.get(payload.task.pair_id, PairAttemptFacts())
    if isinstance(payload, DiagnosticPairStartedPayload):
        state.diagnostic_attempts[payload.task.pair_id] = PairAttemptFacts(
            facts.started_attempts + 1,
            facts.failed_attempts,
            facts.censored_attempts,
            facts.completed_attempts,
        )
    elif isinstance(payload, DiagnosticPairFailedPayload):
        if facts.failed_attempts + facts.completed_attempts >= facts.started_attempts:
            raise ValueError("diagnostic failure lacks a started attempt")
        state.diagnostic_attempts[payload.task.pair_id] = PairAttemptFacts(
            facts.started_attempts,
            facts.failed_attempts + 1,
            facts.censored_attempts,
            facts.completed_attempts,
        )
    else:
        if facts.completed_attempts or facts.failed_attempts >= facts.started_attempts:
            raise ValueError("diagnostic completion lacks a started attempt")
        state.diagnostic_pairs.append(_decode_diagnostic_completion(payload))
        state.diagnostic_attempts[payload.task.pair_id] = PairAttemptFacts(
            facts.started_attempts, facts.failed_attempts, facts.censored_attempts, 1
        )
        state.pending = None


def _apply_cohort(state: _Replay, payload: CohortCompletedPayload) -> None:
    cohort_index = len(state.completed_cohorts)
    candidates = state.active()
    tuning = comparable_prefix_observations(
        tuple(state.observations), candidates, state.manifest.tuning_prefix
    )
    expected = CohortCompletedPayload(
        cohort_index,
        tuple(item.candidate_id for item in candidates),
        tuple(item.candidate_id for item in state.active_elites),
        tuple(
            proposal.source
            for proposal in state.proposals
            if proposal.cohort_index == cohort_index
            and state.dispositions.get(proposal.proposal_index) == "accepted"
        ),
        POLICY_VERSION,
        tuning_frontier(tuning).frontier_id,
    )
    if len(candidates) < state.manifest.finalists or payload != expected:
        raise ValueError("cohort completion does not bind final tuning observations")
    state.completed_cohorts.append(
        CohortRecord(
            cohort_index, candidates, tuple(item.candidate_id for item in state.active_elites)
        )
    )
    # After any cohort completes, clear the active elites (they were just recorded).
    state.active_elites = ()


def _apply_finalists(state: _Replay, payload: FinalistsSelectedPayload) -> None:
    if (
        state.pending is None
        or getattr(state.pending, "tuning_prefix_id", None)
        != state.manifest.tuning_prefix.prefix_id
    ):
        raise ValueError("finalist selection does not match pending allocation")
    cohort = latest_completed_cohort(state.state())
    if len(state.completed_cohorts) < 1 or cohort is None or state.finalists is not None:
        raise ValueError("finalist selection is premature")
    tuning = comparable_prefix_observations(
        tuple(state.observations), cohort.candidates, state.manifest.tuning_prefix
    )
    ordered = select_top_candidates(cohort.candidates, tuning, len(cohort.candidates))
    rank = {item.candidate_id: index for index, item in enumerate(ordered)}
    graph = build_diagnostic_graph(cohort.candidates, tuple(state.diagnostic_pairs), rank)
    finalists, _reserve, _displaced = select_validation_shortlist(
        cohort.candidates, tuning, state.manifest.finalists, graph
    )
    context = _context(state.manifest, "tuning", state.state())
    expected = FinalistsSelectedPayload(
        tuple(item.candidate_id for item in finalists),
        {item.candidate_id: item.estimate.mean for item in tuning},
        context.objective_epoch_id,
        context.task_prefix.corpus_id,
        context.task_prefix.prefix_id,
        context.task_prefix.task_ids,
        context.search_effort,
        "objective-top-with-one-cycle-reserve-v1",
    )
    if payload != expected:
        raise ValueError("finalist selection does not match tuning evidence")
    state.finalists, state.pending = finalists, None


def _apply_completion(state: _Replay, payload: RunCompletedPayload) -> None:
    cohort = latest_completed_cohort(state.state())
    if (
        state.finalists is None
        or cohort is None
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
        tuple(state.observations), cohort.candidates, state.manifest.tuning_prefix
    )
    expected = RunCompletedPayload(
        state.manifest.fingerprint,
        tuple(item.candidate_id for item in accepted_proposal_candidates(state.state())),
        tuple(item.candidate_id for item in state.finalists),
        {"events": _scientific_count(state)},
        claim,
        state.manifest.epoch.epoch_id,
        state.manifest.validation_prefix.prefix_id,
        state.manifest.efforts["validation"],
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
        + len(state.superseded_pairs)
        + len(state.observations)
        + len(state.superseded_observations)
        + len(state.shadow_races)
        + len(state.candidate_failures)
        + len(state.budget_extensions)
        + state.allocations
        + len(state.completed_cohorts)
        + (state.finalists is not None)
        # Each re-open leaves a prior finalists_selected and run_completed in the
        # log as superseded scientific evidence.
        + 2 * len(state.superseded_finalists)
        + 1
    )


def _operational_pair(state: _Replay, payload: PairStartedPayload | PairFailedPayload) -> None:
    tasks = ready_pairs(state.manifest, state.state())
    task = next((item for item in tasks if _matches_pair_identity(payload, item)), None)
    if task is None:
        raise ValueError("operational pair record does not match pending pair")
    if isinstance(payload, PairStartedPayload) and payload.task_seed != task.task_case.seed:
        raise ValueError("pair start seed does not match pending pair")
    facts = state.pair_attempts.get(payload.identity.pair_id, PairAttemptFacts())
    if isinstance(payload, PairStartedPayload):
        if (
            payload.identity.phase == "tuning"
            and facts.started_attempts >= state.manifest.candidate_failure_policy.max_pair_attempts
        ):
            raise ValueError("pair start exceeds the tuning attempt limit")
        state.pair_attempts[payload.identity.pair_id] = PairAttemptFacts(
            facts.started_attempts + 1,
            facts.failed_attempts,
            facts.censored_attempts,
            facts.completed_attempts,
        )
    else:
        if facts.failed_attempts + facts.completed_attempts >= facts.started_attempts:
            raise ValueError("pair failure lacks a started attempt")
        state.pair_attempts[payload.identity.pair_id] = PairAttemptFacts(
            facts.started_attempts,
            facts.failed_attempts + 1,
            facts.censored_attempts,
            facts.completed_attempts,
        )


def _matches_pair_identity(payload: PairStartedPayload | PairFailedPayload, task: PairTask) -> bool:
    """Match an operational record to one exact active ready task."""
    return (
        payload.identity.phase == task.task_case.phase
        and payload.identity.candidate_id == task.candidate_id
        and payload.identity.task_id == task.task_case.task_id
        and payload.identity.pair_id == task.pair_id
        and payload.identity.opponent_id == task.task_case.opponent_id
        and payload.identity.search_effort == task.budget
    )


def _apply_budget_extension(state: _Replay, payload: BudgetExtendedPayload) -> None:
    if state.terminal == "configuration_failed":
        raise ValueError("a configuration-failed run cannot be extended")
    state.budget_extensions.append(payload)
    budget = state.effective_budget()
    finalists = state.manifest.finalists
    if budget.validation_pair_attempts % finalists:
        raise ValueError("extended validation budget must divide finalists")
    if budget.validation_pair_attempts // finalists > len(
        state.manifest.production_validation_corpus.cases
    ):
        raise ValueError("extended validation budget exceeds the frozen validation corpus")
    if state.terminal == "complete":
        # Re-open a completed run: the prior finalists_selected / validation
        # pairs / run_completed events stay in the log as factual evidence, but
        # the active replay state rewinds to the last cohort boundary so the
        # allocator can fund a fresh challenger cohort and a fresh finalist
        # selection and validation from the raised budget. The superseded
        # finalists, validation pairs, and validation observations are set aside
        # (still counted for the scientific event total) so the fresh pass does
        # not collide with them.
        state.terminal = "open"
        if state.finalists is not None:
            state.superseded_finalists.append(state.finalists)
        state.finalists = None
        superseded_pair_ids = {
            pair.task.pair_id
            for pair in state.completed
            if pair.task.task_case.phase == "validation"
        }
        state.superseded_pairs.extend(
            pair for pair in state.completed if pair.task.pair_id in superseded_pair_ids
        )
        state.completed = [
            pair for pair in state.completed if pair.task.pair_id not in superseded_pair_ids
        ]
        state.superseded_observations.extend(
            item for item in state.observations if item.phase == "validation"
        )
        state.observations = [item for item in state.observations if item.phase != "validation"]
        state.pair_attempts = {
            pair_id: facts
            for pair_id, facts in state.pair_attempts.items()
            if pair_id not in superseded_pair_ids
        }


def _apply(state: _Replay, event: EvidenceEvent) -> None:
    if isinstance(event.payload, BudgetExtendedPayload):
        _apply_budget_extension(state, event.payload)
        return
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
            facts = state.pair_attempts.get(payload.identity.pair_id, PairAttemptFacts())
            if facts.completed_attempts or facts.failed_attempts >= facts.started_attempts:
                raise ValueError("pair completion lacks a started attempt")
            state.pair_attempts[payload.identity.pair_id] = PairAttemptFacts(
                facts.started_attempts, facts.failed_attempts, facts.censored_attempts, 1
            )
        case ObservationCompletedPayload() as payload:
            _apply_observation(state, payload)
        case ShadowRaceDecidedPayload() as payload:
            _apply_shadow_race(state, payload)
        case CohortCompletedPayload() as payload:
            _apply_cohort(state, payload)
        case FinalistsSelectedPayload() as payload:
            _apply_finalists(state, payload)
        case RunCompletedPayload() as payload:
            _apply_completion(state, payload)
        case CandidateFailedPayload() as payload:
            _apply_candidate_failure(state, payload)
        case RunFailedPayload():
            state.terminal = "configuration_failed"
        case PairStartedPayload() | PairFailedPayload() as payload:
            _operational_pair(state, payload)
        case (
            DiagnosticPairStartedPayload()
            | DiagnosticPairCompletedPayload()
            | DiagnosticPairFailedPayload()
        ) as payload:
            _apply_diagnostic(state, payload)
        case RunInterruptedPayload():
            return


def fold_events(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    state = _Replay(manifest)
    for event in events:
        state.ledger.apply(event)
        _apply(state, event)
    return state.state()


def replay(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    return fold_events(manifest, events)
