"""The pure policy that orders foreground tuning work."""

from __future__ import annotations

from .artifacts import Manifest
from .cohort import (
    accepted_proposal_candidates_for_cohort,
    current_active_candidates,
    current_admitted_candidates,
    latest_completed_cohort,
    pending_proposal,
    proposal_source,
)
from .domain import (
    AllocationDecision,
    BeginValidation,
    Candidate,
    CandidateFailure,
    ChooseDiagnosticPair,
    CompleteCohort,
    CompleteRun,
    DeepenCohort,
    DeepenCohortAllocation,
    EmitObservation,
    EmitShadowRace,
    EnforceElimination,
    ExecutePair,
    FailCandidate,
    IntroduceCandidate,
    IntroduceProposal,
    NoDecision,
    PairResult,
    PairTask,
    Phase,
    Proposal,
    RefillCandidate,
    ReplayState,
    ResolveProposal,
    ResourceAllocation,
    RetainElites,
    SelectFinalists,
    StartNextCohort,
    SuspendActiveElimination,
    SuspendElimination,
    TaskPrefix,
)
from .elimination import active_elimination_allocation, audited_boundary_reversals
from .identity import pair_task
from .observations import comparable_prefix_observations
from .selection import select_top_candidates
from .race_policy import shadow_prefix_eligible

ALLOCATION_POLICY_VERSION = "budgeted-multi-cohort-v1"


def allocation_policy_version(manifest: Manifest) -> str:
    return (
        "audited-active-elimination-diagnostic-v2"
        if manifest.active_elimination
        else "budgeted-multi-cohort-diagnostic-v2"
    )


def decide_allocation(manifest: Manifest, state: ReplayState) -> AllocationDecision:
    if proposal := pending_proposal(state):
        return ResolveProposal(proposal.proposal_index)
    cohort_index = len(state.completed_cohorts)
    accepted = accepted_proposal_candidates_for_cohort(state, cohort_index)
    active = current_active_candidates(state)
    admitted = current_admitted_candidates(state)
    # Bootstrap phase: fill the initial slots before guided proposals.
    if cohort_index == 0 and len(accepted) < manifest.bootstrap_candidates:
        return IntroduceProposal()
    if failure := candidate_failure_due(manifest, state):
        return FailCandidate(failure)
    if task := pending_pair(manifest, state):
        return ExecutePair(task)
    if candidate := observation_due(manifest, state):
        return EmitObservation(candidate.candidate_id, pair_phase(state))
    # Fill the active cohort to target size once the block-0 frontier is available.
    if len(admitted) < manifest.cohort_size and _frontier(
        state, active, manifest.tuning_blocks[0].prefix_id
    ):
        return IntroduceProposal()
    # Deepen or complete the current cohort when every active candidate has the
    # latest prefix observation.
    if len(admitted) == manifest.cohort_size and _frontier(
        state, active, active_prefix(manifest, state).prefix_id
    ):
        if state.tuning_block_index + 1 < len(manifest.tuning_blocks):
            prefix = active_prefix(manifest, state)
            observation_ids = tuple(
                item.observation_id
                for item in comparable_prefix_observations(state.observations, active, prefix)
            )
            if shadow_prefix_eligible(manifest, prefix) and not any(
                item.cohort_index == len(state.completed_cohorts)
                and item.prefix_id == prefix.prefix_id
                and item.observation_ids == observation_ids
                for item in state.shadow_races
            ):
                return EmitShadowRace(len(state.completed_cohorts), prefix.prefix_id)
            if (
                manifest.active_elimination
                and state.active_elimination_suspension is None
                and shadow_prefix_eligible(manifest, prefix)
                and not any(
                    item.cohort_index == len(state.completed_cohorts)
                    and item.prefix_id == prefix.prefix_id
                    for item in state.elimination_allocations
                )
            ):
                return EnforceElimination(len(state.completed_cohorts), prefix.prefix_id)
            prefix = manifest.tuning_blocks[state.tuning_block_index + 1]
            return DeepenCohort(state.tuning_block_index + 1, prefix.prefix_id)
        return CompleteCohort()
    # At a completed-cohort boundary with no committed finalists, decide between
    # another challenger cohort or validation.
    if state.finalists is None and latest_completed_cohort(state) is not None:
        cohort = latest_completed_cohort(state)
        if (
            manifest.active_elimination
            and state.active_elimination_suspension is None
            and cohort is not None
            and audited_boundary_reversals(manifest, state, cohort)
        ):
            return SuspendElimination(cohort.cohort_index)
        return _cohort_boundary_decision(manifest, state)
    if state.finalists is not None and _validation_complete(manifest, state):
        return CompleteRun()
    return NoDecision()


def _cohort_boundary_decision(manifest: Manifest, state: ReplayState) -> AllocationDecision:
    """Admit another challenger cohort when the remaining tuning pair budget
    can fund all of its planned new pairs; otherwise select finalists and begin
    validation. Reached only with at least one completed cohort, so the check
    applies to every retention boundary alike."""
    prefix_length = manifest.tuning_prefix.length
    challenger_pairs = (manifest.cohort_size - manifest.finalists) * prefix_length
    used = state.compute.tuning.pair_attempts
    budget = manifest.compute_budget.tuning_pair_attempts
    if used + challenger_pairs <= budget:
        return StartNextCohort()
    from .diagnostic_matchmaking import next_diagnostic_allocation

    if next_diagnostic_allocation(manifest, state) is not None:
        cohort = latest_completed_cohort(state)
        assert cohort is not None
        return ChooseDiagnosticPair(cohort.cohort_index)
    return SelectFinalists()


def resource_allocation(
    decision: AllocationDecision, manifest: Manifest, state: ReplayState
) -> ResourceAllocation | None:
    match decision:
        case IntroduceProposal():
            cohort_index = len(state.completed_cohorts)
            slot = len(accepted_proposal_candidates_for_cohort(state, cohort_index))
            source = proposal_source(manifest, cohort_index, slot)
            if failure := pending_refill_failure(state):
                return RefillCandidate(slot, source, failure.candidate_id)
            return IntroduceCandidate(slot, source)
        case DeepenCohort(block_index, prefix_id):
            return DeepenCohortAllocation(block_index, prefix_id)
        case EnforceElimination(cohort_index, prefix_id):
            race = next(
                item
                for item in state.shadow_races
                if item.cohort_index == cohort_index and item.prefix_id == prefix_id
            )
            return active_elimination_allocation(manifest, state, race)
        case SuspendElimination(after_cohort_index):
            cohort = latest_completed_cohort(state)
            if cohort is None or manifest.active_elimination is None:
                return None
            reversals = audited_boundary_reversals(manifest, state, cohort)
            return SuspendActiveElimination(
                after_cohort_index,
                tuple(item.candidate_id for item in reversals),
                tuple(item.prefix_id for item in reversals),
                manifest.active_elimination.safety_rule_version,
            )
        case SelectFinalists():
            return BeginValidation(manifest.tuning_prefix.prefix_id)
        case ChooseDiagnosticPair():
            from .diagnostic_matchmaking import next_diagnostic_allocation

            return next_diagnostic_allocation(manifest, state)
        case StartNextCohort():
            cohort = latest_completed_cohort(state)
            if cohort is None:
                return None
            tuning = comparable_prefix_observations(
                state.observations, cohort.candidates, manifest.tuning_prefix
            )
            elites = select_top_candidates(cohort.candidates, tuning, manifest.finalists)
            next_index = len(state.completed_cohorts)
            return RetainElites(
                next_index,
                tuple(item.candidate_id for item in elites),
                manifest.tuning_prefix.prefix_id,
            )
        case _:
            return None


def proposal_at(state: ReplayState, proposal_index: int) -> Proposal:
    return state.proposals[proposal_index]


def pair_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    return state.finalists if state.finalists is not None else current_active_candidates(state)


def pair_phase(state: ReplayState) -> Phase:
    return "validation" if state.finalists is not None else "tuning"


def active_prefix(manifest: Manifest, state: ReplayState) -> TaskPrefix:
    return (
        manifest.validation_prefix
        if pair_phase(state) == "validation"
        else manifest.tuning_blocks[state.tuning_block_index]
    )


def ready_pairs(
    manifest: Manifest, state: ReplayState, limit: int | None = None
) -> tuple[PairTask, ...]:
    """Return incomplete tasks in the active prefix's canonical order."""
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("pair limit must be a positive integer")
    phase, prefix = pair_phase(state), active_prefix(manifest, state)
    start = (
        0
        if phase == "validation" or state.tuning_block_index == 0
        else manifest.tuning_blocks[state.tuning_block_index - 1].length
    )
    completed = {pair.task.pair_id for pair in state.completed_pairs}
    ready: list[PairTask] = []
    for case in manifest.prefix_cases(phase)[start : prefix.length]:
        for candidate in pair_candidates(state):
            task = pair_task(candidate, case, manifest.efforts[phase])
            if task.pair_id not in completed:
                ready.append(task)
                if limit is not None and len(ready) == limit:
                    return tuple(ready)
    return tuple(ready)


def candidate_failure_due(manifest: Manifest, state: ReplayState) -> CandidateFailure | None:
    if pair_phase(state) != "tuning":
        return None
    facts = dict(state.pair_attempts)
    failed = {item.candidate_id for item in state.candidate_failures}
    for task in ready_pairs(manifest, state):
        if task.candidate_id in failed:
            continue
        attempt = facts.get(task.pair_id)
        if (
            attempt is not None
            and attempt.started_attempts >= manifest.candidate_failure_policy.max_pair_attempts
        ):
            completed_ids = tuple(
                pair.task.pair_id
                for pair in state.completed_pairs
                if pair.task.candidate_id == task.candidate_id
                and pair.task.task_case.phase == "tuning"
            )
            return CandidateFailure(
                len(state.completed_cohorts),
                task.candidate_id,
                task.pair_id,
                attempt.started_attempts,
                attempt.failed_attempts,
                attempt.censored_attempts,
                completed_ids,
            )
    return None


def pending_refill_failure(state: ReplayState) -> CandidateFailure | None:
    accepted = dict(state.dispositions)
    attempts = dict(state.refill_attempts)
    filled = {
        failed_id for index, failed_id in attempts.items() if accepted.get(index) == "accepted"
    }
    return next(
        (item for item in state.candidate_failures if item.candidate_id not in filled), None
    )


def pending_pair(manifest: Manifest, state: ReplayState) -> PairTask | None:
    """Compatibility view of the first currently ready pair."""
    return next(iter(ready_pairs(manifest, state, limit=1)), None)


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
