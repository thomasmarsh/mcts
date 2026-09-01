"""Proposal creation and disposition for the fixed mixed-source cohort."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from .artifacts import Manifest
from .domain import (
    Candidate,
    CohortRecord,
    ModelAttempt,
    ObservationFrontier,
    Proposal,
    ProposalProvenance,
    ProposalRequest,
    ProposalSource,
    ProposedConfiguration,
    ReplayState,
    ValidationResult,
)
from .event_payloads import (
    PanelFieldError,
    PanelRejection,
    ProposalAcceptedPayload,
    ProposalCreatedPayload,
    ProposalIdentity,
    ProposalRejectedPayload,
)
from .family_exclusions import require_candidate_family_allowed
from .identity import candidate_from_config, canonical_json
from .observations import comparable_prefix_observations
from .proposer import (
    ModelProposer,
    derived_seed,
    empty_frontier,
    model_observations,
    tuning_frontier,
)
from .schema import GameSpec
from .space import build_space, random_values
from .target import Target


def accepted_proposal_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    """All accepted candidates in global proposal order, including duplicates."""
    dispositions = dict(state.dispositions)
    return tuple(
        proposal.candidate
        for proposal in state.proposals
        if dispositions.get(proposal.proposal_index) == "accepted"
    )


def globally_accepted_block0_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    """Every globally accepted unique proposal candidate that has reached block 0,
    in global proposal order. This includes non-elites from completed cohorts and
    earlier challengers in the active cohort."""
    dispositions = dict(state.dispositions)
    seen: set[str] = set()
    result: list[Candidate] = []
    for proposal in state.proposals:
        if dispositions.get(proposal.proposal_index) != "accepted":
            continue
        if proposal.candidate.candidate_id in seen:
            continue
        # Candidate must have a block-0 tuning observation.
        if not any(
            item.candidate_id == proposal.candidate.candidate_id and item.phase == "tuning"
            for item in state.observations
        ):
            continue
        seen.add(proposal.candidate.candidate_id)
        result.append(proposal.candidate)
    return tuple(result)


def accepted_proposal_candidates_for_cohort(
    state: ReplayState, cohort_index: int
) -> tuple[Candidate, ...]:
    dispositions = dict(state.dispositions)
    return tuple(
        proposal.candidate
        for proposal in state.proposals
        if proposal.cohort_index == cohort_index
        and dispositions.get(proposal.proposal_index) == "accepted"
    )


def current_active_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    return current_surviving_candidates(state)


def current_admitted_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    failed = {item.candidate_id for item in state.candidate_failures}
    return (
        *(item for item in state.active_elites if item.candidate_id not in failed),
        *(
            item
            for item in accepted_proposal_candidates_for_cohort(state, len(state.completed_cohorts))
            if item.candidate_id not in failed
        ),
    )


def current_surviving_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    admitted = current_admitted_candidates(state)
    pruned = {
        action.candidate_id
        for allocation in state.elimination_allocations
        for action in allocation.actions
        if action.action == "prune"
    }
    return tuple(item for item in admitted if item.candidate_id not in pruned)


def proposal_source(manifest: Manifest, cohort_index: int, accepted_slot: int) -> ProposalSource:
    schedule = (
        manifest.source_schedule if cohort_index == 0 else manifest.challenger_source_schedule
    )
    return schedule[accepted_slot] if accepted_slot < len(schedule) else "random_reserve"


def latest_completed_cohort(state: ReplayState) -> CohortRecord | None:
    return state.completed_cohorts[-1] if state.completed_cohorts else None


def pending_proposal(state: ReplayState) -> Proposal | None:
    if not state.proposals:
        return None
    proposal = state.proposals[-1]
    return proposal if proposal.proposal_index not in dict(state.dispositions) else None


def _identity(proposal: Proposal) -> ProposalIdentity:
    candidate, provenance = proposal.candidate, proposal.provenance
    return ProposalIdentity(
        proposal.proposal_index,
        proposal.cohort_index,
        proposal.cohort_slot,
        provenance.source,
        provenance.source_attempt,
        candidate.candidate_id,
        candidate.fingerprint,
        candidate.canonical_config,
    )


def proposal_payload(proposal: Proposal) -> ProposalCreatedPayload:
    provenance = proposal.provenance
    return ProposalCreatedPayload(
        _identity(proposal),
        proposal.frontier.frontier_id,
        proposal.frontier.observation_ids,
        provenance.proposer_version,
        provenance.origin,
        provenance.acquisition,
        provenance.prediction,
        provenance.uncertainty,
        provenance.parent_candidate_id,
    )


def create_proposal(
    manifest: Manifest,
    state: ReplayState,
    default: Candidate,
    spec: GameSpec,
    model: ModelProposer,
) -> Proposal:
    cohort_index = len(state.completed_cohorts)
    slot = len(accepted_proposal_candidates_for_cohort(state, cohort_index))
    source = proposal_source(manifest, cohort_index, slot)
    attempt = _source_attempt(state, source)
    proposed = _candidate_for_source(manifest, state, default, spec, model, source, attempt)
    require_candidate_family_allowed(proposed.candidate, manifest.excluded_families)
    frontier = _frontier(manifest, state, slot)
    version = (
        getattr(model, "adapter_version", "smac-2.4-public-ask-v1")
        if source in {"smac_model", "qmc_search", "irace_model"}
        else "configspace-independent-v1"
    )
    provenance = ProposalProvenance(
        source,
        version,
        attempt,
        proposed.origin,
        proposed.acquisition,
        proposed.prediction,
        proposed.uncertainty,
        proposed.parent_candidate_id,
    )
    return Proposal(
        len(state.proposals), cohort_index, slot, proposed.candidate, frontier, provenance
    )


def _source_attempt(state: ReplayState, source: str) -> int:
    return 1 + sum(proposal.source == source for proposal in state.proposals)


def _candidate_for_source(
    manifest: Manifest,
    state: ReplayState,
    default: Candidate,
    spec: GameSpec,
    model: ModelProposer,
    source: str,
    attempt: int,
) -> ProposedConfiguration:
    if source == "schema_default":
        return ProposedConfiguration(default, None)
    if source in {"bootstrap_random", "random_reserve", "random_search"}:
        return ProposedConfiguration(_random_candidate(manifest, spec, source, attempt), None)
    return _model_candidate(manifest, state, spec, model, attempt)


def _random_candidate(manifest: Manifest, spec: GameSpec, source: str, attempt: int) -> Candidate:
    namespace = {
        "bootstrap_random": "bootstrap",
        "random_reserve": "reserve",
        "random_search": "random_search",
    }[source]
    space = build_space(
        spec.tuning, derived_seed(manifest.seed, namespace, attempt - 1), manifest.excluded_families
    )
    return candidate_from_config(random_values(space))


def _model_candidate(
    manifest: Manifest,
    state: ReplayState,
    spec: GameSpec,
    model: ModelProposer,
    attempt: int,
) -> ProposedConfiguration:
    block0_candidates = globally_accepted_block0_candidates(state)
    if not block0_candidates:
        block0_candidates = current_active_candidates(state)
    observations = comparable_prefix_observations(
        state.observations, block0_candidates, manifest.tuning_blocks[0]
    )
    frontier = tuning_frontier(observations)
    candidates = {item.candidate_id: item for item in block0_candidates}
    source = proposal_source(
        manifest,
        len(state.completed_cohorts),
        len(accepted_proposal_candidates_for_cohort(state, len(state.completed_cohorts))),
    )
    namespace = {"smac_model": "smac", "qmc_search": "qmc", "irace_model": "irace"}[source]
    parents = _ranked_parents(state, block0_candidates, observations)
    guided = (
        manifest.source_schedule.count(source)
        if not state.completed_cohorts
        else manifest.challenger_source_schedule.count(source)
    )
    request = ProposalRequest(
        model_observations(observations, candidates, frontier),
        frontier,
        frozenset(item.candidate.fingerprint for item in state.proposals),
        ModelAttempt(attempt, derived_seed(manifest.seed, namespace, attempt - 1)),
        len(state.completed_cohorts),
        parents,
        guided,
    )
    try:
        proposed = model.ask(request)
    except TypeError as error:
        # The injected test seam predates ProposalRequest; production adapters
        # are always called through the immutable request above.
        try:
            proposed = model.ask(
                request.observations,
                request.frontier,
                request.excluded_fingerprints,
                request.attempt,
            )  # type: ignore[call-arg]
        except TypeError as legacy_error:
            raise error from legacy_error
    return proposed


def _ranked_parents(
    state: ReplayState, block0_candidates: tuple[Candidate, ...], observations: tuple[object, ...]
) -> tuple[Candidate, ...]:
    if state.completed_cohorts:
        return tuple(
            item
            for item in state.active_elites
            if item.candidate_id in {x.candidate_id for x in block0_candidates}
        )
    means = {item.candidate_id: item.estimate.mean for item in observations}
    return tuple(
        sorted(block0_candidates, key=lambda item: (-means[item.candidate_id], item.fingerprint))
    )


def _frontier(manifest: Manifest, state: ReplayState, slot: int) -> ObservationFrontier:
    if len(state.completed_cohorts) == 0 and slot < manifest.bootstrap_candidates:
        from .domain import ObservationContext

        return empty_frontier(
            ObservationContext(
                manifest.epoch.epoch_id,
                "tuning",
                manifest.tuning_blocks[0],
                manifest.efforts["tuning"],
            )
        )
    block0_candidates = globally_accepted_block0_candidates(state)
    if not block0_candidates:
        block0_candidates = current_active_candidates(state)
    return tuning_frontier(
        comparable_prefix_observations(
            state.observations, block0_candidates, manifest.tuning_blocks[0]
        )
    )


def proposal_disposition(
    target: Target, manifest: Manifest, state: ReplayState, proposal: Proposal
) -> ProposalAcceptedPayload | ProposalRejectedPayload:
    if proposal.candidate.fingerprint in {
        item.fingerprint for item in accepted_proposal_candidates(state)
    }:
        return ProposalRejectedPayload(_identity(proposal), "duplicate", ())
    results = _panel_results(target, manifest, proposal.candidate)
    if all(result.valid for result in results):
        return ProposalAcceptedPayload(_identity(proposal), tuple(_response_fingerprints(results)))
    return _semantic_rejection(manifest, proposal, results)


def _panel_results(
    target: Target, manifest: Manifest, candidate: Candidate
) -> tuple[ValidationResult, ...]:
    return tuple(
        target.validate((candidate,), opponent, manifest.spec.default_game_config)
        for opponent in manifest.panel.opponents
    )


def _response_fingerprints(results: tuple[ValidationResult, ...]) -> list[str]:
    return [
        hashlib.sha256(canonical_json(asdict(result)).encode()).hexdigest() for result in results
    ]


def _semantic_rejection(
    manifest: Manifest, proposal: Proposal, results: tuple[ValidationResult, ...]
) -> ProposalRejectedPayload:
    errors = tuple(
        PanelRejection(
            opponent.opponent_id,
            tuple(
                PanelFieldError(error.field, error.message, error.candidate_index)
                for error in result.errors
            ),
        )
        for opponent, result in zip(manifest.panel.opponents, results, strict=True)
        if not result.valid
    )
    return ProposalRejectedPayload(_identity(proposal), "semantic_validation", errors)
