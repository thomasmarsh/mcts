"""Proposal creation and disposition for the fixed mixed-source cohort."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from .artifacts import Manifest
from .domain import (
    Candidate,
    ModelAttempt,
    Proposal,
    ProposalProvenance,
    ProposedConfiguration,
    ReplayState,
    ValidationResult,
)
from .identity import candidate_from_config, canonical_json
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


def accepted_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    dispositions = dict(state.dispositions)
    return tuple(
        proposal.candidate
        for proposal in state.proposals
        if dispositions.get(proposal.proposal_index) == "accepted"
    )


def pending_proposal(state: ReplayState) -> Proposal | None:
    if not state.proposals:
        return None
    proposal = state.proposals[-1]
    return proposal if proposal.proposal_index not in dict(state.dispositions) else None


def proposal_payload(proposal: Proposal) -> dict[str, object]:
    provenance, candidate = proposal.provenance, proposal.candidate
    return {
        "proposal_index": proposal.proposal_index,
        "cohort_slot": proposal.cohort_slot,
        "source": provenance.source,
        "source_attempt": provenance.source_attempt,
        "candidate_id": candidate.candidate_id,
        "fingerprint": candidate.fingerprint,
        "canonical_config": candidate.canonical_config,
        "frontier_id": proposal.frontier.frontier_id,
        "frontier_observation_ids": list(proposal.frontier.observation_ids),
        "proposer_version": provenance.proposer_version,
        "origin": provenance.origin,
        "acquisition": provenance.acquisition,
        "prediction": provenance.prediction,
        "uncertainty": provenance.uncertainty,
        "parent_candidate_id": provenance.parent_candidate_id,
    }


def create_proposal(
    manifest: Manifest,
    state: ReplayState,
    default: Candidate,
    spec: GameSpec,
    model: ModelProposer,
) -> Proposal:
    slot = len(accepted_candidates(state))
    source = manifest.source_schedule[slot]
    attempt = _source_attempt(state, source)
    proposed = _candidate_for_source(manifest, state, default, spec, model, source, attempt)
    frontier = _frontier(manifest, state, slot)
    version = "smac-2.4-public-ask-v1" if source == "smac_model" else "configspace-independent-v1"
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
    return Proposal(len(state.proposals), slot, proposed.candidate, frontier, provenance)


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
    if source in {"bootstrap_random", "random_reserve"}:
        return ProposedConfiguration(_random_candidate(manifest, spec, source, attempt), None)
    return _model_candidate(manifest, state, spec, model, attempt)


def _random_candidate(manifest: Manifest, spec: GameSpec, source: str, attempt: int) -> Candidate:
    namespace = "bootstrap" if source == "bootstrap_random" else "reserve"
    space = build_space(spec.tuning, derived_seed(manifest.seed, namespace, attempt - 1))
    return candidate_from_config(random_values(space))


def _model_candidate(
    manifest: Manifest,
    state: ReplayState,
    spec: GameSpec,
    model: ModelProposer,
    attempt: int,
) -> ProposedConfiguration:
    observations = tuple(item for item in state.observations if item.phase == "tuning")
    frontier = tuning_frontier(observations)
    candidates = {item.candidate_id: item for item in accepted_candidates(state)}
    proposed = model.ask(
        model_observations(observations, candidates, frontier),
        frontier,
        frozenset(item.candidate.fingerprint for item in state.proposals),
        ModelAttempt(attempt, derived_seed(manifest.seed, "smac", attempt - 1)),
    )
    return proposed


def _frontier(manifest: Manifest, state: ReplayState, slot: int):
    if slot < manifest.bootstrap_candidates:
        from .domain import ObservationContext

        return empty_frontier(
            ObservationContext(
                manifest.epoch.epoch_id,
                "tuning",
                manifest.tuning_prefix,
                manifest.efforts["tuning"],
            )
        )
    return tuning_frontier(tuple(item for item in state.observations if item.phase == "tuning"))


def disposition_payload(proposal: Proposal) -> dict[str, object]:
    candidate, provenance = proposal.candidate, proposal.provenance
    return {
        "proposal_index": proposal.proposal_index,
        "cohort_slot": proposal.cohort_slot,
        "source": proposal.source,
        "source_attempt": provenance.source_attempt,
        "candidate_id": candidate.candidate_id,
        "fingerprint": candidate.fingerprint,
        "canonical_config": candidate.canonical_config,
    }


def proposal_disposition(
    target: Target, manifest: Manifest, state: ReplayState, proposal: Proposal
) -> tuple[str, dict[str, object]]:
    if proposal.candidate.fingerprint in {item.fingerprint for item in accepted_candidates(state)}:
        return "proposal_rejected", {
            **disposition_payload(proposal),
            "reason": "duplicate",
            "errors": [],
        }
    results = _panel_results(target, manifest, proposal.candidate)
    if all(result.valid for result in results):
        return "proposal_accepted", {
            **disposition_payload(proposal),
            "panel_response_fingerprints": _response_fingerprints(results),
        }
    return "proposal_rejected", _semantic_rejection(manifest, proposal, results)


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
) -> dict[str, object]:
    errors = [
        {"opponent_id": opponent.opponent_id, "errors": [asdict(error) for error in result.errors]}
        for opponent, result in zip(manifest.panel.opponents, results, strict=True)
        if not result.valid
    ]
    return {**disposition_payload(proposal), "reason": "semantic_validation", "errors": errors}
