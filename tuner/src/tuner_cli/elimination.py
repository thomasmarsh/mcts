"""Deterministic active-elimination sampling over recorded shadow decisions."""

from __future__ import annotations

from hashlib import sha256

from .artifacts import Manifest, PairedBootstrapPolicySpecification
from .domain import (
    ApplyElimination,
    AuditedBoundaryReversal,
    CandidateEliminationAction,
    CohortRecord,
    ReplayState,
    PairedBootstrapEvidence,
    ShadowRaceDecision,
)
from .identity import canonical_json
from .observations import comparable_prefix_observations, paired_difference


def active_elimination_allocation(
    manifest: Manifest, state: ReplayState, race: ShadowRaceDecision
) -> ApplyElimination:
    if manifest.active_elimination is None:
        raise ValueError("active elimination is disabled")
    if race.policy_kind != "paired_bootstrap":
        raise ValueError("active elimination requires paired bootstrap evidence")
    if not isinstance(manifest.shadow_policy, PairedBootstrapPolicySpecification):
        raise ValueError("active elimination requires paired bootstrap policy")
    protected = {item.candidate_id for item in state.active_elites}
    protected.add(race.boundary_candidate_id)
    for prior in state.shadow_races:
        if prior.cohort_index == race.cohort_index:
            protected.add(prior.boundary_candidate_id)
    for allocation in state.elimination_allocations:
        if allocation.cohort_index == race.cohort_index:
            protected.update(
                item.candidate_id for item in allocation.actions if item.action == "audit_continue"
            )
    actions: list[CandidateEliminationAction] = []
    for decision in race.decisions:
        if decision.disposition != "eliminate" or decision.candidate_id in protected:
            continue
        if not isinstance(decision.evidence, PairedBootstrapEvidence):
            raise ValueError("active elimination requires paired bootstrap evidence")
        payload = canonical_json(
            [
                manifest.active_elimination.sampler_version,
                manifest.epoch.epoch_id,
                manifest.seed,
                race.cohort_index,
                race.prefix_id,
                decision.candidate_id,
            ]
        )
        draw = int.from_bytes(sha256(payload.encode()).digest(), "big") / 2**256
        action = (
            "audit_continue" if draw < manifest.active_elimination.audit_probability else "prune"
        )
        actions.append(
            CandidateEliminationAction(
                decision.candidate_id,
                action,
                manifest.shadow_policy.elimination_probability_threshold
                - decision.evidence.favorable_resamples / decision.evidence.total_resamples,
            )
        )
    return ApplyElimination(race.cohort_index, race.prefix_id, tuple(actions))


def audited_boundary_reversals(
    manifest: Manifest, state: ReplayState, cohort: CohortRecord
) -> tuple[AuditedBoundaryReversal, ...]:
    """Return completed-cohort audit continuations that reach their recorded boundary."""
    reversals: list[AuditedBoundaryReversal] = []
    for allocation in state.elimination_allocations:
        if allocation.cohort_index != cohort.cohort_index:
            continue
        race = next(
            item
            for item in state.shadow_races
            if item.cohort_index == allocation.cohort_index
            and item.prefix_id == allocation.prefix_id
        )
        prefix = next(
            item for item in manifest.tuning_blocks if item.prefix_id == allocation.prefix_id
        )
        observations = {
            item.candidate_id: item
            for item in comparable_prefix_observations(
                state.observations, cohort.candidates, manifest.tuning_prefix
            )
        }
        boundary = observations[race.boundary_candidate_id]
        audited = {
            item.candidate_id for item in allocation.actions if item.action == "audit_continue"
        }
        for candidate in cohort.candidates:
            if candidate.candidate_id not in audited:
                continue
            difference = paired_difference(observations[candidate.candidate_id], boundary).mean
            if difference >= -manifest.shadow_policy.practical_effect_margin:
                reversals.append(
                    AuditedBoundaryReversal(
                        cohort.cohort_index,
                        candidate.candidate_id,
                        prefix.prefix_id,
                        race.boundary_candidate_id,
                        difference,
                    )
                )
    return tuple(reversals)
