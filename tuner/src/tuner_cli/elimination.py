"""Deterministic active-elimination sampling over recorded shadow decisions."""

from __future__ import annotations

from hashlib import sha256

from .artifacts import Manifest
from .domain import ApplyElimination, CandidateEliminationAction, ReplayState, ShadowRaceDecision
from .identity import canonical_json


def active_elimination_allocation(
    manifest: Manifest, state: ReplayState, race: ShadowRaceDecision
) -> ApplyElimination:
    if manifest.active_elimination is None:
        raise ValueError("active elimination is disabled")
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
    actions = []
    for decision in race.decisions:
        if decision.disposition != "eliminate" or decision.candidate_id in protected:
            continue
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
                - decision.favorable_resamples / decision.total_resamples,
            )
        )
    return ApplyElimination(race.cohort_index, race.prefix_id, tuple(actions))
