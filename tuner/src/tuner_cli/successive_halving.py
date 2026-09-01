"""Fixed eta-two shadow comparator over common-prefix observations.

The rank cut keeps ``ceil(n/2)`` survivors and eliminates the rest. When the
policy carries a positive ``spare_margin`` (method version
``successive-halving-spare-near-tie-v1``), a would-be-eliminated candidate whose
paired mean at the cut prefix is within the margin of the last kept candidate is
carried to the next look rather than cut, so a near-coin-flip boundary pair is
not resolved on a short common prefix. ``spare_margin`` of ``0.0`` is exactly the
plain eta-2 cut.
"""

from __future__ import annotations

from .artifacts import Manifest, SuccessiveHalvingPolicySpecification
from .cohort import current_active_candidates
from .domain import (
    ReplayState,
    ShadowCandidateDecision,
    ShadowRaceDecision,
    SuccessiveHalvingEvidence,
    TaskPrefix,
)
from .observations import comparable_prefix_observations, paired_difference
from .selection import select_top_candidates


def decide_successive_halving_shadow_race(
    manifest: Manifest, state: ReplayState, cohort_index: int, prefix: TaskPrefix
) -> ShadowRaceDecision:
    policy = manifest.shadow_policy
    if not isinstance(policy, SuccessiveHalvingPolicySpecification):
        raise ValueError("successive halving requires its selected policy")
    cohort = current_active_candidates(state)
    if (
        cohort_index != len(state.completed_cohorts)
        or not manifest.finalists <= len(cohort) <= manifest.cohort_size
    ):
        raise ValueError("shadow race does not reference the active complete cohort")
    observations = comparable_prefix_observations(state.observations, cohort, prefix)
    observation_ids = tuple(item.observation_id for item in observations)
    if any(
        item.cohort_index == cohort_index
        and item.prefix_id == prefix.prefix_id
        and item.observation_ids == observation_ids
        for item in state.shadow_races
    ):
        raise ValueError("shadow race is already recorded")
    eliminated: set[str] = set()
    for prior in sorted(
        (
            item
            for item in state.shadow_races
            if item.cohort_index == cohort_index and item.policy_kind == "successive_halving"
        ),
        key=lambda item: next(
            index
            for index, block in enumerate(manifest.tuning_blocks)
            if block.prefix_id == item.prefix_id
        ),
    ):
        eliminated.update(
            item.candidate_id for item in prior.decisions if item.disposition == "eliminate"
        )
    protected = {item.candidate_id for item in state.active_elites}
    survivors = tuple(item for item in cohort if item.candidate_id not in eliminated) + tuple(
        item
        for item in cohort
        if item.candidate_id in protected and item.candidate_id in eliminated
    )
    # Preserve roster order while ensuring elites cannot be lost to an earlier hypothetical cut.
    survivors = tuple(
        item
        for item in cohort
        if item.candidate_id in {candidate.candidate_id for candidate in survivors}
    )
    survivor_ids = {candidate.candidate_id for candidate in survivors}
    ranked = select_top_candidates(
        survivors,
        tuple(item for item in observations if item.candidate_id in survivor_ids),
        len(survivors),
    )
    target = max(policy.survivor_floor, (len(survivors) + 1) // 2, len(protected))
    target = min(target, len(survivors))
    kept = {item.candidate_id for item in ranked[:target]}
    if policy.spare_margin > 0.0 and target < len(ranked):
        observation_by_id = {item.candidate_id: item for item in observations}
        boundary_observation = observation_by_id[ranked[target - 1].candidate_id]
        for candidate in ranked[target:]:
            difference = paired_difference(
                observation_by_id[candidate.candidate_id], boundary_observation
            ).mean
            if difference >= -policy.spare_margin:
                kept.add(candidate.candidate_id)
    ranks = {item.candidate_id: index + 1 for index, item in enumerate(ranked)}
    decisions = tuple(
        ShadowCandidateDecision(
            candidate.candidate_id,
            "protected"
            if candidate.candidate_id in protected
            else ("continue" if candidate.candidate_id in kept else "eliminate"),
            SuccessiveHalvingEvidence(
                ranks.get(candidate.candidate_id),
                len(survivors),
                target,
                candidate.candidate_id in survivor_ids
                and candidate.candidate_id not in kept
                and candidate.candidate_id not in protected,
            ),
        )
        for candidate in cohort
    )
    return ShadowRaceDecision(
        cohort_index,
        prefix.prefix_id,
        observation_ids,
        ranked[target - 1].candidate_id,
        decisions,
        "successive_halving",
        policy.method_version,
    )
