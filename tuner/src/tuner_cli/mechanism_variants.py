"""Candidate softenings of the eta-2 rank cut, for the mechanism sweep only.

The shipped `decide_successive_halving_shadow_race` keeps `ceil(n/2)` candidates
per look and cuts the rest purely on rank. These variants keep the same shape
but expose two knobs:

- ``keep_fraction`` -- keep ``ceil(n * keep_fraction)`` instead of half, so the
  cut boundary sits between more clearly separated candidates.
- ``spare_margin`` -- after the rank cut, put back any would-be-eliminated
  candidate whose paired mean at the cut prefix is within ``spare_margin`` of
  the last kept candidate, so a near-tie is carried rather than resolved.

With ``keep_fraction=0.5`` and ``spare_margin=0.0`` the decision is identical to
the shipped policy (a test pins this).
"""

from __future__ import annotations

import math

from .artifacts import Manifest
from .cohort import current_active_candidates
from .domain import (
    ReplayState,
    ShadowCandidateDecision,
    ShadowMethodVersion,
    ShadowRaceDecision,
    SuccessiveHalvingEvidence,
    TaskPrefix,
)
from .observations import comparable_prefix_observations, paired_difference
from .selection import select_top_candidates

_ETA2_VERSION: ShadowMethodVersion = "successive-halving-common-prefix-eta2-v1"


def decide_halving_variant(
    manifest: Manifest,
    state: ReplayState,
    prefix: TaskPrefix,
    *,
    keep_fraction: float = 0.5,
    spare_margin: float = 0.0,
) -> ShadowRaceDecision:
    cohort = current_active_candidates(state)
    observations = comparable_prefix_observations(state.observations, cohort, prefix)
    ranked = select_top_candidates(cohort, observations, len(cohort))
    by_id = {item.candidate_id: item for item in observations}

    target = max(manifest.finalists, math.ceil(len(cohort) * keep_fraction))
    target = min(target, len(cohort))
    boundary = ranked[target - 1]
    kept = {item.candidate_id for item in ranked[:target]}
    if spare_margin > 0.0:
        for candidate in ranked[target:]:
            difference = paired_difference(
                by_id[candidate.candidate_id], by_id[boundary.candidate_id]
            ).mean
            if difference >= -spare_margin:
                kept.add(candidate.candidate_id)

    ranks = {item.candidate_id: index + 1 for index, item in enumerate(ranked)}
    decisions = tuple(
        ShadowCandidateDecision(
            candidate.candidate_id,
            "continue" if candidate.candidate_id in kept else "eliminate",
            SuccessiveHalvingEvidence(
                ranks[candidate.candidate_id],
                len(cohort),
                target,
                candidate.candidate_id not in kept,
            ),
        )
        for candidate in cohort
    )
    return ShadowRaceDecision(
        0,
        prefix.prefix_id,
        tuple(item.observation_id for item in observations),
        boundary.candidate_id,
        decisions,
        "successive_halving",
        _ETA2_VERSION,
    )
