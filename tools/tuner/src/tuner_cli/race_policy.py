"""Typed dispatch for the frozen shadow-race policies."""

from __future__ import annotations

from .artifacts import Manifest, PairedBootstrapPolicySpecification
from .domain import ReplayState, ShadowRaceDecision, TaskPrefix
from .shadow import decide_paired_bootstrap_shadow_race
from .successive_halving import decide_successive_halving_shadow_race


def shadow_prefix_eligible(manifest: Manifest, prefix: TaskPrefix) -> bool:
    return (
        prefix in manifest.tuning_blocks
        and prefix.length >= manifest.shadow_policy.minimum_eligible_prefix_pairs
        and prefix != manifest.tuning_prefix
    )


def decide_shadow_race(
    manifest: Manifest, state: ReplayState, cohort_index: int, prefix: TaskPrefix
) -> ShadowRaceDecision:
    if not shadow_prefix_eligible(manifest, prefix):
        raise ValueError("shadow race uses an ineligible tuning prefix")
    if isinstance(manifest.shadow_policy, PairedBootstrapPolicySpecification):
        return decide_paired_bootstrap_shadow_race(manifest, state, cohort_index, prefix)
    return decide_successive_halving_shadow_race(manifest, state, cohort_index, prefix)
