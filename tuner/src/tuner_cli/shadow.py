"""Deterministic evidence-only paired shadow race policy."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .artifacts import Manifest, PairedBootstrapPolicySpecification
from .cohort import current_active_candidates
from .domain import (
    Observation,
    PairedBootstrapEvidence,
    ReplayState,
    ShadowCandidateDecision,
    ShadowRaceDecision,
    TaskPrefix,
)
from .identity import canonical_json
from .observations import comparable, comparable_prefix_observations
from .selection import select_top_candidates


@dataclass(frozen=True, slots=True)
class StratumDifferences:
    stratum_id: str
    task_ids: tuple[str, ...]
    values: tuple[float, ...]


def paired_stratum_differences(
    manifest: Manifest, left: Observation, right: Observation
) -> tuple[StratumDifferences, ...]:
    """Group the paired candidate-minus-boundary utilities by task stratum.

    This is policy-neutral: the maximum-prefix audit computes per-stratum mean
    differences and reversal labels for every shadow policy. Only the bootstrap
    resampling below is specific to the paired-bootstrap policy.
    """
    comparable(left, right)
    prefix = left.context.task_prefix
    cases = {case.task_id: case for case in manifest.prefix_cases("tuning")[: prefix.length]}
    if prefix.task_ids != tuple(
        case.task_id for case in manifest.prefix_cases("tuning")[: prefix.length]
    ):
        raise ValueError("shadow observations do not use a declared tuning prefix")
    grouped: dict[str, list[tuple[str, float]]] = {}
    for task_id, left_value, right_value in zip(
        prefix.task_ids, left.pair_utilities, right.pair_utilities, strict=True
    ):
        grouped.setdefault(cases[task_id].stratum_id, []).append(
            (task_id, left_value - right_value)
        )
    return tuple(
        StratumDifferences(
            stratum, tuple(task for task, _ in values), tuple(value for _, value in values)
        )
        for stratum, values in grouped.items()
    )


def _source_index(
    manifest: Manifest,
    cohort_index: int,
    prefix: TaskPrefix,
    stratum_id: str,
    replicate: int,
    draw: int,
    length: int,
) -> int:
    payload = canonical_json(
        [
            manifest.shadow_policy.method_version,
            manifest.epoch.epoch_id,
            cohort_index,
            prefix.prefix_id,
            stratum_id,
            replicate,
            draw,
        ]
    )
    return int.from_bytes(sha256(payload.encode()).digest()[:8], "big") % length


def _resample_indexes(
    manifest: Manifest,
    cohort_index: int,
    prefix: TaskPrefix,
    strata: tuple[StratumDifferences, ...],
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Build the common bootstrap draws once so every candidate shares a replicate."""
    if not isinstance(manifest.shadow_policy, PairedBootstrapPolicySpecification):
        raise ValueError("paired bootstrap evidence requires paired policy")
    return {
        stratum.stratum_id: tuple(
            tuple(
                _source_index(
                    manifest,
                    cohort_index,
                    prefix,
                    stratum.stratum_id,
                    replicate,
                    draw,
                    len(stratum.values),
                )
                for draw in range(len(stratum.values))
            )
            for replicate in range(manifest.shadow_policy.resamples)
        )
        for stratum in strata
    }


def favorable_resamples_by_stratum(
    manifest: Manifest,
    cohort_index: int,
    prefix: TaskPrefix,
    differences: tuple[StratumDifferences, ...],
) -> tuple[tuple[str, int], ...]:
    """Return deterministic within-stratum favorable bootstrap counts."""
    if not isinstance(manifest.shadow_policy, PairedBootstrapPolicySpecification):
        raise ValueError("paired bootstrap evidence requires paired policy")
    indexes = _resample_indexes(manifest, cohort_index, prefix, differences)
    return tuple(
        (
            stratum.stratum_id,
            sum(
                sum(stratum.values[index] for index in indexes[stratum.stratum_id][replicate])
                / len(stratum.values)
                >= -manifest.shadow_policy.practical_effect_margin
                for replicate in range(manifest.shadow_policy.resamples)
            ),
        )
        for stratum in differences
    )


def decide_paired_bootstrap_shadow_race(
    manifest: Manifest, state: ReplayState, cohort_index: int, prefix: TaskPrefix
) -> ShadowRaceDecision:
    if not isinstance(manifest.shadow_policy, PairedBootstrapPolicySpecification):
        raise ValueError("paired bootstrap policy is not selected")
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
    top = select_top_candidates(cohort, observations, manifest.finalists)
    boundary = top[-1]
    by_id = {item.candidate_id: item for item in observations}
    differences = {
        candidate.candidate_id: paired_stratum_differences(
            manifest, by_id[candidate.candidate_id], by_id[boundary.candidate_id]
        )
        for candidate in cohort
    }
    decisions: list[ShadowCandidateDecision] = []
    for candidate in cohort:
        candidate_differences = differences[candidate.candidate_id]
        indexes = _resample_indexes(manifest, cohort_index, prefix, candidate_differences)
        favorable = 0
        for replicate in range(manifest.shadow_policy.resamples):
            total = 0.0
            for stratum in candidate_differences:
                total += sum(
                    stratum.values[index] for index in indexes[stratum.stratum_id][replicate]
                )
            if total / prefix.length >= -manifest.shadow_policy.practical_effect_margin:
                favorable += 1
        if candidate.candidate_id in {item.candidate_id for item in state.active_elites}:
            disposition = "protected"
        elif candidate.candidate_id in {item.candidate_id for item in top}:
            disposition = "continue"
        elif _eliminated(manifest, cohort_index, prefix, candidate_differences, favorable):
            disposition = "eliminate"
        else:
            disposition = "continue"
        decisions.append(
            ShadowCandidateDecision(
                candidate.candidate_id,
                disposition,
                PairedBootstrapEvidence(favorable, manifest.shadow_policy.resamples),
            )
        )
    return ShadowRaceDecision(
        cohort_index,
        prefix.prefix_id,
        observation_ids,
        boundary.candidate_id,
        tuple(decisions),
        "paired_bootstrap",
        manifest.shadow_policy.method_version,
    )


def _eliminated(
    manifest: Manifest,
    cohort_index: int,
    prefix: TaskPrefix,
    differences: tuple[StratumDifferences, ...],
    favorable: int,
) -> bool:
    if not isinstance(manifest.shadow_policy, PairedBootstrapPolicySpecification):
        raise ValueError("paired bootstrap policy is not selected")
    threshold = manifest.shadow_policy.elimination_probability_threshold
    if manifest.shadow_policy.method_version == "stratified-paired-bootstrap-v1":
        return favorable / manifest.shadow_policy.resamples < threshold
    return all(
        count / manifest.shadow_policy.resamples < threshold
        for _, count in favorable_resamples_by_stratum(manifest, cohort_index, prefix, differences)
    )


# Compatibility imports remain for callers that only exercise the paired numerical policy.
# Allocation and replay use race_policy's typed dispatcher.
def shadow_prefix_eligible(manifest: Manifest, prefix: TaskPrefix) -> bool:
    from .race_policy import shadow_prefix_eligible as eligible

    return eligible(manifest, prefix)


def decide_shadow_race(
    manifest: Manifest, state: ReplayState, cohort_index: int, prefix: TaskPrefix
) -> ShadowRaceDecision:
    from .race_policy import decide_shadow_race as decide

    return decide(manifest, state, cohort_index, prefix)
