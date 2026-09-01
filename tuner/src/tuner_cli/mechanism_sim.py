"""Deterministic simulation of the shadow-race rule over synthetic cohorts.

The mechanism question -- does the eta-2 rank cut at the 12-pair common prefix
evict a candidate that would have reached the cohort's top set at the maximum
prefix -- is a property of a selection rule over noisy paired estimates, not of
any game. This module samples synthetic cohorts whose realism comes from
`mechanism_calibration` (recorded Druid pair outcomes), draws consistent 12- and
18-pair observation sets from the same latent draws, and runs the shipped
`decide_shadow_race` on a minimal hand-built `ReplayState`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Literal

from .artifacts import (
    Manifest,
    PairedBootstrapPolicySpecification,
    SuccessiveHalvingPolicySpecification,
)
from .domain import (
    Candidate,
    Observation,
    ObservationContext,
    ObservationFrontier,
    Proposal,
    ProposalProvenance,
    ReplayState,
    ShadowRaceDecision,
    TaskPrefix,
)
from .mechanism_calibration import Calibration
from .observations import observation, paired_difference
from .race_policy import decide_shadow_race
from .selection import select_top_candidates
from .shadow import paired_stratum_differences
from .shadow_audit import DANGEROUS_FLIP_MARGIN

# The task-09 gate froze the paired policy at 4096 resamples. The sweep uses
# paired elimination only as a same-trial *rate* baseline (clause 3), and only a
# borderline fraction of decisions flip between 512 and 4096 draws, so the sweep
# runs it at 512 to keep tens of thousands of trials tractable. This is a
# deliberate, documented deviation from the frozen count.
SWEEP_PAIRED_RESAMPLES = 512


def as_paired(manifest: Manifest, resamples: int = SWEEP_PAIRED_RESAMPLES) -> Manifest:
    policy = PairedBootstrapPolicySpecification(
        kind="paired_bootstrap",
        practical_effect_margin=0.0,
        elimination_probability_threshold=0.05,
        resamples=resamples,
        method_version="stratified-paired-bootstrap-all-strata-v2",
        minimum_eligible_prefix_pairs=12,
    )
    return replace(manifest, shadow_policy=policy)


def as_halving(manifest: Manifest) -> Manifest:
    if isinstance(manifest.shadow_policy, SuccessiveHalvingPolicySpecification):
        return manifest
    policy = SuccessiveHalvingPolicySpecification(
        kind="successive_halving",
        method_version="successive-halving-common-prefix-eta2-v1",
        reduction_factor=2,
        practical_effect_margin=manifest.shadow_policy.practical_effect_margin,
        minimum_eligible_prefix_pairs=manifest.shadow_policy.minimum_eligible_prefix_pairs,
        survivor_floor=manifest.finalists,
        ranking_rule="tuning-point-estimate-fingerprint-v1",
    )
    return replace(manifest, shadow_policy=policy)


def draw_pair(calibration: Calibration, propensity: float, rng: random.Random) -> float:
    """Sample one pair utility from the calibrated CDF for the bin holding
    `propensity`; the nearest populated bin is used at the edges."""
    chosen = calibration.bin_for(propensity)
    draw = rng.random()
    for value, cumulative in chosen.cdf:
        if draw <= cumulative:
            return value
    return chosen.cdf[-1][0]


@dataclass(frozen=True, slots=True)
class SyntheticCohort:
    candidates: tuple[Candidate, ...]
    propensities: tuple[dict[str, float], ...]
    latent_strength: tuple[float, ...]


def _stratum_order(manifest: Manifest) -> tuple[str, ...]:
    seen: list[str] = []
    for case in manifest.prefix_cases("tuning"):
        if case.stratum_id not in seen:
            seen.append(case.stratum_id)
    return tuple(seen)


def _candidate(rng: random.Random) -> Candidate:
    token = f"{rng.getrandbits(64):016x}"
    return Candidate(f"candidate-sim-{token}", f"fingerprint-sim-{token}", "{}")


def sample_cohort(
    calibration: Calibration,
    manifest: Manifest,
    rng: random.Random,
    boundary_gap: float,
    spread_scale: float,
) -> SyntheticCohort:
    """Draw `cohort_size` candidates. `boundary_gap` fixes the latent strength
    gap across the eta-2 cut -- between the last kept and first eliminated
    candidate -- exactly, so the near-tie (and inverted) regime is sampled on
    purpose rather than by chance; `spread_scale` scales the strength spread."""
    size = manifest.cohort_size
    strata = _stratum_order(manifest)
    mean = calibration.strength_mean
    std = calibration.strength_std * spread_scale
    rho = max(0.0, min(1.0, calibration.deviation_correlation))
    deviation = calibration.deviation_std

    kept = max(manifest.finalists, (size + 1) // 2)
    strengths = sorted((rng.gauss(mean, std) for _ in range(size)), reverse=True)
    shift = boundary_gap - (strengths[kept - 1] - strengths[kept])
    strengths = strengths[:kept] + [value + shift for value in strengths[kept:]]

    candidates: list[Candidate] = []
    propensities: list[dict[str, float]] = []
    for strength in strengths:
        shared = rng.gauss(0.0, 1.0)
        vector: dict[str, float] = {}
        for stratum in strata:
            idiosyncratic = rng.gauss(0.0, 1.0)
            noise = deviation * (math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * idiosyncratic)
            vector[stratum] = min(0.98, max(0.02, strength + noise))
        candidates.append(_candidate(rng))
        propensities.append(vector)
    return SyntheticCohort(tuple(candidates), tuple(propensities), tuple(strengths))


def draw_trial(
    cohort: SyntheticCohort,
    calibration: Calibration,
    manifest: Manifest,
    rng: random.Random,
) -> dict[int, tuple[Observation, ...]]:
    """One full latent draw per candidate; the 12-pair prefix is a true prefix
    of the 18-pair maximum, so within a trial the two are consistent."""
    cases = manifest.prefix_cases("tuning")
    strata_by_case = [case.stratum_id for case in cases]
    draws: dict[str, list[float]] = {}
    for candidate, vector in zip(cohort.candidates, cohort.propensities, strict=True):
        draws[candidate.candidate_id] = [
            draw_pair(calibration, vector[stratum], rng) for stratum in strata_by_case
        ]
    result: dict[int, tuple[Observation, ...]] = {}
    for block in manifest.tuning_blocks:
        context = ObservationContext(
            manifest.epoch.epoch_id, "tuning", block, manifest.efforts["tuning"]
        )
        result[block.length] = tuple(
            observation(
                candidate.candidate_id,
                context,
                tuple(draws[candidate.candidate_id][: block.length]),
            )
            for candidate in cohort.candidates
        )
    return result


def _dummy_frontier(manifest: Manifest) -> ObservationFrontier:
    block = manifest.tuning_blocks[0]
    return ObservationFrontier(
        "frontier-sim",
        manifest.epoch.epoch_id,
        block.prefix_id,
        block.task_ids,
        manifest.efforts["tuning"],
        (),
    )


def _dummy_provenance() -> ProposalProvenance:
    return ProposalProvenance("bootstrap_random", "sim-v1", 1, None, None, None, None, None)


def build_active_state(
    manifest: Manifest,
    cohort: SyntheticCohort,
    observations: dict[int, tuple[Observation, ...]],
) -> ReplayState:
    """A minimal open state with one active complete cohort, no completed
    cohorts and no elites -- exactly what `decide_shadow_race` inspects."""
    frontier = _dummy_frontier(manifest)
    provenance = _dummy_provenance()
    proposals = tuple(
        Proposal(index, 0, index, candidate, frontier, provenance)
        for index, candidate in enumerate(cohort.candidates)
    )
    accepted: Literal["accepted", "rejected"] = "accepted"
    dispositions: list[tuple[int, Literal["accepted", "rejected"]]] = [
        (index, accepted) for index in range(len(cohort.candidates))
    ]
    flat = tuple(item for group in observations.values() for item in group)
    return ReplayState(
        proposals=proposals,
        dispositions=tuple(dispositions),
        completed_cohorts=(),
        active_elites=(),
        completed_pairs=(),
        observations=flat,
        finalists=None,
        terminal_status="open",
        tuning_block_index=0,
        pending_resource_allocation=None,
    )


@dataclass(frozen=True, slots=True)
class TrialClassification:
    policy: str
    eliminated: int
    top_set_false_evictions: int
    boundary_reversals: int
    rule_tie_evictions: int
    per_stratum_dangerous_flips: int
    unique_pairs_saved: int


def _stratum_means(manifest: Manifest, left: Observation, right: Observation) -> dict[str, float]:
    return {
        item.stratum_id: sum(item.values) / len(item.values)
        for item in paired_stratum_differences(manifest, left, right)
    }


def classify_trial(
    manifest: Manifest,
    decision: ShadowRaceDecision,
    early: tuple[Observation, ...],
    maximum: tuple[Observation, ...],
    cohort: tuple[Candidate, ...],
    early_prefix: TaskPrefix,
    maximum_prefix: TaskPrefix,
) -> TrialClassification:
    """Label every eliminated candidate against the maximum-prefix evidence with
    the same eviction metrics `shadow_audit` reports (top-set false eviction,
    boundary reversal, fingerprint-tie eviction, per-stratum dangerous flip)."""
    early_by_id = {item.candidate_id: item for item in early}
    maximum_by_id = {item.candidate_id: item for item in maximum}
    top_ids = {
        item.candidate_id for item in select_top_candidates(cohort, maximum, manifest.finalists)
    }
    boundary_id = decision.boundary_candidate_id
    eliminated = [
        item.candidate_id for item in decision.decisions if item.disposition == "eliminate"
    ]

    top_false = reversals = ties = flips = 0
    for candidate_id in eliminated:
        overall = paired_difference(maximum_by_id[candidate_id], maximum_by_id[boundary_id]).mean
        if overall > 0.0:
            reversals += 1
        elif overall == 0.0:
            ties += 1
        if candidate_id in top_ids:
            top_false += 1
        early_means = _stratum_means(manifest, early_by_id[candidate_id], early_by_id[boundary_id])
        maximum_means = _stratum_means(
            manifest, maximum_by_id[candidate_id], maximum_by_id[boundary_id]
        )
        flips += sum(
            1
            for stratum, value in early_means.items()
            if value <= -DANGEROUS_FLIP_MARGIN and maximum_means[stratum] >= 0.0
        )

    saved = len(eliminated) * (maximum_prefix.length - early_prefix.length)
    return TrialClassification(
        decision.policy_kind, len(eliminated), top_false, reversals, ties, flips, saved
    )


def run_trial(
    calibration: Calibration,
    manifest: Manifest,
    rng: random.Random,
    boundary_gap: float,
    spread_scale: float,
    paired_resamples: int = SWEEP_PAIRED_RESAMPLES,
) -> dict[str, TrialClassification]:
    halving_manifest = as_halving(manifest)
    paired_manifest = as_paired(manifest, paired_resamples)
    early_prefix = next(block for block in manifest.tuning_blocks if block.length == 12)
    maximum_prefix = manifest.tuning_prefix

    cohort = sample_cohort(calibration, manifest, rng, boundary_gap, spread_scale)
    observations = draw_trial(cohort, calibration, manifest, rng)
    state = build_active_state(manifest, cohort, observations)

    early = observations[early_prefix.length]
    maximum = observations[maximum_prefix.length]

    result: dict[str, TrialClassification] = {}
    for name, policy_manifest in (("halving", halving_manifest), ("paired", paired_manifest)):
        decision = decide_shadow_race(policy_manifest, state, 0, early_prefix)
        result[name] = classify_trial(
            policy_manifest,
            decision,
            early,
            maximum,
            cohort.candidates,
            early_prefix,
            maximum_prefix,
        )
    return result
