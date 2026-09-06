"""Deterministic active-elimination sampling over recorded shadow decisions."""

from __future__ import annotations

from hashlib import sha256

from .artifacts import (
    Manifest,
    PairedBootstrapPolicySpecification,
    SuccessiveHalvingPolicySpecification,
)
from .domain import (
    ApplyElimination,
    AuditedBoundaryReversal,
    CandidateEliminationAction,
    CohortRecord,
    EliminationDecisionMargin,
    PairedProbabilityMargin,
    ReplayState,
    ShadowCandidateDecision,
    ShadowRaceDecision,
    SuccessiveHalvingEvidence,
    SuccessiveHalvingRankMargin,
)
from .identity import canonical_json
from .observations import comparable_prefix_observations, paired_difference


def _spared_count(race: ShadowRaceDecision) -> int:
    """Near-tie candidates the spare-margin rule carried past the cut on this look."""
    return sum(
        1
        for item in race.decisions
        if isinstance(item.evidence, SuccessiveHalvingEvidence)
        and item.evidence.rank is not None
        and item.evidence.rank > item.evidence.target_survivor_count
        and item.disposition == "continue"
    )


def _decision_margin(
    manifest: Manifest, race: ShadowRaceDecision, decision: ShadowCandidateDecision
) -> EliminationDecisionMargin:
    evidence = decision.evidence
    if isinstance(evidence, SuccessiveHalvingEvidence):
        if evidence.rank is None:
            raise ValueError("a newly eliminated rank decision must carry a rank")
        return SuccessiveHalvingRankMargin(
            evidence.rank,
            evidence.target_survivor_count,
            evidence.rank - evidence.target_survivor_count,
            _spared_count(race),
        )
    if not isinstance(manifest.shadow_policy, PairedBootstrapPolicySpecification):
        raise ValueError("a paired probability margin requires the paired bootstrap policy")
    favorable_probability = evidence.favorable_resamples / evidence.total_resamples
    threshold = manifest.shadow_policy.elimination_probability_threshold
    return PairedProbabilityMargin(
        threshold, favorable_probability, threshold - favorable_probability
    )


def _newly_eliminated(decision: ShadowCandidateDecision) -> bool:
    if isinstance(decision.evidence, SuccessiveHalvingEvidence):
        return decision.evidence.newly_eliminated
    return True


def active_elimination_allocation(
    manifest: Manifest, state: ReplayState, race: ShadowRaceDecision
) -> ApplyElimination:
    if manifest.active_elimination is None:
        raise ValueError("active elimination is disabled")
    if race.policy_kind == "paired_bootstrap":
        if not isinstance(manifest.shadow_policy, PairedBootstrapPolicySpecification):
            raise ValueError("active elimination requires paired bootstrap policy")
    elif race.policy_kind == "successive_halving":
        if not isinstance(manifest.shadow_policy, SuccessiveHalvingPolicySpecification):
            raise ValueError("active elimination requires the successive halving policy")
    else:
        raise ValueError("active elimination does not support this shadow policy")
    if race.policy_version != manifest.active_elimination.shadow_method_version:
        raise ValueError("active elimination race policy version does not match the manifest")
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
        if not _newly_eliminated(decision):
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
                _decision_margin(manifest, race, decision),
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
