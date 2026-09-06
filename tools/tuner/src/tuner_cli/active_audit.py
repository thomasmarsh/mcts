"""Completed-run projection for enforced active elimination."""

from __future__ import annotations

from .artifacts import Manifest
from .codec import JsonObject, JsonValue
from .domain import (
    CandidateEliminationAction,
    EliminationDecisionMargin,
    PairedProbabilityMargin,
    ReplayState,
)
from .elimination import audited_boundary_reversals


def _margin_json(margin: EliminationDecisionMargin) -> JsonObject:
    if isinstance(margin, PairedProbabilityMargin):
        return {
            "kind": "paired_probability",
            "elimination_probability_threshold": margin.elimination_probability_threshold,
            "favorable_probability": margin.favorable_probability,
            "threshold_minus_probability": margin.threshold_minus_probability,
        }
    return {
        "kind": "successive_halving_rank",
        "rank": margin.rank,
        "target_survivor_count": margin.target_survivor_count,
        "ranks_below_cutoff": margin.ranks_below_cutoff,
        "spared_count": margin.spared_count,
    }


def _action_json(action: CandidateEliminationAction) -> JsonObject:
    return {
        "candidate_id": action.candidate_id,
        "action": action.action,
        "margin": _margin_json(action.margin),
    }


def _suffix_unique_pairs(manifest: Manifest, prefix_id: str) -> int:
    """Manifest tuning cases strictly after the given elimination prefix."""
    block = next(item for item in manifest.tuning_blocks if item.prefix_id == prefix_id)
    return manifest.tuning_prefix.length - block.length


def build_active_audit(manifest: Manifest, state: ReplayState) -> JsonObject:
    if manifest.active_elimination is None:
        raise ValueError("active elimination is disabled")
    batches: list[JsonValue] = [
        {
            "cohort_index": batch.cohort_index,
            "prefix_id": batch.prefix_id,
            "actions": [_action_json(action) for action in batch.actions],
        }
        for batch in state.elimination_allocations
    ]
    actions = [action for batch in state.elimination_allocations for action in batch.actions]
    suffix_by_action = [
        (action, _suffix_unique_pairs(manifest, batch.prefix_id))
        for batch in state.elimination_allocations
        for action in batch.actions
    ]
    gross_nominal_suffix_unique_pairs = sum(count for _, count in suffix_by_action)
    audit_continuation_suffix_unique_pairs = sum(
        count for action, count in suffix_by_action if action.action == "audit_continue"
    )
    reversals = [
        reversal
        for cohort in state.completed_cohorts
        for reversal in audited_boundary_reversals(manifest, state, cohort)
    ]
    elimination_decisions = len(actions)
    estimated_reversals = sum(1 / manifest.active_elimination.audit_probability for _ in reversals)
    suspension = state.active_elimination_suspension
    return {
        "policy": {
            "policy_kind": manifest.active_elimination.shadow_policy_kind,
            "policy_version": manifest.active_elimination.shadow_method_version,
            "audit_probability": manifest.active_elimination.audit_probability,
            "sampler_version": manifest.active_elimination.sampler_version,
            "safety_rule_version": manifest.active_elimination.safety_rule_version,
        },
        "applied_batches": batches,
        "summary": {
            "pruned": sum(action.action == "prune" for action in actions),
            "audit_continued": sum(action.action == "audit_continue" for action in actions),
            "nominal_eliminations": elimination_decisions,
            "gross_nominal_suffix_unique_pairs": gross_nominal_suffix_unique_pairs,
            "audit_continuation_suffix_unique_pairs": audit_continuation_suffix_unique_pairs,
            "planned_unique_pair_savings": (
                gross_nominal_suffix_unique_pairs - audit_continuation_suffix_unique_pairs
            ),
            "elimination_decisions": elimination_decisions,
            "audited_continuations": sum(action.action == "audit_continue" for action in actions),
            "audited_boundary_reversals": len(reversals),
            "observed_audit_reversal_rate": (
                None
                if not any(action.action == "audit_continue" for action in actions)
                else len(reversals) / sum(action.action == "audit_continue" for action in actions)
            ),
            "estimated_boundary_reversals": estimated_reversals,
            "estimated_reversal_rate": (
                None
                if elimination_decisions == 0
                else min(1.0, estimated_reversals / elimination_decisions)
            ),
        },
        "suspended": suspension is not None,
        "active_interval": {
            "first_cohort_index": 0,
            "last_cohort_index": (None if suspension is None else suspension.after_cohort_index),
        },
        "suspension": (
            None
            if suspension is None
            else {
                "after_cohort_index": suspension.after_cohort_index,
                "triggering_candidate_ids": list(suspension.triggering_candidate_ids),
                "triggering_prefix_ids": list(suspension.triggering_prefix_ids),
            }
        ),
        "audited_boundary_reversals": [
            {
                "cohort_index": reversal.cohort_index,
                "candidate_id": reversal.candidate_id,
                "prefix_id": reversal.prefix_id,
                "boundary_candidate_id": reversal.boundary_candidate_id,
                "maximum_prefix_paired_mean_difference": (
                    reversal.maximum_prefix_paired_mean_difference
                ),
            }
            for reversal in reversals
        ],
        "actual_compute": {
            "tuning_pair_attempts": state.compute.tuning.pair_attempts,
            "tuning_physical_games": state.compute.tuning.physical_games,
            "tuning_search_iterations": state.compute.tuning.search_iterations,
            "tuning_wall_time_ms": state.compute.tuning.wall_time_ms,
        },
    }
