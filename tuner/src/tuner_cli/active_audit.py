"""Completed-run projection for enforced active elimination."""

from __future__ import annotations

from .artifacts import Manifest
from .codec import JsonObject, JsonValue
from .domain import ReplayState


def build_active_audit(manifest: Manifest, state: ReplayState) -> JsonObject:
    if manifest.active_elimination is None:
        raise ValueError("active elimination is disabled")
    batches: list[JsonValue] = [
        {
            "cohort_index": batch.cohort_index,
            "prefix_id": batch.prefix_id,
            "actions": [
                {
                    "candidate_id": action.candidate_id,
                    "action": action.action,
                    "decision_margin": action.decision_margin,
                }
                for action in batch.actions
            ],
        }
        for batch in state.elimination_allocations
    ]
    actions = [action for batch in state.elimination_allocations for action in batch.actions]
    return {
        "policy": {
            "audit_probability": manifest.active_elimination.audit_probability,
            "sampler_version": manifest.active_elimination.sampler_version,
            "automatic_suspension": False,
        },
        "applied_batches": batches,
        "summary": {
            "pruned": sum(action.action == "prune" for action in actions),
            "audit_continued": sum(action.action == "audit_continue" for action in actions),
        },
    }
