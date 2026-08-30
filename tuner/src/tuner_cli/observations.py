"""Contextual observations and their comparability guard."""

from __future__ import annotations

from .domain import Estimate, Observation, ObservationContext
from .statistics import marginal_interval, paired_difference_values


def observation(
    candidate_id: str, context: ObservationContext, utilities: tuple[float, ...]
) -> Observation:
    if len(utilities) != context.task_prefix.length:
        raise ValueError("observation utility count does not match task prefix")
    return Observation(candidate_id, context, utilities, marginal_interval(utilities))


def comparable(left: Observation, right: Observation) -> None:
    a, b = left.context, right.context
    if a.objective_epoch_id != b.objective_epoch_id:
        raise ValueError("observations differ on epoch")
    if a.phase != b.phase:
        raise ValueError("observations differ on phase")
    if (
        a.task_prefix.prefix_id != b.task_prefix.prefix_id
        or a.task_prefix.task_ids != b.task_prefix.task_ids
    ):
        raise ValueError("observations differ on task_prefix")
    if a.search_effort != b.search_effort:
        raise ValueError("observations differ on search_effort")


def paired_difference(left: Observation, right: Observation) -> Estimate:
    comparable(left, right)
    return paired_difference_values(left.pair_utilities, right.pair_utilities)
