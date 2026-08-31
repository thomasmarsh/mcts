"""Contextual observations and their comparability guard."""

from __future__ import annotations

from .domain import Candidate, Estimate, Observation, ObservationContext, PairResult, TaskPrefix
from .effort import encode_effort
from .identity import observation_id
from .statistics import marginal_interval, pair_utility, paired_difference_values


def observation(
    candidate_id: str, context: ObservationContext, utilities: tuple[float, ...]
) -> Observation:
    if len(utilities) != context.task_prefix.length:
        raise ValueError("observation utility count does not match task prefix")
    estimate = marginal_interval(utilities)
    identity_context = {
        "objective_epoch_id": context.objective_epoch_id,
        "phase": context.phase,
        "prefix_id": context.task_prefix.prefix_id,
        "task_ids": context.task_prefix.task_ids,
        "search_effort": encode_effort(context.search_effort),
    }
    return Observation(
        observation_id(candidate_id, identity_context, utilities, estimate),
        candidate_id,
        context,
        utilities,
        estimate,
    )


def contextual_observation(
    candidate: Candidate, context: ObservationContext, pairs: list[PairResult]
) -> Observation:
    """Build an observation from a complete, ordered common task prefix."""
    by_task = {pair.task.task_case.task_id: pair for pair in pairs}
    if set(by_task) != set(context.task_prefix.task_ids):
        raise ValueError("observation needs a complete common task prefix")
    utilities = tuple(pair_utility(by_task[task_id]) for task_id in context.task_prefix.task_ids)
    return observation(candidate.candidate_id, context, utilities)


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


def comparable_prefix_observations(
    observations: tuple[Observation, ...], candidates: tuple[Candidate, ...], prefix: TaskPrefix
) -> tuple[Observation, ...]:
    """Select exactly one tuning observation per candidate at one common prefix."""
    result: list[Observation] = []
    for candidate in candidates:
        matches = [
            item
            for item in observations
            if item.phase == "tuning"
            and item.candidate_id == candidate.candidate_id
            and item.context.task_prefix.prefix_id == prefix.prefix_id
        ]
        if len(matches) != 1:
            raise ValueError("comparable prefix needs one observation per candidate")
        result.append(matches[0])
    if result and any(item.context.task_prefix != prefix for item in result):
        raise ValueError("comparable prefix evidence is mixed")
    return tuple(result)
