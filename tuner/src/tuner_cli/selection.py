"""Deterministic finalist selection over comparable tuning observations."""

from __future__ import annotations

from .domain import Candidate, Observation
from .observations import comparable


def select_finalists(
    cohort: tuple[Candidate, ...], values: tuple[Observation, ...], count: int
) -> tuple[Candidate, ...]:
    if len(values) != len(cohort) or {item.candidate_id for item in values} != {
        item.candidate_id for item in cohort
    }:
        raise ValueError("finalist selection needs all tuning observations")
    for value in values[1:]:
        comparable(values[0], value)
    means = {item.candidate_id: item.estimate.mean for item in values}
    return tuple(
        sorted(cohort, key=lambda item: (-means[item.candidate_id], item.fingerprint))[:count]
    )
