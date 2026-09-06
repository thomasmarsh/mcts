"""Deterministic finalist selection over comparable tuning observations."""

from __future__ import annotations

from .diagnostic_graph import DiagnosticGraph
from .domain import Candidate, Observation
from .observations import comparable


def select_top_candidates(
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


def select_validation_shortlist(
    cohort: tuple[Candidate, ...],
    values: tuple[Observation, ...],
    count: int,
    graph: DiagnosticGraph,
) -> tuple[tuple[Candidate, ...], str | None, str | None]:
    baseline = select_top_candidates(cohort, values, count)
    if count < 2:
        return baseline, None, None
    baseline_ids = {item.candidate_id for item in baseline}
    eligible = {
        candidate_id
        for component in graph.material_cycle_components
        if baseline_ids.intersection(component.candidate_ids)
        for candidate_id in component.candidate_ids
        if candidate_id not in baseline_ids
    }
    if not eligible:
        return baseline, None, None
    means = {item.candidate_id: item.estimate.mean for item in values}
    by_id = {item.candidate_id: item for item in cohort}
    reserve = min(eligible, key=lambda item: (-means[item], by_id[item].fingerprint))
    return baseline[:-1] + (by_id[reserve],), reserve, baseline[-1].candidate_id
