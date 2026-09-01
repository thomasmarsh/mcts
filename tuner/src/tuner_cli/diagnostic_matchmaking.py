"""Pure deterministic direct-match allocation policy."""

from __future__ import annotations

from .artifacts import Manifest
from .cohort import latest_completed_cohort
from .diagnostic_graph import build_diagnostic_graph
from .domain import EvaluateDiagnosticPair, ReplayState
from .identity import diagnostic_edge_id, diagnostic_pair_task
from .observations import comparable_prefix_observations


def next_diagnostic_allocation(
    manifest: Manifest, state: ReplayState
) -> EvaluateDiagnosticPair | None:
    cohort = latest_completed_cohort(state)
    if (
        cohort is None
        or state.finalists is not None
        or state.pending_resource_allocation is not None
    ):
        return None
    if state.compute.diagnostic.pair_attempts >= manifest.compute_budget.diagnostic_pair_attempts:
        return None
    observations = comparable_prefix_observations(
        state.observations, cohort.candidates, manifest.tuning_prefix
    )
    means = {item.candidate_id: item.estimate.mean for item in observations}
    ordered = tuple(
        sorted(cohort.candidates, key=lambda item: (-means[item.candidate_id], item.fingerprint))
    )
    rank = {item.candidate_id: index for index, item in enumerate(ordered)}
    graph = build_diagnostic_graph(cohort.candidates, state.diagnostic_pairs, rank)
    observed = {edge.edge_id: edge for edge in graph.edges}
    choices = [
        (
            left,
            right,
            diagnostic_edge_id(manifest.epoch.epoch_id, left, right),
            observed.get(diagnostic_edge_id(manifest.epoch.epoch_id, left, right)),
        )
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    ]
    unresolved = [item for item in choices if item[3] is None or item[3].unresolved]
    if not unresolved:
        return None
    unobserved = [item for item in unresolved if item[3] is None]
    if unobserved:
        left, right, _, _ = min(
            unobserved,
            key=lambda item: (
                abs(rank[item[0].candidate_id] - rank[item[1].candidate_id]),
                rank[item[0].candidate_id],
                rank[item[1].candidate_id],
                item[2],
            ),
        )
        reason = "graph_connectivity"
    else:
        boundary = [
            item
            for item in unresolved
            if (rank[item[0].candidate_id] < manifest.finalists)
            != (rank[item[1].candidate_id] < manifest.finalists)
        ]
        left, right, _, _edge = min(
            boundary or unresolved,
            key=lambda item: (
                len(item[3].pair_results) if item[3] is not None else 0,
                abs(rank[item[0].candidate_id] - rank[item[1].candidate_id]),
                rank[item[0].candidate_id],
                rank[item[1].candidate_id],
                item[2],
            ),
        )
        reason = "ranking_boundary" if boundary else "unresolved_edge"
    task = diagnostic_pair_task(
        manifest.epoch.epoch_id,
        manifest.task_seed,
        cohort.cohort_index,
        len(state.diagnostic_pairs),
        left,
        right,
        manifest.efforts["tuning"],
    )
    return EvaluateDiagnosticPair(cohort.cohort_index, reason, task)
