"""Read models for direct, candidate-vs-candidate diagnostic evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .domain import Candidate, DiagnosticPairResult, Estimate
from .identity import fingerprint
from .statistics import marginal_interval


@dataclass(frozen=True, slots=True)
class DiagnosticEdge:
    edge_id: str
    left_candidate_id: str
    right_candidate_id: str
    pair_results: tuple[DiagnosticPairResult, ...]
    estimate: Estimate | None
    material_direction: str | None

    @property
    def unresolved(self) -> bool:
        return self.estimate is None or self.estimate.lower <= 0.5 <= self.estimate.upper


@dataclass(frozen=True, slots=True)
class MaterialCycleComponent:
    candidate_ids: tuple[str, ...]
    witness_cycle_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticGraph:
    edges: tuple[DiagnosticEdge, ...]
    material_cycle_components: tuple[MaterialCycleComponent, ...]
    fingerprint: str


def build_diagnostic_graph(
    candidates: tuple[Candidate, ...], pairs: tuple[DiagnosticPairResult, ...], rank: dict[str, int]
) -> DiagnosticGraph:
    grouped: dict[str, list[DiagnosticPairResult]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.task.edge_id].append(pair)
    by_id = {item.candidate_id: item for item in candidates}
    edges: list[DiagnosticEdge] = []
    for edge_id, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda item: item.task.ordinal))
        task = ordered[0].task
        if task.left_candidate_id not in by_id or task.right_candidate_id not in by_id:
            raise ValueError("diagnostic pair is outside the cohort")
        utilities = tuple(_utility(item) for item in ordered)
        estimate = marginal_interval(utilities)
        direction = (
            "left_to_right"
            if estimate.lower > 0.5
            else "right_to_left"
            if estimate.upper < 0.5
            else None
        )
        edges.append(
            DiagnosticEdge(
                edge_id,
                task.left_candidate_id,
                task.right_candidate_id,
                ordered,
                estimate,
                direction,
            )
        )
    edges.sort(
        key=lambda item: (
            min(rank[item.left_candidate_id], rank[item.right_candidate_id]),
            max(rank[item.left_candidate_id], rank[item.right_candidate_id]),
            item.edge_id,
        )
    )
    components = _components(tuple(edges), rank)
    digest = fingerprint(
        {
            "edges": [(x.edge_id, x.estimate, x.material_direction) for x in edges],
            "components": components,
        }
    )
    return DiagnosticGraph(tuple(edges), components, digest)


def _utility(result: DiagnosticPairResult) -> float:
    wins = sum(game.outcome == "candidate_win" for game in result.games)
    draws = sum(game.outcome == "draw" for game in result.games)
    return (wins + 0.5 * draws) / 2


def _components(
    edges: tuple[DiagnosticEdge, ...], rank: dict[str, int]
) -> tuple[MaterialCycleComponent, ...]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    nodes = {node for edge in edges for node in (edge.left_candidate_id, edge.right_candidate_id)}
    for edge in edges:
        if edge.material_direction == "left_to_right":
            adjacency[edge.left_candidate_id].append(edge.right_candidate_id)
        elif edge.material_direction == "right_to_left":
            adjacency[edge.right_candidate_id].append(edge.left_candidate_id)

    def reachable(left: str, right: str) -> tuple[str, ...] | None:
        return _path(adjacency, left, right, rank)

    found: list[MaterialCycleComponent] = []
    unseen = set(nodes)
    while unseen:
        root = min(unseen, key=lambda x: rank[x])
        component = {
            node
            for node in nodes
            if reachable(root, node) is not None and reachable(node, root) is not None
        }
        unseen -= component or {root}
        if len(component) >= 3:
            members = tuple(sorted(component, key=lambda x: rank[x]))
            witness = _witness(adjacency, members, rank)
            found.append(MaterialCycleComponent(members, witness))
    return tuple(sorted(found, key=lambda x: min(rank[item] for item in x.candidate_ids)))


def _path(
    adjacency: dict[str, list[str]], start: str, target: str, rank: dict[str, int]
) -> tuple[str, ...] | None:
    queue: list[tuple[str, tuple[str, ...]]] = [(start, (start,))]
    seen = {start}
    while queue:
        node, path = queue.pop(0)
        if node == target:
            return path
        for next_node in sorted(adjacency[node], key=lambda x: rank[x]):
            if next_node not in seen:
                seen.add(next_node)
                queue.append((next_node, path + (next_node,)))
    return None


def _witness(
    adjacency: dict[str, list[str]], members: tuple[str, ...], rank: dict[str, int]
) -> tuple[str, ...]:
    for start in members:
        for next_node in sorted(adjacency[start], key=lambda x: rank[x]):
            path = _path(adjacency, next_node, start, rank)
            if path is not None:
                return (start,) + path
    raise ValueError("cyclic component has no witness")
