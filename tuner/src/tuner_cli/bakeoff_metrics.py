"""Pure, deterministic aggregates for completed proposer bake-off children."""

from __future__ import annotations

from dataclasses import dataclass

from .bakeoff_artifacts import BAKEOFF_BASELINE, BAKEOFF_CHALLENGER, BakeoffDecision
from .codec import JsonObject, JsonValue
from .domain import Estimate
from .identity import canonical_json, fingerprint
from .statistics import bootstrap_mean_interval

REFERENCE_SET_RULE = "union-returned-finalists-v1"
DECISION_RULE = "irace-vs-smac-largest-budget-v1"
_CONTRASTS: tuple[tuple[str, str], ...] = (
    (BAKEOFF_CHALLENGER, BAKEOFF_BASELINE),
    (BAKEOFF_BASELINE, "random"),
    (BAKEOFF_BASELINE, "qmc"),
    (BAKEOFF_CHALLENGER, "random"),
    (BAKEOFF_CHALLENGER, "qmc"),
)
_METRICS = ("held_out_best_score", "simple_regret", "top_set_recall")


@dataclass(frozen=True, slots=True)
class ChildFact:
    cell_id: str
    budget: int
    seed: int
    policy: str
    manifest_fingerprint: str
    best_candidate_fingerprint: str
    finalist_fingerprints: tuple[str, ...]
    held_out_means: tuple[tuple[str, float], ...]
    held_out_best_score: float
    tuning_pair_attempts: int
    tuning_physical_games: int
    tuning_search_iterations: int
    tuning_wall_time_ms: int


@dataclass(frozen=True, slots=True)
class _CellMetrics:
    fact: ChildFact
    simple_regret: float
    top_set_recall: float

    def value(self, metric: str) -> float:
        if metric == "held_out_best_score":
            return self.fact.held_out_best_score
        if metric == "simple_regret":
            return self.simple_regret
        return self.top_set_recall


def _reference_means(facts: list[ChildFact]) -> dict[str, float]:
    means: dict[str, float] = {}
    for fact in facts:
        for candidate_fingerprint, mean in fact.held_out_means:
            existing = means.get(candidate_fingerprint)
            if existing is not None and existing != mean:
                raise ValueError("bakeoff reference evidence disagrees for one candidate")
            means[candidate_fingerprint] = mean
    return means


def _reference_top_set(means: dict[str, float], top_set_k: int) -> tuple[str, ...]:
    ordered = sorted(means.items(), key=lambda item: (-item[1], item[0]))
    return tuple(candidate for candidate, _ in ordered[:top_set_k])


def _cell_metrics(facts: list[ChildFact], top_set_k: int) -> list[_CellMetrics]:
    means = _reference_means(facts)
    if not means:
        raise ValueError("bakeoff budget has no returned finalists")
    top_set = frozenset(_reference_top_set(means, top_set_k))
    reference_best = max(means.values())
    result: list[_CellMetrics] = []
    for fact in facts:
        recovered = len(top_set.intersection(fact.finalist_fingerprints))
        result.append(
            _CellMetrics(
                fact,
                reference_best - fact.held_out_best_score,
                recovered / len(top_set),
            )
        )
    return result


def _interval(values: tuple[float, ...], experiment: str, label: str) -> JsonObject:
    seed = int(fingerprint({"experiment": experiment, "metric": label})[:8], 16)
    estimate: Estimate = bootstrap_mean_interval(values, seed)
    return {"mean": estimate.mean, "lower": estimate.lower, "upper": estimate.upper}


def _by_seed(cells: list[_CellMetrics], policy: str, metric: str) -> dict[int, float]:
    return {cell.fact.seed: cell.value(metric) for cell in cells if cell.fact.policy == policy}


def _cell_row(cell: _CellMetrics) -> JsonObject:
    fact = cell.fact
    return {
        "cell_id": fact.cell_id,
        "budget": fact.budget,
        "seed": fact.seed,
        "policy": fact.policy,
        "manifest_fingerprint": fact.manifest_fingerprint,
        "candidate_fingerprint": fact.best_candidate_fingerprint,
        "held_out_best_score": fact.held_out_best_score,
        "simple_regret": cell.simple_regret,
        "top_set_recall": cell.top_set_recall,
        "tuning_pair_attempts": fact.tuning_pair_attempts,
        "tuning_physical_games": fact.tuning_physical_games,
        "tuning_search_iterations": fact.tuning_search_iterations,
        "tuning_wall_time_ms": fact.tuning_wall_time_ms,
    }


def _policy_summary(
    cells: list[_CellMetrics], budget: int, policy: str, experiment: str
) -> JsonObject:
    rows = [cell for cell in cells if cell.fact.policy == policy]
    summary: JsonObject = {
        "budget": budget,
        "policy": policy,
        "rows": [_cell_row(cell) for cell in rows],
    }
    for metric in _METRICS:
        values = tuple(cell.value(metric) for cell in rows)
        summary[metric] = _interval(values, experiment, f"{budget}:{policy}:{metric}")
    return summary


def _paired_differences(
    cells: list[_CellMetrics], left: str, right: str, metric: str
) -> tuple[float, ...]:
    left_by_seed = _by_seed(cells, left, metric)
    right_by_seed = _by_seed(cells, right, metric)
    shared = sorted(set(left_by_seed) & set(right_by_seed))
    if shared != sorted(left_by_seed) or shared != sorted(right_by_seed):
        raise ValueError("bakeoff policy contrast is not seed-aligned")
    return tuple(left_by_seed[seed] - right_by_seed[seed] for seed in shared)


def _contrast(
    cells: list[_CellMetrics], budget: int, left: str, right: str, experiment: str
) -> JsonObject:
    contrast: JsonObject = {"budget": budget, "left_policy": left, "right_policy": right}
    for metric in _METRICS:
        differences = _paired_differences(cells, left, right, metric)
        contrast[metric] = _interval(differences, experiment, f"{budget}:{left}-{right}:{metric}")
    return contrast


def _decision_interval(
    cells: list[_CellMetrics], metric: str, experiment: str, budget: int
) -> Estimate:
    differences = _paired_differences(cells, BAKEOFF_CHALLENGER, BAKEOFF_BASELINE, metric)
    seed = int(
        fingerprint({"experiment": experiment, "metric": f"decision:{budget}:{metric}"})[:8], 16
    )
    return bootstrap_mean_interval(differences, seed)


def _decide(
    cells: list[_CellMetrics], decision: BakeoffDecision, experiment: str, budget: int
) -> str:
    score = _decision_interval(cells, "held_out_best_score", experiment, budget)
    recall = _decision_interval(cells, "top_set_recall", experiment, budget)
    change = (
        score.lower > decision.score_practical_margin
        and recall.lower >= -decision.recall_noninferiority_margin
    )
    reject = (
        score.upper < -decision.score_practical_margin
        or recall.upper < -decision.recall_noninferiority_margin
    )
    if change:
        return "change_to_challenger"
    return "reject_challenger" if reject else "keep_current"


def aggregate(
    facts: list[ChildFact], experiment_fingerprint: str, decision: BakeoffDecision
) -> str:
    by_budget: dict[int, list[ChildFact]] = {}
    for fact in sorted(facts, key=lambda item: (item.budget, item.policy, item.seed)):
        by_budget.setdefault(fact.budget, []).append(fact)
    cells_by_budget = {
        budget: _cell_metrics(budget_facts, decision.top_set_k)
        for budget, budget_facts in sorted(by_budget.items())
    }
    summaries: list[JsonValue] = []
    contrasts: list[JsonValue] = []
    for budget, cells in cells_by_budget.items():
        policies = sorted({cell.fact.policy for cell in cells})
        summaries.extend(
            _policy_summary(cells, budget, policy, experiment_fingerprint) for policy in policies
        )
        contrasts.extend(
            _contrast(cells, budget, left, right, experiment_fingerprint)
            for left, right in _CONTRASTS
        )
    largest_budget = max(cells_by_budget)
    outcome = _decide(
        cells_by_budget[largest_budget], decision, experiment_fingerprint, largest_budget
    )
    raw: JsonObject = {
        "schema_version": 1,
        "experiment_fingerprint": experiment_fingerprint,
        "status": "complete",
        "reference_set_rule": REFERENCE_SET_RULE,
        "policy_budget_summaries": summaries,
        "paired_policy_contrasts": contrasts,
        "decision": {"outcome": outcome, "rule": DECISION_RULE},
        "limitations": [
            "Reference sets are within-experiment finalist unions, not global optima.",
            "Bootstrap intervals are across-seed empirical uncertainty, not distribution-free.",
        ],
    }
    return canonical_json(raw) + "\n"
