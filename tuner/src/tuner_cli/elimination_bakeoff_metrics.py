"""Pure, deterministic aggregates for completed elimination bake-off children.

Held-out quality, regret, and recall reuse the proposer bake-off's definitions
(union-of-returned-finalists reference set, across-seed percentile bootstrap,
seed-paired policy contrasts). On top of those this module reports the
elimination-specific facts drawn from typed replay and the active audit --
nominal eliminations, prunes, audited boundary reversals, projected unique-pair
savings, and budget reinvestment -- and applies one frozen largest-budget rule
that emits exactly one of keep / change / reject.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bakeoff_metrics import (
    across_seed_interval,
    held_out_reference_means,
    held_out_reference_top_set,
)
from .codec import JsonObject, JsonValue
from .domain import Estimate
from .identity import canonical_json, fingerprint
from .statistics import bootstrap_mean_interval

REFERENCE_SET_RULE = "union-returned-finalists-v1"
DECISION_RULE = "elimination-largest-budget-keep-change-reject-v1"

NO_ELIMINATION = "no_elimination"
PAIRED_ELIMINATION = "paired_elimination"
SPARE_NEAR_TIE = "spare_near_tie"
_ACTIVE_ARMS = (SPARE_NEAR_TIE, PAIRED_ELIMINATION)
_METRICS = ("held_out_best_score", "simple_regret", "top_set_recall")
_CONTRASTS: tuple[tuple[str, str], ...] = (
    (SPARE_NEAR_TIE, PAIRED_ELIMINATION),
    (SPARE_NEAR_TIE, NO_ELIMINATION),
    (PAIRED_ELIMINATION, NO_ELIMINATION),
)


@dataclass(frozen=True, slots=True)
class EliminationDecision:
    score_practical_margin: float
    recall_noninferiority_margin: float
    top_set_k: int


@dataclass(frozen=True, slots=True)
class EliminationChildFact:
    cell_id: str
    budget: int
    seed: int
    policy: str
    manifest_fingerprint: str
    best_candidate_fingerprint: str
    finalist_fingerprints: tuple[str, ...]
    held_out_means: tuple[tuple[str, float], ...]
    held_out_best_score: float
    completed_cohorts: int
    accepted_unique_candidates: int
    terminal_candidate_failures: int
    censored_tuning_attempts: int
    tuning_pair_attempts: int
    tuning_physical_games: int
    tuning_search_iterations: int
    tuning_wall_time_ms: int
    unspent_pair_attempts: int
    overrun_pair_attempts: int
    nominal_eliminations: int
    pruned: int
    audit_continued: int
    audited_boundary_reversals: int
    estimated_boundary_reversals: float
    gross_nominal_suffix_unique_pairs: int
    audit_continuation_suffix_unique_pairs: int
    planned_unique_pair_savings: int
    suspended: bool


@dataclass(frozen=True, slots=True)
class _CellMetrics:
    fact: EliminationChildFact
    simple_regret: float
    top_set_recall: float

    def value(self, metric: str) -> float:
        if metric == "held_out_best_score":
            return self.fact.held_out_best_score
        if metric == "simple_regret":
            return self.simple_regret
        return self.top_set_recall


def _cell_metrics(facts: list[EliminationChildFact], top_set_k: int) -> list[_CellMetrics]:
    means = held_out_reference_means(facts)
    if not means:
        raise ValueError("elimination bake-off budget has no returned finalists")
    top_set = frozenset(held_out_reference_top_set(means, top_set_k))
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


def _by_seed(cells: list[_CellMetrics], policy: str, metric: str) -> dict[int, float]:
    return {cell.fact.seed: cell.value(metric) for cell in cells if cell.fact.policy == policy}


def _paired_differences(
    cells: list[_CellMetrics], left: str, right: str, metric: str
) -> tuple[float, ...]:
    left_by_seed = _by_seed(cells, left, metric)
    right_by_seed = _by_seed(cells, right, metric)
    shared = sorted(set(left_by_seed) & set(right_by_seed))
    if shared != sorted(left_by_seed) or shared != sorted(right_by_seed):
        raise ValueError("elimination bake-off policy contrast is not seed-aligned")
    return tuple(left_by_seed[seed] - right_by_seed[seed] for seed in shared)


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
        "completed_cohorts": fact.completed_cohorts,
        "accepted_unique_candidates": fact.accepted_unique_candidates,
        "terminal_candidate_failures": fact.terminal_candidate_failures,
        "actual_compute": {
            "tuning_pair_attempts": fact.tuning_pair_attempts,
            "tuning_physical_games": fact.tuning_physical_games,
            "tuning_search_iterations": fact.tuning_search_iterations,
            "tuning_wall_time_ms": fact.tuning_wall_time_ms,
            "censored_tuning_attempts": fact.censored_tuning_attempts,
            "unspent_pair_attempts": fact.unspent_pair_attempts,
            "overrun_pair_attempts": fact.overrun_pair_attempts,
        },
        "active_elimination": {
            "nominal_eliminations": fact.nominal_eliminations,
            "pruned": fact.pruned,
            "audit_continued": fact.audit_continued,
            "audited_boundary_reversals": fact.audited_boundary_reversals,
            "estimated_boundary_reversals": fact.estimated_boundary_reversals,
            "suspended": fact.suspended,
        },
        "projected_unique_pair_savings": {
            "gross_nominal_suffix_unique_pairs": fact.gross_nominal_suffix_unique_pairs,
            "audit_continuation_suffix_unique_pairs": (fact.audit_continuation_suffix_unique_pairs),
            "planned_unique_pair_savings": fact.planned_unique_pair_savings,
        },
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
        summary[metric] = across_seed_interval(values, experiment, f"{budget}:{policy}:{metric}")
    return summary


def _contrast(
    cells: list[_CellMetrics], budget: int, left: str, right: str, experiment: str
) -> JsonObject:
    contrast: JsonObject = {"budget": budget, "left_policy": left, "right_policy": right}
    for metric in _METRICS:
        differences = _paired_differences(cells, left, right, metric)
        contrast[metric] = across_seed_interval(
            differences, experiment, f"{budget}:{left}-{right}:{metric}"
        )
    return contrast


def _safe_in_bakeoff(cells: list[_CellMetrics], arm: str) -> bool:
    rows = [cell.fact for cell in cells if cell.fact.policy == arm]
    return bool(rows) and all(
        fact.audited_boundary_reversals == 0 and not fact.suspended for fact in rows
    )


def _decision_contrast(
    cells: list[_CellMetrics], experiment: str, budget: int, left: str, right: str, metric: str
) -> Estimate:
    differences = _paired_differences(cells, left, right, metric)
    seed = int(
        fingerprint(
            {"experiment": experiment, "metric": f"decision:{budget}:{left}-{right}:{metric}"}
        )[:8],
        16,
    )
    return bootstrap_mean_interval(differences, seed)


def _interval_json(estimate: Estimate) -> JsonObject:
    return {"mean": estimate.mean, "lower": estimate.lower, "upper": estimate.upper}


def _decide(
    cells: list[_CellMetrics], decision: EliminationDecision, experiment: str, budget: int
) -> JsonObject:
    margin = decision.score_practical_margin
    recall_margin = decision.recall_noninferiority_margin

    def contrast(left: str, right: str, metric: str) -> Estimate:
        return _decision_contrast(cells, experiment, budget, left, right, metric)

    safe = {arm: _safe_in_bakeoff(cells, arm) for arm in _ACTIVE_ARMS}
    sh_paired_score = contrast(SPARE_NEAR_TIE, PAIRED_ELIMINATION, "held_out_best_score")
    sh_paired_recall = contrast(SPARE_NEAR_TIE, PAIRED_ELIMINATION, "top_set_recall")
    sh_none_score = contrast(SPARE_NEAR_TIE, NO_ELIMINATION, "held_out_best_score")
    sh_none_recall = contrast(SPARE_NEAR_TIE, NO_ELIMINATION, "top_set_recall")
    paired_none_score = contrast(PAIRED_ELIMINATION, NO_ELIMINATION, "held_out_best_score")
    paired_none_recall = contrast(PAIRED_ELIMINATION, NO_ELIMINATION, "top_set_recall")

    change = (
        safe[SPARE_NEAR_TIE]
        and sh_paired_score.lower > margin
        and sh_paired_recall.lower >= -recall_margin
        and sh_none_score.lower >= -margin
        and sh_none_recall.lower >= -recall_margin
    )

    def arm_worse(score: Estimate, recall: Estimate) -> bool:
        return score.upper < -margin or recall.upper < -recall_margin

    reject = (not any(safe.values())) or (
        arm_worse(sh_none_score, sh_none_recall)
        and arm_worse(paired_none_score, paired_none_recall)
    )

    if change:
        outcome = "change_to_spare_near_tie"
    elif reject:
        outcome = "reject_active_elimination"
    else:
        outcome = "keep_paired_elimination"

    return {
        "outcome": outcome,
        "rule": DECISION_RULE,
        "budget": budget,
        "safe_in_bakeoff": {arm: safe[arm] for arm in _ACTIVE_ARMS},
        "clauses": {
            "spare_near_tie_minus_paired": {
                "held_out_best_score": _interval_json(sh_paired_score),
                "top_set_recall": _interval_json(sh_paired_recall),
            },
            "spare_near_tie_minus_no_elimination": {
                "held_out_best_score": _interval_json(sh_none_score),
                "top_set_recall": _interval_json(sh_none_recall),
            },
            "paired_minus_no_elimination": {
                "held_out_best_score": _interval_json(paired_none_score),
                "top_set_recall": _interval_json(paired_none_recall),
            },
        },
        "score_practical_margin": margin,
        "recall_noninferiority_margin": recall_margin,
    }


def _active_safety_summary(cells: list[_CellMetrics], arm: str) -> JsonObject:
    rows = [cell.fact for cell in cells if cell.fact.policy == arm]
    return {
        "policy": arm,
        "completed_cells": len(rows),
        "nominal_eliminations": sum(fact.nominal_eliminations for fact in rows),
        "pruned": sum(fact.pruned for fact in rows),
        "audit_continued": sum(fact.audit_continued for fact in rows),
        "audited_boundary_reversals": sum(fact.audited_boundary_reversals for fact in rows),
        "estimated_boundary_reversals": sum(fact.estimated_boundary_reversals for fact in rows),
        "suspended_cells": sum(1 for fact in rows if fact.suspended),
        "safe_in_bakeoff": _safe_in_bakeoff(cells, arm),
    }


def _reinvestment(cells: list[_CellMetrics], arm: str) -> JsonObject:
    arm_by_seed = {cell.fact.seed: cell.fact for cell in cells if cell.fact.policy == arm}
    base_by_seed = {
        cell.fact.seed: cell.fact for cell in cells if cell.fact.policy == NO_ELIMINATION
    }
    shared = sorted(set(arm_by_seed) & set(base_by_seed))
    cohort_gains = [
        arm_by_seed[seed].completed_cohorts - base_by_seed[seed].completed_cohorts
        for seed in shared
    ]
    gains_json: list[JsonValue] = []
    gains_json.extend(cohort_gains)
    return {
        "policy": arm,
        "vs": NO_ELIMINATION,
        "seed_paired_completed_cohort_gain": gains_json,
        "total_planned_unique_pair_savings": sum(
            arm_by_seed[seed].planned_unique_pair_savings for seed in shared
        ),
        "funded_additional_cohorts": any(gain > 0 for gain in cohort_gains),
    }


def aggregate(
    facts: list[EliminationChildFact], experiment_fingerprint: str, decision: EliminationDecision
) -> str:
    by_budget: dict[int, list[EliminationChildFact]] = {}
    for fact in sorted(facts, key=lambda item: (item.budget, item.policy, item.seed)):
        by_budget.setdefault(fact.budget, []).append(fact)
    if not by_budget:
        raise ValueError("elimination bake-off has no child facts")
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
    largest_cells = cells_by_budget[largest_budget]
    raw: JsonObject = {
        "schema_version": 1,
        "experiment_fingerprint": experiment_fingerprint,
        "status": "complete",
        "reference_set_rule": REFERENCE_SET_RULE,
        "policy_budget_summaries": summaries,
        "paired_policy_contrasts": contrasts,
        "active_safety_summaries": [
            _active_safety_summary(largest_cells, arm) for arm in _ACTIVE_ARMS
        ],
        "budget_reinvestment": [_reinvestment(largest_cells, arm) for arm in _ACTIVE_ARMS],
        "decision": _decide(largest_cells, decision, experiment_fingerprint, largest_budget),
        "limitations": [
            "Reference sets are within-experiment finalist unions, not global optima.",
            "Bootstrap intervals are across-seed empirical uncertainty, not distribution-free.",
            (
                "Projected unique-pair savings are prefix arithmetic, not observed wall time, "
                "and do not model retries or failures that pruned candidates would have incurred."
            ),
            (
                "Zero audited boundary reversals in a finite bake-off is not a universal "
                "safety guarantee for either active policy."
            ),
        ],
    }
    return canonical_json(raw) + "\n"
