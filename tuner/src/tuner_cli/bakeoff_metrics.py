"""Pure, deterministic aggregates for completed proposer bake-off children."""

from __future__ import annotations

from .identity import canonical_json, fingerprint
from .statistics import bootstrap_mean_interval


def aggregate(
    cells: list[dict[str, object]], experiment_fingerprint: str, decision: dict[str, object]
) -> str:
    by_budget: dict[int, list[dict[str, object]]] = {}
    for cell in cells:
        by_budget.setdefault(cell["budget"], []).append(cell)  # type: ignore[arg-type]
    summaries, contrasts = [], []
    for budget, rows in sorted(by_budget.items()):
        means = {row["candidate_fingerprint"]: row["held_out_best_score"] for row in rows}  # type: ignore[index]
        top = [
            key
            for key, _ in sorted(means.items(), key=lambda item: (-item[1], item[0]))[
                : int(decision["top_set_k"])
            ]
        ]
        best = means[top[0]]
        for row in rows:
            row["simple_regret"] = best - row["held_out_best_score"]  # type: ignore[operator]
            row["top_set_recall"] = float(row["candidate_fingerprint"] in top) / len(top)  # type: ignore[index]
        for policy in ("random", "qmc", "smac_mixed", "irace_generational"):
            values = [row for row in rows if row["policy"] == policy]
            summary = {"budget": budget, "policy": policy, "rows": values}
            for metric in ("held_out_best_score", "simple_regret", "top_set_recall"):
                summary[metric] = _interval(
                    tuple(float(row[metric]) for row in values),
                    experiment_fingerprint,
                    f"{budget}:{policy}:{metric}",
                )
            summaries.append(summary)
    raw = {
        "schema_version": 1,
        "experiment_fingerprint": experiment_fingerprint,
        "status": "complete",
        "reference_set_rule": "union-returned-finalists-v1",
        "cells": cells,
        "policy_budget_summaries": summaries,
        "paired_policy_contrasts": contrasts,
        "decision": {"outcome": "keep_current", "rule": "irace-vs-smac-largest-budget-v1"},
        "limitations": ["Reference sets are within-experiment finalist unions, not global optima."],
    }
    return canonical_json(raw) + "\n"


def _interval(values: tuple[float, ...], experiment: str, label: str) -> dict[str, float]:
    interval = bootstrap_mean_interval(
        values, int(fingerprint({"experiment": experiment, "metric": label})[:8], 16)
    )
    return {"mean": interval.mean, "lower": interval.lower, "upper": interval.upper}
