"""Deterministic checks for the proposer bake-off aggregate instrumentation."""

from __future__ import annotations

import json

import pytest

from tuner_cli.bakeoff_artifacts import BakeoffDecision
from tuner_cli.bakeoff_metrics import ChildFact, aggregate

_POLICIES = ("random", "qmc", "smac_mixed", "irace_generational")


def _fact(policy: str, seed: int, own_score: float) -> ChildFact:
    """One cell whose finalists are a policy/seed-unique candidate and a shared one."""
    own = f"cand-{policy}-{seed}"
    shared = "cand-shared"
    means = ((own, own_score), (shared, 0.5))
    best = max(score for _, score in means)
    return ChildFact(
        cell_id=f"100:{seed}:{policy}",
        budget=100,
        seed=seed,
        policy=policy,
        manifest_fingerprint=f"mf-{policy}-{seed}",
        best_candidate_fingerprint=max(means, key=lambda item: item[1])[0],
        finalist_fingerprints=tuple(sorted(name for name, _ in means)),
        held_out_means=tuple(sorted(means)),
        held_out_best_score=best,
        tuning_pair_attempts=100,
        tuning_physical_games=200,
        tuning_search_iterations=300,
        tuning_wall_time_ms=400,
    )


def _decision(score_margin: float = 0.0, recall_margin: float = 0.1) -> BakeoffDecision:
    return BakeoffDecision("smac_mixed", "irace_generational", score_margin, recall_margin, 1)


def _run(scores: dict[str, float]) -> dict[str, object]:
    facts = [_fact(policy, seed, scores[policy]) for policy in _POLICIES for seed in (1, 2, 3)]
    return json.loads(aggregate(facts, "experiment-fp", _decision()))


def test_aggregate_rebuilds_byte_identically() -> None:
    facts = [_fact(policy, seed, 0.6) for policy in _POLICIES for seed in (1, 2, 3)]
    first = aggregate(facts, "experiment-fp", _decision())
    second = aggregate(list(reversed(facts)), "experiment-fp", _decision())
    assert first == second


def test_regret_and_recall_use_the_within_budget_reference_set() -> None:
    out = _run({"random": 0.2, "qmc": 0.3, "smac_mixed": 0.55, "irace_generational": 0.9})
    rows = {
        row["policy"]: row
        for summary in out["policy_budget_summaries"]
        for row in summary["rows"]
        if row["seed"] == 1
    }
    # Reference best is irace's seed-1 unique 0.9 candidate. Every cell also
    # returns the shared 0.5 candidate, so each cell's best score is at least 0.5.
    assert rows["random"]["simple_regret"] == pytest.approx(0.4)
    assert rows["irace_generational"]["simple_regret"] == pytest.approx(0.0)
    # The single-element reference top set is only recovered by the cell that
    # actually returned that candidate.
    assert rows["irace_generational"]["top_set_recall"] == 1.0
    assert all(rows[policy]["top_set_recall"] == 0.0 for policy in ("random", "qmc", "smac_mixed"))


def test_largest_budget_rule_selects_change_when_irace_dominates() -> None:
    out = _run({"random": 0.2, "qmc": 0.3, "smac_mixed": 0.5, "irace_generational": 0.9})
    assert out["decision"] == {
        "outcome": "change_to_challenger",
        "rule": "irace-vs-smac-largest-budget-v1",
    }


def test_largest_budget_rule_rejects_when_irace_is_worse() -> None:
    out = _run({"random": 0.2, "qmc": 0.3, "smac_mixed": 0.9, "irace_generational": 0.4})
    assert out["decision"]["outcome"] == "reject_challenger"


def test_largest_budget_rule_keeps_current_on_a_tie() -> None:
    out = _run({"random": 0.2, "qmc": 0.3, "smac_mixed": 0.6, "irace_generational": 0.6})
    assert out["decision"]["outcome"] == "keep_current"


def test_contrast_requires_seed_alignment() -> None:
    facts = [_fact(policy, seed, 0.6) for policy in _POLICIES for seed in (1, 2, 3)]
    facts = [fact for fact in facts if not (fact.policy == "random" and fact.seed == 3)]
    with pytest.raises(ValueError, match="seed-aligned"):
        aggregate(facts, "experiment-fp", _decision())


def test_disagreeing_reference_evidence_is_rejected() -> None:
    facts = [_fact(policy, seed, 0.6) for policy in _POLICIES for seed in (1, 2)]
    facts.append(
        ChildFact(
            cell_id="100:9:random",
            budget=100,
            seed=9,
            policy="random",
            manifest_fingerprint="mf",
            best_candidate_fingerprint="cand-shared",
            finalist_fingerprints=("cand-shared",),
            held_out_means=(("cand-shared", 0.99),),
            held_out_best_score=0.99,
            tuning_pair_attempts=1,
            tuning_physical_games=1,
            tuning_search_iterations=1,
            tuning_wall_time_ms=1,
        )
    )
    with pytest.raises(ValueError, match="reference evidence disagrees"):
        aggregate(facts, "experiment-fp", _decision())
