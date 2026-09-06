"""Hand-counted checks for the elimination bake-off aggregate and frozen decision."""

from __future__ import annotations

import json

import pytest

from tuner_cli.elimination_bakeoff_metrics import (
    NO_ELIMINATION,
    PAIRED_ELIMINATION,
    SPARE_NEAR_TIE,
    EliminationChildFact,
    EliminationDecision,
    aggregate,
)

_POLICIES = (NO_ELIMINATION, PAIRED_ELIMINATION, SPARE_NEAR_TIE)
_SEEDS = (1, 2, 3, 4)


def _fact(
    policy: str,
    seed: int,
    own_score: float,
    *,
    budget: int = 100,
    completed_cohorts: int = 2,
    pruned: int = 0,
    audit_continued: int = 0,
    audited_boundary_reversals: int = 0,
    planned_unique_pair_savings: int = 0,
    suspended: bool = False,
) -> EliminationChildFact:
    # Fingerprint by score so equally-scoring cells share a reference candidate
    # and top-set recall is not degenerate under a single-element reference set.
    own = f"cand-{own_score}"
    means = ((own, own_score), ("cand-shared", 0.5))
    nominal = pruned + audit_continued
    return EliminationChildFact(
        cell_id=f"{budget}:{seed}:{policy}",
        budget=budget,
        seed=seed,
        policy=policy,
        manifest_fingerprint=f"mf-{policy}-{seed}",
        best_candidate_fingerprint=max(means, key=lambda item: item[1])[0],
        finalist_fingerprints=tuple(sorted(name for name, _ in means)),
        held_out_means=tuple(sorted(means)),
        held_out_best_score=max(score for _, score in means),
        completed_cohorts=completed_cohorts,
        accepted_unique_candidates=completed_cohorts * 4,
        terminal_candidate_failures=0,
        censored_tuning_attempts=0,
        tuning_pair_attempts=100,
        tuning_physical_games=200,
        tuning_search_iterations=300,
        tuning_wall_time_ms=400,
        unspent_pair_attempts=0,
        overrun_pair_attempts=0,
        nominal_eliminations=nominal,
        pruned=pruned,
        audit_continued=audit_continued,
        audited_boundary_reversals=audited_boundary_reversals,
        estimated_boundary_reversals=float(audited_boundary_reversals) * 4.0,
        gross_nominal_suffix_unique_pairs=planned_unique_pair_savings + 2 * audit_continued,
        audit_continuation_suffix_unique_pairs=2 * audit_continued,
        planned_unique_pair_savings=planned_unique_pair_savings,
        suspended=suspended,
    )


def _decision(score_margin: float = 0.0, recall_margin: float = 0.1) -> EliminationDecision:
    return EliminationDecision(score_margin, recall_margin, 1)


def _run(
    scores: dict[str, float], *, sh_reversal: int = 0, paired_suspended: bool = False
) -> dict[str, object]:
    facts: list[EliminationChildFact] = []
    for policy in _POLICIES:
        for seed in _SEEDS:
            facts.append(
                _fact(
                    policy,
                    seed,
                    scores[policy],
                    audited_boundary_reversals=(sh_reversal if policy == SPARE_NEAR_TIE else 0),
                    suspended=paired_suspended and policy == PAIRED_ELIMINATION,
                )
            )
    return json.loads(aggregate(facts, "experiment-fp", _decision()))


def test_aggregate_rebuilds_byte_identically() -> None:
    facts = [_fact(policy, seed, 0.6) for policy in _POLICIES for seed in _SEEDS]
    first = aggregate(facts, "experiment-fp", _decision())
    second = aggregate(list(reversed(facts)), "experiment-fp", _decision())
    assert first == second


def test_regret_and_recall_use_the_within_budget_reference_set() -> None:
    out = _run({NO_ELIMINATION: 0.55, PAIRED_ELIMINATION: 0.3, SPARE_NEAR_TIE: 0.9})
    rows = {
        row["policy"]: row
        for summary in out["policy_budget_summaries"]
        for row in summary["rows"]
        if row["seed"] == 1
    }
    assert rows[SPARE_NEAR_TIE]["simple_regret"] == pytest.approx(0.0)
    assert rows[PAIRED_ELIMINATION]["simple_regret"] == pytest.approx(0.4)
    assert rows[SPARE_NEAR_TIE]["top_set_recall"] == 1.0
    assert rows[PAIRED_ELIMINATION]["top_set_recall"] == 0.0


def test_change_when_spare_near_tie_dominates_and_is_safe() -> None:
    out = _run({NO_ELIMINATION: 0.9, PAIRED_ELIMINATION: 0.5, SPARE_NEAR_TIE: 0.9})
    assert out["decision"]["outcome"] == "change_to_spare_near_tie"
    assert out["decision"]["safe_in_bakeoff"] == {
        SPARE_NEAR_TIE: True,
        PAIRED_ELIMINATION: True,
    }


def test_reject_when_neither_active_arm_is_safe() -> None:
    out = _run(
        {NO_ELIMINATION: 0.9, PAIRED_ELIMINATION: 0.9, SPARE_NEAR_TIE: 0.9},
        sh_reversal=1,
        paired_suspended=True,
    )
    assert out["decision"]["outcome"] == "reject_active_elimination"
    assert out["decision"]["safe_in_bakeoff"] == {
        SPARE_NEAR_TIE: False,
        PAIRED_ELIMINATION: False,
    }


def test_keep_paired_on_a_tie() -> None:
    out = _run({NO_ELIMINATION: 0.6, PAIRED_ELIMINATION: 0.6, SPARE_NEAR_TIE: 0.6})
    assert out["decision"]["outcome"] == "keep_paired_elimination"


def test_active_safety_and_reinvestment_are_hand_countable() -> None:
    facts = []
    for seed in _SEEDS:
        facts.append(_fact(NO_ELIMINATION, seed, 0.6, completed_cohorts=2))
        facts.append(_fact(PAIRED_ELIMINATION, seed, 0.6, pruned=1, planned_unique_pair_savings=3))
        facts.append(
            _fact(
                SPARE_NEAR_TIE,
                seed,
                0.6,
                pruned=2,
                audit_continued=1,
                planned_unique_pair_savings=5,
                completed_cohorts=3,
            )
        )
    out = json.loads(aggregate(facts, "experiment-fp", _decision()))
    safety = {row["policy"]: row for row in out["active_safety_summaries"]}
    assert safety[SPARE_NEAR_TIE]["pruned"] == 8
    assert safety[SPARE_NEAR_TIE]["audit_continued"] == 4
    assert safety[SPARE_NEAR_TIE]["nominal_eliminations"] == 12
    assert safety[PAIRED_ELIMINATION]["pruned"] == 4
    reinvest = {row["policy"]: row for row in out["budget_reinvestment"]}
    assert reinvest[SPARE_NEAR_TIE]["seed_paired_completed_cohort_gain"] == [1, 1, 1, 1]
    assert reinvest[SPARE_NEAR_TIE]["funded_additional_cohorts"] is True
    assert reinvest[SPARE_NEAR_TIE]["total_planned_unique_pair_savings"] == 20
    assert reinvest[PAIRED_ELIMINATION]["funded_additional_cohorts"] is False


def test_contrast_requires_seed_alignment() -> None:
    facts = [_fact(policy, seed, 0.6) for policy in _POLICIES for seed in _SEEDS]
    facts = [fact for fact in facts if not (fact.policy == SPARE_NEAR_TIE and fact.seed == 4)]
    with pytest.raises(ValueError, match="seed-aligned"):
        aggregate(facts, "experiment-fp", _decision())
