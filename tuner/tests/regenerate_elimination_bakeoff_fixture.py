"""Rebuild the canonical elimination bake-off aggregate fixtures.

Run from the tuner project root::

    uv run python tests/regenerate_elimination_bakeoff_fixture.py

The fixtures are small hand-built aggregates (no target calls): a strict
`experiment.json` and a `results.json` whose largest-budget rule resolves to
`change_to_spare_near_tie`. `test_elimination_bakeoff_fixture.py` asserts the
checked-in bytes match this builder exactly.
"""

from __future__ import annotations

from pathlib import Path

from tuner_cli.domain import SearchEffort
from tuner_cli.elimination_bakeoff import (
    EliminationBakeoffSpec,
    EliminationGate,
    EliminationSharedRun,
    _cell_summary,
    _cells,
    encode_experiment,
    experiment_fingerprint,
)
from tuner_cli.elimination_bakeoff_metrics import (
    NO_ELIMINATION,
    PAIRED_ELIMINATION,
    SPARE_NEAR_TIE,
    EliminationChildFact,
    EliminationDecision,
    aggregate,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "elimination-bakeoff-v1"
_SEEDS = (101, 102, 103, 104)
_BUDGETS = (168, 224)
_POLICIES = (NO_ELIMINATION, PAIRED_ELIMINATION, SPARE_NEAR_TIE)


def build_spec() -> EliminationBakeoffSpec:
    shared = EliminationSharedRun(
        proposer_policy="smac_mixed",
        cohort_size=8,
        finalists=3,
        bootstrap_candidates=3,
        random_reserve_candidates=2,
        tuning_pairs=14,
        validation_pair_budget=36,
        production_validation_pairs=12,
        diagnostic_pair_budget=0,
        tuning_effort=SearchEffort("iterations", 400),
        validation_effort=SearchEffort("iterations", 2000),
        production_effort=SearchEffort("iterations", 2000),
        excluded_families=("meta_mcts", "negamax"),
        evaluator_workers=3,
        pair_timeout_seconds=600,
        active_audit_probability=0.25,
    )
    return EliminationBakeoffSpec(
        experiment_id="druid-elimination-bakeoff-v1",
        game_binary=Path("target/release/game-druid"),
        objective_file=Path("tuner/objectives/druid-reference-v1.json"),
        proposal_seeds=_SEEDS,
        task_seed=43,
        tuning_pair_budgets=_BUDGETS,
        shared_run=shared,
        decision=EliminationDecision(0.01, 0.1, 2),
        gate=EliminationGate(
            "task-11-successive-halving-shadow-gate.md",
            "PASS",
            "successive-halving-spare-near-tie-v1",
        ),
    )


def _child_fact(policy: str, budget: int, seed: int) -> EliminationChildFact:
    own = 0.70 if policy == SPARE_NEAR_TIE else 0.60
    means = ((f"cand-{own:.3f}", own), ("cand-shared-0.500", 0.5))
    active = policy != NO_ELIMINATION
    pruned = 2 if policy == SPARE_NEAR_TIE else 1
    audited = 1 if policy == SPARE_NEAR_TIE else 0
    savings = 6 if policy == SPARE_NEAR_TIE else 2
    return EliminationChildFact(
        cell_id=f"{budget}:{seed}:{policy}",
        budget=budget,
        seed=seed,
        policy=policy,
        manifest_fingerprint=f"mf-{policy}-{budget}-{seed}",
        best_candidate_fingerprint=f"cand-{own:.3f}",
        finalist_fingerprints=tuple(sorted(name for name, _ in means)),
        held_out_means=tuple(sorted(means)),
        held_out_best_score=own,
        completed_cohorts=4 if policy == SPARE_NEAR_TIE else 3,
        accepted_unique_candidates=24,
        terminal_candidate_failures=0,
        censored_tuning_attempts=0,
        tuning_pair_attempts=budget - (savings if active else 0),
        tuning_physical_games=budget * 2,
        tuning_search_iterations=budget * 400,
        tuning_wall_time_ms=budget * 5,
        unspent_pair_attempts=savings if active else 0,
        overrun_pair_attempts=0,
        nominal_eliminations=(pruned + audited) if active else 0,
        pruned=pruned if active else 0,
        audit_continued=audited if active else 0,
        audited_boundary_reversals=0,
        estimated_boundary_reversals=0.0,
        gross_nominal_suffix_unique_pairs=(savings + 2 * audited) if active else 0,
        audit_continuation_suffix_unique_pairs=(2 * audited) if active else 0,
        planned_unique_pair_savings=savings if active else 0,
        suspended=False,
    )


def build() -> tuple[str, str]:
    spec = build_spec()
    cells = _cells(spec, FIXTURE_DIR)
    experiment_text = encode_experiment(spec, [_cell_summary(cell) for cell in cells])
    fingerprint_value = experiment_fingerprint(experiment_text)
    facts = [
        _child_fact(policy, budget, seed)
        for budget in _BUDGETS
        for seed in _SEEDS
        for policy in _POLICIES
    ]
    return experiment_text, aggregate(facts, fingerprint_value, spec.decision)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    experiment_text, results_text = build()
    (FIXTURE_DIR / "experiment.json").write_text(experiment_text)
    (FIXTURE_DIR / "results.json").write_text(results_text)
    print(f"wrote canonical elimination bake-off fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
