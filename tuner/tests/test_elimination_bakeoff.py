"""Matched three-arm elimination bake-off orchestration over mocked runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_run import FakeModel, FakeTarget, _fake_binary, _objective

from tuner_cli.domain import SearchEffort
from tuner_cli.elimination_bakeoff import (
    EliminationBakeoffSpec,
    EliminationCell,
    EliminationGate,
    EliminationSharedRun,
    _options,
    read_elimination_spec,
    run_elimination_bakeoff,
)
from tuner_cli.elimination_bakeoff_metrics import EliminationDecision

_SEEDS = (11, 12, 13, 14)
_BUDGETS = (8, 14)


def _shared_run() -> EliminationSharedRun:
    return EliminationSharedRun(
        proposer_policy="smac_mixed",
        cohort_size=4,
        finalists=1,
        bootstrap_candidates=2,
        random_reserve_candidates=1,
        tuning_pairs=2,
        validation_pair_budget=2,
        production_validation_pairs=2,
        diagnostic_pair_budget=0,
        tuning_effort=SearchEffort("iterations", 3),
        validation_effort=SearchEffort("iterations", 9),
        production_effort=SearchEffort("iterations", 9),
        excluded_families=(),
        evaluator_workers=1,
        pair_timeout_seconds=600,
        active_audit_probability=0.25,
    )


def _spec(tmp_path: Path) -> EliminationBakeoffSpec:
    return EliminationBakeoffSpec(
        experiment_id="druid-elimination-smoke",
        game_binary=_fake_binary(tmp_path),
        objective_file=_objective(tmp_path),
        proposal_seeds=_SEEDS,
        task_seed=9,
        tuning_pair_budgets=_BUDGETS,
        shared_run=_shared_run(),
        decision=EliminationDecision(0.0, 0.1, 1),
        gate=EliminationGate(
            "task-11-successive-halving-shadow-gate.md",
            "PASS",
            "successive-halving-spare-near-tie-v1",
        ),
    )


def _spec_dict(tmp_path: Path) -> dict[str, object]:
    shared = _shared_run()
    return {
        "schema_version": 1,
        "experiment_id": "druid-elimination-smoke",
        "game_binary": str(_fake_binary(tmp_path)),
        "objective_file": str(_objective(tmp_path)),
        "policies": ["no_elimination", "paired_elimination", "spare_near_tie"],
        "proposal_seeds": list(_SEEDS),
        "task_seed": 9,
        "tuning_pair_budgets": list(_BUDGETS),
        "shared_run": {
            "proposer_policy": "smac_mixed",
            "cohort_size": shared.cohort_size,
            "finalists": shared.finalists,
            "bootstrap_candidates": shared.bootstrap_candidates,
            "random_reserve_candidates": shared.random_reserve_candidates,
            "tuning_pairs": shared.tuning_pairs,
            "validation_pair_budget": shared.validation_pair_budget,
            "production_validation_pairs": shared.production_validation_pairs,
            "diagnostic_pair_budget": 0,
            "tuning_effort": {"kind": "iterations", "value": 3},
            "validation_effort": {"kind": "iterations", "value": 9},
            "production_effort": {"kind": "iterations", "value": 9},
            "excluded_families": [],
            "evaluator_workers": 1,
            "pair_timeout_seconds": 600,
            "active_audit_probability": 0.25,
        },
        "decision": {
            "score_practical_margin": 0.0,
            "recall_noninferiority_margin": 0.1,
            "top_set_k": 1,
        },
        "gate": {
            "document_id": "task-11-successive-halving-shadow-gate.md",
            "decision": "PASS",
            "authorized_policy_version": "successive-halving-spare-near-tie-v1",
        },
    }


def test_spec_rejects_a_wrong_gate_authorization(tmp_path: Path) -> None:
    raw = _spec_dict(tmp_path)
    gate = raw["gate"]
    assert isinstance(gate, dict)
    gate["authorized_policy_version"] = "successive-halving-common-prefix-eta2-v1"
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="gate block does not match"):
        read_elimination_spec(path)


def test_spec_rejects_a_non_point_two_five_audit_probability(tmp_path: Path) -> None:
    raw = _spec_dict(tmp_path)
    shared = raw["shared_run"]
    assert isinstance(shared, dict)
    shared["active_audit_probability"] = 0.1
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="audit probability must be"):
        read_elimination_spec(path)


def test_spec_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_spec_dict(tmp_path)))
    spec = read_elimination_spec(path)
    assert spec.proposal_seeds == _SEEDS
    assert spec.tuning_pair_budgets == _BUDGETS
    assert spec.shared_run.active_audit_probability == 0.25


def test_matched_arms_differ_only_in_elimination_policy(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    cell_no = EliminationCell("8:11:no_elimination", 8, 11, "no_elimination", tmp_path / "n")
    cell_paired = EliminationCell(
        "8:11:paired_elimination", 8, 11, "paired_elimination", tmp_path / "p"
    )
    cell_spare = EliminationCell("8:11:spare_near_tie", 8, 11, "spare_near_tie", tmp_path / "s")
    opt_no, opt_paired, opt_spare = (
        _options(spec, cell_no),
        _options(spec, cell_paired),
        _options(spec, cell_spare),
    )
    assert opt_no.active_elimination_audit_probability is None
    assert opt_paired.active_elimination_audit_probability == 0.25
    assert opt_spare.active_elimination_audit_probability == 0.25
    assert opt_paired.shadow_policy == "paired_bootstrap"
    assert opt_spare.shadow_policy == "successive_halving"
    assert opt_spare.shadow_halving_spare_margin == 0.10
    for left in (opt_paired, opt_spare):
        assert (left.seed, left.task_seed, left.cohort_size, left.tuning_pair_budget) == (
            opt_no.seed,
            opt_no.task_seed,
            opt_no.cohort_size,
            opt_no.tuning_pair_budget,
        )


def test_bakeoff_completes_resumes_and_rebuilds_byte_identically(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    experiment_dir = tmp_path / "experiment"
    results_path = run_elimination_bakeoff(
        spec, experiment_dir, target=FakeTarget(), model_proposer=FakeModel()
    )
    experiment_bytes = (experiment_dir / "experiment.json").read_bytes()
    results_bytes = results_path.read_bytes()
    out = json.loads(results_bytes)
    assert out["status"] == "complete"
    assert out["decision"]["rule"] == "elimination-largest-budget-keep-change-reject-v1"

    # Matched child manifests differ only in the elimination policy fields.
    def manifest(policy: str) -> dict[str, object]:
        path = experiment_dir / "runs" / policy / "budget-8" / "seed-11" / "manifest.json"
        return json.loads(path.read_text())

    base, paired, spare = (
        manifest("no_elimination"),
        manifest("paired_elimination"),
        manifest("spare_near_tie"),
    )
    for other in (paired, spare):
        differing = {key for key in base if base[key] != other.get(key)}
        assert differing <= {"shadow_policy", "active_elimination", "fingerprint"}
    assert base["active_elimination"] is None
    assert paired["active_elimination"] is not None

    # Resume performs no work and rebuilds identical aggregate artifacts.
    recording = FakeTarget()
    run_elimination_bakeoff(
        spec, experiment_dir, resume=True, target=recording, model_proposer=FakeModel()
    )
    assert not recording.calls
    assert (experiment_dir / "experiment.json").read_bytes() == experiment_bytes
    assert results_path.read_bytes() == results_bytes
