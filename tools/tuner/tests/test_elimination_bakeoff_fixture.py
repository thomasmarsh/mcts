"""The checked-in canonical elimination bake-off fixtures stay in lock-step."""

from __future__ import annotations

import json

from regenerate_elimination_bakeoff_fixture import FIXTURE_DIR, build, build_spec

from tuner_cli.elimination_bakeoff import _cells, read_experiment


def test_fixtures_match_the_deterministic_builder() -> None:
    experiment_text, results_text = build()
    assert (FIXTURE_DIR / "experiment.json").read_text() == experiment_text
    assert (FIXTURE_DIR / "results.json").read_text() == results_text


def test_experiment_fixture_is_a_valid_strict_manifest() -> None:
    raw = read_experiment((FIXTURE_DIR / "experiment.json").read_text())
    spec = build_spec()
    expected = [
        {"cell_id": cell.cell_id, "budget": cell.budget, "seed": cell.seed, "policy": cell.policy}
        for cell in _cells(spec, FIXTURE_DIR)
    ]
    assert raw["cells"] == expected
    assert raw["spec"]["gate"]["authorized_policy_version"] == (
        "successive-halving-spare-near-tie-v1"
    )


def test_results_fixture_carries_the_frozen_decision() -> None:
    results = json.loads((FIXTURE_DIR / "results.json").read_text())
    assert results["decision"]["rule"] == "elimination-largest-budget-keep-change-reject-v1"
    assert results["decision"]["outcome"] == "change_to_spare_near_tie"
    assert {row["policy"] for row in results["active_safety_summaries"]} == {
        "spare_near_tie",
        "paired_elimination",
    }
    assert all(row["safe_in_bakeoff"] for row in results["active_safety_summaries"])
