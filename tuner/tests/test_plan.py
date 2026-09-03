"""`tuner_cli plan` — the resolved-shape preview a launch form renders.

Reuses `test_run.py`'s fake binary / objective / target, so no real game
process is spawned and nothing touches a real run directory.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from test_run import FakeTarget, _fake_binary, _objective

from tuner_cli.domain import SearchEffort
from tuner_cli.plan import plan_launch
from tuner_cli.run import RunOptions
from tuner_cli.space_overrides import decode_space_overrides


def _options(tmp_path: Path, **over: object) -> RunOptions:
    base = RunOptions(
        _fake_binary(tmp_path),
        tmp_path / "runs" / "run",
        objective_file=_objective(tmp_path),
        task_seed=9,
        finalists=1,
        cohort_size=4,
        bootstrap_candidates=2,
        random_reserve_candidates=1,
        tuning_pairs=2,
        tuning_pair_budget=16,
        validation_pair_budget=2,
        production_validation_pairs=2,
        tuning_effort=SearchEffort("iterations", 3),
        validation_effort=SearchEffort("iterations", 5),
        production_effort=SearchEffort("iterations", 9),
    )
    return replace(base, **over)  # type: ignore[arg-type]


def test_resolves_schema_default_opponent(tmp_path: Path) -> None:
    plan = plan_launch(_options(tmp_path), target=FakeTarget())
    assert plan["ok"] is True
    by_id = {opponent["id"]: opponent for opponent in plan["opponents"]}
    # The schema-default opponent is expanded to the actual resolved config,
    # not left as a `{"source": "schema_default"}` reference.
    default = by_id["schema-default"]
    assert default["source"] == "schema_default"
    assert default["role"] == "default"
    assert default["config"] == '{"family":"a"}'
    assert default["fingerprint"]
    assert by_id["historical"]["config"] == '{"family":"b"}'


def test_space_reflects_constraints(tmp_path: Path) -> None:
    options = _options(
        tmp_path,
        excluded_families=("d",),
        space_overrides=decode_space_overrides({"family": {"choices": ["a", "b", "c", "d"]}}),
    )
    plan = plan_launch(options, target=FakeTarget())
    # `space_overrides` narrows the declared choice set; `--exclude-family`
    # then removes `d` from what is left.
    assert plan["space"]["families"] == ["a", "b", "c"]
    assert plan["space"]["excluded_families"] == ["d"]
    family = next(p for p in plan["space"]["parameters"] if p["name"] == "family")
    assert family["choices"] == ["a", "b", "c"]


def test_efforts_and_budgets_are_reported(tmp_path: Path) -> None:
    plan = plan_launch(_options(tmp_path), target=FakeTarget())
    assert plan["efforts"]["production"] == {"kind": "iterations", "value": 9}
    budgets = plan["budgets"]
    assert budgets["derived"]["initial_cohort_pairs"] == 8
    assert budgets["derived"]["validation_pairs_per_finalist"] == 2
    assert plan["epoch"]["fingerprint"]


def test_no_side_effects(tmp_path: Path) -> None:
    options = _options(tmp_path)
    plan_launch(options, target=FakeTarget())
    assert not options.run_dir.exists()
    assert not (options.run_dir.parent / "plan-preview").exists()
    assert list(options.run_dir.parent.glob("*")) == []


def test_surfaces_preflight_errors(tmp_path: Path) -> None:
    # finalists must be smaller than the cohort — a validate_options rule that
    # preflight catches and plan re-surfaces verbatim.
    plan = plan_launch(_options(tmp_path, finalists=4, cohort_size=4), target=FakeTarget())
    assert plan["ok"] is False
    assert "finalists must be smaller than cohort size" in plan["errors"][0]
    # The plan stays partial rather than raising.
    assert "opponents" not in plan
