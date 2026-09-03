"""`tuner_cli preflight` — the dry-run every launch problem the form must
catch. Reuses `test_run.py`'s fake binary / objective / target so no real
game process is spawned."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from test_run import FakeTarget, _fake_binary, _objective

from tuner_cli.domain import SearchEffort, ValidationResult
from tuner_cli.preflight import preflight_launch
from tuner_cli.run import RunOptions


def _options(tmp_path: Path, **over: object) -> RunOptions:
    base = RunOptions(
        _fake_binary(tmp_path),
        tmp_path / "run",
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


def test_coherent_options_pass(tmp_path: Path) -> None:
    assert preflight_launch(_options(tmp_path), target=FakeTarget()) == {"ok": True, "errors": []}


def test_argument_coherence_failure_is_reported(tmp_path: Path) -> None:
    # finalists must be smaller than the cohort — a validate_options rule.
    result = preflight_launch(_options(tmp_path, finalists=4, cohort_size=4), target=FakeTarget())
    assert result["ok"] is False
    assert "finalists must be smaller than cohort size" in result["errors"][0]


def test_validation_budget_relationship_failure_is_reported(tmp_path: Path) -> None:
    # validation pairs (budget // finalists) may not exceed production pairs —
    # a validate_objective_options rule, so this exercises the game-spec /
    # objective-resolution stage too.
    result = preflight_launch(
        _options(tmp_path, validation_pair_budget=240, production_validation_pairs=60, finalists=1),
        target=FakeTarget(),
    )
    assert result["ok"] is False
    assert "cannot exceed production validation pairs" in result["errors"][0]


def test_panel_opponent_rejected_by_binary_is_reported(tmp_path: Path) -> None:
    # A historical-reference config the binary's `compare validate` rejects
    # (e.g. a half-specified family config) must fail preflight, not launch --
    # this is the `preflight_default` stage.
    class RejectingTarget(FakeTarget):  # type: ignore[misc,valid-type]
        def validate(self, candidates, opponent, game_config):  # type: ignore[no-untyped-def]
            return ValidationResult(False, ())

    result = preflight_launch(_options(tmp_path), target=RejectingTarget())
    assert result["ok"] is False
    assert "schema default failed panel preflight" in result["errors"][0]


def test_validation_budget_must_divide_finalists(tmp_path: Path) -> None:
    result = preflight_launch(
        _options(tmp_path, validation_pair_budget=5, finalists=2),
        target=FakeTarget(),
    )
    assert result["ok"] is False
    assert "validation pair budget must divide finalists" in result["errors"][0]
