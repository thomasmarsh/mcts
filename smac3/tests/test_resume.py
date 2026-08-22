"""Reusing a run id reloads both the Optuna study and its frozen opponent pool."""

from __future__ import annotations

from pathlib import Path

from smac3_cli.__main__ import run_optimization
from smac3_cli.config import OptimizerConfig, SearchConfig, TargetConfig


def _cfg(binary: Path, n_trials: int) -> SearchConfig:
    return SearchConfig(
        optimizer=OptimizerConfig(n_trials=n_trials, deterministic=True, seed=7),
        target=TargetConfig(binary=binary, rounds=1),
    )


def test_reusing_run_id_completes_only_new_trials_and_reloads_pool(
    game_nim_binary: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    first, first_pool = run_optimization(_cfg(game_nim_binary, 1), run_id="resume")
    first_ids = [anchor.id for anchor in first_pool.anchors]
    assert len(first.trials) == 1

    second, second_pool = run_optimization(_cfg(game_nim_binary, 2), run_id="resume")
    assert len(second.trials) == 2
    assert [anchor.id for anchor in second_pool.anchors][: len(first_ids)] == first_ids
    assert (tmp_path / "optuna_output" / "resume" / "study.db").is_file()
    assert (tmp_path / "optuna_output" / "resume" / "pool.json").is_file()
