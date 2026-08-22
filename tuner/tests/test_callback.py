"""The Optuna driver emits benchmark-compatible trial and incumbent records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuner_cli.__main__ import run_optimization
from tuner_cli.config import OptimizerConfig, SearchConfig, TargetConfig


def _cfg(binary: Path) -> SearchConfig:
    return SearchConfig(
        optimizer=OptimizerConfig(n_trials=2, deterministic=True, seed=7),
        target=TargetConfig(binary=binary, rounds=1),
    )


def _records(raw_stdout: str) -> list[dict]:
    records = []
    for line in raw_stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def test_ask_tell_loop_emits_true_skill_jsonl(game_nim_binary: Path, tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    study, _pool = run_optimization(_cfg(game_nim_binary), run_id="records", git_sha="test-sha")

    records = _records(capsys.readouterr().out)
    trials = [record for record in records if record.get("type") == "trial"]
    incumbents = [record for record in records if record.get("type") == "incumbent"]
    assert len(trials) == 2
    assert incumbents
    for record in trials:
        assert record["cost"] == pytest.approx(
            -(record["extra"]["mu"] - 3 * record["extra"]["sigma"])
        )
        assert record["extra"]["git_sha"] == "test-sha"
        assert isinstance(record["extra"]["opponents"], list)
    assert incumbents[-1]["config"] == study.best_trial.user_attrs["config"]
    assert incumbents[-1]["cost"] == pytest.approx(-study.best_value)
