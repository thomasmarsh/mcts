"""`--baseline-config <id>=<json>` CLI parsing and `target.py`'s `train()`
dispatching a raw-config-backed instance id to `--baseline-config` instead
of `--baseline`.

`test_train_dispatches_...` mocks `subprocess.run` rather than shelling out
to a real game binary -- unlike `test_resume.py`/`test_callback.py` (which
exist specifically to catch drift in SMAC3's own behavior and would be
pointless to fake), this is purely about which argv `train()` builds, not
about what a real search returns.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from smac3_cli.__main__ import _parse_baseline_configs
from smac3_cli.config import OptimizerConfig, SearchConfig, TargetConfig
from smac3_cli.target import make_target


def test_parse_baseline_configs_parses_id_equals_json():
    parsed = _parse_baseline_configs(
        ['strong-plus=' + json.dumps({"family": "ucb1", "c": 1.5})]
    )
    assert parsed == {"strong-plus": {"family": "ucb1", "c": 1.5}}


def test_parse_baseline_configs_rejects_missing_equals():
    with pytest.raises(ValueError):
        _parse_baseline_configs(["not-a-kv-pair"])


def test_parse_baseline_configs_empty_list_is_empty_dict():
    assert _parse_baseline_configs([]) == {}


class _FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def test_train_dispatches_named_baseline_as_dash_dash_baseline(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompletedProcess(json.dumps({"cost": 0.25}))

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(binary=binary, rounds=4, baselines=["strong"]),
    )
    train = make_target(cfg)
    cost = train({"family": "ucb1"}, instance="strong", seed=0)

    assert cost == pytest.approx(0.25)
    assert "--baseline" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--baseline") + 1] == "strong"
    assert "--baseline-config" not in captured["cmd"]


def test_train_dispatches_ladder_instance_as_dash_dash_baseline_config(
    monkeypatch, tmp_path: Path
):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompletedProcess(json.dumps({"cost": 0.1}))

    monkeypatch.setattr(subprocess, "run", fake_run)

    ladder_config = {"family": "ucb1", "final_action": "robust_child", "c": 1.4}
    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(
            binary=binary,
            rounds=4,
            baselines=["strong"],
            baseline_configs={"ladder1": ladder_config},
        ),
    )
    train = make_target(cfg)
    cost = train({"family": "ucb1"}, instance="ladder1", seed=0)

    assert cost == pytest.approx(0.1)
    assert "--baseline-config" in captured["cmd"]
    sent = json.loads(captured["cmd"][captured["cmd"].index("--baseline-config") + 1])
    assert sent == ladder_config
    assert "--baseline" not in captured["cmd"]


@pytest.mark.parametrize("floor_id", ["flat_mc", "random"])
def test_train_dispatches_floor_baseline_as_dash_dash_baseline_config(
    monkeypatch, tmp_path: Path, floor_id: str
):
    # A game's `tune_eval` only recognizes `--baseline <id>` as one of its
    # *own* named presets (Druid: easy/medium/strong/master) -- routing a
    # floor family that way fails with "unknown baseline" on every trial,
    # which `train()`'s own non-zero-exit handling below scores as
    # `cost = 1.0`, an apparent 100%-loss streak that's actually every
    # trial silently erroring. This is exactly what surfaced as a real
    # regression: launching with "flat_mc" as the starting baseline (a
    # plain `target.baselines=['flat_mc']` override, not a
    # `baseline_configs` entry) produced 100% loss on every trial.
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompletedProcess(json.dumps({"cost": 0.1}))

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(binary=binary, rounds=4, baselines=[floor_id]),
    )
    train = make_target(cfg)
    cost = train({"family": "ucb1"}, instance=floor_id, seed=0)

    assert cost == pytest.approx(0.1)
    assert "--baseline-config" in captured["cmd"]
    sent = json.loads(captured["cmd"][captured["cmd"].index("--baseline-config") + 1])
    assert sent == {"family": floor_id, "q_init": "Infinity"}
    assert "--baseline" not in captured["cmd"]


def test_train_forwards_game_config_when_set(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompletedProcess(json.dumps({"cost": 0.5}))

    monkeypatch.setattr(subprocess, "run", fake_run)

    game_config = {"size": {"w": 9, "h": 9}}
    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(binary=binary, rounds=4, game_config=game_config),
    )
    train = make_target(cfg)
    cost = train({"family": "ucb1"}, seed=0)

    assert cost == pytest.approx(0.5)
    assert "--game-config" in captured["cmd"]
    sent = json.loads(captured["cmd"][captured["cmd"].index("--game-config") + 1])
    assert sent == game_config


def test_train_omits_game_config_when_unset(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompletedProcess(json.dumps({"cost": 0.5}))

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    train = make_target(cfg)
    train({"family": "ucb1"}, seed=0)

    assert "--game-config" not in captured["cmd"]


def test_train_forwards_max_iterations_when_set(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompletedProcess(json.dumps({"cost": 0.5}))

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(binary=binary, rounds=4, max_iterations=1000),
    )
    train = make_target(cfg)
    cost = train({"family": "ucb1"}, seed=0)

    assert cost == pytest.approx(0.5)
    assert "--max-iterations" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--max-iterations") + 1] == "1000"


def test_train_omits_max_iterations_when_unset(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompletedProcess(json.dumps({"cost": 0.5}))

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    train = make_target(cfg)
    train({"family": "ucb1"}, seed=0)

    assert "--max-iterations" not in captured["cmd"]


def test_optimizer_config_termination_cost_threshold_defaults_to_inf():
    import math

    assert OptimizerConfig().termination_cost_threshold == math.inf


def test_target_config_baseline_configs_defaults_empty():
    assert TargetConfig().baseline_configs == {}


def test_target_config_max_iterations_defaults_none():
    assert TargetConfig().max_iterations is None


def test_target_config_game_config_defaults_none():
    assert TargetConfig().game_config is None


def test_search_config_from_dict_reads_game_config_from_yaml():
    cfg = SearchConfig._from_dict({"target": {"game_config": {"size": {"w": 7, "h": 7}}}})
    assert cfg.target.game_config == {"size": {"w": 7, "h": 7}}
