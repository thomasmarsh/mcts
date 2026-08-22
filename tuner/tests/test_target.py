"""`target.py`'s `play_game`: argv construction, wins/losses/draws parsing,
and timeout/crashed/unparseable-output status tagging.

Mocks `subprocess.Popen` (what `play_game`'s `_run_with_heartbeat` actually
drives via `.communicate()`) rather than shelling out to a real game binary
-- unlike `test_resume.py`/`test_callback.py` (which exist specifically to
catch drift in the real ask/tell loop's behavior and would be pointless to
fake), this is purely about which argv `play_game` builds and how it parses
a match result, not about what a real search returns.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tuner_cli.__main__ import _parse_baseline_configs
from tuner_cli.config import OptimizerConfig, SearchConfig, TargetConfig
from tuner_cli.target import play_game


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


class _FakePopen:
    """Stand-in for `subprocess.Popen`, driven the same way `target.py`'s
    `_run_with_heartbeat` drives a real one: repeated `.communicate(timeout=
    ...)` calls until one returns instead of raising, then (only after a
    real timeout) one final no-timeout drain call. `fake_run(cmd)` is called
    on every timed `communicate()` -- if it returns a completed-process-like
    object, that call succeeds immediately (as if the process had already
    exited); if it raises `subprocess.TimeoutExpired`, this mirrors that so
    the heartbeat loop keeps polling. The no-timeout drain call after
    `kill()` never calls `fake_run` again -- a real killed process's final
    `communicate()` doesn't re-run the command, just collects already-
    buffered output.
    """

    def __init__(self, cmd, fake_run):
        self._cmd = cmd
        self._fake_run = fake_run
        self.returncode = 0

    def communicate(self, timeout=None):
        if timeout is None:
            return "", ""
        result = self._fake_run(self._cmd)
        self.returncode = getattr(result, "returncode", 0)
        return result.stdout, result.stderr

    def kill(self):
        pass


def _patch_popen(monkeypatch, fake_run):
    """Patch `subprocess.Popen` so `target.py`'s `_run_with_heartbeat` (the
    only thing that spawns a subprocess in `play_game`) drives `fake_run`
    through `_FakePopen` instead of a real process.
    """
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: _FakePopen(cmd, fake_run))


def _result(wins=1, losses=0, draws=0):
    return _FakeCompletedProcess(json.dumps({"wins": wins, "losses": losses, "draws": draws}))


def test_play_game_forwards_candidate_and_opponent_configs(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result(wins=3, losses=1, draws=0)

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    opponent = {"family": "random", "q_init": "Infinity"}
    wins, losses, draws, status = play_game(
        cfg, binary, {"family": "ucb1"}, opponent, seed=0
    )

    assert (wins, losses, draws, status) == (3, 1, 0, None)
    assert json.loads(captured["cmd"][captured["cmd"].index("--config") + 1]) == {
        "family": "ucb1"
    }
    assert json.loads(captured["cmd"][captured["cmd"].index("--baseline-config") + 1]) == opponent
    assert "--baseline" not in captured["cmd"]


def test_play_game_tags_timeout_as_status(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 600))

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    wins, losses, draws, status = play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert (wins, losses, draws) == (0, 0, 0)
    assert status == "timeout"


def test_play_game_tags_nonzero_exit_as_crashed(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    class _FakeFailedProcess:
        stdout = ""
        stderr = "boom"
        returncode = 1

    _patch_popen(monkeypatch, lambda cmd: _FakeFailedProcess())

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    wins, losses, draws, status = play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert (wins, losses, draws) == (0, 0, 0)
    assert status == "crashed"


def test_play_game_tags_unparseable_output_as_crashed(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    _patch_popen(monkeypatch, lambda cmd: _FakeCompletedProcess("not json"))

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    wins, losses, draws, status = play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert (wins, losses, draws) == (0, 0, 0)
    assert status == "crashed"


def test_play_game_forwards_game_config_when_set(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result()

    _patch_popen(monkeypatch, fake_run)

    game_config = {"size": {"w": 9, "h": 9}}
    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(binary=binary, rounds=4, game_config=game_config),
    )
    play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert "--game-config" in captured["cmd"]
    sent = json.loads(captured["cmd"][captured["cmd"].index("--game-config") + 1])
    assert sent == game_config


def test_play_game_omits_game_config_when_unset(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result()

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert "--game-config" not in captured["cmd"]


def test_play_game_forwards_max_iterations_when_set(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result()

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(binary=binary, rounds=4, max_iterations=1000),
    )
    play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert "--max-iterations" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--max-iterations") + 1] == "1000"


def test_play_game_omits_max_iterations_when_unset(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    _patch_popen(monkeypatch, lambda cmd, **kwargs: _result())

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result()

    _patch_popen(monkeypatch, fake_run)
    play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert "--max-iterations" not in captured["cmd"]


def test_play_game_forwards_max_time_ms_when_set(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result()

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(binary=binary, rounds=4, max_time_ms=5000),
    )
    play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert "--max-time-ms" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--max-time-ms") + 1] == "5000"
    assert "--max-iterations" not in captured["cmd"]


def test_play_game_omits_max_time_ms_when_unset(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result()

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert "--max-time-ms" not in captured["cmd"]


def test_play_game_rejects_both_max_iterations_and_max_time_ms(tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(binary=binary, rounds=4, max_iterations=1000, max_time_ms=5000),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)


def test_target_config_max_time_ms_defaults_none():
    assert TargetConfig().max_time_ms is None


def test_play_game_logs_heartbeat_while_waiting_on_a_slow_trial(monkeypatch, tmp_path: Path):
    """A trial that takes longer than one heartbeat tick to finish should
    log at least one "still running" heartbeat instead of blocking silently
    -- this is what makes a slow-but-alive trial distinguishable from a hung
    one in `stdout.log` before the full timeout kills it."""
    binary = tmp_path / "game-fake"
    binary.touch()

    calls: list[int] = []

    def fake_run(cmd, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise subprocess.TimeoutExpired(cmd, 30)
        return _result(wins=2, losses=2, draws=0)

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    wins, losses, draws, status = play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert (wins, losses, draws, status) == (2, 2, 0, None)
    assert len(calls) == 3


def test_play_game_forwards_trace_path_when_set(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result()

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    trace_path = str(tmp_path / "moves.jsonl")
    play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0, trace_path=trace_path)

    assert "--trace-path" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--trace-path") + 1] == trace_path


def test_play_game_omits_trace_path_when_unset(monkeypatch, tmp_path: Path):
    binary = tmp_path / "game-fake"
    binary.touch()

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _result()

    _patch_popen(monkeypatch, fake_run)

    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=binary, rounds=4))
    play_game(cfg, binary, {"family": "ucb1"}, {"family": "random"}, seed=0)

    assert "--trace-path" not in captured["cmd"]


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
