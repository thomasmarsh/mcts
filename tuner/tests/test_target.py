"""Strict configured-comparison pair transport tests."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tuner_cli.__main__ import _parse_baseline_configs
from tuner_cli.config import OptimizerConfig, SearchConfig, TargetConfig
from tuner_cli.evaluation import (
    OpponentSnapshot,
    PairTask,
    Rating,
    configured_game_seed,
)
from tuner_cli.lifecycle import SessionId, TrialId
from tuner_cli.target import (
    PairExecutionError,
    _build_pair_cmd,
    evaluate_pair,
    parse_pair_output,
)


def _task() -> PairTask:
    return PairTask(
        SessionId("session"),
        TrialId("trial"),
        "pair-1",
        0,
        42,
        {"family": "ucb1"},
        OpponentSnapshot("random", {"family": "random"}, 0.0, 0.5),
        "pool",
        Rating(25.0, 8.3),
    )


def _output() -> str:
    seed = configured_game_seed(42)
    metrics = {"iterations_total": 12, "iterations_first_half": 5, "move_time_ms": 7}
    records = [
        {
            "type": "configured_match_result",
            "seq": 1,
            "round": 1,
            "seed": seed,
            "candidate_side": "first",
            "outcome": "candidate_win",
            "trace_game_seq": 40,
            "plies": 8,
            "elapsed_ms": 9,
            "candidate": metrics,
            "baseline": metrics,
        },
        {
            "type": "configured_match_result",
            "seq": 2,
            "round": 1,
            "seed": seed,
            "candidate_side": "second",
            "outcome": "baseline_win",
            "trace_game_seq": 41,
            "plies": 10,
            "elapsed_ms": 11,
            "candidate": metrics,
            "baseline": metrics,
        },
        {
            "type": "configured_comparison_summary",
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
        },
    ]
    return "\n".join(json.dumps(record) for record in records)


def _cfg() -> SearchConfig:
    return SearchConfig(
        optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("game-fake"))
    )


def test_parse_baseline_configs_preserves_id_and_raw_config():
    parsed = _parse_baseline_configs(
        ["strong-plus=" + json.dumps({"family": "ucb1", "c": 1.5})]
    )
    assert parsed == {"strong-plus": {"family": "ucb1", "c": 1.5}}


def test_parse_baseline_configs_rejects_missing_separator():
    with pytest.raises(ValueError):
        _parse_baseline_configs(["not-a-kv-pair"])


def test_pair_command_uses_one_configured_comparison_and_default_budget():
    cmd = _build_pair_cmd(_cfg(), Path("game-fake"), _task())
    assert cmd[:3] == ["game-fake", "compare", "eval"]
    assert cmd[cmd.index("--rounds") + 1] == "1"
    assert "--config" not in cmd
    assert cmd[cmd.index("--max-iterations") + 1] == "10000"


def test_pair_command_forwards_configs_time_budget_and_game_config():
    cfg = SearchConfig(
        optimizer=OptimizerConfig(),
        target=TargetConfig(
            binary=Path("game-fake"),
            game_config={"size": 7},
            max_time_ms=25,
        ),
    )
    task = _task()
    cmd = _build_pair_cmd(cfg, Path("game-fake"), task)
    assert json.loads(cmd[cmd.index("--candidate-config") + 1]) == task.candidate_config
    assert json.loads(cmd[cmd.index("--baseline-config") + 1]) == task.opponent.config
    assert json.loads(cmd[cmd.index("--game-config") + 1]) == {"size": 7}
    assert cmd[cmd.index("--max-time-ms") + 1] == "25"
    assert "--trace-path" not in cmd
    assert "--max-iterations" not in cmd


def test_parser_returns_two_ordered_physical_games_with_trace_ids():
    pair = parse_pair_output(_output(), _task())
    assert [game.candidate_side for game in pair.games] == ["first", "second"]
    assert [game.trace_game_seq for game in pair.games] == [40, 41]
    assert [game.outcome for game in pair.games] == ["candidate_win", "baseline_win"]
    assert pair.games[0].game_id != pair.games[1].game_id


def test_parser_requires_descriptor_assigned_trace_game_sequences():
    task = replace(_task(), trace_game_sequence_start=40)
    assert [
        game.trace_game_seq for game in parse_pair_output(_output(), task).games
    ] == [
        40,
        41,
    ]
    records = [json.loads(line) for line in _output().splitlines()]
    records[1]["trace_game_seq"] = 42
    with pytest.raises(ValueError, match="trace_game_seq"):
        parse_pair_output("\n".join(json.dumps(record) for record in records), task)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda records: records.pop(1), "exactly two"),
        (lambda records: records.insert(1, dict(records[0])), "exactly two"),
        (
            lambda records: records.__setitem__(
                1, {**records[1], "candidate_side": "first"}
            ),
            "sequence",
        ),
        (lambda records: records.__setitem__(slice(0, 2), records[1::-1]), "ordered"),
        (
            lambda records: records.__setitem__(1, {**records[1], "seed": 1}),
            "unexpected round or seed",
        ),
        (lambda records: records.__setitem__(2, {**records[2], "wins": 2}), "summary"),
    ],
)
def test_parser_rejects_incomplete_or_inconsistent_pairs(mutate, message):
    records = [json.loads(line) for line in _output().splitlines()]
    mutate(records)
    with pytest.raises(ValueError, match=message):
        parse_pair_output("\n".join(json.dumps(record) for record in records), _task())


def test_evaluate_pair_turns_timeout_and_invalid_output_into_pair_errors(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("game-fake", 1)

    monkeypatch.setattr("tuner_cli.target._run_with_heartbeat", timeout)
    with pytest.raises(PairExecutionError, match="timed out"):
        evaluate_pair(_cfg(), Path("game-fake"), _task())


def test_evaluate_pair_rejects_nonzero_exit_and_mutually_exclusive_budgets(monkeypatch):
    failed = subprocess.CompletedProcess([], 2, "", "bad config")
    monkeypatch.setattr(
        "tuner_cli.target._run_with_heartbeat", lambda *_args, **_kwargs: failed
    )
    with pytest.raises(PairExecutionError, match="code 2"):
        evaluate_pair(_cfg(), Path("game-fake"), _task())

    cfg = _cfg()
    cfg.target.max_iterations = 10
    cfg.target.max_time_ms = 10
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate_pair(cfg, Path("game-fake"), _task())
