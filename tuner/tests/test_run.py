from __future__ import annotations

import json
from pathlib import Path

from tuner_cli.domain import (
    GameResult,
    PairResult,
    PairTask,
    StrategyMetrics,
    ValidationResult,
)
from tuner_cli.identity import canonical_json, stable_id
from tuner_cli.report import write_report
from tuner_cli.run import RunOptions, run_foreground


def _fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "game-fake"
    binary.touch()
    binary.chmod(0o755)
    return binary


class FakeTarget:
    def __init__(self) -> None:
        self.calls: list[PairTask] = []

    def describe(self) -> dict[str, object]:
        return {
            "kind": "druid",
            "label": "Druid",
            "description": "fake",
            "default_config": {"size": 5},
            "ai_presets": [],
            "tuning": {
                "id": "strategy",
                "baselines": [],
                "eval_rounds": 1,
                "game_config": {"size": 5},
                "parameters": [
                    {
                        "name": "family",
                        "type": "categorical",
                        "choices": ["a", "b"],
                        "default": "a",
                    },
                ],
                "conditions": [],
            },
        }

    def validate(self, candidates, opponent, game_config):  # type: ignore[no-untyped-def]
        return ValidationResult(True, ())

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        self.calls.append(task)
        outcome = (
            "candidate_win" if json.loads(candidate.canonical_config)["family"] == "b" else "draw"
        )
        games = []
        for seq, side in ((1, "first"), (2, "second")):
            raw = {
                "type": "configured_match_result",
                "seq": seq,
                "round": 1,
                "seed": 0,
                "candidate_side": side,
                "outcome": outcome,
                "trace_game_seq": None,
                "plies": 1,
                "elapsed_ms": 1,
                "candidate": {"iterations_total": 1, "iterations_first_half": 1, "move_time_ms": 1},
                "baseline": {"iterations_total": 1, "iterations_first_half": 1, "move_time_ms": 1},
            }
            games.append(
                GameResult(
                    stable_id("game", {"pair": task.pair_id, "side": side}),
                    side,
                    outcome,
                    0,
                    1,
                    seq,
                    None,
                    1,
                    1,
                    StrategyMetrics(1, 1, 1),
                    StrategyMetrics(1, 1, 1),
                    canonical_json(raw),
                )
            )
        return PairResult(task, tuple(games))


def test_foreground_fake_run_has_common_blocks_and_rebuildable_report(tmp_path: Path) -> None:
    target = FakeTarget()
    run_dir = tmp_path / "run"
    run_foreground(
        RunOptions(
            _fake_binary(tmp_path),
            run_dir,
            cohort_size=2,
            finalists=1,
            tuning_pairs=2,
            validation_pairs=1,
            tuning_max_iterations=3,
            validation_max_iterations=5,
            production_max_iterations=9,
        ),
        target,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = [json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["type"] == "run_completed"
    assert not {item["seed"] for item in manifest["tuning_tasks"]["cases"]} & {
        item["seed"] for item in manifest["validation_tasks"]["cases"]
    }
    tuning_starts = [
        event["payload"]
        for event in events
        if event["type"] == "pair_started" and event["payload"]["phase"] == "tuning"
    ]
    assert [item["budget"] for item in tuning_starts] == [3] * 4
    report = (run_dir / "report.json").read_bytes()
    write_report(run_dir)
    assert (run_dir / "report.json").read_bytes() == report


def test_validation_claim_depends_only_on_iteration_budgets(tmp_path: Path) -> None:
    run_dir = tmp_path / "production"
    run_foreground(
        RunOptions(
            _fake_binary(tmp_path),
            run_dir,
            cohort_size=2,
            finalists=1,
            tuning_pairs=1,
            validation_pairs=1,
            tuning_max_iterations=3,
            validation_max_iterations=5,
            production_max_iterations=5,
        ),
        FakeTarget(),
    )
    assert json.loads((run_dir / "report.json").read_text())["validation_claim"] == "production"
