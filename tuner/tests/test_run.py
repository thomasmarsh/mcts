from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from tuner_cli.domain import (
    GameResult,
    PairResult,
    PairTask,
    StrategyMetrics,
    ValidationResult,
)
from tuner_cli.evidence import read_events, scientific_projection
from tuner_cli.identity import canonical_json, game_id
from tuner_cli.report import write_report
from tuner_cli.run import RunOptions, run_foreground
from tuner_cli.target import _splitmix_seed


def _fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "game-fake"
    binary.touch()
    binary.chmod(0o755)
    return binary


def _objective(tmp_path: Path) -> Path:
    path = tmp_path / "objective.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "objective_id": "fake-reference-v1",
                "game_kind": "druid",
                "opponents": [
                    {
                        "id": "schema-default",
                        "label": "Default",
                        "role": "default",
                        "weight": 1,
                        "config": {"source": "schema_default"},
                    },
                    {
                        "id": "historical",
                        "label": "Historical",
                        "role": "historical_reference",
                        "weight": 1,
                        "config": {"source": "inline", "value": {"family": "b"}},
                    },
                ],
                "start_distribution": {"kind": "default_only"},
            }
        )
    )
    return path


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
                        "choices": ["a", "b", "c", "d", "e"],
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
                "seed": _splitmix_seed(task.task_case.seed),
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
                    game_id(task, side),
                    side,
                    outcome,
                    _splitmix_seed(task.task_case.seed),
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


class InterruptingTarget(FakeTarget):
    def __init__(self, interrupt_on_call: int) -> None:
        super().__init__()
        self.interrupt_on_call = interrupt_on_call

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        if len(self.calls) + 1 == self.interrupt_on_call:
            self.calls.append(task)
            raise KeyboardInterrupt
        return super().evaluate(task, candidate, opponent, game_config, timeout_seconds)


def test_foreground_fake_run_has_common_blocks_and_rebuildable_report(tmp_path: Path) -> None:
    target = FakeTarget()
    run_dir = tmp_path / "run"
    run_foreground(
        RunOptions(
            _fake_binary(tmp_path),
            run_dir,
            objective_file=_objective(tmp_path),
            task_seed=9,
            cohort_size=4,
            finalists=1,
            bootstrap_candidates=2,
            random_reserve_candidates=1,
            tuning_pairs=2,
            validation_pairs=2,
            production_validation_pairs=2,
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
    assert not {item["seed"] for item in manifest["corpora"]["tuning"]["cases"]} & {
        item["seed"] for item in manifest["corpora"]["production_validation"]["cases"]
    }
    tuning_starts = [
        event["payload"]
        for event in events
        if event["type"] == "pair_started" and event["payload"]["phase"] == "tuning"
    ]
    assert [item["budget"] for item in tuning_starts] == [3] * 8
    report = (run_dir / "report.json").read_bytes()
    write_report(run_dir)
    assert (run_dir / "report.json").read_bytes() == report


def test_validation_claim_depends_only_on_iteration_budgets(tmp_path: Path) -> None:
    run_dir = tmp_path / "production"
    run_foreground(
        RunOptions(
            _fake_binary(tmp_path),
            run_dir,
            objective_file=_objective(tmp_path),
            task_seed=9,
            cohort_size=4,
            finalists=1,
            bootstrap_candidates=2,
            random_reserve_candidates=1,
            tuning_pairs=2,
            validation_pairs=2,
            production_validation_pairs=2,
            tuning_max_iterations=3,
            validation_max_iterations=5,
            production_max_iterations=5,
        ),
        FakeTarget(),
    )
    assert (
        json.loads((run_dir / "report.json").read_text())["validation_claim"]["claim"]
        == "production"
    )


def test_interrupted_pair_resumes_to_the_same_scientific_artifact(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    options = RunOptions(
        binary,
        tmp_path / "control" / "run",
        objective_file=_objective(tmp_path),
        task_seed=9,
        cohort_size=4,
        finalists=1,
        bootstrap_candidates=2,
        random_reserve_candidates=1,
        tuning_pairs=2,
        validation_pairs=2,
        production_validation_pairs=2,
        tuning_max_iterations=3,
        validation_max_iterations=5,
        production_max_iterations=9,
    )
    run_foreground(options, FakeTarget())
    interrupted = InterruptingTarget(interrupt_on_call=2)
    resumed_dir = tmp_path / "resumed" / "run"
    try:
        run_foreground(replace(options, run_dir=resumed_dir), interrupted)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the injected interruption should escape the foreground run")
    before = list(interrupted.calls)
    run_foreground(replace(options, run_dir=resumed_dir, resume=True), interrupted)
    control_events = read_events(options.run_dir / "evidence.jsonl")
    resumed_events = read_events(resumed_dir / "evidence.jsonl")
    assert scientific_projection(control_events) == scientific_projection(resumed_events)
    assert (options.run_dir / "report.json").read_bytes() == (
        resumed_dir / "report.json"
    ).read_bytes()
    completed = [
        event.payload["pair_id"] for event in resumed_events if event.type == "pair_completed"
    ]
    assert len(completed) == len(set(completed))
    assert interrupted.calls[len(before)].pair_id == before[-1].pair_id
    report = (resumed_dir / "report.json").read_bytes()
    (resumed_dir / "report.json").unlink()
    completed_target = FakeTarget()
    run_foreground(replace(options, run_dir=resumed_dir, resume=True), completed_target)
    assert not completed_target.calls
    assert (resumed_dir / "report.json").read_bytes() == report
