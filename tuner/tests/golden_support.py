"""Deterministic fake target and canonical options for the version-4 golden run.

The golden fixtures under ``tests/fixtures/version4`` are produced only from this
module so that a behavior change in manifest construction, evidence encoding,
replay, scientific projection, or report projection shows up as a fixture diff.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from tuner_cli.codec import JsonValue
from tuner_cli.domain import (
    Candidate,
    GameResult,
    Opponent,
    PairResult,
    PairTask,
    StrategyMetrics,
    ValidationError,
    ValidationResult,
)
from tuner_cli.identity import canonical_json, game_id
from tuner_cli.run import RunOptions
from tuner_cli.target import _splitmix_seed

FIXTURES = Path(__file__).parent / "fixtures" / "version4"

# The two cohorts need more valid configurations than either cohort alone, while
# still exercising one deterministic semantic rejection.
_FAMILIES = ("a", "b", "c", "d", "e", "f", "g", "h")
_WINNING_FAMILIES = frozenset({"b", "c"})
_INVALID_FAMILY = "e"


def golden_options(binary: Path, run_dir: Path, objective_file: Path) -> RunOptions:
    return RunOptions(
        game_binary=binary,
        run_dir=run_dir,
        objective_file=objective_file,
        seed=7,
        task_seed=9,
        cohort_size=4,
        finalists=2,
        bootstrap_candidates=2,
        random_reserve_candidates=1,
        tuning_pairs=4,
        validation_pairs=2,
        production_validation_pairs=2,
        tuning_max_iterations=3,
        validation_max_iterations=5,
        production_max_iterations=9,
    )


def write_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "game-druid"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return binary


def write_objective(tmp_path: Path) -> Path:
    path = tmp_path / "objective.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "objective_id": "druid-golden-v1",
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


def normalize_operational(text: str, manifest: dict[str, object]) -> str:
    """Mask the machine-specific, non-scientific fields of a version-4 artifact.

    The manifest embeds absolute binary/objective paths and a fingerprint that
    covers them, and that fingerprint reappears verbatim in the evidence and
    report. None of these are scientific content, so golden byte comparisons
    replace them with stable tokens; the excluded fields are asserted directly.
    """
    binary = manifest["binary"]
    objective = manifest["objective"]
    assert isinstance(binary, dict) and isinstance(objective, dict)
    replacements = {
        str(manifest["fingerprint"]): "<MANIFEST_FINGERPRINT>",
        str(binary["path"]): "<BINARY_PATH>",
        str(objective["source_path"]): "<OBJECTIVE_SOURCE_PATH>",
    }
    for actual, token in replacements.items():
        text = text.replace(actual, token)
    return text


def _family(candidate: Candidate) -> str:
    value = json.loads(candidate.canonical_config)["family"]
    return value if isinstance(value, str) else ""


class GoldenTarget:
    """A pure fake target whose per-candidate verdicts and outcomes are fixed."""

    def __init__(self) -> None:
        self.calls: list[PairTask] = []

    def describe(self) -> JsonValue:
        return {
            "kind": "druid",
            "label": "Druid",
            "description": "golden fixture target",
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
                        "choices": list(_FAMILIES),
                        "default": "a",
                    }
                ],
                "conditions": [],
            },
        }

    def validate(
        self, candidates: Sequence[Candidate], opponent: Opponent, game_config: str
    ) -> ValidationResult:
        invalid = [
            index
            for index, candidate in enumerate(candidates)
            if _family(candidate) == _INVALID_FAMILY
        ]
        if invalid:
            return ValidationResult(
                False,
                tuple(
                    ValidationError("family", "family 'e' is not supported", index)
                    for index in invalid
                ),
            )
        return ValidationResult(True, ())

    def evaluate(
        self,
        task: PairTask,
        candidate: Candidate,
        opponent: Opponent,
        game_config: str,
        timeout_seconds: int,
    ) -> PairResult:
        self.calls.append(task)
        outcome = "candidate_win" if _family(candidate) in _WINNING_FAMILIES else "draw"
        seed = _splitmix_seed(task.task_case.seed)

        def _game(seq: int, side: Literal["first", "second"]) -> GameResult:
            raw = {
                "type": "configured_match_result",
                "seq": seq,
                "round": 1,
                "seed": seed,
                "candidate_side": side,
                "outcome": outcome,
                "trace_game_seq": None,
                "plies": 1,
                "elapsed_ms": 1,
                "candidate": {
                    "iterations_total": 1,
                    "iterations_first_half": 1,
                    "move_time_ms": 1,
                },
                "baseline": {
                    "iterations_total": 1,
                    "iterations_first_half": 1,
                    "move_time_ms": 1,
                },
            }
            return GameResult(
                game_id(task, side),
                side,
                outcome,
                seed,
                1,
                seq,
                None,
                1,
                1,
                StrategyMetrics(1, 1, 1),
                StrategyMetrics(1, 1, 1),
                canonical_json(raw),
            )

        return PairResult(task, (_game(1, "first"), _game(2, "second")))
