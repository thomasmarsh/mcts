"""Deterministic fake target and canonical options for the version-4 golden run.

The golden fixtures under ``tests/fixtures/version4`` are produced only from this
module so that a behavior change in manifest construction, evidence encoding,
replay, scientific projection, or report projection shows up as a fixture diff.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

from tuner_cli.codec import JsonValue
from tuner_cli.domain import (
    Candidate,
    GameResult,
    Opponent,
    PairResult,
    PairTask,
    SearchEffort,
    StrategyMetrics,
    ValidationError,
    ValidationResult,
)
from tuner_cli.identity import canonical_json, game_id
from tuner_cli.run import RunOptions
from tuner_cli.target import _splitmix_seed

FIXTURES = Path(__file__).parent / "fixtures" / "version4"
ACTIVE_FIXTURES = Path(__file__).parent / "fixtures" / "version4-active-halving"

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
        tuning_pairs=14,
        tuning_pair_budget=84,
        validation_pair_budget=4,
        production_validation_pairs=2,
        tuning_effort=SearchEffort("iterations", 3),
        validation_effort=SearchEffort("iterations", 5),
        production_effort=SearchEffort("iterations", 9),
    )


def active_halving_golden_options(binary: Path, run_dir: Path, objective_file: Path) -> RunOptions:
    """Golden options that enforce the gate-approved spare-near-tie halving policy."""
    return replace(
        golden_options(binary, run_dir, objective_file),
        shadow_policy="successive_halving",
        shadow_halving_spare_margin=0.10,
        active_elimination_audit_probability=0.25,
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

    def cancel(self) -> None:
        return None

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

    def _outcome(self, task: PairTask, candidate: Candidate) -> str:
        del task
        return "candidate_win" if _family(candidate) in _WINNING_FAMILIES else "draw"

    def evaluate(
        self,
        task: PairTask,
        candidate: Candidate,
        opponent: Opponent,
        game_config: str,
        timeout_seconds: int,
    ) -> PairResult:
        del opponent, game_config, timeout_seconds
        self.calls.append(task)
        outcome = self._outcome(task, candidate)
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


class ActiveHalvingGoldenTarget(GoldenTarget):
    """Graded per-family win rates so the eta-2 rank cut resolves a real order.

    Earlier families in ``_FAMILIES`` win more tuning tasks, producing a strict
    objective ranking the successive-halving policy can cut on -- unlike the
    two-tier :class:`GoldenTarget`, whose ties the spare-margin rule would carry.
    """

    def _outcome(self, task: PairTask, candidate: Candidate) -> str:
        if task.task_case.phase != "tuning":
            return "draw"
        rank = _FAMILIES.index(_family(candidate))
        return "candidate_win" if task.task_case.ordinal % len(_FAMILIES) >= rank else "draw"
