"""Game-binary subprocess boundary and strict configured-comparison wire decoder."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol

from .codec import JsonObject, JsonValue, is_json_object, literal, strict_json, string
from .domain import (
    Candidate,
    GameResult,
    Opponent,
    PairResult,
    PairTask,
    StrategyMetrics,
    ValidationError,
    ValidationResult,
)
from .identity import canonical_json, game_id


class PairExecutionError(RuntimeError):
    """A subprocess failed before it produced one valid, complete pair."""

    def __init__(
        self,
        kind: str,
        message: str,
        command: list[str],
        *,
        returncode: int | None = None,
        stderr: str = "",
        stdout: str = "",
    ) -> None:
        super().__init__(message)
        self.kind, self.command, self.returncode = kind, command, returncode
        self.stderr, self.stdout = stderr, stdout


class Target(Protocol):
    def describe(self) -> JsonValue: ...

    def validate(
        self, candidates: Sequence[Candidate], opponent: Opponent, game_config: str
    ) -> ValidationResult: ...

    def evaluate(
        self,
        task: PairTask,
        candidate: Candidate,
        opponent: Opponent,
        game_config: str,
        timeout_seconds: int,
    ) -> PairResult: ...

    def cancel(self) -> None: ...


def _splitmix_seed(seed: int) -> int:
    value = seed & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 53) - 1)


class GameBinaryTarget:
    def __init__(self, binary_path: Path) -> None:
        self.binary_path = binary_path
        self._children: set[subprocess.Popen[str]] = set()
        self._children_lock = Lock()

    def cancel(self) -> None:
        with self._children_lock:
            children = tuple(self._children)
        for process in children:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

    def describe(self) -> JsonValue:
        completed = subprocess.run(
            [str(self.binary_path), "describe"], capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.binary_path} describe exited {completed.returncode}: {completed.stderr}"
            )
        try:
            response = _single_object(completed.stdout)
        except ValueError as error:
            raise RuntimeError(
                f"{self.binary_path} describe did not emit one JSON object: {error}"
            ) from error
        return response

    def validate(
        self, candidates: Sequence[Candidate], opponent: Opponent, game_config: str
    ) -> ValidationResult:
        command = [str(self.binary_path), "compare", "validate"]
        for candidate in candidates:
            command += ["--candidate-config", candidate.canonical_config]
        command += ["--baseline-config", opponent.canonical_config, "--game-config", game_config]
        completed = subprocess.run(command, capture_output=True, text=True)
        try:
            valid, errors = _validation_response(_single_object(completed.stdout))
        except ValueError as error:
            raise RuntimeError(
                f"invalid validation response from {self.binary_path}: {error}; "
                f"stderr: {completed.stderr}"
            ) from error
        if completed.returncode == 0 and valid and not errors:
            return ValidationResult(True, ())
        if completed.returncode == 1 and not valid:
            return ValidationResult(False, tuple(errors))
        raise RuntimeError(
            f"validation transport failure from {self.binary_path} ({completed.returncode}): "
            f"{completed.stderr}"
        )

    def evaluate(
        self,
        task: PairTask,
        candidate: Candidate,
        opponent: Opponent,
        game_config: str,
        timeout_seconds: int,
    ) -> PairResult:
        command = [
            str(self.binary_path),
            "compare",
            "eval",
            "--candidate-config",
            candidate.canonical_config,
            "--baseline-config",
            opponent.canonical_config,
            "--game-config",
            game_config,
            "--rounds",
            "1",
            "--seed",
            str(task.task_case.seed),
            "--max-iterations",
            str(task.budget.max_iterations),
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        with self._children_lock:
            self._children.add(process)
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                process.kill()
                stdout, stderr = process.communicate()
                raise PairExecutionError(
                    "timeout",
                    f"pair timed out after {timeout_seconds}s",
                    command,
                    stderr=stderr,
                    stdout=stdout,
                ) from error
            except KeyboardInterrupt:
                process.kill()
                process.communicate()
                raise
            if process.returncode != 0:
                raise PairExecutionError(
                    "nonzero_exit",
                    f"comparison exited {process.returncode}",
                    command,
                    returncode=process.returncode,
                    stderr=stderr,
                    stdout=stdout,
                )
            try:
                return parse_pair_output(stdout, task)
            except ValueError as error:
                raise PairExecutionError(
                    "malformed_output",
                    str(error),
                    command,
                    returncode=process.returncode,
                    stderr=stderr,
                    stdout=stdout,
                ) from error
        finally:
            with self._children_lock:
                self._children.discard(process)


def _single_object(stdout: str) -> JsonObject:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("expected exactly one JSON object")
    try:
        value = strict_json(lines[0], "response")
    except ValueError as error:
        raise ValueError("response is not strict JSON") from error
    if not is_json_object(value):
        raise ValueError("response is not an object")
    return value


def _validation_response(response: JsonObject) -> tuple[bool, list[ValidationError]]:
    valid, raw_errors = response.get("valid"), response.get("errors")
    if not isinstance(valid, bool) or not isinstance(raw_errors, list):
        raise ValueError("response needs Boolean valid and errors array")
    errors: list[ValidationError] = []
    for raw in raw_errors:
        if not is_json_object(raw):
            raise ValueError("malformed validation error")
        field = string(raw.get("field"), "validation field")
        message = string(raw.get("message"), "validation message")
        index = raw.get("candidate_index")
        if index is not None and (
            not isinstance(index, int) or isinstance(index, bool) or index < 0
        ):
            raise ValueError("invalid candidate_index")
        errors.append(ValidationError(field, message, index))
    return valid, errors


def parse_pair_output(stdout: str, task: PairTask) -> PairResult:
    records: list[JsonObject] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            value = strict_json(line, "comparison output")
        except ValueError as error:
            raise ValueError("output contains non-JSON text") from error
        if not is_json_object(value):
            raise ValueError("output record is not an object")
        records.append(value)
    if len(records) != 3 or [record.get("type") for record in records] != [
        "configured_match_result",
        "configured_match_result",
        "configured_comparison_summary",
    ]:
        raise ValueError("expected exactly two game records followed by one summary")
    expected_seed = _splitmix_seed(task.task_case.seed)
    games = (
        _decode_game(records[0], task, expected_seed, "first", 1),
        _decode_game(records[1], task, expected_seed, "second", 2),
    )
    expected = {
        "games": 2,
        "wins": sum(game.outcome == "candidate_win" for game in games),
        "losses": sum(game.outcome == "baseline_win" for game in games),
        "draws": sum(game.outcome == "draw" for game in games),
    }
    if any(_integer(records[2], key) != value for key, value in expected.items()):
        raise ValueError("summary does not match physical game outcomes")
    return PairResult(task, games)


def _decode_game(
    record: JsonObject, task: PairTask, seed: int, side: Literal["first", "second"], seq: int
) -> GameResult:
    outcome_value = string(record.get("outcome"), "outcome")
    if outcome_value == "candidate_win":
        outcome: Literal["candidate_win", "baseline_win", "draw"] = "candidate_win"
    elif outcome_value == "baseline_win":
        outcome = "baseline_win"
    elif outcome_value == "draw":
        outcome = "draw"
    else:
        raise ValueError("game has invalid outcome")
    if record.get("candidate_side") != side or outcome not in {
        "candidate_win",
        "baseline_win",
        "draw",
    }:
        raise ValueError("game has invalid candidate side or outcome")
    if (
        _integer(record, "round") != 1
        or _integer(record, "seed") != seed
        or _integer(record, "seq") != seq
    ):
        raise ValueError("game has unexpected round, seed, or sequence")
    trace = record.get("trace_game_seq")
    if trace is not None and (not isinstance(trace, int) or isinstance(trace, bool) or trace < 0):
        raise ValueError("trace_game_seq must be non-negative integer or null")
    return GameResult(
        game_id(task, side),
        literal(side, ("first", "second"), "candidate side"),
        outcome,
        seed,
        1,
        seq,
        trace,
        _integer(record, "plies"),
        _integer(record, "elapsed_ms"),
        _metrics(record.get("candidate"), "candidate"),
        _metrics(record.get("baseline"), "baseline"),
        canonical_json(record),
    )


def _metrics(value: object, label: str) -> StrategyMetrics:
    if not is_json_object(value):
        raise ValueError(f"{label} metrics must be an object")
    return StrategyMetrics(
        _integer(value, "iterations_total"),
        _integer(value, "iterations_first_half"),
        _integer(value, "move_time_ms"),
    )


def _integer(record: JsonObject, key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value
