"""Execute one committed evaluation descriptor into its task-owned bundle."""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from .artifact_layout import (
    ARTIFACT_LAYOUT_SCHEMA_VERSION,
    TaskIdentity,
    game_sequences_for,
    parse_descriptor_filename,
)
from .config import json_dumps
from .evaluation import (
    GameResult,
    OpponentSnapshot,
    PairResult,
    PairTask,
    Rating,
    StrategyMetrics,
    configured_game_seed,
    game_id_for,
)
from .lifecycle import AttemptId, SessionId, TrialId
from .target import parse_pair_output
from .task_artifacts import (
    CompletionMember,
    Heartbeat,
    TaskCompletion,
    canonical_json_bytes,
    read_completion,
    sha256_digest,
    write_completion,
    write_heartbeat,
    write_immutable,
)

TASK_RESULT_SCHEMA_VERSION: Final = 1
TASK_FAILURE_SCHEMA_VERSION: Final = 1
_HEARTBEAT_INTERVAL_S: Final = 30
_TRIAL_TIMEOUT_S: Final = 600


class TaskDescriptorError(ValueError):
    """A descriptor is not a committed, executable task description."""


class TaskResultError(ValueError):
    """A committed task bundle cannot be consumed as its scheduled result."""


@dataclass(frozen=True)
class TaskArtifactReference:
    """Small terminal reference to one task bundle, without game payloads."""

    task_id: str
    attempt_id: str
    descriptor_digest: str
    outcome: str
    completion_digest: str


@dataclass(frozen=True)
class _Execution:
    descriptor_path: Path
    task_directory: Path
    identity: TaskIdentity
    descriptor_digest: str
    task: PairTask
    binary: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class _CapturedProcess:
    returncode: int
    stdout: bytes
    stderr: bytes


class _TimedOut(Exception):
    def __init__(self, stdout: bytes, stderr: bytes):
        self.stdout = stdout
        self.stderr = stderr


def execute_task_bundle(
    descriptor_path: str | Path,
    descriptor_digest: str,
    *,
    clock: Callable[[], datetime] | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
) -> TaskArtifactReference:
    """Run one validated descriptor and publish its task terminal marker last.

    The descriptor digest comes from the coordinator's immutable commit.  It
    is deliberately required here so an arbitrary JSON file can never become
    executable merely because it looks like a descriptor.
    """
    execution = _load_execution(descriptor_path, descriptor_digest)
    now = clock or (lambda: datetime.now(UTC))
    execution.task_directory.mkdir(parents=True, exist_ok=True)
    _require_real_directory(execution.task_directory)
    heartbeat = _HeartbeatWriter(execution, now)
    heartbeat.write()

    try:
        captured = _run_process(
            _build_pair_cmd_for_execution(execution), popen, heartbeat.write
        )
    except _TimedOut as error:
        return _write_failure(
            execution,
            error.stdout,
            error.stderr,
            "timeout",
            "pair timed out after 600s",
        )
    except OSError as error:
        return _write_failure(execution, b"", b"", "process_launch", str(error))

    if captured.returncode != 0:
        return _write_failure(
            execution,
            captured.stdout,
            captured.stderr,
            "process_exit",
            f"comparison exited with code {captured.returncode}",
        )

    try:
        stdout = captured.stdout.decode("utf-8")
        pair = parse_pair_output(stdout, execution.task)
    except (UnicodeDecodeError, ValueError) as error:
        return _write_failure(
            execution, captured.stdout, captured.stderr, "malformed_output", str(error)
        )

    stdout_member, stderr_member, trace_member = _write_logs(execution, captured)
    result_contents = canonical_json_bytes(
        {
            "attempt_id": execution.identity.attempt_id,
            "descriptor_digest": execution.descriptor_digest,
            "games": [_game_payload(game) for game in pair.games],
            "pair_id": execution.identity.pair_id,
            "schema_version": TASK_RESULT_SCHEMA_VERSION,
            "task_id": execution.identity.task_id,
        }
    )
    write_immutable(execution.task_directory / "result.json", result_contents)
    completion = TaskCompletion(
        task_id=execution.identity.task_id,
        attempt_id=execution.identity.attempt_id,
        descriptor_digest=execution.descriptor_digest,
        outcome="completed",
        terminal=CompletionMember.for_contents("result.json", result_contents),
        stdout=stdout_member,
        stderr=stderr_member,
        trace=trace_member,
    )
    completion_digest = write_completion(execution.task_directory, completion)
    return TaskArtifactReference(
        execution.identity.task_id,
        execution.identity.attempt_id,
        execution.descriptor_digest,
        "completed",
        completion_digest,
    )


def read_task_bundle(
    descriptor_path: str | Path,
    descriptor_digest: str,
    reference: Any,
    scheduled_task: PairTask,
) -> PairResult:
    """Validate one completed bundle and reconstruct its scheduled pair result.

    The descriptor is reloaded here instead of trusting the worker's small
    reference.  That binds the terminal files to the coordinator's session,
    trial, pair, and immutable task identity before any rating state changes.
    """
    execution = _load_execution(descriptor_path, descriptor_digest)
    _validate_scheduled_task(execution, scheduled_task)
    _validate_reference(reference, execution)
    completion = read_completion(
        execution.task_directory, execution.identity, execution.descriptor_digest
    )
    complete_contents = (execution.task_directory / "complete.json").read_bytes()
    if sha256_digest(complete_contents) != reference.completion_digest:
        raise TaskResultError("completion digest does not match worker reference")
    if completion.outcome != reference.outcome:
        raise TaskResultError("completion outcome does not match worker reference")
    terminal_contents = (
        execution.task_directory / completion.terminal.filename
    ).read_bytes()
    if completion.outcome == "failed":
        raise TaskResultError(
            f"committed task failed: {_failure_message(terminal_contents, execution)}"
        )
    return PairResult(execution.task, _decode_result(terminal_contents, execution))


def _validate_scheduled_task(execution: _Execution, scheduled_task: PairTask) -> None:
    actual = execution.task
    if (
        actual.session_id,
        actual.trial_id,
        actual.pair_id,
        actual.pair_index,
        actual.seed,
        actual.candidate_config,
        actual.opponent,
        actual.pool_snapshot_fingerprint,
        actual.rating_before,
    ) != (
        scheduled_task.session_id,
        scheduled_task.trial_id,
        scheduled_task.pair_id,
        scheduled_task.pair_index,
        scheduled_task.seed,
        scheduled_task.candidate_config,
        scheduled_task.opponent,
        scheduled_task.pool_snapshot_fingerprint,
        scheduled_task.rating_before,
    ):
        raise TaskResultError("descriptor does not match the scheduled pair")


def _validate_reference(reference: Any, execution: _Execution) -> None:
    if not isinstance(reference, TaskArtifactReference):
        raise TaskResultError("worker did not return a task artifact reference")
    if (
        reference.task_id,
        reference.attempt_id,
        reference.descriptor_digest,
    ) != (
        execution.identity.task_id,
        execution.identity.attempt_id,
        execution.descriptor_digest,
    ):
        raise TaskResultError(
            "worker reference identity or descriptor digest mismatches"
        )
    if reference.outcome not in ("completed", "failed"):
        raise TaskResultError("worker reference has an invalid outcome")
    if len(reference.completion_digest) != 64 or any(
        char not in "0123456789abcdef" for char in reference.completion_digest
    ):
        raise TaskResultError("worker reference has an invalid completion digest")


def _failure_message(contents: bytes, execution: _Execution) -> str:
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TaskResultError("failure artifact is not UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "attempt_id",
        "descriptor_digest",
        "kind",
        "message",
        "schema_version",
        "task_id",
    }:
        raise TaskResultError("failure artifact has an invalid schema")
    if not isinstance(payload["kind"], str) or not isinstance(payload["message"], str):
        raise TaskResultError("failure artifact has invalid values")
    if (
        payload["schema_version"],
        payload["task_id"],
        payload["attempt_id"],
        payload["descriptor_digest"],
    ) != (
        TASK_FAILURE_SCHEMA_VERSION,
        execution.identity.task_id,
        execution.identity.attempt_id,
        execution.descriptor_digest,
    ):
        raise TaskResultError(
            "failure artifact identity or descriptor digest mismatches"
        )
    return f"{payload['kind']}: {payload['message']}"


def _decode_result(
    contents: bytes, execution: _Execution
) -> tuple[GameResult, GameResult]:
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TaskResultError("result artifact is not UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "attempt_id",
        "descriptor_digest",
        "games",
        "pair_id",
        "schema_version",
        "task_id",
    }:
        raise TaskResultError("result artifact has an invalid schema")
    if payload["schema_version"] != TASK_RESULT_SCHEMA_VERSION:
        raise TaskResultError("result artifact has an unsupported schema version")
    if (
        payload["task_id"],
        payload["attempt_id"],
        payload["descriptor_digest"],
        payload["pair_id"],
    ) != (
        execution.identity.task_id,
        execution.identity.attempt_id,
        execution.descriptor_digest,
        execution.identity.pair_id,
    ):
        raise TaskResultError(
            "result artifact identity or descriptor digest mismatches"
        )
    if not isinstance(payload["games"], list) or len(payload["games"]) != 2:
        raise TaskResultError("result artifact must contain exactly two games")
    return (
        _decode_result_game(payload["games"][0], execution.task, "first"),
        _decode_result_game(payload["games"][1], execution.task, "second"),
    )


def _decode_result_game(value: Any, task: PairTask, side: str) -> GameResult:
    required = {
        "baseline",
        "candidate",
        "candidate_side",
        "elapsed_ms",
        "game_id",
        "outcome",
        "plies",
        "round",
        "seed",
        "trace_game_seq",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise TaskResultError("result game has an invalid schema")
    expected_trace = task.trace_game_sequence_start
    if expected_trace is None:
        raise TaskResultError("descriptor task lacks its trace game sequence")
    if side == "second":
        expected_trace += 1
    if (
        value["game_id"] != game_id_for(task.pair_id, side)
        or value["candidate_side"] != side
        or value["seed"] != configured_game_seed(task.seed)
        or value["round"] != 1
        or value["trace_game_seq"] != expected_trace
        or value["outcome"] not in ("candidate_win", "baseline_win", "draw")
    ):
        raise TaskResultError("result game does not match its scheduled identity")
    return GameResult(
        value["game_id"],
        value["candidate_side"],
        value["outcome"],
        value["seed"],
        value["round"],
        value["trace_game_seq"],
        _result_integer(value, "plies"),
        _result_integer(value, "elapsed_ms"),
        _decode_result_metrics(value["candidate"]),
        _decode_result_metrics(value["baseline"]),
    )


def _decode_result_metrics(value: Any) -> StrategyMetrics:
    if not isinstance(value, dict) or set(value) != {
        "iterations_first_half",
        "iterations_total",
        "move_time_ms",
    }:
        raise TaskResultError("result metrics have an invalid schema")
    return StrategyMetrics(
        _result_integer(value, "iterations_total"),
        _result_integer(value, "iterations_first_half"),
        _result_integer(value, "move_time_ms"),
    )


def _result_integer(value: dict[str, Any], field: str) -> int:
    result = value.get(field)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise TaskResultError(f"result {field} must be a non-negative integer")
    return result


def _load_execution(descriptor_path: str | Path, descriptor_digest: str) -> _Execution:
    path = Path(descriptor_path)
    _require_regular_file(path)
    if path.parent.name != "descriptors":
        raise TaskDescriptorError("descriptor must live directly in descriptors")
    artifact_root = path.parent.parent
    _require_real_directory(artifact_root)
    try:
        sequence, task_id = parse_descriptor_filename(path.name)
        contents = path.read_bytes()
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise TaskDescriptorError("descriptor is not valid canonical JSON") from error
    if sha256_digest(contents) != descriptor_digest:
        raise TaskDescriptorError("descriptor digest does not match committed bytes")
    if canonical_json_bytes(payload) != contents:
        raise TaskDescriptorError("descriptor is not canonical JSON")
    identity, task, binary = _decode_descriptor(payload, sequence, task_id)
    task_directory = artifact_root / "tasks" / identity.task_id
    if payload["task_directory"] != f"tasks/{identity.task_id}":
        raise TaskDescriptorError(
            "descriptor task directory does not match task identity"
        )
    tasks_directory = task_directory.parent
    if tasks_directory.exists():
        _require_real_directory(tasks_directory)
    if task_directory.exists():
        _require_real_directory(task_directory)
    return _Execution(
        path, task_directory, identity, descriptor_digest, task, binary, payload
    )


def _decode_descriptor(
    payload: Any, sequence: int, task_id: str
) -> tuple[TaskIdentity, PairTask, Path]:
    required = {
        "artifact_layout_schema_version",
        "attempt_id",
        "bench_run_id",
        "binary",
        "candidate_config",
        "created_at",
        "game",
        "game_ids",
        "manifest_fingerprint",
        "opponent",
        "optimizer_id",
        "pair_id",
        "pair_index",
        "pool_snapshot",
        "pool_snapshot_fingerprint",
        "rating_before",
        "schema_version",
        "search_budget",
        "seed",
        "session_id",
        "task_directory",
        "task_id",
        "task_sequence",
        "trace_game_sequences",
        "trial_id",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise TaskDescriptorError("descriptor has an invalid schema")
    if (
        payload["artifact_layout_schema_version"] != ARTIFACT_LAYOUT_SCHEMA_VERSION
        or payload["schema_version"] != 1
    ):
        raise TaskDescriptorError("descriptor has an unsupported schema version")
    try:
        identity = TaskIdentity.for_pair(
            AttemptId(_string(payload, "attempt_id")),
            _positive_int(payload, "task_sequence"),
            _string(payload, "pair_id"),
        )
    except ValueError as error:
        raise TaskDescriptorError("descriptor has an invalid task identity") from error
    if (identity.task_sequence, identity.task_id) != (sequence, task_id):
        raise TaskDescriptorError("descriptor filename does not match task identity")
    if payload["task_id"] != identity.task_id:
        raise TaskDescriptorError("descriptor task_id does not match task identity")
    sequences = game_sequences_for(identity.task_sequence)
    if payload["trace_game_sequences"] != {
        "candidate_first": sequences.candidate_first,
        "candidate_second": sequences.candidate_second,
    }:
        raise TaskDescriptorError(
            "descriptor trace sequences do not match task sequence"
        )
    _validate_descriptor_values(payload)
    opponent = payload["opponent"]
    rating = payload["rating_before"]
    task = PairTask(
        SessionId(payload["session_id"]),
        TrialId(payload["trial_id"]),
        identity.pair_id,
        payload["pair_index"],
        payload["seed"],
        payload["candidate_config"],
        OpponentSnapshot(
            opponent["anchor_id"], opponent["config"], opponent["mu"], opponent["sigma"]
        ),
            payload["pool_snapshot_fingerprint"],
            Rating(rating["mu"], rating["sigma"]),
            sequences.candidate_first,
        )
    return identity, task, Path(payload["binary"]["path"])


def _validate_descriptor_values(payload: dict[str, Any]) -> None:
    for field in (
        "session_id",
        "trial_id",
        "manifest_fingerprint",
        "optimizer_id",
        "pool_snapshot_fingerprint",
        "created_at",
    ):
        _string(payload, field)
    if payload["bench_run_id"] is not None:
        _string(payload, "bench_run_id")
    for field in ("pair_index", "seed"):
        _nonnegative_int(payload, field)
    if not isinstance(payload["candidate_config"], dict) or not isinstance(
        payload["pool_snapshot"], list
    ):
        raise TaskDescriptorError(
            "descriptor configs and pool snapshot have invalid types"
        )
    if not isinstance(payload["binary"], dict) or set(payload["binary"]) != {"path"}:
        raise TaskDescriptorError("descriptor binary has an invalid schema")
    binary = _string(payload["binary"], "path")
    if not Path(binary).is_absolute():
        raise TaskDescriptorError("descriptor binary path must be absolute")
    if (
        not isinstance(payload["game"], dict)
        or set(payload["game"]) != {"game_config", "rounds"}
        or payload["game"]["rounds"] != 1
    ):
        raise TaskDescriptorError("descriptor game must specify exactly one round")
    if payload["game"]["game_config"] is not None and not isinstance(
        payload["game"]["game_config"], dict
    ):
        raise TaskDescriptorError("descriptor game config has an invalid type")
    _opponent(payload["opponent"])
    _rating(payload["rating_before"])
    if not isinstance(payload["game_ids"], dict) or set(payload["game_ids"]) != {
        "candidate_first",
        "candidate_second",
    }:
        raise TaskDescriptorError("descriptor game IDs have an invalid schema")
    if not all(
        isinstance(value, str) and value for value in payload["game_ids"].values()
    ):
        raise TaskDescriptorError("descriptor game IDs have invalid values")
    if payload["game_ids"] != {
        "candidate_first": game_id_for(payload["pair_id"], "first"),
        "candidate_second": game_id_for(payload["pair_id"], "second"),
    }:
        raise TaskDescriptorError("descriptor game IDs do not match the pair")
    budget = payload["search_budget"]
    if not isinstance(budget, dict) or set(budget) not in (
        {"kind", "max_iterations"},
        {"kind", "max_time_ms"},
    ):
        raise TaskDescriptorError("descriptor search budget has an invalid schema")
    if budget.get("kind") == "max_iterations":
        _positive_int(budget, "max_iterations")
    elif budget.get("kind") == "max_time_ms":
        _positive_int(budget, "max_time_ms")
    else:
        raise TaskDescriptorError("descriptor search budget kind is invalid")


def _opponent(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "anchor_id",
        "config",
        "mu",
        "sigma",
    }:
        raise TaskDescriptorError("descriptor opponent has an invalid schema")
    if not isinstance(value["anchor_id"], str) or not isinstance(value["config"], dict):
        raise TaskDescriptorError("descriptor opponent has invalid values")
    _numeric_rating(value)


def _rating(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"mu", "sigma"}:
        raise TaskDescriptorError("descriptor rating has invalid values")
    _numeric_rating(value)


def _numeric_rating(value: dict[str, Any]) -> None:
    if not all(
        isinstance(value[key], (int, float)) and not isinstance(value[key], bool)
        for key in ("mu", "sigma")
    ):
        raise TaskDescriptorError("descriptor rating has invalid values")


def _string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise TaskDescriptorError(f"descriptor {field} must be a nonempty string")
    return result


def _nonnegative_int(value: dict[str, Any], field: str) -> int:
    result = value.get(field)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise TaskDescriptorError(f"descriptor {field} must be a nonnegative integer")
    return result


def _positive_int(value: dict[str, Any], field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise TaskDescriptorError(f"descriptor {field} must be positive")
    return result


def _build_pair_cmd_for_execution(execution: _Execution) -> list[str]:
    budget = execution.payload["search_budget"]
    command = [
        str(execution.binary),
        "compare",
        "eval",
        "--candidate-config",
        json_dumps(execution.task.candidate_config),
        "--baseline-config",
        json_dumps(execution.task.opponent.config),
        "--rounds",
        "1",
        "--seed",
        str(execution.task.seed),
    ]
    game = execution.payload["game"]
    if game["game_config"] is not None:
        command += [
            "--game-config",
            json_dumps(game["game_config"]),
        ]
    if budget["kind"] == "max_time_ms":
        command += ["--max-time-ms", str(budget["max_time_ms"])]
    else:
        command += ["--max-iterations", str(budget["max_iterations"])]
    command += [
        "--trace-path",
        str(execution.task_directory / "trace.jsonl"),
        "--trace-game-sequence-start",
        str(execution.payload["trace_game_sequences"]["candidate_first"]),
    ]
    return command


def _run_process(
    cmd: list[str], popen: Callable[..., Any], heartbeat: Callable[[], None]
) -> _CapturedProcess:
    proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = 0
    while True:
        wait = min(_HEARTBEAT_INTERVAL_S, _TRIAL_TIMEOUT_S - elapsed)
        try:
            stdout, stderr = proc.communicate(timeout=wait)
            return _CapturedProcess(proc.returncode, _bytes(stdout), _bytes(stderr))
        except subprocess.TimeoutExpired:
            elapsed += wait
            heartbeat()
            if elapsed >= _TRIAL_TIMEOUT_S:
                proc.kill()
                stdout, stderr = proc.communicate()
                raise _TimedOut(_bytes(stdout), _bytes(stderr)) from None


def _bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else value.encode()


@dataclass
class _HeartbeatWriter:
    execution: _Execution
    clock: Callable[[], datetime]
    sequence: int = 0

    def write(self) -> None:
        observed = self.clock().astimezone(UTC)
        write_heartbeat(
            self.execution.task_directory / "heartbeat.json",
            Heartbeat(
                self.execution.identity.task_id,
                self.execution.identity.attempt_id,
                self.sequence,
                _iso_time(observed),
                _iso_time(observed + timedelta(seconds=2 * _HEARTBEAT_INTERVAL_S)),
            ),
        )
        self.sequence += 1


def _iso_time(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _write_failure(
    execution: _Execution, stdout: bytes, stderr: bytes, kind: str, message: str
) -> TaskArtifactReference:
    stdout_member, stderr_member, trace_member = _write_logs(
        execution, _CapturedProcess(-1, stdout, stderr)
    )
    failure_contents = canonical_json_bytes(
        {
            "attempt_id": execution.identity.attempt_id,
            "descriptor_digest": execution.descriptor_digest,
            "kind": kind,
            "message": message,
            "schema_version": TASK_FAILURE_SCHEMA_VERSION,
            "task_id": execution.identity.task_id,
        }
    )
    write_immutable(execution.task_directory / "failure.json", failure_contents)
    completion = TaskCompletion(
        task_id=execution.identity.task_id,
        attempt_id=execution.identity.attempt_id,
        descriptor_digest=execution.descriptor_digest,
        outcome="failed",
        terminal=CompletionMember.for_contents("failure.json", failure_contents),
        stdout=stdout_member,
        stderr=stderr_member,
        trace=trace_member,
    )
    completion_digest = write_completion(execution.task_directory, completion)
    return TaskArtifactReference(
        execution.identity.task_id,
        execution.identity.attempt_id,
        execution.descriptor_digest,
        "failed",
        completion_digest,
    )


def _game_payload(game: Any) -> dict[str, Any]:
    return {
        "baseline": {
            "iterations_first_half": game.baseline.iterations_first_half,
            "iterations_total": game.baseline.iterations_total,
            "move_time_ms": game.baseline.move_time_ms,
        },
        "candidate": {
            "iterations_first_half": game.candidate.iterations_first_half,
            "iterations_total": game.candidate.iterations_total,
            "move_time_ms": game.candidate.move_time_ms,
        },
        "candidate_side": game.candidate_side,
        "elapsed_ms": game.elapsed_ms,
        "game_id": game.game_id,
        "outcome": game.outcome,
        "plies": game.plies,
        "round": game.round,
        "seed": game.seed,
        "trace_game_seq": game.trace_game_seq,
    }


def _write_logs(
    execution: _Execution, captured: _CapturedProcess
) -> tuple[CompletionMember, CompletionMember, CompletionMember | None]:
    write_immutable(execution.task_directory / "stdout.log", captured.stdout)
    write_immutable(execution.task_directory / "stderr.log", captured.stderr)
    trace_path = execution.task_directory / "trace.jsonl"
    trace_member = None
    if trace_path.exists():
        _require_regular_file(trace_path)
        trace_member = CompletionMember.for_contents(
            "trace.jsonl", trace_path.read_bytes()
        )
    return (
        CompletionMember.for_contents("stdout.log", captured.stdout),
        CompletionMember.for_contents("stderr.log", captured.stderr),
        trace_member,
    )


def _require_regular_file(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise TaskDescriptorError(f"required file is missing: {path.name}") from error
    if not stat.S_ISREG(info.st_mode):
        raise TaskDescriptorError(f"required file is not regular: {path.name}")


def _require_real_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise TaskDescriptorError(
            f"required directory is missing: {path.name}"
        ) from error
    if not stat.S_ISDIR(info.st_mode):
        raise TaskDescriptorError(f"required directory is not real: {path.name}")
