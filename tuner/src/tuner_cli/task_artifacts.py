"""Durable, per-task artifacts for one submitted tuning evaluation.

The layout module owns the names.  This module owns the small filesystem
protocol which publishes their contents: task leaves are immutable apart from
the worker heartbeat, and a completion record is the sole terminal marker.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Final, Literal

from .artifact_layout import (
    ARTIFACT_LAYOUT_SCHEMA_VERSION,
    ArtifactLayout,
    TaskIdentity,
    game_sequences_for,
    validate_task_id,
)
from .evaluation import game_id_for
from .lifecycle import strict_json_dumps

TASK_COMPLETION_SCHEMA_VERSION: Final = 1
ATTEMPT_DESCRIPTOR_SCHEMA_VERSION: Final = 1
TASK_DESCRIPTOR_SCHEMA_VERSION: Final = 1
_DIGEST_LENGTH: Final = 64

TaskOutcome = Literal["completed", "failed"]


class ArtifactIntegrityError(ValueError):
    """An artifact exists but does not satisfy its immutable contract."""


@dataclass(frozen=True)
class DescriptorCommit:
    """The identity and immutable digest published before worker submission."""

    identity: TaskIdentity
    digest: str


@dataclass
class TaskDescriptorAllocator:
    """Coordinator-owned attempt evidence and task identity allocation."""

    layout: ArtifactLayout
    session_id: str
    optimizer_id: str
    attempt_id: str
    bench_run_id: str | None
    manifest_fingerprint: str
    _next_task_sequence: int = 1

    @classmethod
    def start(
        cls,
        physical_attempt_root: str | Path,
        *,
        session_id: str,
        optimizer_id: str,
        attempt_id: str,
        bench_run_id: str | None,
        manifest_fingerprint: str,
    ) -> TaskDescriptorAllocator:
        """Publish one immutable attempt record before any task can be allocated."""
        allocator = cls(
            ArtifactLayout.for_attempt_root(physical_attempt_root),
            session_id,
            optimizer_id,
            attempt_id,
            bench_run_id,
            manifest_fingerprint,
        )
        write_immutable_json(allocator.layout.attempt, allocator.attempt_payload())
        return allocator

    def attempt_payload(self) -> dict[str, Any]:
        """Build the immutable physical-attempt record without a task root path."""
        return {
            "artifact_layout_schema_version": ARTIFACT_LAYOUT_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "bench_run_id": self.bench_run_id,
            "created_at": _created_at(),
            "manifest_fingerprint": self.manifest_fingerprint,
            "optimizer_id": self.optimizer_id,
            "schema_version": ATTEMPT_DESCRIPTOR_SCHEMA_VERSION,
            "session_id": self.session_id,
        }

    def commit_task(
        self,
        task: Any,
        *,
        cfg: Any,
        binary: Path,
        pool_snapshot: list[Any],
    ) -> DescriptorCommit:
        """Allocate and publish one complete task descriptor exactly once."""
        identity = TaskIdentity.for_pair(
            self.attempt_id, self._next_task_sequence, task.pair_id
        )
        payload = self.task_payload(identity, task, cfg, binary, pool_snapshot)
        digest = write_immutable_json(self.layout.descriptor(identity), payload)
        self._next_task_sequence += 1
        return DescriptorCommit(identity, digest)

    def task_payload(
        self,
        identity: TaskIdentity,
        task: Any,
        cfg: Any,
        binary: Path,
        pool_snapshot: list[Any],
    ) -> dict[str, Any]:
        """Freeze all causal submission inputs in a relocatable descriptor."""
        sequences = game_sequences_for(identity.task_sequence)
        task_directory = f"tasks/{identity.task_id}"
        if cfg.target.max_time_ms is not None:
            search_budget: dict[str, int | str] = {
                "kind": "max_time_ms",
                "max_time_ms": cfg.target.max_time_ms,
            }
        else:
            search_budget = {
                "kind": "max_iterations",
                "max_iterations": cfg.target.max_iterations or 10_000,
            }
        return {
            "artifact_layout_schema_version": ARTIFACT_LAYOUT_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "bench_run_id": self.bench_run_id,
            "binary": {"path": str(binary.resolve())},
            "candidate_config": deepcopy(task.candidate_config),
            "created_at": _created_at(),
            "game": {
                "game_config": deepcopy(cfg.target.game_config),
                "rounds": 1,
            },
            "game_ids": {
                "candidate_first": game_id_for(task.pair_id, "first"),
                "candidate_second": game_id_for(task.pair_id, "second"),
            },
            "manifest_fingerprint": self.manifest_fingerprint,
            "opponent": _opponent_payload(task.opponent),
            "optimizer_id": self.optimizer_id,
            "pair_id": task.pair_id,
            "pair_index": task.pair_index,
            "pool_snapshot": [_opponent_payload(anchor) for anchor in pool_snapshot],
            "pool_snapshot_fingerprint": task.pool_snapshot_fingerprint,
            "rating_before": {
                "mu": task.rating_before.mu,
                "sigma": task.rating_before.sigma,
            },
            "schema_version": TASK_DESCRIPTOR_SCHEMA_VERSION,
            "search_budget": search_budget,
            "seed": task.seed,
            "session_id": self.session_id,
            "task_directory": task_directory,
            "task_id": identity.task_id,
            "task_sequence": identity.task_sequence,
            "trace_game_sequences": {
                "candidate_first": sequences.candidate_first,
                "candidate_second": sequences.candidate_second,
            },
            "trial_id": task.trial_id,
        }


def _created_at() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _opponent_payload(opponent: Any) -> dict[str, Any]:
    return {
        "anchor_id": opponent.anchor_id,
        "config": deepcopy(opponent.config),
        "mu": opponent.mu,
        "sigma": opponent.sigma,
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically as UTF-8 without a presentation newline."""
    return strict_json_dumps(value, sort_keys=True).encode("utf-8")


def sha256_digest(contents: bytes) -> str:
    """Return the lowercase SHA-256 digest used in task evidence."""
    if not isinstance(contents, bytes):
        raise TypeError("artifact contents must be bytes")
    return hashlib.sha256(contents).hexdigest()


def write_immutable(path: str | Path, contents: bytes) -> str:
    """Durably publish bytes once, accepting an identical replay only.

    A hard link publishes the fully fsynced sibling temporary without allowing
    a concurrent writer to replace an existing final name.  The temporary is
    private to this call; readers never discover files by scanning for it.
    """
    destination = Path(path)
    if not destination.name:
        raise ValueError("immutable artifact path must name a file")
    if not isinstance(contents, bytes):
        raise TypeError("artifact contents must be bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_digest(contents)

    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            _require_identical(destination, digest)
            # A prior writer can have published the link and then lost power
            # before its directory sync.  An idempotent replay finishes that
            # durability step without ever replacing the final file.
            _fsync_directory(destination.parent)
            return digest
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(destination.parent)
        return digest
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_immutable_json(path: str | Path, value: Any) -> str:
    """Publish canonical UTF-8 JSON as an immutable task artifact."""
    return write_immutable(path, canonical_json_bytes(value))


def _replace_heartbeat_bytes(path: str | Path, contents: bytes) -> str:
    """Atomically replace the one mutable task leaf with fully written bytes."""
    destination = Path(path)
    if destination.name != "heartbeat.json":
        raise ValueError("only heartbeat.json may be replaced")
    if not isinstance(contents, bytes):
        raise TypeError("artifact contents must be bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(destination.parent)
        return sha256_digest(contents)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class Heartbeat:
    """Replaceable liveness evidence, deliberately separate from completion."""

    task_id: str
    attempt_id: str
    update_sequence: int
    observed_at: str
    expires_at: str

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("heartbeat attempt_id must be a nonempty string")
        if (
            not isinstance(self.update_sequence, int)
            or isinstance(self.update_sequence, bool)
            or self.update_sequence < 0
        ):
            raise ValueError("heartbeat update_sequence must be a nonnegative integer")
        if not isinstance(self.observed_at, str) or not self.observed_at:
            raise ValueError("heartbeat observed_at must be a nonempty string")
        if not isinstance(self.expires_at, str) or not self.expires_at:
            raise ValueError("heartbeat expires_at must be a nonempty string")

    def payload(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "expires_at": self.expires_at,
            "observed_at": self.observed_at,
            "task_id": self.task_id,
            "update_sequence": self.update_sequence,
        }

    @classmethod
    def from_payload(cls, value: Any) -> Heartbeat:
        if not isinstance(value, dict) or set(value) != {
            "attempt_id",
            "expires_at",
            "observed_at",
            "task_id",
            "update_sequence",
        }:
            raise ArtifactIntegrityError("heartbeat has an invalid schema")
        try:
            return cls(**value)
        except (TypeError, ValueError) as error:
            raise ArtifactIntegrityError("heartbeat has invalid values") from error


def replace_heartbeat(path: str | Path, heartbeat: Heartbeat) -> str:
    """Replace heartbeat evidence while requiring a strictly newer sequence."""
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        existing = _read_heartbeat(destination)
        if (existing.task_id, existing.attempt_id) != (
            heartbeat.task_id,
            heartbeat.attempt_id,
        ):
            raise ArtifactIntegrityError("heartbeat identity changed")
        if heartbeat.update_sequence <= existing.update_sequence:
            raise ArtifactIntegrityError("heartbeat update_sequence did not advance")
    return _replace_heartbeat_bytes(
        destination, canonical_json_bytes(heartbeat.payload())
    )


# A verb which reads naturally at the worker call site while retaining the
# protocol name used by callers that reason about the replaceable leaf.
write_heartbeat = replace_heartbeat


@dataclass(frozen=True)
class CompletionMember:
    """One regular task file committed before ``complete.json``."""

    filename: str
    digest: str
    byte_length: int

    def __post_init__(self) -> None:
        _validate_filename(self.filename)
        _validate_digest(self.digest)
        if (
            not isinstance(self.byte_length, int)
            or isinstance(self.byte_length, bool)
            or self.byte_length < 0
        ):
            raise ValueError("completion member byte_length must be nonnegative")

    @classmethod
    def for_contents(cls, filename: str, contents: bytes) -> CompletionMember:
        return cls(filename, sha256_digest(contents), len(contents))

    def payload(self) -> dict[str, Any]:
        return {
            "byte_length": self.byte_length,
            "digest": self.digest,
            "filename": self.filename,
        }

    @classmethod
    def from_payload(cls, value: Any) -> CompletionMember:
        if not isinstance(value, dict) or set(value) != {
            "byte_length",
            "digest",
            "filename",
        }:
            raise ArtifactIntegrityError("completion member has an invalid schema")
        try:
            return cls(**value)
        except (TypeError, ValueError) as error:
            raise ArtifactIntegrityError(
                "completion member has invalid values"
            ) from error


@dataclass(frozen=True)
class TaskCompletion:
    """The immutable terminal record readers validate before accepting a task."""

    task_id: str
    attempt_id: str
    descriptor_digest: str
    outcome: TaskOutcome
    terminal: CompletionMember
    stdout: CompletionMember
    stderr: CompletionMember
    trace: CompletionMember | None = None
    schema_version: int = TASK_COMPLETION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("completion attempt_id must be a nonempty string")
        _validate_digest(self.descriptor_digest)
        if self.outcome not in {"completed", "failed"}:
            raise ValueError("completion outcome must be completed or failed")
        if self.schema_version != TASK_COMPLETION_SCHEMA_VERSION:
            raise ValueError("unsupported completion schema version")
        expected_terminal = (
            "result.json" if self.outcome == "completed" else "failure.json"
        )
        if self.terminal.filename != expected_terminal:
            raise ValueError("completion terminal member does not match its outcome")
        if self.stdout.filename != "stdout.log" or self.stderr.filename != "stderr.log":
            raise ValueError("completion stdout and stderr members must be canonical")
        if self.trace is not None and self.trace.filename != "trace.jsonl":
            raise ValueError("completion trace member must be canonical")

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "attempt_id": self.attempt_id,
            "descriptor_digest": self.descriptor_digest,
            "outcome": self.outcome,
            "schema_version": self.schema_version,
            "stderr": self.stderr.payload(),
            "stdout": self.stdout.payload(),
            "task_id": self.task_id,
            "terminal": self.terminal.payload(),
        }
        if self.trace is not None:
            payload["trace"] = self.trace.payload()
        return payload

    @classmethod
    def from_payload(cls, value: Any) -> TaskCompletion:
        required = {
            "attempt_id",
            "descriptor_digest",
            "outcome",
            "schema_version",
            "stderr",
            "stdout",
            "task_id",
            "terminal",
        }
        if not isinstance(value, dict) or set(value) not in (
            required,
            required | {"trace"},
        ):
            raise ArtifactIntegrityError("completion has an invalid schema")
        try:
            trace = (
                CompletionMember.from_payload(value["trace"])
                if "trace" in value
                else None
            )
            return cls(
                task_id=value["task_id"],
                attempt_id=value["attempt_id"],
                descriptor_digest=value["descriptor_digest"],
                outcome=value["outcome"],
                terminal=CompletionMember.from_payload(value["terminal"]),
                stdout=CompletionMember.from_payload(value["stdout"]),
                stderr=CompletionMember.from_payload(value["stderr"]),
                trace=trace,
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as error:
            raise ArtifactIntegrityError("completion has invalid values") from error


def write_completion(task_directory: str | Path, completion: TaskCompletion) -> str:
    """Validate terminal members, then write the task's terminal marker last."""
    directory = Path(task_directory)
    _validate_completion_members(directory, completion)
    return write_immutable_json(directory / "complete.json", completion.payload())


def read_completion(
    task_directory: str | Path,
    identity: TaskIdentity,
    descriptor_digest: str,
) -> TaskCompletion:
    """Read an accepted terminal task only after checking every listed member."""
    _validate_digest(descriptor_digest)
    directory = Path(task_directory)
    complete_path = directory / "complete.json"
    contents = _read_regular_file(complete_path, directory)
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("completion is not UTF-8 JSON") from error
    completion = TaskCompletion.from_payload(payload)
    if (completion.task_id, completion.attempt_id) != (
        identity.task_id,
        identity.attempt_id,
    ):
        raise ArtifactIntegrityError("completion identity does not match its task")
    if completion.descriptor_digest != descriptor_digest:
        raise ArtifactIntegrityError("completion descriptor digest does not match")
    _validate_completion_members(directory, completion)
    return completion


def _require_identical(path: Path, expected_digest: str) -> None:
    try:
        existing = _read_regular_file(path)
    except ArtifactIntegrityError as error:
        raise ArtifactIntegrityError(
            f"immutable destination is not a regular file: {path}"
        ) from error
    if sha256_digest(existing) != expected_digest:
        raise ArtifactIntegrityError(f"immutable artifact already differs: {path}")


def _read_heartbeat(path: Path) -> Heartbeat:
    contents = _read_regular_file(path)
    try:
        return Heartbeat.from_payload(json.loads(contents.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("heartbeat is not UTF-8 JSON") from error


def _validate_completion_members(directory: Path, completion: TaskCompletion) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ArtifactIntegrityError("task directory must be a real directory")
    for member in (
        completion.terminal,
        completion.stdout,
        completion.stderr,
        completion.trace,
    ):
        if member is None:
            continue
        contents = _read_regular_file(directory / member.filename, directory)
        if len(contents) != member.byte_length:
            raise ArtifactIntegrityError(
                f"completion member length differs: {member.filename}"
            )
        if sha256_digest(contents) != member.digest:
            raise ArtifactIntegrityError(
                f"completion member digest differs: {member.filename}"
            )


def _read_regular_file(path: Path, containing_directory: Path | None = None) -> bytes:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactIntegrityError(
            f"required artifact is missing: {path.name}"
        ) from error
    if not stat.S_ISREG(info.st_mode):
        raise ArtifactIntegrityError(f"artifact is not a regular file: {path.name}")
    if containing_directory is not None:
        try:
            directory = containing_directory.resolve(strict=True)
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise ArtifactIntegrityError(
                f"artifact cannot be resolved: {path.name}"
            ) from error
        if resolved.parent != directory:
            raise ArtifactIntegrityError(
                f"artifact escapes task directory: {path.name}"
            )
    return path.read_bytes()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_filename(value: str) -> None:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("completion member filename must be a basename")


def _validate_digest(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("artifact digest must be lowercase SHA-256 hex")
