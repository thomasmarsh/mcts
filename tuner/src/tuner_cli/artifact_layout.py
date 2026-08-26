"""Typed, relocatable paths for version-1 tuning evaluation artifacts.

This module describes the on-disk contract only.  It intentionally performs no
I/O: the coordinator, evaluation worker, game child, and ingestor keep their
separate runtime responsibilities.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, NewType
from uuid import UUID, uuid5

from .evaluation import PairId
from .lifecycle import AttemptId

ARTIFACT_LAYOUT_SCHEMA_VERSION: Final = 1
ARTIFACT_DIRECTORY_NAME: Final = "tuning-artifacts"
TASK_SEQUENCE_MAX: Final = (2**63 - 1) // 2
TASK_SEQUENCE_WIDTH: Final = 19

# A schema-owned namespace makes the v1 name independent of URL or deployment
# namespaces.  Changing this UUID requires a new artifact-layout schema.
TASK_ID_NAMESPACE: Final = UUID("ee182139-6adb-5b25-baad-8063aa139ce1")

TaskId = NewType("TaskId", str)
ArtifactOwner = Literal["coordinator", "evaluation", "worker", "game_child"]

_SAFE_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_TASK_ID = re.compile(r"^task-[0-9a-f]{32}$")
_DESCRIPTOR_NAME = re.compile(r"^(?P<sequence>[0-9]{19})-(?P<task>task-[0-9a-f]{32})\.json$")


def validate_task_sequence(task_sequence: int) -> int:
    """Return a positive, trace-sequence-safe task sequence.

    The upper bound leaves room for both reserved game sequences in a signed
    64-bit consumer without wrapping.
    """
    if (
        not isinstance(task_sequence, int)
        or isinstance(task_sequence, bool)
        or not 1 <= task_sequence <= TASK_SEQUENCE_MAX
    ):
        raise ValueError(f"task_sequence must be an integer from 1 through {TASK_SEQUENCE_MAX}")
    return task_sequence


def task_id_for(attempt_id: AttemptId, task_sequence: int, pair_id: PairId) -> TaskId:
    """Derive an opaque v1 task identity from its one submitted execution."""
    attempt = _validate_opaque_id(attempt_id, "attempt_id")
    sequence = validate_task_sequence(task_sequence)
    pair = _validate_opaque_id(pair_id, "pair_id")
    name = json.dumps([attempt, sequence, pair], separators=(",", ":"), ensure_ascii=True)
    return TaskId(f"task-{uuid5(TASK_ID_NAMESPACE, name).hex}")


def validate_task_id(task_id: TaskId | str) -> TaskId:
    """Reject anything that is not a generated task identity, including paths."""
    if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
        raise ValueError("task_id must be a canonical task-<32 lowercase hex> identifier")
    return TaskId(task_id)


def descriptor_filename(task_sequence: int, task_id: TaskId | str) -> str:
    """Return the canonically sortable descriptor filename for one task."""
    sequence = validate_task_sequence(task_sequence)
    task = validate_task_id(task_id)
    return f"{sequence:0{TASK_SEQUENCE_WIDTH}d}-{task}.json"


def parse_descriptor_filename(name: str) -> tuple[int, TaskId]:
    """Validate and decode one descriptor basename without normalizing it."""
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("descriptor filename must be a basename")
    match = _DESCRIPTOR_NAME.fullmatch(name)
    if match is None:
        raise ValueError("descriptor filename is not canonical")
    sequence = validate_task_sequence(int(match["sequence"]))
    task_id = validate_task_id(match["task"])
    if descriptor_filename(sequence, task_id) != name:
        raise ValueError("descriptor filename is not canonical")
    return sequence, task_id


@dataclass(frozen=True)
class PairGameSequences:
    """The two trace sequences reserved for one candidate seat-swapped pair."""

    candidate_first: int
    candidate_second: int


def game_sequences_for(task_sequence: int) -> PairGameSequences:
    """Reserve checked, noncolliding trace sequences for a task's two games."""
    sequence = validate_task_sequence(task_sequence)
    return PairGameSequences(2 * sequence - 1, 2 * sequence)


@dataclass(frozen=True)
class TaskIdentity:
    """Typed identity for one execution attempt of one logical evaluation pair."""

    attempt_id: AttemptId
    task_sequence: int
    pair_id: PairId
    task_id: TaskId

    @classmethod
    def for_pair(cls, attempt_id: AttemptId, task_sequence: int, pair_id: PairId) -> TaskIdentity:
        """Build a task identity whose opaque ID is tied to all causal IDs."""
        return cls(
            AttemptId(_validate_opaque_id(attempt_id, "attempt_id")),
            validate_task_sequence(task_sequence),
            PairId(_validate_opaque_id(pair_id, "pair_id")),
            task_id_for(attempt_id, task_sequence, pair_id),
        )

    def __post_init__(self) -> None:
        expected = task_id_for(self.attempt_id, self.task_sequence, self.pair_id)
        if validate_task_id(self.task_id) != expected:
            raise ValueError("task_id does not match attempt_id, task_sequence, and pair_id")


@dataclass(frozen=True)
class TaskMemberPaths:
    """Relative paths stored in a task descriptor; never an artifact root."""

    heartbeat: PurePosixPath
    stdout: PurePosixPath
    stderr: PurePosixPath
    trace: PurePosixPath
    result: PurePosixPath
    failure: PurePosixPath
    complete: PurePosixPath

    @classmethod
    def for_task(cls, task_id: TaskId | str) -> TaskMemberPaths:
        """Return every v1 leaf below the task's relative directory."""
        task = validate_task_id(task_id)
        directory = PurePosixPath("tasks") / task
        return cls(
            directory / "heartbeat.json",
            directory / "stdout.log",
            directory / "stderr.log",
            directory / "trace.jsonl",
            directory / "result.json",
            directory / "failure.json",
            directory / "complete.json",
        )

    def __post_init__(self) -> None:
        for member in self.__dict__.values():
            _validate_relative_member_path(member)


@dataclass(frozen=True)
class TaskDescriptor:
    """Versioned, path-relocatable descriptor schema for one task execution."""

    schema_version: int
    identity: TaskIdentity
    members: TaskMemberPaths

    @classmethod
    def for_identity(cls, identity: TaskIdentity) -> TaskDescriptor:
        """Build the complete v1 descriptor without committing it to disk."""
        return cls(
            ARTIFACT_LAYOUT_SCHEMA_VERSION,
            identity,
            TaskMemberPaths.for_task(identity.task_id),
        )

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_LAYOUT_SCHEMA_VERSION:
            raise ValueError("unsupported artifact layout schema version")
        if self.members != TaskMemberPaths.for_task(self.identity.task_id):
            raise ValueError("task descriptor members do not match its task_id")


@dataclass(frozen=True)
class ArtifactOwnership:
    """One leaf or directory's exclusive writer and any read-only ingestor."""

    relative_pattern: str
    owner: ArtifactOwner
    ingestor_reads: bool = True


ARTIFACT_OWNERSHIP: Final = (
    ArtifactOwnership("attempt.json", "coordinator"),
    ArtifactOwnership("descriptors/<19-digit-task-sequence>-<task-id>.json", "coordinator"),
    ArtifactOwnership("tasks/<task-id>/", "evaluation"),
    ArtifactOwnership("tasks/<task-id>/heartbeat.json", "worker"),
    ArtifactOwnership("tasks/<task-id>/stdout.log", "worker"),
    ArtifactOwnership("tasks/<task-id>/stderr.log", "worker"),
    ArtifactOwnership("tasks/<task-id>/trace.jsonl", "game_child"),
    ArtifactOwnership("tasks/<task-id>/result.json", "worker"),
    ArtifactOwnership("tasks/<task-id>/failure.json", "worker"),
    ArtifactOwnership("tasks/<task-id>/complete.json", "worker"),
)


@dataclass(frozen=True)
class ArtifactLayout:
    """Absolute access paths derived from one validated artifact root."""

    root: Path

    @classmethod
    def for_artifact_root(cls, artifact_root: str | Path) -> ArtifactLayout:
        """Use the complete server-chosen artifact root without rewriting it."""
        return cls(_validate_root(artifact_root))

    def __post_init__(self) -> None:
        root = _validate_root(self.root)
        if root.name != ARTIFACT_DIRECTORY_NAME:
            raise ValueError(f"artifact root must end in {ARTIFACT_DIRECTORY_NAME!r}")
        object.__setattr__(self, "root", root)

    @property
    def attempt(self) -> Path:
        return self.root / "attempt.json"

    def descriptor(self, identity: TaskIdentity) -> Path:
        return (
            self.root
            / "descriptors"
            / descriptor_filename(identity.task_sequence, identity.task_id)
        )

    def task_members(self, task_id: TaskId | str) -> TaskMemberPaths:
        """Return descriptor-relative member paths for one validated task ID."""
        return TaskMemberPaths.for_task(task_id)

    def task_member_path(self, member: PurePosixPath) -> Path:
        """Resolve one validated descriptor member under this layout's root."""
        relative = _validate_relative_member_path(member)
        return self.root.joinpath(*relative.parts)


def _validate_opaque_id(value: str, field: str) -> str:
    if not isinstance(value, str) or _SAFE_OPAQUE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a nonempty safe ASCII opaque identifier")
    return value


def _validate_root(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("artifact root must be a path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact root must be an absolute path without traversal")
    return path


def _validate_relative_member_path(value: PurePosixPath) -> PurePosixPath:
    if (
        not isinstance(value, PurePosixPath)
        or value.is_absolute()
        or any(part in ("", ".", "..") for part in value.parts)
    ):
        raise ValueError("artifact member path must be a nonempty relative path")
    return value
