"""Filesystem contract tests for durable per-task evaluation artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tuner_cli.artifact_layout import TaskIdentity
from tuner_cli.evaluation import PairId
from tuner_cli.lifecycle import AttemptId
from tuner_cli.task_artifacts import (
    ArtifactIntegrityError,
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


def _identity() -> TaskIdentity:
    return TaskIdentity.for_pair(
        AttemptId("attempt-a"), 1, PairId("pair-0123456789abcdef0123456789abcdef")
    )


def _completion(
    directory: Path, identity: TaskIdentity, *, outcome: str = "completed"
) -> TaskCompletion:
    files = {
        "result.json" if outcome == "completed" else "failure.json": b'{"ok":true}',
        "stdout.log": b"standard output\n",
        "stderr.log": b"standard error\n",
        "trace.jsonl": b'{"trace":1}\n',
    }
    for name, contents in files.items():
        write_immutable(directory / name, contents)
    terminal_name = "result.json" if outcome == "completed" else "failure.json"
    return TaskCompletion(
        task_id=identity.task_id,
        attempt_id=identity.attempt_id,
        descriptor_digest=sha256_digest(b"descriptor"),
        outcome=outcome,  # type: ignore[arg-type]
        terminal=CompletionMember.for_contents(terminal_name, files[terminal_name]),
        stdout=CompletionMember.for_contents("stdout.log", files["stdout.log"]),
        stderr=CompletionMember.for_contents("stderr.log", files["stderr.log"]),
        trace=CompletionMember.for_contents("trace.jsonl", files["trace.jsonl"]),
    )


def test_canonical_json_and_digest_are_stable():
    first = canonical_json_bytes({"z": [1, 2], "a": {"b": True}})
    second = canonical_json_bytes({"a": {"b": True}, "z": [1, 2]})
    assert first == second == b'{"a":{"b":true},"z":[1,2]}'
    assert sha256_digest(first) == sha256_digest(second)


def test_immutable_create_has_no_partial_final_visibility(tmp_path: Path, monkeypatch):
    path = tmp_path / "result.json"
    observed: list[bytes] = []
    actual_link = os.link

    def inspect_then_link(source, destination, *args, **kwargs):
        assert not path.exists()
        observed.append(Path(source).read_bytes())
        return actual_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", inspect_then_link)
    write_immutable(path, b"complete bytes")
    assert observed == [b"complete bytes"]
    assert path.read_bytes() == b"complete bytes"
    assert not list(tmp_path.glob(".result.json.tmp-*"))


def test_immutable_replay_accepts_identical_bytes_and_rejects_conflict(tmp_path: Path):
    path = tmp_path / "stdout.log"
    digest = write_immutable(path, b"first")
    assert write_immutable(path, b"first") == digest
    with pytest.raises(ArtifactIntegrityError, match="already differs"):
        write_immutable(path, b"second")


def test_completion_ignores_partial_and_temporary_files(tmp_path: Path):
    identity = _identity()
    task = tmp_path / "tasks" / identity.task_id
    completion = _completion(task, identity)
    (task / ".result.json.tmp-interrupted").write_bytes(b"incomplete")
    (task / "unrelated.partial").write_bytes(b"incomplete")
    write_completion(task, completion)
    assert read_completion(task, identity, completion.descriptor_digest) == completion


def test_failed_completion_commits_the_failure_member(tmp_path: Path):
    identity = _identity()
    task = tmp_path / "task"
    completion = _completion(task, identity, outcome="failed")
    write_completion(task, completion)
    assert read_completion(task, identity, completion.descriptor_digest) == completion


@pytest.mark.parametrize("member", ["result.json", "stdout.log", "stderr.log", "trace.jsonl"])
def test_completion_rejects_missing_or_swapped_members(tmp_path: Path, member: str):
    identity = _identity()
    task = tmp_path / "task"
    completion = _completion(task, identity)
    if member == "stdout.log":
        (task / member).write_bytes((task / "stderr.log").read_bytes())
    else:
        (task / member).unlink()
    with pytest.raises(ArtifactIntegrityError):
        write_completion(task, completion)


def test_reader_rejects_traversing_and_symlinked_completion_members(tmp_path: Path):
    identity = _identity()
    task = tmp_path / "task"
    completion = _completion(task, identity)
    payload = completion.payload()
    payload["stdout"]["filename"] = "../outside.log"
    (task / "complete.json").write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ArtifactIntegrityError):
        read_completion(task, identity, completion.descriptor_digest)

    payload = completion.payload()
    (task / "complete.json").write_bytes(canonical_json_bytes(payload))
    (task / "stdout.log").unlink()
    (task / "stdout.log").symlink_to(tmp_path / "outside.log")
    with pytest.raises(ArtifactIntegrityError, match="regular"):
        read_completion(task, identity, completion.descriptor_digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", "task-" + "0" * 32),
        ("attempt_id", "other"),
        ("descriptor_digest", "0" * 64),
    ],
)
def test_reader_rejects_wrong_completion_identities_and_digest(
    tmp_path: Path, field: str, value: str
):
    identity = _identity()
    task = tmp_path / "task"
    completion = _completion(task, identity)
    payload = completion.payload()
    payload[field] = value
    (task / "complete.json").write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ArtifactIntegrityError):
        read_completion(task, identity, completion.descriptor_digest)


@pytest.mark.parametrize("field,value", [("digest", "0" * 64), ("byte_length", 0)])
def test_reader_rejects_wrong_member_digest_or_length(tmp_path: Path, field: str, value: str | int):
    identity = _identity()
    task = tmp_path / "task"
    completion = _completion(task, identity)
    payload = completion.payload()
    payload["terminal"][field] = value
    (task / "complete.json").write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ArtifactIntegrityError):
        read_completion(task, identity, completion.descriptor_digest)


def test_heartbeat_replaces_only_with_a_newer_sequence(tmp_path: Path):
    identity = _identity()
    path = tmp_path / "heartbeat.json"
    first = Heartbeat(identity.task_id, identity.attempt_id, 0, "observed-1", "expiry-1")
    second = Heartbeat(identity.task_id, identity.attempt_id, 1, "observed-2", "expiry-2")
    write_heartbeat(path, first)
    write_heartbeat(path, second)
    assert json.loads(path.read_text()) == second.payload()
    with pytest.raises(ArtifactIntegrityError, match="did not advance"):
        write_heartbeat(path, second)


def test_directory_fsync_failure_cleans_the_owned_temporary(tmp_path: Path, monkeypatch):
    path = tmp_path / "result.json"
    actual_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(fd: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        actual_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        write_immutable(path, b"durable")
    assert path.read_bytes() == b"durable"
    assert not list(tmp_path.glob(".result.json.tmp-*"))
