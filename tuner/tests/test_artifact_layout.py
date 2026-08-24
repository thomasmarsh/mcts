"""Pure contract tests for partitioned tuning evaluation artifacts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from tuner_cli.artifact_layout import (
    ARTIFACT_LAYOUT_SCHEMA_VERSION,
    ARTIFACT_OWNERSHIP,
    TASK_SEQUENCE_MAX,
    ArtifactLayout,
    TaskDescriptor,
    TaskIdentity,
    descriptor_filename,
    game_sequences_for,
    parse_descriptor_filename,
    task_id_for,
)
from tuner_cli.evaluation import PairId
from tuner_cli.lifecycle import AttemptId
from tuner_cli.manifest import build_session_manifest
from tuner_cli.config import SearchConfig


def _identity(sequence: int = 7) -> TaskIdentity:
    return TaskIdentity.for_pair(
        AttemptId("tuning-attempt-run-7"),
        sequence,
        PairId("pair-0123456789abcdef0123456789abcdef"),
    )


def test_task_identity_is_deterministic_and_a_retry_gets_a_distinct_identity():
    first = _identity()
    assert first.task_id == task_id_for(
        first.attempt_id, first.task_sequence, first.pair_id
    )
    retry = TaskIdentity.for_pair(first.attempt_id, 8, first.pair_id)
    assert retry.task_id != first.task_id


def test_descriptor_names_are_canonical_and_sort_in_task_sequence_order():
    names = [
        descriptor_filename(sequence, _identity(sequence).task_id)
        for sequence in (19, 2, 11)
    ]
    assert sorted(names) == [
        descriptor_filename(sequence, _identity(sequence).task_id)
        for sequence in (2, 11, 19)
    ]
    assert parse_descriptor_filename(names[0]) == (19, _identity(19).task_id)


def test_pair_game_sequences_are_checked_and_do_not_collide():
    first = game_sequences_for(1)
    last = game_sequences_for(TASK_SEQUENCE_MAX)
    assert (first.candidate_first, first.candidate_second) == (1, 2)
    assert last.candidate_second == 2 * TASK_SEQUENCE_MAX
    assert (
        len(
            {
                first.candidate_first,
                first.candidate_second,
                last.candidate_first,
                last.candidate_second,
            }
        )
        == 4
    )


def test_paths_are_exact_relative_to_the_artifact_root_and_relocatable(tmp_path: Path):
    identity = _identity()
    descriptor = TaskDescriptor.for_identity(identity)
    layout = ArtifactLayout.for_artifact_root(
        tmp_path / "bench-runs" / "run-a" / "tuning-artifacts"
    )
    moved = ArtifactLayout.for_artifact_root(
        tmp_path / "other-bench-runs" / "run-a" / "tuning-artifacts"
    )

    assert layout.attempt == tmp_path / "bench-runs/run-a/tuning-artifacts/attempt.json"
    assert layout.descriptor(identity).relative_to(layout.root) == Path(
        "descriptors/0000000000000000007-" + str(identity.task_id) + ".json"
    )
    assert (
        descriptor.members.trace
        == PurePosixPath("tasks") / identity.task_id / "trace.jsonl"
    )
    assert layout.task_member_path(descriptor.members.complete).relative_to(
        layout.root
    ) == Path("tasks", identity.task_id, "complete.json")
    assert moved.task_member_path(descriptor.members.complete).relative_to(
        moved.root
    ) == layout.task_member_path(descriptor.members.complete).relative_to(layout.root)


@pytest.mark.parametrize(
    "attempt_id, sequence, pair_id",
    [
        ("", 1, "pair-0123456789abcdef0123456789abcdef"),
        ("../attempt", 1, "pair-0123456789abcdef0123456789abcdef"),
        ("attempt-" + "x" * 121, 1, "pair-0123456789abcdef0123456789abcdef"),
        ("attempt-ok", 0, "pair-0123456789abcdef0123456789abcdef"),
        ("attempt-ok", TASK_SEQUENCE_MAX + 1, "pair-0123456789abcdef0123456789abcdef"),
        ("attempt-ok", 1, "pair/escape"),
    ],
)
def test_identity_rejects_empty_traversal_and_oversize_values(
    attempt_id, sequence, pair_id
):
    with pytest.raises(ValueError):
        TaskIdentity.for_pair(AttemptId(attempt_id), sequence, PairId(pair_id))


@pytest.mark.parametrize(
    "name",
    [
        "0000000000000000000-task-0123456789abcdef0123456789abcdef.json",
        "7-task-0123456789abcdef0123456789abcdef.json",
        "0000000000000000007-task-0123456789abcdef0123456789abcdef.json/extra",
        "0000000000000000007-task-0123456789ABCDEF0123456789abcdef.json",
    ],
)
def test_descriptor_parser_rejects_noncanonical_names(name):
    with pytest.raises(ValueError):
        parse_descriptor_filename(name)


def test_owner_declarations_assign_one_writer_and_keep_the_ingestor_read_only():
    owners = {
        declaration.relative_pattern: declaration.owner
        for declaration in ARTIFACT_OWNERSHIP
    }
    assert owners["attempt.json"] == "coordinator"
    assert (
        owners["descriptors/<19-digit-task-sequence>-<task-id>.json"] == "coordinator"
    )
    assert owners["tasks/<task-id>/"] == "evaluation"
    assert owners["tasks/<task-id>/trace.jsonl"] == "game_child"
    assert all(declaration.owner != "ingestor" for declaration in ARTIFACT_OWNERSHIP)
    assert all(declaration.ingestor_reads for declaration in ARTIFACT_OWNERSHIP)


def test_manifest_advertises_layout_schema_without_physical_task_details(
    tmp_path: Path,
):
    manifest = build_session_manifest(
        SearchConfig.defaults(),
        game_kind="nim",
        binary=Path("game-nim"),
        git_sha="abc",
        study_name="study",
        storage="sqlite:///study.db",
    )
    semantic = manifest["semantic_inputs"]
    assert (
        semantic["schema_versions"]["artifact_layout"] == ARTIFACT_LAYOUT_SCHEMA_VERSION
    )
    assert "task_id" not in str(semantic)
    assert str(tmp_path) not in str(semantic)
