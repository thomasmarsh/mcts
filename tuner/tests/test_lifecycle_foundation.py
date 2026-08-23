from __future__ import annotations

import json
from pathlib import Path

import pytest
from tuner_cli.config import SearchConfig

from tuner_cli.lifecycle import (
    AttemptId,
    LifecycleWriter,
    SessionId,
    TrialId,
    strict_json_dumps,
    trial_id_for,
)
from tuner_cli.manifest import (
    build_session_manifest,
    manifest_fingerprint,
    write_manifest_atomic,
)


def test_strict_v1_serialization_is_portable_and_rejects_unknown_event_type(
    tmp_path: Path,
):
    assert json.loads(strict_json_dumps({"z": float("nan"), "a": float("inf")})) == {
        "a": "infinity",
        "z": "nan",
    }
    writer = LifecycleWriter(tmp_path / "events.jsonl", SessionId("s"), AttemptId("a"))
    try:
        with pytest.raises(ValueError, match="unsupported lifecycle event type"):
            writer.emit("unknown", {})
    finally:
        writer.close()


def test_writer_accepts_trial_reported_events(tmp_path: Path):
    with LifecycleWriter(tmp_path / "events.jsonl", SessionId("s"), AttemptId("a")) as writer:
        record = writer.emit(
            "trial_reported",
            {
                "trial_id": TrialId("trial-1"),
                "trial_number": 0,
                "completed_pairs": 1,
                "mu": 25.0,
                "sigma": 1.5,
                "score": 20.5,
                "score_formula_version": 1,
                "conservative_k": 3.0,
                "outcome": "continue",
                "reason": "below_min_pairs",
                "pruning_exempt": False,
                "bracket_id": None,
                "rung_resource": None,
            },
        )
    assert record["event_type"] == "trial_reported"


def test_fingerprint_is_deterministic_independent_of_mapping_order():
    left = {"game": {"kind": "nim", "config": {"size": 5}}, "seed": 7}
    right = {"seed": 7, "game": {"config": {"size": 5}, "kind": "nim"}}
    assert manifest_fingerprint(left) == manifest_fingerprint(right)


def test_manifest_records_resolved_policy_and_fingerprints_semantic_changes():
    cfg = SearchConfig.defaults()
    kwargs = {
        "game_kind": "traffic-lights",
        "binary": Path("/games/traffic-lights"),
        "git_sha": "abc",
        "study_name": "study",
        "storage": "sqlite:///study.db",
    }
    first = build_session_manifest(cfg, **kwargs)

    policy = first["semantic_inputs"]["optimizer"]
    assert policy["resource"] == {"min_pairs": 5, "max_pairs": 15}
    assert policy["sampler"]["kind"] == "tpe"
    assert policy["sampler"]["startup_trials"] == 10
    assert policy["pruning"]["enabled"] is False
    assert first["semantic_inputs"]["rating"]["sigma_stop"] == 2.0
    assert first["semantic_inputs"]["rating"]["conservative_k"] == 3.0

    cfg.optimizer.n_trials = 2000
    assert build_session_manifest(cfg, **kwargs)["fingerprint"] == first["fingerprint"]

    cfg.optimizer.sampler.startup_trials = 11
    assert build_session_manifest(cfg, **kwargs)["fingerprint"] != first["fingerprint"]


def test_manifest_is_atomic_and_immutable(tmp_path: Path):
    path = tmp_path / "session-manifest.json"
    first = {"schema_version": 1, "fingerprint": "abc", "semantic_inputs": {"x": 1}}
    write_manifest_atomic(path, first)
    assert json.loads(path.read_text()) == first
    write_manifest_atomic(path, first)
    with pytest.raises(ValueError, match="different fingerprint"):
        write_manifest_atomic(path, {**first, "fingerprint": "changed"})


def test_sequence_and_terminal_state_continue_across_reopen(tmp_path: Path):
    path = tmp_path / "lifecycle.jsonl"
    session, attempt, trial = (
        SessionId("session-1"),
        AttemptId("attempt-1"),
        trial_id_for(SessionId("session-1"), 1),
    )
    with LifecycleWriter(path, session, attempt) as writer:
        first = writer.emit(
            "session_started", {"manifest": {}, "manifest_fingerprint": "f"}
        )
        writer.emit("attempt_started", {})
        writer.emit(
            "trial_created", {"trial_id": trial, "trial_number": 1, "config": {}}
        )
        writer.emit("trial_started", {"trial_id": trial, "trial_number": 1})
        terminal = writer.emit_trial_terminal("trial_failed", trial, {"error": "boom"})
    with LifecycleWriter(path, session, attempt) as reopened:
        assert reopened.has_session_started
        assert reopened._sequence == terminal["session_sequence"]
        assert reopened.has_trial_terminal(trial)
        with pytest.raises(ValueError, match="already has terminal"):
            reopened.emit_trial_terminal("trial_completed", trial, {})
        next_record = reopened.emit("attempt_failed", {"error": "boom"})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["session_sequence"] for record in records] == list(
        range(1, len(records) + 1)
    )
    assert first["schema_version"] == 1
    assert next_record["session_sequence"] == len(records)


def test_trial_ids_are_typed_and_stable():
    assert trial_id_for(SessionId("s"), 2) == trial_id_for(SessionId("s"), 2)
    assert trial_id_for(SessionId("s"), 2) != trial_id_for(SessionId("s"), 3)
    assert isinstance(trial_id_for(SessionId("s"), 2), str)


def test_legacy_trial_and_incumbent_records_remain_compatible(capsys):
    from tuner_cli.callback import emit_incumbent_record, emit_trial_record

    emit_trial_record(3, {"c": 1.5}, 7, 25.0, 2.0, [{"outcome": "win"}], "sha")
    emit_incumbent_record({"c": 1.5}, 25.0, 2.0)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["type"] == "trial"
    assert lines[0]["trial_id"] == 3
    assert lines[0]["cost"] == pytest.approx(-(25.0 - 3 * 2.0))
    assert lines[0]["extra"]["git_sha"] == "sha"
    assert lines[1]["type"] == "incumbent"
    assert lines[1]["config"] == {"c": 1.5}
