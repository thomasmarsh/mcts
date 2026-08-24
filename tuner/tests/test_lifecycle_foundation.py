from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tuner_cli.config import SearchConfig
import tuner_cli.coordinator as coordinator

from tuner_cli.lifecycle import (
    AttemptId,
    LifecycleWriter,
    SessionId,
    TrialId,
    strict_json_dumps,
    trial_id_for,
    replay_journal,
)
from tuner_cli.pool import Anchor, OpponentPool
from tuner_cli.attempt import save_inserted_pool_anchor
from tuner_cli.manifest import (
    SessionForkRequired,
    build_session_manifest,
    manifest_fingerprint,
    write_manifest_atomic,
)
from tuner_cli.coordinator import _emit_session_started
from tuner_cli.coordinator import _recover_orphaned_attempt


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
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("s"), AttemptId("a")
    ) as writer:
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


def test_pool_revisions_follow_attempt_start_and_duplicate_loaded_snapshot(
    tmp_path: Path,
):
    path = tmp_path / "events.jsonl"
    pool = OpponentPool(
        [
            Anchor(
                "default",
                {"family": "rave"},
                25.0,
                0.5,
                "bootstrap_default",
                "bootstrap",
            )
        ]
    )
    with LifecycleWriter(path, SessionId("s"), AttemptId("a1")) as writer:
        writer.emit("session_started", {"manifest": {}, "manifest_fingerprint": "f"})
        writer.emit("attempt_started", {})
        first = writer.emit("pool_revised", pool.revision_payload())
    with LifecycleWriter(path, SessionId("s"), AttemptId("a2")) as writer:
        writer.emit("attempt_started", {})
        second = writer.emit("pool_revised", pool.revision_payload())

    assert first["payload"] == second["payload"]
    assert first["payload"]["anchors"] == [
        {
            "anchor_id": "default",
            "config": {"family": "rave"},
            "mu": 25.0,
            "sigma": 0.5,
            "provenance": "bootstrap_default",
            "insertion_reason": "bootstrap",
            "source_trial_id": None,
        }
    ]


def test_pool_revision_cannot_precede_session_start(tmp_path: Path):
    pool = OpponentPool([Anchor("old", {}, 1.0, 1.0)])
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("s"), AttemptId("a")
    ) as writer:
        with pytest.raises(ValueError, match="requires session_started"):
            writer.emit("pool_revised", pool.revision_payload())


def test_inserted_anchor_emits_a_revision_after_durable_pool_save(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    pool_path = tmp_path / "pool.json"
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    active_trial = SimpleNamespace(
        config={"family": "rave"}, trial_id=TrialId("trial-9")
    )
    with LifecycleWriter(path, SessionId("s"), AttemptId("a")) as writer:
        writer.emit("session_started", {"manifest": {}, "manifest_fingerprint": "f"})
        writer.emit("attempt_started", {})
        save_inserted_pool_anchor(pool, pool_path, writer, active_trial, 30.0, 2.0)
        assert pool_path.exists()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    revision = records[-1]
    assert revision["event_type"] == "pool_revised"
    assert revision["payload"]["anchors"][-1] == {
        "anchor_id": "trial-1",
        "config": {"family": "rave"},
        "mu": 30.0,
        "sigma": 2.0,
        "provenance": "trial",
        "insertion_reason": "champion",
        "source_trial_id": "trial-9",
    }


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
    with pytest.raises(SessionForkRequired, match="fork required"):
        write_manifest_atomic(path, {**first, "fingerprint": "changed"})


def test_manifest_conflict_stops_before_opening_or_recovering_a_study(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = SearchConfig.defaults()
    monkeypatch.setattr(coordinator, "_resolve_search_space", lambda *_: None)
    monkeypatch.setattr(
        coordinator,
        "_write_session_manifest",
        lambda *_: (_ for _ in ()).throw(SessionForkRequired("fork required")),
    )
    monkeypatch.setattr(
        coordinator,
        "_open_study",
        lambda *_: pytest.fail("manifest conflict opened the study"),
    )
    with pytest.raises(SessionForkRequired, match="fork required"):
        coordinator.run_optimization(cfg, optimizer_id="session-conflict")


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


def test_attempt_writers_share_one_journal_and_preserve_a_partial_tail(tmp_path: Path):
    path = tmp_path / "lifecycle.jsonl"
    session = SessionId("session-1")
    manifest = {"fingerprint": "manifest-1"}
    manifest_path = tmp_path / "session-manifest.json"
    manifest_path.write_text("{}")
    with LifecycleWriter(path, session, AttemptId("attempt-1")) as first:
        _emit_session_started(first, manifest, manifest_path, "optimizer-1", 1)
        first.emit("attempt_started", {"optimizer_id": "optimizer-1"})
        with pytest.raises(RuntimeError, match="already locked"):
            LifecycleWriter(path, session, AttemptId("attempt-2"))

    with path.open("ab") as artifact:
        artifact.write(b'{"partial":')
    with LifecycleWriter(path, session, AttemptId("attempt-2")) as second:
        assert second.has_session_started
        second.emit("attempt_started", {"optimizer_id": "optimizer-1"})
        with pytest.raises(SessionForkRequired, match="fork required"):
            _emit_session_started(
                second,
                {"fingerprint": "manifest-2"},
                manifest_path,
                "optimizer-1",
                1,
            )

    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines if line != '{"partial":']
    assert [record["event_type"] for record in records] == [
        "session_started",
        "attempt_started",
        "attempt_started",
    ]
    assert [record["attempt_id"] for record in records[1:]] == [
        "attempt-1",
        "attempt-2",
    ]
    assert [record["session_sequence"] for record in records] == [1, 2, 3]


def test_trial_ids_are_typed_and_stable():
    assert trial_id_for(SessionId("s"), 2) == trial_id_for(SessionId("s"), 2)
    assert trial_id_for(SessionId("s"), 2) != trial_id_for(SessionId("s"), 3)
    assert isinstance(trial_id_for(SessionId("s"), 2), str)


def _orphaned_trial_journal(path: Path, session: SessionId, trial: TrialId) -> None:
    with LifecycleWriter(path, session, AttemptId("prior")) as writer:
        writer.emit("session_started", {"manifest": {}, "manifest_fingerprint": "f"})
        writer.emit("attempt_started", {"bench_run_id": "physical-prior"})
        writer.emit(
            "trial_created", {"trial_id": trial, "trial_number": 0, "config": {}}
        )
        writer.emit("trial_started", {"trial_id": trial, "trial_number": 0})
        writer.emit("pair_started", {"trial_id": trial, "pair_id": "pair-0"})


@pytest.mark.parametrize(
    ("optuna_state", "reason"),
    [
        ("running", "abrupt_attempt_recovery"),
        ("complete", "recovery_evidence_gap"),
        ("pruned", "recovery_evidence_gap"),
    ],
)
def test_recovery_emits_exact_scope_and_consumes_an_orphaned_optuna_slot(
    tmp_path: Path, optuna_state: str, reason: str
):
    import optuna
    from optuna.trial import TrialState

    session = SessionId("session")
    trial_id = trial_id_for(session, 0)
    path = tmp_path / "lifecycle.jsonl"
    _orphaned_trial_journal(path, session, trial_id)
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    if optuna_state == "complete":
        study.tell(trial, 1.0)
    elif optuna_state == "pruned":
        study.tell(trial, state=TrialState.PRUNED)

    with LifecycleWriter(path, session, AttemptId("current")) as writer:
        writer.emit("attempt_started", {"bench_run_id": "physical-current"})
        _recover_orphaned_attempt(writer, study)
        writer.emit("pool_revised", {"after": "recovery"})
        writer.emit("attempt_completed", {})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event_type"] for record in records][-4:-1] == [
        "attempt_started",
        "attempt_recovered",
        "pool_revised",
    ]
    recovered = records[-3]["payload"]
    assert recovered["prior_attempt_id"] == "prior"
    assert recovered["prior_bench_run_id"] == "physical-prior"
    assert recovered["pair_ids"] == ["pair-0"]
    assert recovered["trials"] == [
        {"trial_id": trial_id, "trial_number": 0, "reason": reason}
    ]
    assert len(study.trials) == 1
    assert study.trials[0].state == (
        TrialState.FAIL
        if optuna_state == "running"
        else TrialState[optuna_state.upper()]
    )

    with LifecycleWriter(path, session, AttemptId("later")) as writer:
        assert writer.journal_snapshot.orphaned_attempt is None


def test_replay_refuses_conflicting_identity_and_multiple_unterminated_attempts(
    tmp_path: Path,
):
    session = SessionId("session")
    trial_id = trial_id_for(session, 0)
    conflict_path = tmp_path / "conflict.jsonl"
    _orphaned_trial_journal(conflict_path, session, trial_id)
    records = [json.loads(line) for line in conflict_path.read_text().splitlines()]
    records[2]["payload"]["trial_id"] = "trial-not-deterministic"
    conflict_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    with pytest.raises(ValueError, match="deterministic"):
        replay_journal(conflict_path, session)

    attempts_path = tmp_path / "attempts.jsonl"
    with LifecycleWriter(attempts_path, session, AttemptId("first")) as writer:
        writer.emit("session_started", {"manifest": {}, "manifest_fingerprint": "f"})
        writer.emit("attempt_started", {})
    with LifecycleWriter(attempts_path, session, AttemptId("second")) as writer:
        writer.emit("attempt_started", {})
    with pytest.raises(ValueError, match="multiple unterminated"):
        replay_journal(attempts_path, session)


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
