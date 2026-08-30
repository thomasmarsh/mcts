"""Freeze the version-4 scientific artifacts against checked-in golden bytes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from golden_support import (
    FIXTURES,
    GoldenTarget,
    golden_options,
    normalize_operational,
    write_binary,
    write_objective,
)

from tuner_cli.artifacts import read_manifest
from tuner_cli.event_payloads import EventType, ProposalRejectedPayload
from tuner_cli.evidence import read_events, scientific_projection
from tuner_cli.replay import replay
from tuner_cli.report import build_report
from tuner_cli.run import run_foreground

_ALL_EVENT_TYPES: set[EventType] = {
    "proposal_created",
    "proposal_accepted",
    "proposal_rejected",
    "cohort_completed",
    "pair_started",
    "pair_completed",
    "observation_completed",
    "finalists_selected",
    "run_completed",
}


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("golden")
    run_dir = tmp / "run"
    run_foreground(golden_options(write_binary(tmp), run_dir, write_objective(tmp)), GoldenTarget())
    return run_dir


def _masked(path: Path, manifest: dict[str, object]) -> str:
    return normalize_operational(path.read_text(encoding="utf-8"), manifest)


def _golden(name: str) -> str:
    fixture_manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    return normalize_operational((FIXTURES / name).read_text(encoding="utf-8"), fixture_manifest)


def test_manifest_matches_golden_after_masking_operational_fields(regenerated: Path) -> None:
    live = json.loads((regenerated / "manifest.json").read_text(encoding="utf-8"))
    assert _masked(regenerated / "manifest.json", live) == _golden("manifest.json")
    # The masked-away fields must still be present and well-formed.
    assert re.fullmatch(r"[0-9a-f]{64}", live["fingerprint"])
    assert Path(live["binary"]["path"]).name == "game-druid"
    assert Path(live["objective"]["source_path"]).name == "objective.json"


def test_evidence_matches_golden_after_masking_operational_fields(regenerated: Path) -> None:
    live = json.loads((regenerated / "manifest.json").read_text(encoding="utf-8"))
    assert _masked(regenerated / "evidence.jsonl", live) == _golden("evidence.jsonl")


def test_report_matches_golden_after_masking_operational_fields(regenerated: Path) -> None:
    live = json.loads((regenerated / "manifest.json").read_text(encoding="utf-8"))
    assert _masked(regenerated / "report.json", live) == _golden("report.json")


def test_golden_evidence_exercises_every_scientific_event() -> None:
    events = read_events(FIXTURES / "evidence.jsonl")
    assert _ALL_EVENT_TYPES <= {event.type for event in events}
    rejected = [event.payload for event in events if event.type == "proposal_rejected"]
    assert rejected and any(
        isinstance(payload, ProposalRejectedPayload) and payload.reason == "semantic_validation"
        for payload in rejected
    )


def test_golden_manifest_round_trips_through_public_codec() -> None:
    manifest = read_manifest(FIXTURES / "manifest.json")
    assert manifest.fingerprint is not None


def test_golden_scientific_projection_matches_fixture() -> None:
    projection = scientific_projection(read_events(FIXTURES / "evidence.jsonl"))
    assert projection + "\n" == (FIXTURES / "scientific_projection.json").read_text(
        encoding="utf-8"
    )


def test_complete_golden_stream_replays_to_a_terminal_state() -> None:
    manifest = read_manifest(FIXTURES / "manifest.json")
    state = replay(manifest, read_events(FIXTURES / "evidence.jsonl"))
    assert state.terminal_status == "complete"
    assert state.finalists is not None
    assert state.cohort is not None and len(state.cohort) == manifest.cohort_size


def test_interrupted_golden_prefix_replays_to_a_pending_pair() -> None:
    manifest = read_manifest(FIXTURES / "manifest.json")
    prefix = read_events(FIXTURES / "evidence.interrupted.jsonl")
    state = replay(manifest, prefix)
    assert state.terminal_status == "open"
    assert state.next_pair_id is not None
    assert not state.completed_pairs


def test_report_projection_is_pure_manifest_plus_replay(regenerated: Path) -> None:
    from_run = build_report(regenerated)
    assert from_run == build_report(regenerated)
