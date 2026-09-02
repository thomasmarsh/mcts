"""Row-builder unit coverage: each builder maps typed replay/report input to
deterministic rows keyed by the existing domain identities."""

from __future__ import annotations

from pathlib import Path

import pytest

from tuner_cli.artifacts import Manifest, read_manifest
from tuner_cli.codec import JsonObject
from tuner_cli.evidence import read_events
from tuner_cli.replay import ReplayState, replay
from tuner_cli.report import build_report
from tuner_projection import rows

FIXTURES = Path(__file__).parent / "fixtures"
COMPLETE = FIXTURES / "version4"
ACTIVE = FIXTURES / "version4-active-halving"


def _load(run_dir: Path) -> tuple[Manifest, ReplayState, JsonObject]:
    manifest = read_manifest(run_dir / "manifest.json")
    state = replay(manifest, read_events(run_dir / "evidence.jsonl"))
    return manifest, state, build_report(run_dir)


@pytest.fixture(scope="module")
def complete() -> tuple[Manifest, ReplayState, JsonObject]:
    return _load(COMPLETE)


@pytest.fixture(scope="module")
def active() -> tuple[Manifest, ReplayState, JsonObject]:
    return _load(ACTIVE)


def test_run_manifest_row_extracts_scalars(
    complete: tuple[Manifest, ReplayState, JsonObject],
) -> None:
    manifest, _state, _report = complete
    row = rows.run_manifest_row("r", manifest)
    assert row.run_id == "r"
    assert row.game_kind == manifest.spec.kind
    assert row.cohort_size == manifest.cohort_size
    assert row.shadow_policy_kind == "paired_bootstrap"
    assert row.active_elimination == 0
    assert row.manifest_json.startswith("{")


def test_candidate_rows_are_accepted_proposals_deduped(
    complete: tuple[Manifest, ReplayState, JsonObject],
) -> None:
    _manifest, state, _report = complete
    result = rows.candidate_rows("r", state)
    accepted = {index for index, disp in state.dispositions if disp == "accepted"}
    assert {row.candidate_id for row in result} == {
        state.proposals[index].candidate.candidate_id for index in accepted
    }
    assert len({row.candidate_id for row in result}) == len(result)


def test_pair_and_game_rows_agree_on_counts(
    complete: tuple[Manifest, ReplayState, JsonObject],
) -> None:
    _manifest, state, _report = complete
    pair_result = rows.pair_rows("r", state)
    game_result = rows.game_rows("r", state)
    assert len(pair_result) == len(state.completed_pairs)
    assert len(game_result) == 2 * len(state.completed_pairs)
    assert {row.pair_id for row in game_result} <= {row.pair_id for row in pair_result}


def test_shadow_decision_rows_key_on_race_index(
    complete: tuple[Manifest, ReplayState, JsonObject],
) -> None:
    _manifest, state, _report = complete
    result = rows.shadow_decision_rows("r", state)
    keys = [(row.run_id, row.race_index, row.candidate_id) for row in result]
    assert len(keys) == len(set(keys))
    assert len(result) == sum(len(race.decisions) for race in state.shadow_races)


def test_active_elimination_rows_tag_margin_kind(
    active: tuple[Manifest, ReplayState, JsonObject],
) -> None:
    _manifest, state, _report = active
    result = rows.active_elimination_decision_rows("r", state)
    assert result, "the active-halving fixture records rank eliminations"
    assert all(row.margin_kind == "SuccessiveHalvingRankMargin" for row in result)


def test_validation_rows_follow_report_order(
    complete: tuple[Manifest, ReplayState, JsonObject],
) -> None:
    _manifest, _state, report = complete
    result = rows.validation_rows("r", report)
    assert [row.rank for row in result] == list(range(len(result)))
    assert rows.validation_rows("r", None) == []


def test_compute_phase_rows_cover_three_phases(
    complete: tuple[Manifest, ReplayState, JsonObject],
) -> None:
    _manifest, state, _report = complete
    result = rows.compute_phase_rows("r", state)
    assert [row.phase for row in result] == ["tuning", "validation", "diagnostic"]
    assert result[0].pair_attempts == state.compute.tuning.pair_attempts


def test_row_builders_are_deterministic(complete: tuple[Manifest, ReplayState, JsonObject]) -> None:
    _manifest, state, report = complete
    assert rows.proposal_rows("r", state) == rows.proposal_rows("r", state)
    assert rows.validation_rows("r", report) == rows.validation_rows("r", report)
