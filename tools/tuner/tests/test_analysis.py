"""The scientific sections shared by the report and the projection must shape
non-empty output from a *partial* replay state -- any completed cohort, not only
the terminal one. The golden ``report.json`` test guards that the bytes are
unchanged; this guards that the sections stand alone.
"""

from __future__ import annotations

from golden_support import FIXTURES

from tuner_cli.analysis import (
    diagnostic_section,
    opponent_response_section,
    shadow_elimination_section,
)
from tuner_cli.artifacts import read_manifest
from tuner_cli.diagnostic_graph import build_diagnostic_graph
from tuner_cli.evidence import read_events
from tuner_cli.observations import comparable_prefix_observations
from tuner_cli.opponent_interactions import build_opponent_response_analysis
from tuner_cli.replay import replay
from tuner_cli.selection import select_top_candidates, select_validation_shortlist
from tuner_cli.shadow_audit import build_shadow_audit


def _cohort_zero_state():
    manifest = read_manifest(FIXTURES / "manifest.json")
    events = read_events(FIXTURES / "evidence.jsonl")
    state = replay(manifest, events)
    assert len(state.completed_cohorts) >= 2, "fixture must have >1 cohort to prove partiality"
    return manifest, events, state, state.completed_cohorts[0]


def test_opponent_response_section_shapes_from_a_completed_cohort() -> None:
    manifest, _events, state, cohort = _cohort_zero_state()
    tuning = comparable_prefix_observations(
        state.observations, cohort.candidates, manifest.tuning_prefix
    )
    analysis = build_opponent_response_analysis(
        manifest.panel, cohort, tuning, tuple(state.completed_pairs)
    )
    section = opponent_response_section(manifest, cohort.cohort_index, tuning[0], analysis)
    assert section["scope"]["cohort_index"] == 0
    assert section["candidates"]
    assert all(entry["opponent_responses"] for entry in section["candidates"])


def test_diagnostic_section_shapes_from_a_completed_cohort() -> None:
    manifest, _events, state, cohort = _cohort_zero_state()
    tuning = comparable_prefix_observations(
        state.observations, cohort.candidates, manifest.tuning_prefix
    )
    order = select_top_candidates(cohort.candidates, tuning, len(cohort.candidates))
    rank = {item.candidate_id: index for index, item in enumerate(order)}
    graph = build_diagnostic_graph(cohort.candidates, state.diagnostic_pairs, rank)
    shortlist = select_top_candidates(cohort.candidates, tuning, manifest.finalists)
    _selected, reserve, displaced = select_validation_shortlist(
        cohort.candidates, tuning, manifest.finalists, graph
    )
    section = diagnostic_section(
        manifest,
        cohort,
        order,
        graph,
        shortlist,
        tuple(order[: manifest.finalists]),
        reserve,
        displaced,
    )
    assert section["scope"]["cohort_index"] == 0
    assert len(section["nodes"]) == len(cohort.candidates)


def test_shadow_elimination_section_shapes_from_a_partial_state() -> None:
    manifest, events, state, _cohort = _cohort_zero_state()
    audit = build_shadow_audit(manifest, state, events)
    section = shadow_elimination_section(manifest, audit)
    assert section["policy"]["kind"] in {"paired_bootstrap", "successive_halving"}
    assert "calibration_bins" in section
    assert isinstance(section["cohorts"], list)
