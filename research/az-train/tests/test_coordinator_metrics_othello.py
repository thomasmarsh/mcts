# ruff: noqa: E501
# pyright: reportIndexIssue=false
"""Deterministic parsing checks for ``coordinator_metrics_othello`` -- the
instrumentation logic, not a real graded run."""

from __future__ import annotations

from az_train.coordinator_metrics_othello import merge_generation_metrics, parse_gate_output

GATE_TEXT = """\
head to head vs baseline: candidate 55-3-42 (W-D-L) over 100, score share 0.565, Wilson LB 0.466  -> FAIL
rollout anchor: candidate 62-2-36 (W-D-L) over 100, score share 0.630, Wilson LB 0.533  -> PASS
edax yardstick (level 3) games=100  W-D-L 40-1-59  win_rate=0.405  ci=[0.311, 0.505]
GATE: FAIL
"""

NO_EDAX_TEXT = """\
head to head vs baseline: candidate 51-0-49 (W-D-L) over 100, score share 0.510, Wilson LB 0.412  -> FAIL
rollout anchor: candidate 51-0-49 (W-D-L) over 100, score share 0.510, Wilson LB 0.412  -> FAIL
edax yardstick: skipped (--edax-binary/--edax-data-dir not given)
GATE: FAIL
"""


def test_parses_both_checks_and_the_edax_line() -> None:
    parsed = parse_gate_output(GATE_TEXT)
    assert parsed["h2h"]["verdict"] == "FAIL"
    assert parsed["h2h"]["wilson_lower_bound"] == 0.466
    assert parsed["rollout_anchor"]["verdict"] == "PASS"
    assert parsed["edax_yardstick"]["level"] == 3
    assert parsed["edax_yardstick"]["win_rate"] == 0.405


def test_tolerates_a_skipped_edax_line() -> None:
    parsed = parse_gate_output(NO_EDAX_TEXT)
    assert "edax_yardstick" not in parsed
    assert parsed["h2h"]["games"] == 100


def test_merge_omits_gate_vs_prev_when_absent() -> None:
    line = merge_generation_metrics(
        generation=0,
        wall_seconds=12.5,
        gate_vs_gen0_text=GATE_TEXT,
        value_meta={"metrics": {"train_bce": 0.6}},
        policy_meta={"metrics": {"train": {"cross_entropy": 1.5}}},
    )
    assert line["generation"] == 0
    assert line["gate_vs_prev"] is None
    assert line["value_metrics"]["train_bce"] == 0.6
    assert line["gate_vs_gen0"]["h2h"]["games"] == 100
