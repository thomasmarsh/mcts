# pyright: reportPrivateUsage=false, reportIndexIssue=false, reportArgumentType=false
# ruff: noqa: E501
"""Fast deterministic checks for the graded CNN coordinator's metrics-line merge."""

from __future__ import annotations

import math

import pytest

from az_train.coordinator_metrics_c4 import (
    merge_generation_metrics,
    parse_gate_line,
    wilson_lower_bound,
)

_GATE = "CNN candidate vs opponent, equal budget: 152-3-45 (W-D-L), score share 0.767, games=200, sims=32"


def test_parse_gate_line_extracts_record_and_derives_share_and_bound() -> None:
    parsed = parse_gate_line("noise\n" + _GATE + "\nmore noise")
    assert parsed["wins"] == 152
    assert parsed["draws"] == 3
    assert parsed["losses"] == 45
    assert parsed["games"] == 200
    assert parsed["sims"] == 32
    # 152 + 0.5*3 = 153.5 successes over 200.
    assert parsed["score_share"] == pytest.approx(0.7675)
    assert 0.0 < parsed["wilson_lower_bound"] < parsed["score_share"]


def test_parse_gate_line_rejects_text_without_a_result() -> None:
    with pytest.raises(ValueError):
        parse_gate_line("no gate here")


def test_wilson_lower_bound_matches_closed_form() -> None:
    lb = wilson_lower_bound(60.0, 100.0)
    z = 1.96
    phat, n = 0.6, 100.0
    expected = (phat + z * z / (2 * n) - z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    assert lb == pytest.approx(expected)
    assert wilson_lower_bound(0.0, 0) == 0.0


def test_merge_generation_metrics_produces_a_flat_reconstructable_line() -> None:
    result = {
        "held_out_proven": {"principled_early_stop": {"value_pearson": 0.6, "sign_agreement": 0.87}},
        "in_replay_mixed_target_pearson": {"principled_early_stop": {"held_out": 0.548, "train": 0.634}},
        "fit_metrics": {
            "train_value_mse": 0.31,
            "validation_value_mse": 0.40,
            "train_policy_cross_entropy": 1.70,
            "validation_policy_cross_entropy": 1.80,
        },
        "replay_split": {"games": 200, "train_games": 160, "held_out_games": 40},
        "reference_proven_counts": {"validation": 956, "validation_wins": 764, "validation_losses": 189},
        "searched_value_summary": {"train_proven_fraction": 0.21},
        "principled_early_stop_epoch": 15,
        "fit_wall_seconds": 120.0,
        "peak_rss_bytes": 1_000_000,
        "weights": {"principled_early_stop": "gen1.c4cnn"},
    }
    line = merge_generation_metrics(
        result, _GATE, _GATE.replace("0.767", "0.512").replace("152-3-45", "101-4-95"),
        generation=1, wall_seconds=999.4,
    )
    assert line["generation"] == 1
    assert line["value_pearson_reference_corpus"] == 0.6
    assert line["value_pearson_selfplay_heldout"] == 0.548
    assert line["fit_metrics"]["validation_policy_cross_entropy"] == 1.80
    assert line["gate_vs_zero"]["score_share"] == pytest.approx(0.7675)
    assert line["gate_vs_gen0"]["wins"] == 101
    assert line["coordinator_wall_seconds"] == 999.4
    assert line["gate_vs_prev"] is None


def test_merge_generation_metrics_includes_gate_vs_prev_when_supplied() -> None:
    result = {
        "held_out_proven": {"principled_early_stop": {"value_pearson": 0.6, "sign_agreement": 0.87}},
        "in_replay_mixed_target_pearson": {"principled_early_stop": {"held_out": 0.5, "train": 0.6}},
        "replay_split": {}, "reference_proven_counts": {}, "searched_value_summary": {},
    }
    line = merge_generation_metrics(
        result, _GATE, _GATE, generation=2, wall_seconds=1.0,
        gate_vs_prev_text=_GATE.replace("0.767", "0.381").replace("152-3-45", "74-4-122"),
    )
    assert line["gate_vs_prev"]["wins"] == 74
    assert line["gate_vs_prev"]["score_share"] == pytest.approx(0.38)
