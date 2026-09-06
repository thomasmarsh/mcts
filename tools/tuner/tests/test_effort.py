from __future__ import annotations

from pathlib import Path

import pytest

from tuner_cli.domain import SearchEffort
from tuner_cli.effort import decode_effort, encode_effort, exceeds_same_kind
from tuner_cli.run import RunOptions, validate_options


@pytest.mark.parametrize("effort", [SearchEffort("iterations", 3), SearchEffort("time_ms", 3)])
def test_effort_codec_round_trips_both_modes(effort: SearchEffort) -> None:
    assert decode_effort(encode_effort(effort)) == effort


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"kind": "iterations"},
        {"value": 1},
        {"kind": "other", "value": 1},
        {"kind": "iterations", "value": True},
        {"kind": "iterations", "value": 0},
        {"kind": "iterations", "value": -1},
        {"kind": "iterations", "value": 1.0},
        {"kind": "iterations", "value": 1, "extra": 1},
    ],
)
def test_effort_codec_rejects_malformed_values(value: object) -> None:
    with pytest.raises(ValueError):
        decode_effort(value)


def test_effort_kind_is_part_of_identity_and_ordering_is_same_kind_only() -> None:
    iterations = SearchEffort("iterations", 100)
    time = SearchEffort("time_ms", 100)
    assert iterations != time
    assert exceeds_same_kind(SearchEffort("iterations", 101), iterations)
    assert not exceeds_same_kind(SearchEffort("time_ms", 101), iterations)


def test_run_options_reject_unresolved_effort_before_setup() -> None:
    options = RunOptions(Path("game"), Path("run"), tuning_effort=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resolved SearchEffort"):
        validate_options(options)
