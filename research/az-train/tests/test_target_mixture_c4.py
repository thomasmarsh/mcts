# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Fast deterministic checks for the searched-value / outcome target mixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from az_train.records_c4 import Positions, encode_records
from az_train.target_mixture_c4 import (
    _replay_rows_with_searched,
    mixed_target,
    read_searched_values,
)


def test_mixed_target_endpoints_and_midpoint() -> None:
    outcome = np.array([1.0, -1.0, 1.0, 0.0])
    searched = np.array([0.0, -1.0, 1.0, -1.0])
    assert np.array_equal(mixed_target(outcome, searched, 0.0), outcome)
    assert np.array_equal(mixed_target(outcome, searched, 1.0), searched)
    np.testing.assert_allclose(
        mixed_target(outcome, searched, 0.5), 0.5 * (outcome + searched)
    )
    np.testing.assert_allclose(
        mixed_target(outcome, searched, 0.25), 0.75 * outcome + 0.25 * searched
    )


def test_mixed_target_rejects_bad_shapes_and_weights() -> None:
    with pytest.raises(ValueError):
        mixed_target(np.zeros((2, 2)), np.zeros((2, 2)), 0.5)
    with pytest.raises(ValueError):
        mixed_target(np.zeros(3), np.zeros(4), 0.5)
    with pytest.raises(ValueError):
        mixed_target(np.zeros(3), np.zeros(3), 1.5)


def test_read_searched_values_round_trip_and_guards(tmp_path: Path) -> None:
    path = tmp_path / "searched.f32"
    values = np.array([1.0, 0.0, -1.0, 0.5], dtype="<f4")
    path.write_bytes(values.tobytes())
    np.testing.assert_allclose(read_searched_values(path, 4), values.astype(np.float64))
    with pytest.raises(ValueError):
        read_searched_values(path, 3)
    bad = tmp_path / "bad.f32"
    bad.write_bytes(np.array([2.0], dtype="<f4").tobytes())
    with pytest.raises(ValueError):
        read_searched_values(bad, 1)


def _tiny_shard(tmp_path: Path) -> Path:
    """One three-ply game: the middle record carries no completed-Q policy."""
    pos = Positions(
        black=np.array([0, 1, 1], dtype=np.uint64),
        white=np.array([0, 0, 2], dtype=np.uint64),
        side=np.array([0, 1, 0], dtype=np.uint8),
        ply=np.array([0, 1, 2], dtype=np.uint8),
        value=np.array([1.0, -1.0, 1.0], dtype=np.float32),
        policy=[[(3, 1.0)], [], [(2, 0.5), (4, 0.5)]],
    )
    path = tmp_path / "gen0.bin"
    path.write_bytes(encode_records(pos))
    return path


def test_replay_rows_align_searched_values_with_policy_rows(tmp_path: Path) -> None:
    shard = _tiny_shard(tmp_path)
    searched = tmp_path / "searched.f32"
    searched.write_bytes(np.array([0.25, -0.5, 0.75], dtype="<f4").tobytes())

    rows = _replay_rows_with_searched([shard], searched)

    # Records 0 and 2 have a policy tail; record 1 is dropped.
    assert rows["outcome"].tolist() == [1.0, 1.0]
    np.testing.assert_allclose(rows["searched"], [0.25, 0.75])
    assert rows["policy"].shape == (2, 7)
    assert rows["legal"].sum() == 3


def test_replay_rows_reject_a_length_mismatch(tmp_path: Path) -> None:
    shard = _tiny_shard(tmp_path)
    searched = tmp_path / "searched.f32"
    searched.write_bytes(np.array([0.0, 0.0], dtype="<f4").tobytes())
    with pytest.raises(ValueError):
        _replay_rows_with_searched([shard], searched)
