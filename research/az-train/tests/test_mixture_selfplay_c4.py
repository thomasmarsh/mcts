# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: E501
"""Fast deterministic checks for the diverse-opening self-play mixture fit driver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from az_train.mixture_selfplay_c4 import load_split_replay, read_concat_searched_values
from az_train.records_c4 import Positions, encode_records
from az_train.target_mixture_c4 import mixed_target


def _write_shard(path: Path, plies: list[int], value: float, with_policy: bool) -> int:
    """Write one synthetic game of ``len(plies)`` records; returns the record count."""
    n = len(plies)
    pos = Positions(
        black=np.zeros(n, dtype=np.uint64),
        white=np.zeros(n, dtype=np.uint64),
        side=np.array([p % 2 for p in plies], dtype=np.uint8),
        ply=np.array(plies, dtype=np.uint8),
        value=np.full(n, value, dtype=np.float32),
        policy=[[(3, 1.0)] if with_policy else [] for _ in plies],
    )
    path.write_bytes(encode_records(pos))
    return n


def test_read_concat_searched_values_joins_shards_in_order(tmp_path: Path) -> None:
    a = tmp_path / "a.f32"
    b = tmp_path / "b.f32"
    a.write_bytes(np.array([1.0, 0.0, -1.0], dtype="<f4").tobytes())
    b.write_bytes(np.array([0.5, -0.5], dtype="<f4").tobytes())
    joined = read_concat_searched_values([a, b], 5)
    np.testing.assert_allclose(joined, [1.0, 0.0, -1.0, 0.5, -0.5])
    with pytest.raises(ValueError):
        read_concat_searched_values([a, b], 4)
    bad = tmp_path / "bad.f32"
    bad.write_bytes(np.array([2.0], dtype="<f4").tobytes())
    with pytest.raises(ValueError):
        read_concat_searched_values([bad], 1)


def test_mixed_target_matches_manual_blend() -> None:
    outcome = np.array([1.0, -1.0, 1.0, 0.0])
    searched = np.array([1.0, 0.0, -1.0, 1.0])
    np.testing.assert_allclose(
        mixed_target(outcome, searched, 0.75), 0.25 * outcome + 0.75 * searched
    )


def test_load_split_replay_keeps_whole_games_and_aligns_searched_values(tmp_path: Path) -> None:
    # Generation zero: one 5-ply game. Generation one: one 4-ply game.
    gen0 = tmp_path / "gen0.bin"
    gen1 = tmp_path / "gen1.bin"
    n0 = _write_shard(gen0, [0, 1, 2, 3, 4], value=1.0, with_policy=True)
    n1 = _write_shard(gen1, [0, 1, 2, 3], value=-1.0, with_policy=True)

    # Searched values stay inside [-1, 1]: gen0 rows non-negative, gen1 non-positive.
    sv0 = tmp_path / "gen0.f32"
    sv1 = tmp_path / "gen1.f32"
    sv0.write_bytes((np.arange(n0, dtype="<f4") / np.float32(100.0)).tobytes())
    sv1.write_bytes((-(np.arange(n1, dtype="<f4") / np.float32(100.0))).tobytes())

    train, held_out, counts = load_split_replay(
        [gen0, gen1], [sv0, sv1], validation_fraction=0.5, split_seed=0
    )
    assert counts["games"] == 2
    assert counts["train_games"] == 1 and counts["held_out_games"] == 1
    assert counts["total_records"] == n0 + n1
    # No game straddles the split: one split holds only +1 outcomes, the other only -1.
    for pack in (train, held_out):
        assert len(set(pack["outcome"].tolist())) == 1
    # The searched value for a +1-outcome row is non-negative (from gen0's sv0),
    # and for a -1-outcome row is non-positive (from gen1's sv1): alignment held.
    for pack in (train, held_out):
        if pack["outcome"][0] > 0:
            assert np.all(pack["searched"] >= 0.0)
        else:
            assert np.all(pack["searched"] <= 0.0)
