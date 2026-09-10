# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: E501
"""Fast checks for the replay-composition scheme construction (no CNN fit)."""

from __future__ import annotations

import numpy as np
import pytest

from az_train.replay_composition_c4 import (
    build_training_indices,
    scheme_shards,
    shard_of_row,
)


def test_scheme_shards_select_expected_generations() -> None:
    assert scheme_shards("full", 4) == [(i, 1.0) for i in range(5)]
    assert scheme_shards("window2", 4) == [(3, 1.0), (4, 1.0)]
    assert scheme_shards("window3", 4) == [(2, 1.0), (3, 1.0), (4, 1.0)]
    assert scheme_shards("window2_plus_gen0", 4) == [(0, 1.0), (3, 1.0), (4, 1.0)]

    # At g=1 every window collapses onto the same {0, 1} pair as `full`.
    for scheme in ("full", "window2", "window3", "window2_plus_gen0"):
        assert scheme_shards(scheme, 1) == [(0, 1.0), (1, 1.0)]

    # window2_plus_gen0 dedupes gen0 when it is already in the last two.
    assert scheme_shards("window2_plus_gen0", 2) == [(0, 1.0), (1, 1.0), (2, 1.0)]


def test_recency_weighted_decays_with_age() -> None:
    spec = scheme_shards("recency_weighted", 4, gamma=0.5)
    assert [s for s, _ in spec] == [0, 1, 2, 3, 4]
    assert [w for _, w in spec] == [0.0625, 0.125, 0.25, 0.5, 1.0]


def test_scheme_shards_rejects_generation_zero() -> None:
    with pytest.raises(ValueError):
        scheme_shards("full", 0)
    with pytest.raises(ValueError):
        scheme_shards("nonsense", 3)


def test_shard_of_row_maps_records_to_generations() -> None:
    got = shard_of_row([3, 2, 4])
    assert got.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]


def test_build_training_indices_keeps_only_retained_shards() -> None:
    # 4 shards of 10 records each; canonical train split is every row.
    row_shard = shard_of_row([10, 10, 10, 10])
    canonical = np.arange(40)
    rng = np.random.default_rng(0)

    spec = scheme_shards("window2", 3)  # keeps shards {2, 3}
    idx = build_training_indices(spec, canonical, row_shard, budget=0, rng=rng)
    assert set(row_shard[idx].tolist()) == {2, 3}
    assert np.array_equal(idx, np.arange(20, 40))  # uniform + no budget -> pool unchanged


def test_recency_resampling_biases_toward_recent_shards() -> None:
    row_shard = shard_of_row([100, 100, 100, 100])
    canonical = np.arange(400)
    spec = scheme_shards("recency_weighted", 3, gamma=0.5)  # weights 1:2:4:8
    idx = build_training_indices(
        spec, canonical, row_shard, budget=8000, rng=np.random.default_rng(1)
    )
    assert idx.size == 8000
    counts = np.bincount(row_shard[idx], minlength=4)
    # newest shard should dominate and the order should be monotone in age.
    assert counts[3] > counts[2] > counts[1] > counts[0]
    assert counts[3] > 3 * counts[0]
