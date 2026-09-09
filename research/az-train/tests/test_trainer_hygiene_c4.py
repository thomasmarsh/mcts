# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: E501
"""Fast deterministic checks for the Connect Four trainer-hygiene sweep."""

from __future__ import annotations

import numpy as np
import pytest

from az_train.trainer_hygiene_c4 import (
    oracle_epoch,
    replay_game_split_indices,
)


def test_split_assigns_whole_games_without_straddling() -> None:
    # Three games of lengths 4, 3, 5 laid out contiguously.
    bounds = [(0, 4), (4, 7), (7, 12)]
    train, held_out, train_games, held_out_games = replay_game_split_indices(bounds, 0.34, seed=1)
    assert train_games + held_out_games == 3
    assert held_out_games == 1
    assert set(train.tolist()).isdisjoint(held_out.tolist())
    assert sorted(train.tolist() + held_out.tolist()) == list(range(12))
    for start, stop in bounds:
        block = set(range(start, stop))
        assert block <= set(train.tolist()) or block <= set(held_out.tolist())


def test_split_is_seed_reproducible_and_seed_sensitive() -> None:
    bounds = [(i * 5, i * 5 + 5) for i in range(10)]
    a = replay_game_split_indices(bounds, 0.2, seed=7)
    b = replay_game_split_indices(bounds, 0.2, seed=7)
    c = replay_game_split_indices(bounds, 0.2, seed=8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_split_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        replay_game_split_indices([(0, 4)], 0.2, seed=0)
    with pytest.raises(ValueError):
        replay_game_split_indices([(0, 4), (4, 8)], 0.0, seed=0)


def test_oracle_epoch_picks_the_argmax_and_breaks_ties_early() -> None:
    trace = [
        {"value_pearson": 0.1},
        {"value_pearson": 0.4},
        {"value_pearson": 0.4},
        {"value_pearson": 0.2},
    ]
    assert oracle_epoch(trace) == 2
    with pytest.raises(ValueError):
        oracle_epoch([])
