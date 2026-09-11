# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: E501
"""Fast checks for per-row policy-loss weighting and its recency schedule."""

from __future__ import annotations

import numpy as np
import pytest

from az_train.convnet_c4 import (
    N_WEIGHTS,
    _head_gradient_components,
    _literal_loss_gradient,
    _literal_loss_gradient_terms,
    initial_weights,
)
from az_train.policy_recency_c4 import recency_policy_weights
from az_train.replay_composition_c4 import shard_of_row


def test_recency_weights_decay_geometrically_and_average_to_one() -> None:
    row_shard = shard_of_row([10, 10, 10, 10, 10])
    weights = recency_policy_weights(row_shard, 4, 0.5)
    per_shard = [float(weights[row_shard == h][0]) for h in range(5)]
    ratios = [per_shard[h + 1] / per_shard[h] for h in range(4)]
    assert ratios == pytest.approx([2.0] * 4)
    assert float(weights.mean()) == pytest.approx(1.0)
    assert np.all(np.diff(per_shard) > 0.0)


def test_gamma_one_is_uniform() -> None:
    row_shard = shard_of_row([3, 7, 5])
    assert recency_policy_weights(row_shard, 2, 1.0) == pytest.approx(np.ones(15))


def test_recency_weights_reject_out_of_range_gamma() -> None:
    row_shard = shard_of_row([4, 4])
    for gamma in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            recency_policy_weights(row_shard, 1, gamma)


def _tiny_batch(rows: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(11)
    me = (rng.random((rows, 42)) < 0.2).astype(np.float32)
    opp = ((rng.random((rows, 42)) < 0.2) & (me == 0.0)).astype(np.float32)
    value = rng.uniform(-1.0, 1.0, rows).astype(np.float32)
    policy = rng.random((rows, 7)).astype(np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    legal = np.ones((rows, 7), dtype=np.float32)
    return me, opp, value, policy, legal


def test_uniform_row_weights_reproduce_the_unweighted_gradient_exactly() -> None:
    """An all-ones weight must be bit-identical, not merely close.

    Long Adam fits amplify tiny numerical differences, so the unweighted control
    arm has to be reproducible through the weighted code path.
    """
    me, opp, value, policy, legal = _tiny_batch(6)
    w = initial_weights(3)
    plain_loss, plain_gradient = _literal_loss_gradient(w, me, opp, value, policy, legal, 1e-4)
    weighted_loss, weighted_gradient = _literal_loss_gradient(
        w, me, opp, value, policy, legal, 1e-4, policy_row_weight=np.ones(6)
    )
    assert weighted_loss == plain_loss
    assert np.array_equal(weighted_gradient, plain_gradient)


def _policy_gradient(w: np.ndarray, batch: tuple[np.ndarray, ...], row_weight: np.ndarray | None) -> np.ndarray:
    me, opp, value, policy, legal = batch
    _, gradient = _literal_loss_gradient_terms(
        w, me, opp, value, policy, legal, 1e-4,
        value_weight=0.0, policy_weight=1.0, include_regularization=False,
        policy_row_weight=row_weight,
    )
    return gradient


def test_row_weights_leave_the_value_gradient_untouched() -> None:
    me, opp, value, policy, legal = _tiny_batch(4)
    w = initial_weights(5)
    row_weight = np.array([0.25, 1.0, 0.0, 2.75])
    plain_value, plain_policy, _ = _head_gradient_components(w, me, opp, value, policy, legal, 1e-4)
    weighted_value, weighted_policy, _ = _head_gradient_components(
        w, me, opp, value, policy, legal, 1e-4, policy_row_weight=row_weight
    )
    assert weighted_value == pytest.approx(plain_value, abs=1e-7)
    assert not np.allclose(weighted_policy, plain_policy)


def test_zero_row_weight_drops_that_row_from_the_policy_gradient() -> None:
    """A zeroed row contributes nothing, leaving the remaining rows' mean rescaled."""
    batch = _tiny_batch(4)
    w = initial_weights(5)
    kept = np.array([0, 1, 3])
    subset = tuple(array[kept] for array in batch)
    zeroed = _policy_gradient(w, batch, np.array([1.0, 1.0, 0.0, 1.0]))
    # The weighted gradient still divides by all four rows, the subset by three.
    assert zeroed == pytest.approx(0.75 * _policy_gradient(w, subset, None), abs=1e-7)


def test_row_weights_scale_the_policy_gradient_linearly() -> None:
    batch = _tiny_batch(5)
    w = initial_weights(7)
    at_one = _policy_gradient(w, batch, np.ones(5))
    at_three = _policy_gradient(w, batch, np.full(5, 3.0))
    assert at_one == pytest.approx(_policy_gradient(w, batch, None), abs=1e-7)
    assert at_three == pytest.approx(3.0 * at_one, abs=1e-7)


def test_row_weights_reject_bad_shapes_and_values() -> None:
    me, opp, value, policy, legal = _tiny_batch(4)
    w = initial_weights(9)
    assert w.size == N_WEIGHTS
    for bad in (np.ones(3), np.array([1.0, -1.0, 1.0, 1.0]), np.array([1.0, np.nan, 1.0, 1.0])):
        with pytest.raises(ValueError):
            _literal_loss_gradient(w, me, opp, value, policy, legal, 1e-4, policy_row_weight=bad)
