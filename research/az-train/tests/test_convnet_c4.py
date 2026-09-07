# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# ruff: noqa: E501
from pathlib import Path

import numpy as np
import pytest

from az_train.convnet_c4 import (
    N_WEIGHTS,
    _literal_loss_gradient,
    fit_value_policy_with_diagnostics,
    predict,
    read_weights,
    write_weights,
)


def test_layout_round_trip_and_reference_prediction(tmp_path: Path) -> None:
    weights = (np.arange(N_WEIGHTS, dtype=np.float32) - N_WEIGHTS / 2) * 1e-6
    path = tmp_path / "fixture.c4cnn"
    write_weights(str(path), weights)
    assert np.array_equal(read_weights(str(path)), weights)
    me = np.zeros((1, 42), dtype=np.float32)
    opp = np.zeros_like(me)
    me[0, [0, 2, 8]] = 1
    opp[0, [1, 7]] = 1
    value, logits = predict(weights, me, opp)
    assert value.shape == (1,)
    assert logits.shape == (1, 7)
    assert np.isfinite(value).all() and np.isfinite(logits).all()
    assert abs(float(value[0]) - 0.0063871006) < 1e-7
    assert np.allclose(
        logits[0],
        [0.0069883447, 0.0069883438, 0.0069883447, 0.0069883442, 0.0069883447, 0.0069883438, 0.0069883447],
        atol=1e-8,
    )


def test_mirror_equivariance_and_legal_masking() -> None:
    weights = np.zeros(N_WEIGHTS, dtype=np.float32)
    me = np.zeros((1, 42), dtype=np.float32)
    opp = np.zeros_like(me)
    me[0, [0, 8, 15]] = 1
    opp[0, [1, 7, 14]] = 1
    value, logits = predict(weights, me, opp)
    mirrored_value, mirrored_logits = predict(weights, me.reshape(1, 6, 7)[:, :, ::-1].reshape(1, 42), opp.reshape(1, 6, 7)[:, :, ::-1].reshape(1, 42))
    assert np.array_equal(value, mirrored_value)
    assert np.array_equal(logits, mirrored_logits[:, ::-1])
    legal = np.array([[True, False, True, False, True, True, False]])
    masked = np.where(legal, logits, -np.inf)
    assert np.isneginf(masked[0, ~legal[0]]).all()
    assert np.isfinite(masked[0, legal[0]]).all()


def test_invalid_layout_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.c4cnn"
    path.write_bytes(b"C4CNN001" + b"\0" * 8)
    with pytest.raises(ValueError, match="C4CNN001"):
        read_weights(str(path))


def test_joint_value_policy_gradient_is_finite_and_deterministic() -> None:
    rng = np.random.default_rng(9)
    me = np.zeros((2, 42), dtype=np.float32)
    opp = np.zeros_like(me)
    me[0, 0], opp[1, 1] = 1.0, 1.0
    value = np.array([1.0, -1.0], dtype=np.float32)
    target = np.full((2, 7), 1.0 / 7.0, dtype=np.float32)
    legal = np.ones((2, 7), dtype=bool)
    weights = (rng.standard_normal(N_WEIGHTS) * 0.03).astype(np.float32)
    loss, gradient = _literal_loss_gradient(weights, me, opp, value, target, legal, 1e-4)
    assert np.isfinite(loss) and np.all(np.isfinite(gradient))
    first, first_meta = fit_value_policy_with_diagnostics(
        me, opp, value, target, legal, (me, opp, value, target, legal), epochs=2, batch_size=2, seed=3
    )
    second, second_meta = fit_value_policy_with_diagnostics(
        me, opp, value, target, legal, (me, opp, value, target, legal), epochs=2, batch_size=2, seed=3
    )
    assert np.array_equal(first, second)
    assert first_meta["optimizer_steps"] == second_meta["optimizer_steps"] == 2
