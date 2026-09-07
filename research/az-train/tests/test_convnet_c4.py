# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# ruff: noqa: E501
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from az_train.convnet_c4 import (
    N_WEIGHTS,
    _gradient_conflict_metrics,
    _head_gradient_components,
    _literal_loss_gradient,
    _unpack,
    fit_value_policy_with_diagnostics,
    orientation_diagnostics,
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


def _non_symmetric_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small legal rows which keep every ReLU on the finite-difference path active."""
    me = np.zeros((3, 42), dtype=np.float32)
    opp = np.zeros_like(me)
    me[0, [0, 7, 1]], opp[0, [2, 9, 16]] = 1.0, 1.0
    me[1, [0, 1, 2]], opp[1, [7, 8, 9]] = 1.0, 1.0
    me[2, [3, 10, 4]], opp[2, [0, 7, 14]] = 1.0, 1.0
    value = np.array([0.7, -0.5, 0.2], dtype=np.float32)
    policy = np.array(
        [[0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0]],
        dtype=np.float32,
    )
    return me, opp, value, policy, policy.astype(bool)


def test_literal_loss_gradient_matches_finite_differences_across_cnn() -> None:
    """Central differences use 1e-3 steps, appropriate for this f32-only objective."""
    me, opp, value, policy, legal = _non_symmetric_batch()
    rng = np.random.default_rng(17)
    weights = (rng.standard_normal(N_WEIGHTS) * 0.02).astype(np.float32)
    parameters = _unpack(weights)
    for tensor in parameters:
        if tensor.ndim == 1:
            tensor[:] += 0.1
    loss, gradient = _literal_loss_gradient(weights, me, opp, value, policy, legal, 2e-4)
    assert np.isfinite(loss)
    offsets = np.cumsum([0, *[tensor.size for tensor in parameters]])
    representatives = {
        "stem": (0, 0),
        "residual_block_1": (2, 19),
        "residual_block_2": (6, 29),
        "value_head": (12, 31),
        "policy_head": (18, 43),
    }
    epsilon = np.float32(1e-3)
    for name, (tensor_index, element_index) in representatives.items():
        index = int(offsets[tensor_index] + element_index)
        plus, minus = weights.copy(), weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric = (
            _literal_loss_gradient(plus, me, opp, value, policy, legal, 2e-4)[0]
            - _literal_loss_gradient(minus, me, opp, value, policy, legal, 2e-4)[0]
        ) / (2.0 * float(epsilon))
        assert np.isclose(gradient[index], numeric, rtol=0.04, atol=2e-4), name


def test_public_fit_substantially_reduces_literal_joint_objective() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    initial_rng = np.random.default_rng(23)
    initial = (initial_rng.standard_normal(N_WEIGHTS) * 0.03).astype(np.float32)
    for tensor in _unpack(initial):
        if tensor.ndim == 1:
            tensor.fill(0.05)
    # The public fitter uses this seed for precisely the same initialization.
    initial_loss, _ = _literal_loss_gradient(initial, me, opp, value, policy, legal, 1e-5)
    fitted, metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=80, learning_rate=5e-3,
    )
    final_loss, _ = _literal_loss_gradient(fitted, me, opp, value, policy, legal, 1e-5)
    assert metadata["optimizer_steps"] == 80
    assert final_loss < initial_loss * 0.6


def test_fit_reports_repeatable_nonzero_shared_trunk_telemetry() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    first, first_metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=2, learning_rate=5e-3,
    )
    second, second_metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=2, learning_rate=5e-3,
    )
    assert np.array_equal(first, second)
    telemetry = first_metadata["parameter_groups"]
    assert telemetry == second_metadata["parameter_groups"]
    assert isinstance(telemetry, dict)
    assert set(telemetry) == {
        "stem", "residual_block_1", "residual_block_2", "value_head", "policy_head",
    }
    for group in telemetry.values():
        assert isinstance(group, dict)
        for measurement in group.values():
            assert isinstance(measurement, float)
            assert np.isfinite(measurement) and measurement >= 0.0
    assert telemetry["stem"]["first_batch_gradient_l2"] > 0.0
    assert telemetry["stem"]["initial_to_final_delta_l2"] > 0.0


def test_head_gradient_components_sum_to_existing_joint_gradient() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    weights = (np.random.default_rng(71).standard_normal(N_WEIGHTS) * 0.03).astype(np.float32)
    _, joint = _literal_loss_gradient(weights, me, opp, value, policy, legal, 1e-4)
    value_gradient, policy_gradient, regularization_gradient = _head_gradient_components(
        weights, me, opp, value, policy, legal, 1e-4,
    )
    assert np.allclose(value_gradient + policy_gradient + regularization_gradient, joint, rtol=0.0, atol=1e-8)


def test_head_gradient_conflict_metrics_are_finite_repeatable_and_neutral_at_zero() -> None:
    value_gradient = np.array([3.0, 4.0], dtype=np.float32)
    policy_gradient = np.array([4.0, -3.0], dtype=np.float32)
    first = _gradient_conflict_metrics(value_gradient, policy_gradient)
    assert first == _gradient_conflict_metrics(value_gradient, policy_gradient)
    assert all(np.isfinite(measurement) for measurement in first.values())
    assert first == {"value_gradient_l2": 5.0, "policy_gradient_l2": 5.0, "cosine_similarity": 0.0}
    assert _gradient_conflict_metrics(np.zeros(2, dtype=np.float32), policy_gradient)["cosine_similarity"] == 0.0


def test_head_gradient_conflict_metrics_distinguish_aligned_and_opposing_vectors() -> None:
    value_gradient = np.array([2.0, -1.0], dtype=np.float32)
    aligned = _gradient_conflict_metrics(value_gradient, 3.0 * value_gradient)
    opposing = _gradient_conflict_metrics(value_gradient, -3.0 * value_gradient)
    assert np.isclose(aligned["cosine_similarity"], 1.0)
    assert np.isclose(opposing["cosine_similarity"], -1.0)


def test_fit_reports_repeatable_shared_head_gradient_conflict() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    _, first = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=2, learning_rate=5e-3,
    )
    _, second = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=2, learning_rate=5e-3,
    )
    telemetry = cast(dict[str, dict[str, float]], first["first_batch_shared_head_gradient_conflict"])
    assert telemetry == cast(dict[str, dict[str, float]], second["first_batch_shared_head_gradient_conflict"])
    assert set(telemetry) == {"stem", "residual_block_1", "residual_block_2"}
    assert all(np.isfinite(measurement) for group in telemetry.values() for measurement in group.values())


def test_orientation_diagnostics_are_finite_repeatable_and_distinguish_averaging() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    weights = (np.random.default_rng(41).standard_normal(N_WEIGHTS) * 0.03).astype(np.float32)
    for tensor in _unpack(weights):
        if tensor.ndim == 1:
            tensor += 0.07
    first = orientation_diagnostics(weights, me, opp, value, policy, legal)
    second = orientation_diagnostics(weights, me, opp, value, policy, legal)
    assert first == second
    assert all(np.isfinite(metric) for group in first.values() for metric in group.values())
    assert first["literal_vs_reflected_remapped"]["value_mae"] > 0.0
    assert first["literal_vs_reflected_remapped"]["policy_logit_mae"] > 0.0
    assert first["literal"]["mse"] != first["mirror_averaged"]["mse"]


def test_orientation_diagnostics_zero_control_is_symmetric_and_finite() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    result = orientation_diagnostics(np.zeros(N_WEIGHTS, dtype=np.float32), me, opp, value, policy, legal)
    assert result["literal_vs_reflected_remapped"] == {
        "value_mae": 0.0, "value_pearson": 0.0, "policy_logit_mae": 0.0,
    }
    assert result["literal"] == result["mirror_averaged"]
    assert all(np.isfinite(metric) for group in result.values() for metric in group.values())
