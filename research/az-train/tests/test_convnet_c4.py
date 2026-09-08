# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# ruff: noqa: E501
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from az_train.convnet_c4 import (
    N_WEIGHTS,
    _activation_health,
    _aggregate_gradient_conflict,
    _epoch_batches,
    _gradient_conflict_metrics,
    _head_gradient_components,
    _literal_loss_gradient,
    _parameter_groups,
    _unpack,
    fit_value_policy_with_diagnostics,
    initial_weights,
    orientation_diagnostics,
    predict,
    read_weights,
    select_validation_epoch,
    write_weights,
)
from az_train.fitability_c4 import _canonical_board_key, select_balanced_unique_rows
from az_train.records_c4 import Positions


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
    trace = cast(list[dict[str, float]], first_meta["validation_epoch_trace"])
    assert trace == cast(list[dict[str, float]], second_meta["validation_epoch_trace"])
    assert len(trace) == 2
    assert all(
        set(epoch) == {
            "value_mse", "value_pearson", "value_sign_agreement", "masked_policy_cross_entropy",
        }
        and all(np.isfinite(measurement) for measurement in epoch.values())
        for epoch in trace
    )


def test_value_loss_weight_default_is_bit_identical_to_explicit_one() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    default, default_metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        seed=19, batch_size=3, epochs=3, learning_rate=5e-3,
    )
    explicit, explicit_metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        seed=19, batch_size=3, epochs=3, learning_rate=5e-3, value_loss_weight=1.0,
    )
    assert np.array_equal(default, explicit)
    assert {
        name: measurement for name, measurement in default_metadata.items()
        if name not in {"fit_wall_seconds", "peak_rss_bytes"}
    } == {
        name: measurement for name, measurement in explicit_metadata.items()
        if name not in {"fit_wall_seconds", "peak_rss_bytes"}
    }


def test_value_loss_weight_scales_gradient_and_matches_finite_difference() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    weights = initial_weights(29)
    value_gradient, _, _ = _head_gradient_components(weights, me, opp, value, policy, legal, 0.0)
    # Use a nonzero direction through both a value-only and shared parameter.
    direction = np.zeros(N_WEIGHTS, dtype=np.float32)
    spans = _parameter_groups()
    direction[spans["stem"][0] : spans["value_head"][1]] = 1.0
    direction /= np.linalg.norm(direction)
    _, one = _literal_loss_gradient(weights, me, opp, value, policy, legal, 0.0)
    _, five = _literal_loss_gradient(
        weights, me, opp, value, policy, legal, 0.0, value_loss_weight=5.0,
    )
    assert np.allclose(five - one, 4.0 * value_gradient, rtol=1e-5, atol=1e-7)
    epsilon = np.float32(1e-3)
    numeric = (
        _literal_loss_gradient(
            weights + epsilon * direction, me, opp, value, policy, legal, 0.0,
            value_loss_weight=5.0,
        )[0]
        - _literal_loss_gradient(
            weights - epsilon * direction, me, opp, value, policy, legal, 0.0,
            value_loss_weight=5.0,
        )[0]
    ) / (2.0 * float(epsilon))
    assert np.isclose(float(np.dot(five, direction)), numeric, rtol=0.06, atol=3e-4)


def test_epoch_batches_visit_each_row_once_and_are_repeatable() -> None:
    first = _epoch_batches(np.random.default_rng(41), 11, 4)
    second = _epoch_batches(np.random.default_rng(41), 11, 4)
    assert len(first) == 3
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert np.array_equal(np.sort(np.concatenate(first)), np.arange(11))
    assert len(np.unique(np.concatenate(first))) == 11


def test_epoch_batches_preserve_single_batch_permutation() -> None:
    expected_rng = np.random.default_rng(59)
    expected = expected_rng.permutation(3)
    actual = _epoch_batches(np.random.default_rng(59), 3, 3)
    assert len(actual) == 1
    assert np.array_equal(actual[0], expected)


def test_initializer_assigns_constants_only_to_declared_bias_roles() -> None:
    seed = 23
    initial = initial_weights(seed)
    repeated = initial_weights(seed)
    raw = (np.random.default_rng(seed).standard_normal(N_WEIGHTS) * 0.03).astype(np.float32)
    parameters = _unpack(initial)
    raw_parameters = _unpack(raw)
    bias_indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}

    assert np.array_equal(initial, repeated)
    for index, (parameter, raw_parameter) in enumerate(zip(parameters, raw_parameters, strict=True)):
        if index in bias_indices:
            assert np.array_equal(parameter, np.full(parameter.shape, 0.05, dtype=np.float32))
        else:
            assert np.array_equal(parameter, raw_parameter)
    assert parameters[14].shape == (32,)
    assert np.array_equal(parameters[14], raw_parameters[14])
    assert not np.all(parameters[14] == 0.05)


def test_fitability_selection_is_seeded_balanced_and_mirror_unique() -> None:
    # Each adjacent pair is a reflected board and only one may be selected.
    pos = Positions(
        black=np.array([1, 1 << 6, 2, 1 << 5, 4, 1 << 4, 8, 8], dtype=np.uint64),
        white=np.zeros(8, dtype=np.uint64),
        side=np.zeros(8, dtype=np.uint8),
        ply=np.zeros(8, dtype=np.uint8),
        value=np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float32),
        policy=[[(0, 1.0)]] * 8,
    )
    first = select_balanced_unique_rows(pos, 2, 71)
    second = select_balanced_unique_rows(pos, 2, 71)
    assert np.array_equal(first, second)
    assert np.count_nonzero(pos.value[first] == -1.0) == 2
    assert np.count_nonzero(pos.value[first] == 1.0) == 2
    keys = {_canonical_board_key(int(pos.black[row]), int(pos.white[row]), int(pos.side[row])) for row in first}
    assert len(keys) == len(first)


def test_validation_checkpoint_selection_uses_earliest_highest_pearson() -> None:
    assert select_validation_epoch([
        {"value_pearson": -0.2}, {"value_pearson": 0.4}, {"value_pearson": 0.4},
    ]) == 1


def test_opt_in_validation_checkpoint_is_reproducible_and_preserves_final_weights(tmp_path: Path) -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    first_path, second_path = tmp_path / "first.c4cnn", tmp_path / "second.c4cnn"
    default_weights, _ = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=3, learning_rate=5e-3,
    )
    first, first_metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=3, learning_rate=5e-3,
        selected_validation_checkpoint_out=str(first_path),
    )
    second, second_metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=3, learning_rate=5e-3,
        selected_validation_checkpoint_out=str(second_path),
    )
    assert np.array_equal(default_weights, first)
    assert np.array_equal(first, second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_metadata["selected_validation_epoch"] == second_metadata["selected_validation_epoch"]


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


def _residual_gate_fixture(
    first_output_bias: float, second_output_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make each residual output gate independently active or inactive."""
    weights = (np.random.default_rng(123).standard_normal(N_WEIGHTS) * 0.03).astype(np.float32)
    parameters = _unpack(weights)
    for tensor in parameters:
        if tensor.ndim == 1:
            tensor.fill(0.1)
    parameters[1].fill(0.2)
    parameters[5].fill(first_output_bias)
    parameters[9].fill(second_output_bias)
    me = np.zeros((2, 42), dtype=np.float32)
    opp = np.zeros_like(me)
    me[0, [0, 8]], opp[1, [1, 9]] = 1.0, 1.0
    value = np.array([0.7, -0.4], dtype=np.float32)
    policy = np.full((2, 7), 1.0 / 7.0, dtype=np.float32)
    return weights, me, opp, value, policy, np.ones((2, 7), dtype=bool)


def _directional_difference(
    weights: np.ndarray, direction: np.ndarray, me: np.ndarray, opp: np.ndarray,
    value: np.ndarray, policy: np.ndarray, legal: np.ndarray,
) -> tuple[float, float]:
    _, gradient = _literal_loss_gradient(weights, me, opp, value, policy, legal, 0.0)
    epsilon = np.float32(3e-3)
    numeric = (
        _literal_loss_gradient(weights + epsilon * direction, me, opp, value, policy, legal, 0.0)[0]
        - _literal_loss_gradient(weights - epsilon * direction, me, opp, value, policy, legal, 0.0)[0]
    ) / (2.0 * float(epsilon))
    return float(np.dot(gradient, direction)), numeric


def test_inactive_residual_output_relus_block_skip_gradients() -> None:
    """An inactive residual output cannot send a gradient through its skip."""
    weights, me, opp, value, policy, legal = _residual_gate_fixture(-1.0, -1.0)
    _, gradient = _literal_loss_gradient(weights, me, opp, value, policy, legal, 0.0)
    spans = _parameter_groups()
    assert np.array_equal(gradient[:spans["value_head"][0]], np.zeros(spans["value_head"][0], dtype=np.float32))
    stem_bias = _unpack(weights)[0].size
    epsilon = np.float32(3e-3)
    plus, minus = weights.copy(), weights.copy()
    plus[stem_bias] += epsilon
    minus[stem_bias] -= epsilon
    numeric = (
        _literal_loss_gradient(plus, me, opp, value, policy, legal, 0.0)[0]
        - _literal_loss_gradient(minus, me, opp, value, policy, legal, 0.0)[0]
    ) / (2.0 * float(epsilon))
    assert abs(numeric) < 3e-5
    assert abs(float(gradient[stem_bias])) < 3e-5


def test_residual_gradient_random_directions_cover_gate_bias_regimes() -> None:
    """Finite differences cover active and inactive residual-output gates."""
    rng = np.random.default_rng(987)
    spans = _parameter_groups()
    cases = (
        ("active", 0.1, 0.1, ("stem", "residual_block_1", "residual_block_2")),
        ("first_inactive", -1.0, 0.1, ("stem",)),
        ("second_inactive", 0.1, -1.0, ("stem", "residual_block_1")),
        ("both_inactive", -1.0, -1.0, ("stem",)),
    )
    for name, first_bias, second_bias, groups in cases:
        weights, me, opp, value, policy, legal = _residual_gate_fixture(first_bias, second_bias)
        for group in groups:
            start, end = spans[group]
            direction = np.zeros(N_WEIGHTS, dtype=np.float32)
            direction[start:end] = rng.standard_normal(end - start).astype(np.float32)
            direction /= np.linalg.norm(direction)
            analytic, numeric = _directional_difference(
                weights, direction, me, opp, value, policy, legal,
            )
            assert np.isclose(analytic, numeric, rtol=0.12, atol=3e-5), f"{name} {group}"


def test_gradient_multiple_seeded_directions_span_relu_masks_and_residual_blocks() -> None:
    """Exercise every residual branch through several stable ReLU-mask regimes."""
    spans = _parameter_groups()
    cases = (
        ("both_active", 0.1, 0.1, ("stem", "residual_block_1", "residual_block_2")),
        ("first_closed", -1.0, 0.1, ("stem",)),
        ("second_closed", 0.1, -1.0, ("stem", "residual_block_1")),
    )
    for name, first_bias, second_bias, groups in cases:
        weights, me, opp, value, policy, legal = _residual_gate_fixture(first_bias, second_bias)
        for seed in (101, 509, 911):
            rng = np.random.default_rng(seed)
            for group in groups:
                start, end = spans[group]
                direction = np.zeros(N_WEIGHTS, dtype=np.float32)
                direction[start:end] = rng.standard_normal(end - start).astype(np.float32)
                direction /= np.linalg.norm(direction)
                analytic, numeric = _directional_difference(
                    weights, direction, me, opp, value, policy, legal,
                )
                assert np.isclose(analytic, numeric, rtol=0.12, atol=3e-5), (
                    f"{name} seed={seed} {group}"
                )


def test_public_fit_substantially_reduces_literal_joint_objective() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    initial = initial_weights(23)
    # The public fitter uses this seed for precisely the same initialization.
    initial_loss, _ = _literal_loss_gradient(initial, me, opp, value, policy, legal, 1e-5)
    fitted, metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=80, learning_rate=5e-3,
    )
    final_loss, _ = _literal_loss_gradient(fitted, me, opp, value, policy, legal, 1e-5)
    assert metadata["optimizer_steps"] == 80
    assert final_loss < initial_loss * 0.8


def test_public_fit_memorizes_legal_asymmetric_value_and_policy_rows() -> None:
    """The public fitter must jointly fit signed values and legal policy targets."""
    # `_non_symmetric_batch` uses gravity-supported, non-symmetric boards.
    # All seven columns remain legal here, as on ordinary early-game replay.
    me, opp, value, _, _ = _non_symmetric_batch()
    policy = np.array(
        [[0.45, 0.05, 0.20, 0.05, 0.05, 0.10, 0.10],
         [0.05, 0.15, 0.05, 0.50, 0.05, 0.10, 0.10],
         [0.10, 0.05, 0.10, 0.05, 0.40, 0.10, 0.20]],
        dtype=np.float32,
    )
    legal = np.ones((len(me), 7), dtype=bool)
    initial = initial_weights(23)
    initial_loss, _ = _literal_loss_gradient(initial, me, opp, value, policy, legal, 1e-5)
    fitted, _ = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=160, learning_rate=5e-3,
        value_loss_weight=5.0,
    )
    final_loss, _ = _literal_loss_gradient(fitted, me, opp, value, policy, legal, 1e-5)
    prediction, logits = predict(fitted, me, opp)
    assert final_loss < initial_loss * 0.75
    assert np.std(prediction) > 0.25
    assert np.any(prediction > 0.25) and np.any(prediction < -0.10)
    assert np.mean(np.sign(prediction) == np.sign(value)) == 1.0
    assert np.isfinite(logits).all()


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
    telemetry = cast(dict[str, dict[str, float]], first_metadata["parameter_groups"])
    assert telemetry == cast(dict[str, dict[str, float]], second_metadata["parameter_groups"])
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


def test_first_batch_activation_health_is_finite_and_repeatable() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    _, first = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=2, learning_rate=5e-3,
    )
    _, second = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=3, epochs=2, learning_rate=5e-3,
    )
    health = cast(dict[str, float], first["first_batch_activation_health"])
    assert health == cast(dict[str, float], second["first_batch_activation_health"])
    assert set(health) == {
        "stem_relu_active_fraction",
        "residual_block_1_output_relu_active_fraction",
        "residual_block_2_output_relu_active_fraction",
        "value_1x1_relu_active_fraction",
        "value_hidden_relu_active_fraction",
        "policy_1x1_relu_active_fraction",
        "value_score_standard_deviation",
        "value_prediction_standard_deviation",
    }
    assert all(np.isfinite(measurement) for measurement in health.values())


def test_activation_health_distinguishes_dead_and_active_relu_paths() -> None:
    me, opp, _, _, _ = _non_symmetric_batch()
    dead = _activation_health(np.zeros(N_WEIGHTS, dtype=np.float32), me, opp)
    active_weights = np.zeros(N_WEIGHTS, dtype=np.float32)
    parameters = _unpack(active_weights)
    for index in (1, 3, 5, 7, 9, 11, 13, 17):
        parameters[index].fill(1.0)
    active = _activation_health(active_weights, me, opp)
    active_fractions = [name for name in active if name.endswith("relu_active_fraction")]
    assert all(dead[name] == 0.0 for name in active_fractions)
    assert all(active[name] == 1.0 for name in active_fractions)
    assert dead["value_score_standard_deviation"] == 0.0
    assert active["value_prediction_standard_deviation"] == 0.0


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
    nonfinite = _gradient_conflict_metrics(
        np.array([np.inf, 1.0], dtype=np.float32), policy_gradient
    )
    assert nonfinite == {
        "value_gradient_l2": 0.0,
        "policy_gradient_l2": 5.0,
        "cosine_similarity": 0.0,
    }


def test_head_gradient_conflict_metrics_distinguish_aligned_and_opposing_vectors() -> None:
    value_gradient = np.array([2.0, -1.0], dtype=np.float32)
    aligned = _gradient_conflict_metrics(value_gradient, 3.0 * value_gradient)
    opposing = _gradient_conflict_metrics(value_gradient, -3.0 * value_gradient)
    assert np.isclose(aligned["cosine_similarity"], 1.0)
    assert np.isclose(opposing["cosine_similarity"], -1.0)


def test_gradient_conflict_timeline_aggregation_is_repeatable_and_exact() -> None:
    cosines = {
        "stem": [0.5, -0.75, 0.25, 0.0],
        "residual_block_1": [],
        "residual_block_2": [-1.0, -0.5],
    }
    first = _aggregate_gradient_conflict(cosines)
    assert first == _aggregate_gradient_conflict(cosines)
    assert first == {
        "stem": {
            "batch_count": 4,
            "mean_cosine_similarity": 0.0,
            "min_cosine_similarity": -0.75,
            "max_cosine_similarity": 0.5,
            "negative_cosine_batch_count": 1,
        },
        "residual_block_1": {
            "batch_count": 0,
            "mean_cosine_similarity": 0.0,
            "min_cosine_similarity": 0.0,
            "max_cosine_similarity": 0.0,
            "negative_cosine_batch_count": 0,
        },
        "residual_block_2": {
            "batch_count": 2,
            "mean_cosine_similarity": -0.75,
            "min_cosine_similarity": -1.0,
            "max_cosine_similarity": -0.5,
            "negative_cosine_batch_count": 2,
        },
    }


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


def test_timeline_telemetry_preserves_default_adam_weights() -> None:
    me, opp, value, policy, legal = _non_symmetric_batch()
    fitted, metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        l2=1e-5, seed=23, batch_size=2, epochs=2, learning_rate=5e-3,
    )
    rng = np.random.default_rng(23)
    expected = initial_weights(23)
    moment, velocity = np.zeros_like(expected), np.zeros_like(expected)
    step = 0
    for _ in range(2):
        order = rng.permutation(len(me))
        for start in range(0, len(me), 2):
            batch = order[start : start + 2]
            _, gradient = _literal_loss_gradient(
                expected, me[batch], opp[batch], value[batch], policy[batch], legal[batch], 1e-5
            )
            step += 1
            moment = 0.9 * moment + 0.1 * gradient
            velocity = 0.999 * velocity + 0.001 * gradient * gradient
            expected -= 5e-3 * (moment / (1.0 - 0.9**step)) / (
                np.sqrt(velocity / (1.0 - 0.999**step)) + 1e-8
            )
    assert np.array_equal(fitted, expected)
    timeline = cast(dict[str, dict[str, float | int]], metadata["shared_head_gradient_conflict_timeline"])
    assert set(timeline) == {"stem", "residual_block_1", "residual_block_2"}
    assert all(group["batch_count"] == 4 for group in timeline.values())
    assert all(
        np.isfinite(float(measurement))
        for group in timeline.values()
        for measurement in group.values()
    )


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
