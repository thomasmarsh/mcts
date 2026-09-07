# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
from pathlib import Path

import numpy as np

from az_train.mlp_c4 import (
    N_WEIGHTS,
    fit_value_head_with_diagnostics,
    inputs,
    loss_and_gradient,
    predict,
    read_weights,
    write_weights,
)


def _planes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    me = np.zeros((4, 42), dtype=np.float32)
    opp = np.zeros_like(me)
    me[:, 0] = [1, 0, 1, 0]
    opp[:, 1] = [0, 1, 0, 1]
    return me, opp, np.array([1, -1, 1, -1], dtype=np.float32)


def test_layout_round_trip_and_prediction_is_perspective_sensitive(tmp_path: Path) -> None:
    w = np.zeros(N_WEIGHTS, dtype=np.float32)
    # First hidden unit observes mover cell 0, then flows directly to output.
    w[0] = 1.0
    w[84 * 128 + 128] = 1.0
    w[84 * 128 + 128 + 128 * 128 + 128] = 1.0
    me, opp, _ = _planes()
    path = tmp_path / "value.mlp"
    write_weights(str(path), w)
    assert np.array_equal(read_weights(str(path)), w)
    assert predict(w, me[:1], opp[:1])[0] > 0.7
    assert abs(float(predict(w, opp[:1], me[:1])[0])) < 1e-7


def test_adam_is_deterministic_and_learns_a_tiny_batch() -> None:
    me, opp, value = _planes()
    first, a = fit_value_head_with_diagnostics(me, opp, value, epochs=12, batch_size=2, seed=7)
    second, b = fit_value_head_with_diagnostics(me, opp, value, epochs=12, batch_size=2, seed=7)
    assert np.array_equal(first, second)
    assert a["optimizer_steps"] == b["optimizer_steps"] == 24
    assert np.mean((predict(first, me, opp) - value) ** 2) < 0.4


def test_minibatch_gradient_matches_finite_difference() -> None:
    me, opp, value = _planes()
    rng = np.random.default_rng(4)
    w = (rng.standard_normal(N_WEIGHTS) * 0.01).astype(np.float32)
    x = inputs(me[:2], opp[:2])
    loss, gradient = loss_and_gradient(w, x, value[:2], 0.03)
    assert np.isfinite(loss) and np.all(np.isfinite(gradient))
    probe, epsilon = 0, 1e-3
    plus, minus = w.copy(), w.copy()
    plus[probe] += epsilon
    minus[probe] -= epsilon
    numeric = (
        loss_and_gradient(plus, x, value[:2], 0.03)[0]
        - loss_and_gradient(minus, x, value[:2], 0.03)[0]
    ) / (2 * epsilon)
    assert abs(float(gradient[probe]) - numeric) < 2e-4
