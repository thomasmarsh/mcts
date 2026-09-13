# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false
"""Fast, deterministic checks for the Othello CNN's forward/backward math --
no self-play, no real training run. Mirrors ``test_ntuple.py``/
``test_policy.py``'s scope for the linear models.
"""

from __future__ import annotations

import numpy as np

from othello_eval.convnet import (
    BOARD,
    D4,
    N_WEIGHTS,
    _literal_loss_gradient,
    _predict_literal,
    fit,
    initial_weights,
    me_opp_planes,
    predict,
    read_weights,
    write_weights,
)
from othello_eval.records import RECORD_DTYPE


def _positions(rows: list[tuple[int, int, int, float]]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=RECORD_DTYPE)
    for i, (black, white, side, target) in enumerate(rows):
        arr[i] = (black, white, side, 0, target)
    return arr


def _rng_batch(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    me = (rng.random((n, BOARD * BOARD)) > 0.7).astype(np.float32)
    opp = (rng.random((n, BOARD * BOARD)) > 0.7).astype(np.float32)
    opp = opp * (1.0 - me)  # a square is never both occupied twice
    value = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    return me, opp, value


def test_weight_count_matches_the_documented_layout() -> None:
    stem = 16 * 2 * 3 * 3 + 16
    blocks = 2 * 2 * (16 * 16 * 3 * 3 + 16)
    value = 16 + 1 + 64 * 32 + 32 + 32 + 1
    assert N_WEIGHTS == stem + blocks + value


def test_gradient_matches_finite_differences() -> None:
    me, opp, value = _rng_batch(seed=1, n=6)
    weights = initial_weights(seed=2)
    _, analytic = _literal_loss_gradient(weights, me, opp, value, l2=1e-3)

    rng = np.random.default_rng(3)
    indices = rng.choice(N_WEIGHTS, size=24, replace=False)
    eps = 1e-3
    for i in indices:
        bumped = weights.copy()
        bumped[i] += eps
        loss_plus, _ = _literal_loss_gradient(bumped, me, opp, value, l2=1e-3)
        bumped[i] -= 2 * eps
        loss_minus, _ = _literal_loss_gradient(bumped, me, opp, value, l2=1e-3)
        numeric = (loss_plus - loss_minus) / (2 * eps)
        assert abs(numeric - analytic[i]) < 5e-3, (i, numeric, analytic[i])


def test_zero_weights_predicts_zero_value() -> None:
    me, opp, _ = _rng_batch(seed=4, n=3)
    weights = np.zeros(N_WEIGHTS, dtype=np.float32)
    assert np.allclose(predict(weights, me, opp), 0.0)


def test_predict_is_d4_equivariant() -> None:
    rng = np.random.default_rng(5)
    weights = (rng.standard_normal(N_WEIGHTS) * 0.05).astype(np.float32)
    me, opp, _ = _rng_batch(seed=6, n=4)
    base = predict(weights, me, opp)
    for sym in range(8):
        cols = D4[sym]
        transformed = predict(weights, me[:, cols], opp[:, cols])
        assert np.allclose(base, transformed, atol=1e-5), sym


def test_fit_reduces_validation_mse_below_a_constant_baseline() -> None:
    me, opp, value = _rng_batch(seed=7, n=200)
    val_me, val_opp, val_value = _rng_batch(seed=8, n=50)
    _weights, meta = fit(
        me, opp, value, (val_me, val_opp, val_value),
        seed=0, epochs=6, batch_size=32, learning_rate=5e-3,
    )
    baseline_mse = float(np.mean(val_value**2))
    final_metrics: dict[str, float] = meta["final_validation_metrics"]  # type: ignore[assignment]
    assert final_metrics["value_mse"] < baseline_mse


def test_weights_round_trip_through_disk() -> None:
    import tempfile
    from pathlib import Path

    weights = initial_weights(seed=9)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "weights.bin"
        write_weights(str(path), weights)
        loaded = read_weights(str(path))
    assert np.array_equal(weights, loaded)


def test_me_opp_planes_matches_direct_bit_unpacking() -> None:
    positions = _positions([(0b101, 0b010, 0, 0.0), (0b1, 0b10, 1, 0.0)])
    me, opp = me_opp_planes(positions)
    assert me[0, 0] == 1.0 and me[0, 2] == 1.0 and me[0, 1] == 0.0
    assert opp[0, 1] == 1.0
    # side == 1: me/opp swap black/white.
    assert me[1, 1] == 1.0 and opp[1, 0] == 1.0


def test_value_matches_the_rust_reference_fixture() -> None:
    """Same weights formula, geometry and state as
    ``games/othello/src/convnet.rs``'s ``value_matches_python_reference_
    fixture`` -- pins that the Rust hot path and this numpy trainer agree
    on the D4-averaged value, not just each independently passing its own
    tests."""
    weights = np.array(
        [(i - N_WEIGHTS / 2) * 1e-6 for i in range(N_WEIGHTS)], dtype=np.float32
    )
    black = (1 << 0) | (1 << 2) | (1 << 8)
    white = (1 << 1) | (1 << 7)
    me = np.array([[(black >> j) & 1 for j in range(64)]], dtype=np.float32)
    opp = np.array([[(white >> j) & 1 for j in range(64)]], dtype=np.float32)
    got = predict(weights, me, opp)[0]
    assert abs(float(got) - 0.007168648) < 1e-6


def test_literal_forward_is_deterministic_and_bounded() -> None:
    me, opp, _ = _rng_batch(seed=10, n=5)
    weights = initial_weights(seed=11)
    a = _predict_literal(weights, me, opp)
    b = _predict_literal(weights, me, opp)
    assert np.array_equal(a, b)
    assert np.all(np.abs(a) <= 1.0)
