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
    COLUMNS,
    D4,
    N_WEIGHTS,
    SQUARES,
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


def _rng_policy_targets(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(N, COLUMNS)`` legal-move mask (a random subset of squares
    plus PASS, always at least one legal column) and a target distribution
    over exactly the legal columns. Unrelated to any board input -- suitable
    for gradient/shape checks, but not for a "the fit learns something"
    assertion (see ``_first_empty_square_targets`` for that)."""
    rng = np.random.default_rng(seed)
    legal = rng.random((n, COLUMNS)) > 0.5
    legal[:, SQUARES] |= ~legal[:, :SQUARES].any(axis=1)  # PASS legal if nothing else is
    target = rng.random((n, COLUMNS)) * legal
    target /= target.sum(axis=1, keepdims=True)
    return target, legal


def _two_square_targets(me: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(N, COLUMNS)`` target/legal pair that *is* learnable from
    ``me``: squares 0 and 1 are always the only legal columns, and all the
    target probability mass sits on square 0 when ``me``'s own square 0 is
    occupied, else square 1. This gives ``test_fit_with_policy_reduces_
    policy_cross_entropy_below_uniform`` a simple, reliably learnable signal
    (a single input feature routed through the shared trunk to two nearby
    output squares) to fit within a small test's epoch/data budget, unlike
    ``_rng_policy_targets``, whose target is independent noise no amount of
    training can predict, or a raster-order "first empty square" rule, which
    this architecture's small residual trunk cannot reliably learn from 300
    examples in a unit test's time budget."""
    n = me.shape[0]
    target = np.zeros((n, COLUMNS), dtype=np.float64)
    legal = np.zeros((n, COLUMNS), dtype=bool)
    legal[:, 0] = True
    legal[:, 1] = True
    choose_zero = me[:, 0] > 0.5
    target[choose_zero, 0] = 1.0
    target[~choose_zero, 1] = 1.0
    return target, legal


def test_weight_count_matches_the_documented_layout() -> None:
    stem = 16 * 2 * 3 * 3 + 16
    blocks = 2 * 2 * (16 * 16 * 3 * 3 + 16)
    value = 16 + 1 + 64 * 32 + 32 + 32 + 1
    policy = 16 + 1 + 64 * 64 + 64
    assert N_WEIGHTS == stem + blocks + value + policy


def test_gradient_matches_finite_differences_value_only() -> None:
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


def test_gradient_matches_finite_differences_value_and_policy() -> None:
    me, opp, value = _rng_batch(seed=12, n=6)
    policy, legal = _rng_policy_targets(seed=13, n=6)
    weights = initial_weights(seed=14)
    _, analytic = _literal_loss_gradient(
        weights, me, opp, value, l2=1e-3, policy=policy, legal=legal,
    )

    rng = np.random.default_rng(15)
    # Sample indices from every parameter tensor, including the new policy
    # head, so a bug isolated to the policy conv/dense weights isn't missed
    # by chance.
    indices = rng.choice(N_WEIGHTS, size=40, replace=False)
    eps = 1e-3
    for i in indices:
        bumped = weights.copy()
        bumped[i] += eps
        loss_plus, _ = _literal_loss_gradient(
            bumped, me, opp, value, l2=1e-3, policy=policy, legal=legal,
        )
        bumped[i] -= 2 * eps
        loss_minus, _ = _literal_loss_gradient(
            bumped, me, opp, value, l2=1e-3, policy=policy, legal=legal,
        )
        numeric = (loss_plus - loss_minus) / (2 * eps)
        assert abs(numeric - analytic[i]) < 5e-3, (i, numeric, analytic[i])


def test_policy_only_gradient_is_included_in_the_joint_gradient() -> None:
    """The policy head's own weights get a zero gradient under value-only
    fitting except for L2 -- i.e. omitting policy/legal doesn't silently
    also skip the policy head's regularization term."""
    me, opp, value = _rng_batch(seed=16, n=4)
    weights = initial_weights(seed=17)
    _, grad_value_only = _literal_loss_gradient(weights, me, opp, value, l2=1e-2)
    # The last 64*64 + 64 entries are the policy dense weight+bias; their
    # gradient under value-only fitting is exactly the L2 term (2 * l2 * w
    # for the dense weight, and 0 for its unregularized bias).
    policy_dense_w_start = N_WEIGHTS - (64 * 64 + 64)
    policy_dense_w_end = policy_dense_w_start + 64 * 64
    expected = 2.0 * 1e-2 * weights[policy_dense_w_start:policy_dense_w_end]
    assert np.allclose(grad_value_only[policy_dense_w_start:policy_dense_w_end], expected)


def test_zero_weights_predicts_zero_value_and_uniform_policy() -> None:
    me, opp, _ = _rng_batch(seed=4, n=3)
    weights = np.zeros(N_WEIGHTS, dtype=np.float32)
    value, policy = predict(weights, me, opp)
    assert np.allclose(value, 0.0)
    assert np.allclose(policy, 0.0)


def test_predict_is_d4_equivariant() -> None:
    rng = np.random.default_rng(5)
    weights = (rng.standard_normal(N_WEIGHTS) * 0.05).astype(np.float32)
    me, opp, _ = _rng_batch(seed=6, n=4)
    base_value, base_policy = predict(weights, me, opp)
    for sym in range(8):
        cols = D4[sym]
        value, policy = predict(weights, me[:, cols], opp[:, cols])
        assert np.allclose(base_value, value, atol=1e-5), sym
        for row in range(base_policy.shape[0]):
            assert np.allclose(base_policy[row, cols], policy[row], atol=1e-5), sym


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


def test_fit_with_policy_reduces_policy_cross_entropy_below_uniform() -> None:
    me, opp, value = _rng_batch(seed=20, n=300)
    policy, legal = _two_square_targets(me)
    val_me, val_opp, val_value = _rng_batch(seed=22, n=60)
    val_policy, val_legal = _two_square_targets(val_me)
    _weights, meta = fit(
        me, opp, value, (val_me, val_opp, val_value),
        seed=0, epochs=30, batch_size=32, learning_rate=2e-3,
        policy=policy, legal=legal,
        validation_policy=val_policy, validation_legal=val_legal,
    )
    final_metrics: dict[str, float] = meta["final_validation_metrics"]  # type: ignore[assignment]
    uniform_ce = float(np.mean(np.log(val_legal.sum(axis=1))))
    assert final_metrics["masked_policy_cross_entropy"] < uniform_ce


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
    value, _policy = predict(weights, me, opp)
    assert abs(float(value[0]) - 0.0042488095) < 1e-6


def test_policy_matches_the_rust_reference_fixture() -> None:
    """Same weights formula, geometry and state as
    ``games/othello/src/convnet.rs``'s ``policy_matches_python_reference_
    fixture`` -- pins that the Rust hot path and this numpy trainer agree
    on the D4-averaged policy logits, not just each independently passing
    its own tests."""
    weights = np.array(
        [(i - N_WEIGHTS / 2) * 1e-6 for i in range(N_WEIGHTS)], dtype=np.float32
    )
    black = (1 << 0) | (1 << 2) | (1 << 8)
    white = (1 << 1) | (1 << 7)
    me = np.array([[(black >> j) & 1 for j in range(64)]], dtype=np.float32)
    opp = np.array([[(white >> j) & 1 for j in range(64)]], dtype=np.float32)
    _value, policy = predict(weights, me, opp)
    expected = [
        0.009362561628222466,
        0.00936256255954504,
        0.00936256255954504,
        0.00936256255954504,
        0.00936256255954504,
        0.00936256255954504,
        0.00936256255954504,
        0.009362561628222466,
    ]
    for actual, want in zip(policy[0, :8], expected, strict=True):
        assert abs(float(actual) - want) < 1e-6, (actual, want)


def test_literal_forward_is_deterministic_and_bounded() -> None:
    me, opp, _ = _rng_batch(seed=10, n=5)
    weights = initial_weights(seed=11)
    value_a, policy_a = _predict_literal(weights, me, opp)
    value_b, policy_b = _predict_literal(weights, me, opp)
    assert np.array_equal(value_a, value_b)
    assert np.array_equal(policy_a, policy_b)
    assert np.all(np.abs(value_a) <= 1.0)
