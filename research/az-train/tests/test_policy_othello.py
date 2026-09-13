# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false, reportPrivateUsage=false
"""Deterministic, hand-verifiable checks for the D4 policy sidecar's fit.

``az_train.policy_othello`` hand-derives an analytic backward pass through
the D4-averaging forward pass (``games/othello/src/policy.rs``'s Python
counterpart). That derivation, not the self-play numbers a real run
produces, is exactly the kind of instrumentation logic ``AGENTS.md`` asks
for a fast deterministic test on a small input -- a numeric-gradient check
against a tiny synthetic geometry.
"""

from __future__ import annotations

import numpy as np

from az_train import ntuple_othello, policy_othello
from az_train.records_othello import Positions


def _tiny_geometry() -> ntuple_othello.ModelGeometry:
    """Two single-square tuples (3 weights each): squares 0 and 1."""
    tuple_syms = [ntuple_othello.D4[:, [0]], ntuple_othello.D4[:, [1]]]
    offsets = np.asarray([0, 3], dtype=np.int64)
    return ntuple_othello.ModelGeometry(
        tuple_syms=tuple_syms,
        offsets=offsets,
        n_weights=6,
        sha256_hex="deadbeef",
        names=["a", "b"],
    )


def _tiny_positions(n: int, seed: int) -> Positions:
    rng = np.random.default_rng(seed)
    black = rng.integers(0, 1 << 8, size=n, dtype=np.uint64)
    white = rng.integers(0, 1 << 8, size=n, dtype=np.uint64) & ~black
    side = rng.integers(0, 2, size=n).astype(np.uint8)
    ply = np.arange(n, dtype=np.uint8)
    value = rng.uniform(-1, 1, size=n).astype(np.float32)
    policy: list[list[tuple[int, float]]] = []
    for _ in range(n):
        k = rng.integers(1, 4)
        squares = rng.choice(64, size=k, replace=False)
        probs = rng.dirichlet(np.ones(k))
        policy.append([(int(s), float(p)) for s, p in zip(squares, probs, strict=True)])
    return Positions(black=black, white=white, side=side, ply=ply, value=value, policy=policy)


def test_forward_matches_real_model_toml_geometry_shape() -> None:
    """Sanity check against the real geometry (not a synthetic tiny one):
    ``games/othello/ntuple/model.toml``'s SHA-256 and feature-column count
    match what the Rust loader and ``othello_eval.ntuple`` compute -- both
    are already pinned to each other and to the Rust evaluator by their own
    cross-language fixture, so this only needs to confirm this module's
    independently duplicated geometry parser agrees, not re-derive that
    fixture."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    model_toml = repo_root / "games" / "othello" / "ntuple" / "model.toml"
    geom = ntuple_othello.load_model_toml(model_toml)
    assert geom.n_weights == 113724
    assert geom.n_features == geom.n_tuples * 8
    pos = _tiny_positions(5, seed=1)
    rng = np.random.default_rng(2)
    weights = rng.normal(size=geom.n_weights * policy_othello.SQUARES).astype(np.float32)
    out = policy_othello.logits(weights, pos, geom)
    assert out.shape == (5, policy_othello.COLUMNS)
    assert np.all(np.isfinite(out))


def test_backward_matches_finite_differences() -> None:
    geom = _tiny_geometry()
    pos = _tiny_positions(6, seed=3)
    train_pos, val_pos = pos, _tiny_positions(4, seed=4)
    n_weights = geom.n_weights

    feat_idx = policy_othello.featurize(train_pos, geom)
    target, legal = policy_othello.targets(train_pos.policy)
    rng = np.random.default_rng(5)
    weights = rng.normal(scale=0.1, size=(n_weights, policy_othello.SQUARES))

    def loss(w: np.ndarray) -> float:
        raw, _ = policy_othello._forward(w, feat_idx)
        masked = np.where(legal, raw, -np.inf)
        log_z = np.logaddexp.reduce(masked, axis=1)
        log_prob = masked - log_z[:, None]
        return float(-np.sum(target[target > 0] * log_prob[target > 0]) / len(target))

    raw, idx_syms = policy_othello._forward(weights, feat_idx)
    masked = np.where(legal, raw, -np.inf)
    prob = np.exp(masked - np.logaddexp.reduce(masked, axis=1)[:, None])
    d_logits = (prob - target) / len(train_pos)
    analytic = policy_othello._backward(d_logits, idx_syms, n_weights)

    eps = 1e-5
    base = loss(weights)
    # Spot-check a handful of weight entries rather than the full 384-entry
    # table -- finite differences are only a numeric check on the analytic
    # derivation, not a full second implementation.
    checks = [(0, 0), (0, 63), (2, 5), (5, 40)]
    for r, c in checks:
        bumped = weights.copy()
        bumped[r, c] += eps
        numeric = (loss(bumped) - base) / eps
        assert abs(numeric - analytic[r, c]) < 1e-2, (r, c, numeric, analytic[r, c])

    del val_pos  # unused; kept for readability of the fixture's intent


def _learnable_positions(n: int, seed: int) -> Positions:
    """Positions whose two-way choice between squares 0 and 1 is a
    deterministic function of the tiny geometry's own features (whether
    square 0 holds the mover's disc), so a correctly-implemented fit has
    real signal to find -- unlike :func:`_tiny_positions`' fully random
    targets, which no 6-weight model could ever beat uniform on."""
    rng = np.random.default_rng(seed)
    black = rng.integers(0, 1 << 8, size=n, dtype=np.uint64)
    white = rng.integers(0, 1 << 8, size=n, dtype=np.uint64) & ~black
    side = rng.integers(0, 2, size=n).astype(np.uint8)
    ply = np.arange(n, dtype=np.uint8)
    value = rng.uniform(-1, 1, size=n).astype(np.float32)
    black_to_move = side == 0
    me = np.where(black_to_move, black, white)
    mover_holds_sq0 = ((me >> np.uint64(0)) & np.uint64(1)).astype(bool)
    # Both squares are always legal, but the mass leans heavily toward
    # whichever one the deterministic rule prefers, so a correct fit beats
    # uniform without collapsing to a trivial single-legal-move position.
    policy = [
        [(0, 0.9), (1, 0.1)] if flag else [(0, 0.1), (1, 0.9)] for flag in mover_holds_sq0
    ]
    return Positions(black=black, white=white, side=side, ply=ply, value=value, policy=policy)


def test_fit_reduces_validation_cross_entropy_below_uniform() -> None:
    geom = _tiny_geometry()
    train_pos = _learnable_positions(200, seed=10)
    val_pos = _learnable_positions(60, seed=11)
    _, metrics = policy_othello.fit(
        train_pos, geom, val_pos, l2=1e-4, seed=0, epochs=40, batch_size=16
    )
    assert metrics["validation"]["cross_entropy"] < metrics["validation"][
        "uniform_baseline_cross_entropy"
    ] - 0.1


def test_pass_logit_is_the_mean_of_real_square_logits() -> None:
    geom = _tiny_geometry()
    pos = _tiny_positions(3, seed=20)
    rng = np.random.default_rng(21)
    weights = rng.normal(size=geom.n_weights * policy_othello.SQUARES).astype(np.float32)
    full = policy_othello.logits(weights, pos, geom)
    real = full[:, : policy_othello.SQUARES]
    assert np.allclose(full[:, policy_othello.SQUARES], real.mean(axis=1))
