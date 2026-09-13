# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""Deterministic checks for the Gumbel-loop value-head trainer's geometry
parsing and featurization -- the instrumentation logic, not a real fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from az_train import ntuple_othello
from az_train.records_othello import Positions

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_TOML = REPO_ROOT / "games" / "othello" / "ntuple" / "model.toml"


def test_geometry_matches_the_documented_weight_count() -> None:
    geom = ntuple_othello.load_model_toml(MODEL_TOML)
    # 6561 + 19683 + 6561 + 2187 + 19683 + 59049, per model.toml's own comment.
    assert geom.n_weights == 113724
    assert geom.n_tuples == 6
    assert geom.n_features == 6 * 8


def test_sha256_matches_the_rust_loader_hash() -> None:
    import hashlib

    geom = ntuple_othello.load_model_toml(MODEL_TOML)
    assert geom.sha256_hex == hashlib.sha256(MODEL_TOML.read_bytes()).hexdigest()


def test_featurize_selects_exactly_one_column_per_tuple_orientation() -> None:
    geom = ntuple_othello.load_model_toml(MODEL_TOML)
    pos = Positions(
        black=np.asarray([1, 0], dtype=np.uint64),
        white=np.asarray([0, 2], dtype=np.uint64),
        side=np.asarray([0, 1], dtype=np.uint8),
        ply=np.asarray([0, 0], dtype=np.uint8),
        value=np.asarray([0.0, 0.0], dtype=np.float32),
        policy=[[], []],
    )
    feat = ntuple_othello.featurize(pos, geom)
    assert feat.shape == (2, geom.n_features)
    # Every selected global index falls inside the geometry's total range.
    assert np.all(feat >= 0)
    assert np.all(feat < geom.n_weights)


def test_fit_drives_bce_below_the_zero_weight_baseline() -> None:
    """A tiny synthetic geometry (one 1-square tuple) with a label
    correlated with the mover holding square 0. The all-zero weight vector
    predicts p=0.5 everywhere (bce = log 2); a correct fit should do
    noticeably better -- not perfectly, since the geometry's D4 averaging
    sums the trit over square 0's whole 8-square orbit, not just square 0
    itself, so the label is only partially explained by what the model
    actually computes."""
    tuple_syms = [ntuple_othello.D4[:, [0]]]
    geom = ntuple_othello.ModelGeometry(
        tuple_syms=tuple_syms,
        offsets=np.asarray([0], dtype=np.int64),
        n_weights=3,
        sha256_hex="deadbeef",
        names=["a"],
    )
    rng = np.random.default_rng(0)
    n = 400
    black = rng.integers(0, 1 << 8, size=n, dtype=np.uint64)
    white = rng.integers(0, 1 << 8, size=n, dtype=np.uint64) & ~black
    side = rng.integers(0, 2, size=n).astype(np.uint8)
    black_to_move = side == 0
    me = np.where(black_to_move, black, white)
    mover_holds_sq0 = ((me >> np.uint64(0)) & np.uint64(1)).astype(bool)
    value = np.where(mover_holds_sq0, 0.8, -0.8).astype(np.float32)
    pos = Positions(
        black=black,
        white=white,
        side=side,
        ply=np.zeros(n, dtype=np.uint8),
        value=value,
        policy=[[] for _ in range(n)],
    )
    feat = ntuple_othello.featurize(pos, geom)
    y01 = (pos.value.astype(np.float64) + 1.0) / 2.0
    cfg = ntuple_othello.TrainConfig(epochs=100, lr=0.5, l2=1e-6, batch=n, report_every=1000)
    w = ntuple_othello.fit(feat, y01, cfg, geom.n_weights)
    bce, acc = ntuple_othello.bce_and_acc(w, feat, y01)
    assert bce < np.log(2) - 0.1
    assert acc > 0.7
