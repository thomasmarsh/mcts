"""Fast, deterministic checks for the n-tuple trainer's instrumentation
logic -- featurisation, symmetry sharing, and that the optimiser math
actually descends. No Othello self-play and no real training run here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from othello_eval.ntuple import (
    D4,
    ModelGeometry,
    TrainConfig,
    bce_and_acc,
    featurize,
    fit,
    load_model_toml,
)
from othello_eval.records import RECORD_DTYPE

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_DIR = REPO_ROOT / "games/othello/ntuple/tests"


def _positions(rows: list[tuple[int, int, int]]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=RECORD_DTYPE)
    for i, (black, white, side) in enumerate(rows):
        arr[i] = (black, white, side, 0, 0.0)
    return arr


def _toy_geom() -> ModelGeometry:
    # One 2-square tuple [0, 9]; no D4 folding matters for the hand check
    # because we look at a single orientation's column.
    return load_model_toml(TINY_DIR / "tiny.toml")


def test_feature_index_matches_hand_calculation() -> None:
    geom = _toy_geom()
    # tiny.toml: tuple 0 = [0] (offset 0), tuple 1 = [0, 9] (offset 3).
    # Position: black disc on square 0, white on square 9, black to move.
    pos = _positions([(1 << 0, 1 << 9, 0)])
    feats = featurize(pos, geom)  # (1, 2 tuples * 8 syms)
    assert feats.shape == (1, 16)

    # Identity orientation of tuple 1 is column index 8 (tuple 1, sym 0).
    # trit(sq0)=1 (own), trit(sq9)=2 (opp) -> feat = 1*1 + 2*3 = 7,
    # global = offset 3 + 7 = 10.
    assert feats[0, 8] == 10
    # Identity orientation of tuple 0 is column 0: trit(sq0)=1 -> global 1.
    assert feats[0, 0] == 1
    # Empty board: every feature index is a tuple offset.
    empty = featurize(_positions([(0, 0, 0)]), geom)
    assert set(empty[0].tolist()) == {0, 3}


def _rotate_bits(bits: int, sym: int) -> int:
    out = 0
    b = bits
    while b:
        i = (b & -b).bit_length() - 1
        b &= b - 1
        out |= 1 << int(D4[sym, i])
    return out


def test_rotation_gives_the_same_multiset_of_feature_indices() -> None:
    geom = load_model_toml(REPO_ROOT / "games/othello/ntuple/model.toml")
    black, white = 0x0000_0081_0000_0000, 0x0000_0010_0800_0000
    for sym in range(1, 8):
        rb, rw = _rotate_bits(black, sym), _rotate_bits(white, sym)
        a = featurize(_positions([(black, white, 0)]), geom)[0]
        r = featurize(_positions([(rb, rw, 0)]), geom)[0]
        assert sorted(a.tolist()) == sorted(r.tolist()), sym


def test_featurize_matches_the_committed_cross_impl_fixture() -> None:
    geom = _toy_geom()
    cases = json.loads((TINY_DIR / "featurize_cases.json").read_text())
    for case in cases:
        pos = _positions([(int(case["black"], 16), int(case["white"], 16), case["side"])])
        got = sorted(featurize(pos, geom)[0].tolist())
        assert got == sorted(case["expected"]), case


def test_fit_drives_bce_down_on_a_separable_synthetic_set() -> None:
    rng = np.random.default_rng(0)
    n, n_weights = 4000, 12
    # Each sample activates 3 of 12 weights; label is a noisy threshold on a
    # fixed "true" weight vector -> linearly separable in this basis.
    feat_idx = rng.integers(0, n_weights, size=(n, 3))
    true_w = rng.normal(size=n_weights)
    score = true_w[feat_idx].sum(axis=1)
    y01 = (score > np.median(score)).astype(np.float64)

    cfg = TrainConfig(
        epochs=60, lr=0.5, l2=0.0, batch=1024, val_fraction=0.0, seed=1, report_every=100
    )
    start_bce = bce_and_acc(np.zeros(n_weights), feat_idx, y01)[0]
    w = fit(feat_idx, y01, cfg, n_weights)
    end_bce, end_acc = bce_and_acc(w.astype(np.float64), feat_idx, y01)
    assert end_bce < start_bce * 0.5
    assert end_acc > 0.9
