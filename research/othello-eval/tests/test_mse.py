"""Instrumentation checks for the held-out MSE diagnostic, on hand-built
inputs -- separate from the slow bake-off runs that use it."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from othello_eval.mse import bootstrap_diff_ci, predict, squared_errors
from othello_eval.records import RECORD_DTYPE

FIX = Path(__file__).resolve().parents[3] / "games/othello/ntuple/tests"


def _tiny_weights_dir(tmp_path: Path) -> Path:
    d = tmp_path / "w"
    d.mkdir()
    shutil.copy(FIX / "tiny.toml", d / "model.toml")
    shutil.copy(FIX / "weights.bin", d / "weights.bin")
    shutil.copy(FIX / "weights.meta.json", d / "weights.meta.json")
    return d


def _positions(rows: list[tuple[int, int, int, int, float]]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=RECORD_DTYPE)
    for i, (b, w, side, ply, tgt) in enumerate(rows):
        arr[i] = (b, w, side, ply, tgt)
    return arr


def test_predict_is_bounded_and_squared_error_matches_by_hand(tmp_path: Path) -> None:
    d = _tiny_weights_dir(tmp_path)
    # Empty board scores exactly 0 in the tiny fixture (all-empty features
    # weighted 0) -> tanh(0) = 0; squared error against a target of 1.0 is 1.0.
    pos = _positions([(0, 0, 0, 0, 1.0), (0, 0, 0, 0, 0.0)])
    se = squared_errors(d, pos)
    assert se[0] == 1.0
    assert se[1] == 0.0


def test_predict_stays_in_range() -> None:
    # tanh keeps every prediction inside (-1, 1) regardless of the logit.
    big = np.array([[0]], dtype=np.int64)
    w = np.array([3.0])
    assert abs(float(predict(w, big)[0])) < 1.0


def test_bootstrap_diff_ci_is_deterministic_and_signed() -> None:
    # Model A always errs by 0.5, model B is perfect: MSE(A) - MSE(B) = 0.25,
    # and with zero variance the CI collapses onto the point estimate.
    se_a = np.full(200, 0.25)
    se_b = np.zeros(200)
    point, lo, hi = bootstrap_diff_ci(se_a, se_b, iters=500, seed=0)
    assert abs(point - 0.25) < 1e-12
    assert lo > 0 and hi < 0.25 + 1e-9
    # Deterministic under a fixed seed.
    assert bootstrap_diff_ci(se_a, se_b, iters=500, seed=0) == (point, lo, hi)
