# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""The linear value head learns something and its weights.bin round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from az_train.model import (
    N_WEIGHTS,
    fit_value_head,
    predict,
    read_weights,
    write_weights,
)
from az_train.records import Positions


def _synthetic(n: int = 4000) -> Positions:
    """Positions whose value is +1 exactly when the mover holds the centre
    cell (cell 4) -- a signal the linear head can fit."""
    rng = np.random.default_rng(0)
    side = rng.integers(0, 2, size=n).astype(np.uint8)
    mover_has_centre = rng.integers(0, 2, size=n).astype(bool)
    board = np.zeros(n, dtype=np.uint32)
    # digit for the mover: 1 if X to move, 2 if O to move.
    mover_digit = np.where(side == 0, 1, 2).astype(np.uint32)
    board = np.where(mover_has_centre, mover_digit << np.uint32(8), board).astype(np.uint32)
    value = np.where(mover_has_centre, 1.0, -1.0).astype(np.float32)
    ply = mover_has_centre.astype(np.uint8)
    return Positions(board=board, side=side, ply=ply, value=value, policy=[[] for _ in range(n)])


def test_value_head_fits_a_learnable_signal() -> None:
    pos = _synthetic()
    w = fit_value_head(pos)
    assert w.shape == (N_WEIGHTS,)
    pred = predict(w, pos)
    # Sign agreement with the target on held-out-shaped data.
    assert np.mean(np.sign(pred) == np.sign(pos.value)) > 0.95


def test_weights_round_trip(tmp_path: Path) -> None:
    w = np.arange(N_WEIGHTS, dtype=np.float32) * 0.5 - 3.0
    p = tmp_path / "weights.bin"
    write_weights(str(p), w)
    assert np.array_equal(read_weights(str(p)), w)
    assert p.stat().st_size == N_WEIGHTS * 4
