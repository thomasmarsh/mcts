# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""The n-tuple value head learns a signal the linear head cannot, and its
weights.bin round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from az_train.ntuple import (
    N_WEIGHTS,
    features,
    fit_value_head,
    predict,
    read_weights,
    write_weights,
)
from az_train.records import Positions


def test_weight_count_matches_documented_layout() -> None:
    assert N_WEIGHTS == 1 + 8 * 27 + 4 * 81 == 541


def _boards_with_values(n: int = 6000) -> Positions:
    """Positions whose value is +1 exactly when the mover completes the top
    row (cells 0,1,2) -- an AND of three cells, which a single linear plane
    cannot represent but the row tuple can."""
    rng = np.random.default_rng(0)
    side = rng.integers(0, 2, size=n).astype(np.uint8)
    mover_digit = np.where(side == 0, 1, 2).astype(np.uint32)
    opp_digit = np.where(side == 0, 2, 1).astype(np.uint32)
    board = np.zeros(n, dtype=np.uint32)
    has_row = rng.integers(0, 2, size=n).astype(bool)
    # has_row: all three cells are the mover's. Otherwise cell 0 is the
    # opponent's (so it is never a win) and cells 1,2 are random -- "own two
    # of the three" is common but decisively not the +1 case.
    for cell in (0, 1, 2):
        if cell == 0:
            give_mover = has_row
        else:
            give_mover = has_row | (rng.random(n) < 0.5)
        digit = np.where(give_mover, mover_digit, opp_digit)
        board = board | (digit << np.uint32(cell * 2))
    value = np.where(has_row, 1.0, -1.0).astype(np.float32)
    ply = np.full(n, 3, dtype=np.uint8)
    return Positions(board=board, side=side, ply=ply, value=value, policy=[[] for _ in range(n)])


def test_ntuple_head_fits_a_conjunction() -> None:
    pos = _boards_with_values()
    w = fit_value_head(pos)
    assert w.shape == (N_WEIGHTS,)
    pred = predict(w, pos)
    assert np.mean(np.sign(pred) == np.sign(pos.value)) > 0.98


def test_features_select_one_column_per_tuple() -> None:
    pos = _boards_with_values(16)
    x = features(pos)
    # bias column plus exactly one hot per tuple (12 tuples).
    assert np.allclose(x.sum(axis=1), 1 + 12)
    assert np.array_equal(x[:, 0], np.ones(16, dtype=np.float32))


def test_weights_round_trip(tmp_path: Path) -> None:
    w = (np.arange(N_WEIGHTS, dtype=np.float32) - N_WEIGHTS / 2) * 0.01
    p = tmp_path / "weights.bin"
    write_weights(str(p), w)
    assert np.array_equal(read_weights(str(p)), w)
    assert p.stat().st_size == N_WEIGHTS * 4
