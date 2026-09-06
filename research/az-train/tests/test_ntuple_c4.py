# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""The Connect Four n-tuple value head fits a four-in-a-row conjunction its
windows can represent, and its weights.bin round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from az_train.ntuple_c4 import (
    CELLS,
    N_WEIGHTS,
    N_WINDOWS,
    OFFSETS,
    WINDOWS,
    features,
    fit_value_head,
    predict,
    read_weights,
    write_weights,
)


def test_layout_matches_documented_shape() -> None:
    assert N_WINDOWS == 69
    assert N_WEIGHTS == 1 + 69 * 81 == 5590
    assert OFFSETS[0] == 1
    for cells in WINDOWS:
        assert len(cells) == 4
        assert all(0 <= c < CELLS for c in cells)
        assert list(cells) == sorted(cells)
    assert len({tuple(w) for w in WINDOWS}) == N_WINDOWS


def test_features_select_one_column_per_window() -> None:
    rng = np.random.default_rng(0)
    me = (rng.random((16, CELLS)) < 0.2).astype(np.float32)
    opp = ((rng.random((16, CELLS)) < 0.2) & (me == 0)).astype(np.float32)
    x = features(me, opp)
    assert np.allclose(x.sum(axis=1), 1 + N_WINDOWS)
    assert np.array_equal(x[:, 0], np.ones(16, dtype=np.float32))


def _bottom_row_win(n: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value is +1 exactly when the mover holds the bottom-row window
    (cells 0,1,2,3) -- an AND of four cells."""
    rng = np.random.default_rng(1)
    me = np.zeros((n, CELLS), dtype=np.float32)
    opp = np.zeros((n, CELLS), dtype=np.float32)
    has_win = rng.integers(0, 2, size=n).astype(bool)
    for cell in range(4):
        give_me = has_win if cell == 0 else has_win | (rng.random(n) < 0.5)
        me[give_me, cell] = 1.0
        opp[~give_me, cell] = 1.0
    value = np.where(has_win, 1.0, -1.0).astype(np.float32)
    return me, opp, value


def test_ntuple_head_fits_a_four_cell_conjunction() -> None:
    me, opp, value = _bottom_row_win()
    w = fit_value_head(me, opp, value)
    assert w.shape == (N_WEIGHTS,)
    pred = predict(w, me, opp)
    assert np.mean(np.sign(pred) == np.sign(value)) > 0.98


def test_weights_round_trip(tmp_path: Path) -> None:
    w = (np.arange(N_WEIGHTS, dtype=np.float32) - N_WEIGHTS / 2) * 0.01
    p = tmp_path / "weights.bin"
    write_weights(str(p), w)
    assert np.array_equal(read_weights(str(p)), w)
    assert p.stat().st_size == N_WEIGHTS * 4


def test_matches_rust_fixture_board() -> None:
    """The board and weights behind
    ``game_connect4::valuenet::tests::value_matches_python_reference_prediction``."""
    w = ((np.arange(N_WEIGHTS, dtype=np.float64) - N_WEIGHTS / 2) * 0.0002).astype(np.float32)
    me = np.zeros((1, CELLS), dtype=np.float32)
    opp = np.zeros((1, CELLS), dtype=np.float32)
    me[0, [0, 2]] = 1.0
    opp[0, [1, 7]] = 1.0
    assert abs(float(predict(w, me, opp)[0]) - (-0.8014309)) < 1e-6
