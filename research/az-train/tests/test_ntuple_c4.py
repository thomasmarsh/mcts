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
    LINES_3,
    LINES_4,
    LINES_5,
    LINES_6,
    N_WEIGHTS,
    N_WINDOWS,
    OFFSETS,
    SQUARES,
    STRUCTURED_ACTIVE_COUNT,
    STRUCTURED_TUPLES,
    STRUCTURED_WEIGHTS,
    WINDOWS,
    active_indices,
    features,
    fit_value_head,
    fit_value_head_with_diagnostics,
    predict,
    predict_structured,
    read_weights,
    structured_active_indices,
    structured_features,
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
    active = active_indices(me, opp)
    assert active.shape == (16, 1 + N_WINDOWS)
    assert np.allclose(x[np.arange(16)[:, None], active], 1.0)


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


def test_direct_target_keeps_terminal_labels_at_unit_magnitude() -> None:
    me = np.zeros((2, CELLS), dtype=np.float32)
    opp = np.zeros_like(me)
    direct, _ = fit_value_head_with_diagnostics(me, opp, np.array([1.0, 1.0]), 1.0, "direct")
    atanh, _ = fit_value_head_with_diagnostics(me, opp, np.array([1.0, 1.0]), 1.0, "atanh")
    # The direct target's normal-equation RHS is +/- 1, never the clipped
    # atanh magnitude (~3.8) used by the legacy target.
    assert np.max(np.abs(direct)) < np.max(np.abs(atanh))


def test_matrix_free_ridge_matches_dense_solution() -> None:
    me, opp, value = _bottom_row_win(32)
    l2 = 0.7
    w, diagnostics = fit_value_head_with_diagnostics(
        me, opp, value, l2, "direct", tolerance=1e-11, max_iterations=1000
    )
    x = features(me, opp).astype(np.float64)
    reg = np.eye(N_WEIGHTS)
    reg[0, 0] = 0.0
    dense = np.linalg.solve(x.T @ x + l2 * reg, x.T @ value).astype(np.float32)
    assert np.allclose(w, dense, atol=2e-5)
    assert diagnostics["cg_final_residual"] < 1e-6


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


def test_structured_geometry_counts_bounds_uniqueness_and_order() -> None:
    assert [len(SQUARES), len(LINES_3), len(LINES_4), len(LINES_5), len(LINES_6)] == [
        30,
        98,
        69,
        44,
        23,
    ]
    assert LINES_4 == WINDOWS
    assert len(STRUCTURED_TUPLES) == 264
    assert len(set(STRUCTURED_TUPLES)) == len(STRUCTURED_TUPLES)
    assert all(0 <= cell < CELLS for cells in STRUCTURED_TUPLES for cell in cells)
    assert STRUCTURED_WEIGHTS == 38_216


def test_structured_indexing_includes_column_height_and_base_parity() -> None:
    me = np.zeros((1, CELLS), dtype=np.float32)
    opp = np.zeros_like(me)
    me[0, [0, 14]] = 1.0
    opp[0, 7] = 1.0
    active = structured_active_indices(me, opp)
    assert active.shape == (1, STRUCTURED_ACTIVE_COUNT)
    # First column table: height 3 and mover at the bottom -> local feature 5.
    assert active[0, -7] == STRUCTURED_WEIGHTS - 7 * 13 + 5
    x = structured_features(me, opp)
    assert x.shape == (1, STRUCTURED_WEIGHTS)
    assert x.sum() == STRUCTURED_ACTIVE_COUNT


def test_structured_prediction_matches_rust_reference_vector() -> None:
    weights = (
        (np.arange(STRUCTURED_WEIGHTS, dtype=np.float64) - STRUCTURED_WEIGHTS / 2) * 0.0000002
    ).astype(np.float32)
    me = np.zeros((1, CELLS), dtype=np.float32)
    opp = np.zeros_like(me)
    me[0, [0, 2]] = 1.0
    opp[0, [1, 7]] = 1.0
    assert abs(float(predict_structured(weights, me, opp)[0]) - (-0.47970423)) < 1e-6
