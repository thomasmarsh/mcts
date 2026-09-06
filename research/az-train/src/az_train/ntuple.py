# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""N-tuple value head for tic-tac-toe.

An n-tuple is an ordered list of board cells. A position maps each tuple to
one base-3 *feature index* by reading a digit per cell -- 0 empty, 1 the
side-to-move piece, 2 the opponent piece, least-significant digit first --
and each tuple owns a table of ``3 ** k`` weights. The pre-``tanh`` score is
a bias term plus the plain sum of the one selected weight per tuple; the
value estimate is that score squashed through ``tanh`` into ``[-1, 1]``.

The geometry is fixed (tic-tac-toe never changes shape), so it is hard-coded
here and mirrored in ``game_ttt::valuenet::NTupleValueNet`` rather than read
from a file:

- 8 structural lines (3 rows, 3 columns, 2 diagonals), 3 cells each, 27
  weights per tuple;
- 4 overlapping 2x2 squares, 4 cells each, 81 weights per tuple.

Weights are exported as a flat little-endian ``f32`` array in this order::

    [bias,
     line[0][0..27], line[1][0..27], ..., line[7][0..27],
     square[0][0..81], square[1][0..81], square[2][0..81], square[3][0..81]]

i.e. 1 + 8*27 + 4*81 = 541 floats. The Rust ``Evaluator`` that consumes
``weights.bin`` needs no schema beyond this doc comment.

Cells are row-major, ``row * 3 + col``. There is no policy head: the
recorded policy tail is carried through the pipeline but not trained
against.
"""

from __future__ import annotations

import numpy as np

from az_train.records import BOARD_CELLS, Positions, me_opp_planes

# Structural lines first, then 2x2 squares. Cell order within a tuple fixes
# the base-3 digit order (first cell is the least-significant trit).
LINES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
SQUARES: tuple[tuple[int, ...], ...] = (
    (0, 1, 3, 4),
    (1, 2, 4, 5),
    (3, 4, 6, 7),
    (4, 5, 7, 8),
)
TUPLES: tuple[tuple[int, ...], ...] = LINES + SQUARES

# Offset of each tuple's table into the flat weight vector; index 0 is the
# bias, so the first tuple starts at 1.
_OFFSETS: list[int] = []
_off = 1
for _t in TUPLES:
    _OFFSETS.append(_off)
    _off += 3 ** len(_t)
OFFSETS: tuple[int, ...] = tuple(_OFFSETS)
N_WEIGHTS = _off  # 1 + 8*27 + 4*81 == 541


def _trits(pos: Positions) -> np.ndarray:
    """``(N, 9)`` int array of per-cell digits: 0 empty, 1 mover, 2 opponent."""
    me, opp = me_opp_planes(pos)
    return (me + 2.0 * opp).astype(np.int64)


def features(pos: Positions) -> np.ndarray:
    """``(N, N_WEIGHTS)`` float32 design matrix: a 1 in the bias column and a
    1 in the single selected column of each tuple's table."""
    trit = _trits(pos)
    n = len(pos)
    rows = np.arange(n)
    x = np.zeros((n, N_WEIGHTS), dtype=np.float32)
    x[:, 0] = 1.0
    for cells, off in zip(TUPLES, OFFSETS, strict=True):
        place = 3 ** np.arange(len(cells), dtype=np.int64)
        feat = (trit[:, list(cells)] * place[None, :]).sum(axis=1)
        x[rows, off + feat] = 1.0
    return x


def fit_value_head(pos: Positions, l2: float = 1e-3) -> np.ndarray:
    """Ridge least-squares fit of the pre-``tanh`` score to
    ``arctanh(value)``. Returns the flat ``float32`` weight vector.

    The bias column is left unregularised; every table weight gets an L2
    penalty (tables are sparse -- most feature indices are seen rarely -- so
    a firmer prior than the linear head's keeps unseen cells near zero)."""
    x = features(pos).astype(np.float64)
    y = np.clip(pos.value.astype(np.float64), -0.999, 0.999)
    target = np.arctanh(y)
    reg = np.eye(x.shape[1])
    reg[0, 0] = 0.0
    a = x.T @ x + l2 * reg
    b = x.T @ target
    w = np.linalg.solve(a, b)
    return w.astype(np.float32)


def predict(w: np.ndarray, pos: Positions) -> np.ndarray:
    """``(N,)`` float32 value estimates in ``[-1, 1]``."""
    return np.tanh(features(pos) @ w.astype(np.float32)).astype(np.float32)


def write_weights(path: str, w: np.ndarray) -> None:
    if w.shape != (N_WEIGHTS,):
        raise ValueError(f"expected {N_WEIGHTS} weights, got {w.shape}")
    w.astype("<f4").tofile(path)


def read_weights(path: str) -> np.ndarray:
    w = np.fromfile(path, dtype="<f4")
    if w.shape != (N_WEIGHTS,):
        raise ValueError(f"{path}: expected {N_WEIGHTS} weights, got {w.shape}")
    return w


assert BOARD_CELLS == 9, "n-tuple geometry assumes a 3x3 board"
