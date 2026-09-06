# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""N-tuple value head for standard 6x7 Connect Four.

Same shape as ``az_train.ntuple`` (tic-tac-toe), retargeted to the Connect
Four board. An n-tuple is an ordered list of board cells; a position maps
each tuple to one base-3 *feature index* by reading a digit per cell -- 0
empty, 1 the side-to-move piece, 2 the opponent piece, least-significant
digit first -- and each tuple owns a table of ``3 ** 4`` weights. The
pre-``tanh`` score is a bias term plus the plain sum of the one selected
weight per tuple; the value estimate is that score squashed through
``tanh`` into ``[-1, 1]``.

The tuples are the 69 four-in-a-row windows of the 6x7 board -- every line
that can win the game: 24 horizontal, 21 vertical, 12 up-right diagonal, 12
up-left diagonal. Cells are row-major with row 0 at the bottom,
``cell = row * 7 + col``, matching ``game_connect4::State``'s bit layout.
The geometry is fixed (the standard board never changes shape), so it is
hard-coded here and mirrored in ``game_connect4::valuenet::NTupleValueNet``
rather than read from a file.

Weights are exported as a flat little-endian ``f32`` array in this order::

    [bias, window[0][0..81], window[1][0..81], ..., window[68][0..81]]

i.e. 1 + 69*81 = 5590 floats. The Rust ``Evaluator`` that consumes
``weights.bin`` needs no schema beyond this doc comment.

This module works on explicit ``(N, 42)`` mover/opponent occupancy planes
rather than on decoded ``Positions``: the Connect Four self-play record
format and its reader are a later slice of the port, and the value-head
geometry and weight layout stand on their own without them.
"""

from __future__ import annotations

import time

import numpy as np

ROWS = 6
COLS = 7
CELLS = ROWS * COLS
TUPLE_LEN = 4


def _windows() -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    # Horizontal: 4 consecutive columns on a row.
    for row in range(ROWS):
        for c0 in range(COLS - 3):
            out.append(tuple(row * COLS + c0 + k for k in range(4)))
    # Vertical: 4 consecutive rows in a column.
    for col in range(COLS):
        for r0 in range(ROWS - 3):
            out.append(tuple((r0 + k) * COLS + col for k in range(4)))
    # Up-right diagonal.
    for r0 in range(ROWS - 3):
        for c0 in range(COLS - 3):
            out.append(tuple((r0 + k) * COLS + (c0 + k) for k in range(4)))
    # Up-left diagonal.
    for r0 in range(ROWS - 3):
        for c0 in range(3, COLS):
            out.append(tuple((r0 + k) * COLS + (c0 - k) for k in range(4)))
    return tuple(out)


WINDOWS: tuple[tuple[int, ...], ...] = _windows()
N_WINDOWS = len(WINDOWS)  # 69

_OFFSETS: list[int] = []
_off = 1
for _ in WINDOWS:
    _OFFSETS.append(_off)
    _off += 3 ** TUPLE_LEN
OFFSETS: tuple[int, ...] = tuple(_OFFSETS)
N_WEIGHTS = _off  # 1 + 69 * 81 == 5590


def _trits(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """``(N, 42)`` int array of per-cell digits: 0 empty, 1 mover, 2 opponent."""
    return (np.asarray(me) + 2.0 * np.asarray(opp)).astype(np.int64)


def features(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """``(N, N_WEIGHTS)`` float32 design matrix: a 1 in the bias column and a
    1 in the single selected column of each window's table."""
    trit = _trits(me, opp)
    n = trit.shape[0]
    rows = np.arange(n)
    x = np.zeros((n, N_WEIGHTS), dtype=np.float32)
    x[:, 0] = 1.0
    place = 3 ** np.arange(TUPLE_LEN, dtype=np.int64)
    for cells, off in zip(WINDOWS, OFFSETS, strict=True):
        feat = (trit[:, list(cells)] * place[None, :]).sum(axis=1)
        x[rows, off + feat] = 1.0
    return x


def active_indices(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Return the bias and one active table entry per window for each row.

    This is the compact representation used by the trainer: unlike a dense
    design matrix it remains practical for six-figure position datasets.
    """
    trit = _trits(me, opp)
    n = trit.shape[0]
    active = np.empty((n, 1 + N_WINDOWS), dtype=np.int32)
    active[:, 0] = 0
    place = 3 ** np.arange(TUPLE_LEN, dtype=np.int64)
    for j, (cells, off) in enumerate(zip(WINDOWS, OFFSETS, strict=True), start=1):
        active[:, j] = off + (trit[:, list(cells)] * place[None, :]).sum(axis=1)
    return active


def _target(value: np.ndarray, value_target: str) -> np.ndarray:
    y = np.asarray(value, dtype=np.float64)
    if value_target == "direct":
        return y
    if value_target == "atanh":
        return np.arctanh(np.clip(y, -0.999, 0.999))
    raise ValueError(f"unknown value target {value_target!r}")


def fit_value_head_with_diagnostics(
    me: np.ndarray,
    opp: np.ndarray,
    value: np.ndarray,
    l2: float = 1e-3,
    value_target: str = "atanh",
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Matrix-free ridge solve over active n-tuple features using CG."""
    active = active_indices(me, opp)
    y = _target(value, value_target)
    regularizer = np.full(N_WEIGHTS, float(l2), dtype=np.float64)
    regularizer[0] = 0.0

    def xt(vector: np.ndarray) -> np.ndarray:
        result = np.zeros(N_WEIGHTS, dtype=np.float64)
        np.add.at(result, active.ravel(), np.repeat(vector, active.shape[1]))
        return result

    def apply(vector: np.ndarray) -> np.ndarray:
        return xt(vector[active].sum(axis=1)) + regularizer * vector

    started = time.perf_counter()
    b = xt(y)
    w = np.zeros(N_WEIGHTS, dtype=np.float64)
    residual = b.copy()
    direction = residual.copy()
    residual_sq = float(residual @ residual)
    initial_residual = residual_sq ** 0.5
    iterations = 0
    converged = initial_residual <= tolerance
    for step in range(1, max_iterations + 1) if initial_residual > tolerance else ():
        iterations = step
        ap = apply(direction)
        denom = float(direction @ ap)
        if denom <= 0.0:
            raise RuntimeError("ridge normal equations were not positive definite")
        alpha = residual_sq / denom
        w += alpha * direction
        residual -= alpha * ap
        next_sq = float(residual @ residual)
        if next_sq ** 0.5 <= tolerance * max(1.0, initial_residual):
            residual_sq = next_sq
            converged = True
            break
        direction = residual + (next_sq / residual_sq) * direction
        residual_sq = next_sq
    diagnostics: dict[str, float | int] = {
        "cg_tolerance": tolerance,
        "cg_max_iterations": max_iterations,
        "cg_iterations": iterations,
        "cg_converged": converged,
        "cg_final_residual": residual_sq ** 0.5,
        "fit_wall_seconds": time.perf_counter() - started,
        # active indices, target, and a handful of working vectors dominate.
        "peak_working_set_estimate_bytes": int(active.nbytes + y.nbytes + N_WEIGHTS * 8 * 7),
    }
    return w.astype(np.float32), diagnostics


def fit_value_head(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray, l2: float = 1e-3,
    value_target: str = "atanh",
) -> np.ndarray:
    """Fit ridge regression to a direct or ``atanh`` outcome target."""
    return fit_value_head_with_diagnostics(me, opp, value, l2, value_target)[0]


def predict(w: np.ndarray, me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """``(N,)`` float32 value estimates in ``[-1, 1]``."""
    return np.tanh(w[active_indices(me, opp)].sum(axis=1)).astype(np.float32)


def write_weights(path: str, w: np.ndarray) -> None:
    if w.shape != (N_WEIGHTS,):
        raise ValueError(f"expected {N_WEIGHTS} weights, got {w.shape}")
    w.astype("<f4").tofile(path)


def read_weights(path: str) -> np.ndarray:
    w = np.fromfile(path, dtype="<f4")
    if w.shape != (N_WEIGHTS,):
        raise ValueError(f"{path}: expected {N_WEIGHTS} weights, got {w.shape}")
    return w
