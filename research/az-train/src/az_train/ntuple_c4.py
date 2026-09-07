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
from pathlib import Path

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
    _off += 3**TUPLE_LEN
OFFSETS: tuple[int, ...] = tuple(_OFFSETS)
N_WEIGHTS = _off  # 1 + 69 * 81 == 5590


def _line_segments(length: int) -> list[tuple[int, ...]]:
    """Lines in the Rust geometry order: horizontal, vertical, /, \\."""
    out: list[tuple[int, ...]] = []
    for row in range(ROWS):
        for c0 in range(COLS - length + 1):
            out.append(tuple(row * COLS + c0 + k for k in range(length)))
    for col in range(COLS):
        for r0 in range(ROWS - length + 1):
            out.append(tuple((r0 + k) * COLS + col for k in range(length)))
    for r0 in range(ROWS - length + 1):
        for c0 in range(COLS - length + 1):
            out.append(tuple((r0 + k) * COLS + c0 + k for k in range(length)))
    for r0 in range(ROWS - length + 1):
        for c0 in range(length - 1, COLS):
            out.append(tuple((r0 + k) * COLS + c0 - k for k in range(length)))
    return out


SQUARES: tuple[tuple[int, ...], ...] = tuple(
    (row * COLS + col, row * COLS + col + 1, (row + 1) * COLS + col, (row + 1) * COLS + col + 1)
    for row in range(ROWS - 1)
    for col in range(COLS - 1)
)
LINES_3 = tuple(_line_segments(3))
LINES_4 = tuple(_line_segments(4))
LINES_5 = tuple(_line_segments(5))
LINES_6 = tuple(_line_segments(6))
STRUCTURED_TUPLES = SQUARES + LINES_3 + LINES_4 + LINES_5 + LINES_6
STRUCTURED_LENGTHS = tuple(map(len, STRUCTURED_TUPLES))
_structured_offsets: list[int] = []
_structured_off = 1
for length in STRUCTURED_LENGTHS:
    _structured_offsets.append(_structured_off)
    _structured_off += 3**length
STRUCTURED_OFFSETS = tuple(_structured_offsets)
COLUMN_TABLE_SIZE = 13
COLUMN_OFFSETS = tuple(_structured_off + COLUMN_TABLE_SIZE * col for col in range(COLS))
STRUCTURED_WEIGHTS = _structured_off + COLS * COLUMN_TABLE_SIZE
STRUCTURED_ACTIVE_COUNT = 1 + len(STRUCTURED_TUPLES) + COLS


def _connected8_tuples() -> tuple[tuple[int, ...], ...]:
    """Read the reviewable tuple list shared with Rust inference."""
    path = Path(__file__).resolve().parents[4] / "games/connect4/src/connected8_tuples.txt"
    tuples: list[tuple[int, ...]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tuples.append(tuple(map(int, line.split())))
    return tuple(tuples)


CONNECTED8_TUPLES = _connected8_tuples()
CONNECTED8_TABLE_SIZE = 3**8
CONNECTED8_OFFSETS = tuple(1 + i * CONNECTED8_TABLE_SIZE for i in range(len(CONNECTED8_TUPLES)))
CONNECTED8_WEIGHTS = 1 + len(CONNECTED8_TUPLES) * CONNECTED8_TABLE_SIZE
CONNECTED8_ACTIVE_COUNT = 1 + len(CONNECTED8_TUPLES)


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


def structured_active_indices(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Sparse structured-head indices in the documented table order.

    A column table has index zero for an empty column.  Otherwise it encodes
    the height and whether the bottom disc belongs to the mover or opponent:
    ``1 + 2 * (height - 1) + (bottom_trit - 1)``.
    """
    trit = _trits(me, opp)
    n = trit.shape[0]
    active = np.empty((n, STRUCTURED_ACTIVE_COUNT), dtype=np.int32)
    active[:, 0] = 0
    pairs = zip(STRUCTURED_TUPLES, STRUCTURED_OFFSETS, strict=True)
    for j, (cells, off) in enumerate(pairs, start=1):
        place = 3 ** np.arange(len(cells), dtype=np.int64)
        active[:, j] = off + (trit[:, list(cells)] * place[None, :]).sum(axis=1)
    base = 1 + len(STRUCTURED_TUPLES)
    for col, off in enumerate(COLUMN_OFFSETS):
        column = trit[:, col::COLS]
        height = np.count_nonzero(column, axis=1)
        active[:, base + col] = off + np.where(
            height == 0, 0, 1 + 2 * (height - 1) + column[:, 0] - 1
        )
    return active


def structured_features(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    active = structured_active_indices(me, opp)
    x = np.zeros((len(active), STRUCTURED_WEIGHTS), dtype=np.float32)
    x[np.arange(len(active))[:, None], active] = 1.0
    return x


def connected8_active_indices(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Sparse indices for the fixed connected eight-cell table layout."""
    trit = _trits(me, opp)
    n = trit.shape[0]
    active = np.empty((n, CONNECTED8_ACTIVE_COUNT), dtype=np.int32)
    active[:, 0] = 0
    place = 3 ** np.arange(8, dtype=np.int64)
    pairs = zip(CONNECTED8_TUPLES, CONNECTED8_OFFSETS, strict=True)
    for j, (cells, off) in enumerate(pairs, start=1):
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
    initial_residual = residual_sq**0.5
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
        if next_sq**0.5 <= tolerance * max(1.0, initial_residual):
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
        "cg_final_residual": residual_sq**0.5,
        "fit_wall_seconds": time.perf_counter() - started,
        # active indices, target, and a handful of working vectors dominate.
        "peak_working_set_estimate_bytes": int(active.nbytes + y.nbytes + N_WEIGHTS * 8 * 7),
    }
    return w.astype(np.float32), diagnostics


def fit_value_head(
    me: np.ndarray,
    opp: np.ndarray,
    value: np.ndarray,
    l2: float = 1e-3,
    value_target: str = "atanh",
) -> np.ndarray:
    """Fit ridge regression to a direct or ``atanh`` outcome target."""
    return fit_value_head_with_diagnostics(me, opp, value, l2, value_target)[0]


def fit_structured_value_head_with_diagnostics(
    me: np.ndarray,
    opp: np.ndarray,
    value: np.ndarray,
    l2: float = 1e-3,
    value_target: str = "direct",
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    mirror_augment: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Matrix-free ridge fit for the structured value layout."""
    active = structured_active_indices(me, opp)
    mirror_active: np.ndarray | None = None
    if mirror_augment:
        mirror_me = np.asarray(me).reshape(-1, ROWS, COLS)[:, :, ::-1].reshape(-1, CELLS)
        mirror_opp = np.asarray(opp).reshape(-1, ROWS, COLS)[:, :, ::-1].reshape(-1, CELLS)
        mirror_active = structured_active_indices(mirror_me, mirror_opp)
    y = _target(value, value_target)
    regularizer = np.full(STRUCTURED_WEIGHTS, float(l2), dtype=np.float64)
    regularizer[0] = 0.0

    def xt(vector: np.ndarray, indices: np.ndarray) -> np.ndarray:
        result = np.zeros(STRUCTURED_WEIGHTS, dtype=np.float64)
        # Avoid a `repeat(vector, 272)` temporary on the clean-data fit.
        for column in range(indices.shape[1]):
            np.add.at(result, indices[:, column], vector)
        return result

    def row_sum(vector: np.ndarray, indices: np.ndarray) -> np.ndarray:
        result = np.zeros(len(indices), dtype=np.float64)
        # Bound the gathered float64 temporary while keeping the inner sum
        # vectorized; the full clean split would otherwise materialize >300 MB.
        for start in range(0, len(indices), 8192):
            stop = min(start + 8192, len(indices))
            result[start:stop] = vector[indices[start:stop]].sum(axis=1)
        return result

    def apply(vector: np.ndarray) -> np.ndarray:
        result = xt(row_sum(vector, active), active)
        if mirror_active is not None:
            result += xt(row_sum(vector, mirror_active), mirror_active)
        return result + regularizer * vector

    started = time.perf_counter()
    b = xt(y, active)
    if mirror_active is not None:
        b += xt(y, mirror_active)
    w = np.zeros(STRUCTURED_WEIGHTS, dtype=np.float64)
    residual = b.copy()
    direction = residual.copy()
    residual_sq = float(residual @ residual)
    initial_residual = residual_sq**0.5
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
        if next_sq**0.5 <= tolerance * max(1.0, initial_residual):
            residual_sq = next_sq
            converged = True
            break
        direction = residual + (next_sq / residual_sq) * direction
        residual_sq = next_sq
    return w.astype(np.float32), {
        "cg_tolerance": tolerance,
        "cg_max_iterations": max_iterations,
        "cg_iterations": iterations,
        "cg_converged": converged,
        "cg_final_residual": residual_sq**0.5,
        "fit_wall_seconds": time.perf_counter() - started,
        "peak_working_set_estimate_bytes": int(
            active.nbytes
            + (mirror_active.nbytes if mirror_active is not None else 0)
            + y.nbytes
            + STRUCTURED_WEIGHTS * 8 * 8
        ),
    }


def predict_structured(w: np.ndarray, me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    if w.shape != (STRUCTURED_WEIGHTS,):
        raise ValueError(f"expected {STRUCTURED_WEIGHTS} weights, got {w.shape}")
    return np.tanh(w[structured_active_indices(me, opp)].sum(axis=1)).astype(np.float32)


def write_structured_weights(path: str, w: np.ndarray) -> None:
    if w.shape != (STRUCTURED_WEIGHTS,):
        raise ValueError(f"expected {STRUCTURED_WEIGHTS} weights, got {w.shape}")
    w.astype("<f4").tofile(path)


def connected8_loss_and_gradient(
    w: np.ndarray, active: np.ndarray, value: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    """Mean tanh-squared loss and sparse gradient for one deterministic batch."""
    score = w[active].sum(axis=1)
    prediction = np.tanh(score)
    error = prediction - value
    loss = float(np.mean(error * error) + l2 * np.dot(w[1:], w[1:]))
    delta = 2.0 * error * (1.0 - prediction * prediction) / len(active)
    gradient = np.zeros(CONNECTED8_WEIGHTS, dtype=np.float64)
    for column in range(active.shape[1]):
        np.add.at(gradient, active[:, column], delta)
    gradient[1:] += 2.0 * l2 * w[1:]
    return loss, gradient


def fit_connected8_value_head_with_diagnostics(
    me: np.ndarray,
    opp: np.ndarray,
    value: np.ndarray,
    l2: float = 10.0,
    value_target: str = "direct",
    seed: int = 0,
    batch_size: int = 1024,
    epochs: int = 80,
    learning_rate: float = 0.03,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Bounded-memory sparse AdaGrad fit for the connected eight-cell layout."""
    active = connected8_active_indices(me, opp)
    y = _target(value, value_target)
    n = len(active)
    if n == 0:
        raise ValueError("cannot fit an empty position set")
    w = np.zeros(CONNECTED8_WEIGHTS, dtype=np.float64)
    accumulated_square = np.zeros_like(w)
    rng = np.random.default_rng(seed)
    step = 0
    started = time.perf_counter()
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            batch_active = active[batch]
            prediction = np.tanh(w[batch_active].sum(axis=1))
            delta = 2.0 * (prediction - y[batch]) * (1.0 - prediction * prediction) / len(batch)
            step += 1
            for column in range(batch_active.shape[1]):
                indices, inverse = np.unique(batch_active[:, column], return_inverse=True)
                gradient = np.bincount(inverse, weights=delta, minlength=len(indices))
                if column != 0:
                    gradient += 2.0 * (float(l2) / n) * w[indices]
                accumulated_square[indices] += gradient * gradient
                denominator = np.sqrt(accumulated_square[indices]) + 1e-8
                w[indices] -= learning_rate * gradient / denominator
    loss, gradient = connected8_loss_and_gradient(w, active, y, float(l2) / n)
    return w.astype(np.float32), {
        "optimizer": "sparse_adagrad_tanh_mse",
        "optimizer_seed": seed,
        "optimizer_batch_size": batch_size,
        "optimizer_epochs": epochs,
        "optimizer_learning_rate": learning_rate,
        "optimizer_steps": step,
        "final_loss": loss,
        "final_gradient_l2": float(np.linalg.norm(gradient)),
        "fit_wall_seconds": time.perf_counter() - started,
        "peak_working_set_estimate_bytes": int(
            active.nbytes + y.nbytes + CONNECTED8_WEIGHTS * 8 * 3 + batch_size * active.shape[1] * 8
        ),
    }


def predict_connected8(w: np.ndarray, me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    if w.shape != (CONNECTED8_WEIGHTS,):
        raise ValueError(f"expected {CONNECTED8_WEIGHTS} weights, got {w.shape}")
    return np.tanh(w[connected8_active_indices(me, opp)].sum(axis=1)).astype(np.float32)


def write_connected8_weights(path: str, w: np.ndarray) -> None:
    if w.shape != (CONNECTED8_WEIGHTS,):
        raise ValueError(f"expected {CONNECTED8_WEIGHTS} weights, got {w.shape}")
    w.astype("<f4").tofile(path)


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
