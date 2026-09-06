# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""Linear value head (no policy head).

The feature vector for a position is, relative to the side to move, a bias
term plus a one-hot "my piece here" plane and an "opponent piece here"
plane over the 9 cells -- 19 weights total. The value estimate is the plain
dot product, squashed through ``tanh`` into ``[-1, 1]``.

Weights are exported as a flat little-endian ``f32`` array in this order::

    [bias, me[0..9], opp[0..9]]

so the Rust ``Evaluator`` that consumes ``weights.bin`` needs no schema
beyond this doc comment. There is no policy head yet: the recorded policy
tail is carried through the pipeline but not trained against.
"""

from __future__ import annotations

import numpy as np

from az_train.records import BOARD_CELLS, Positions, me_opp_planes

N_WEIGHTS = 1 + 2 * BOARD_CELLS


def features(pos: Positions) -> np.ndarray:
    """``(N, 19)`` float32 design matrix."""
    me, opp = me_opp_planes(pos)
    bias = np.ones((len(pos), 1), dtype=np.float32)
    return np.concatenate([bias, me, opp], axis=1)


def fit_value_head(pos: Positions, l2: float = 1e-4) -> np.ndarray:
    """Ridge least-squares fit of the linear pre-tanh score to
    ``arctanh(value)``. Returns the flat ``float32`` weight vector."""
    x = features(pos).astype(np.float64)
    # Targets live in {-1, 0, 1}; pull them off the tanh asymptotes before
    # inverting so arctanh stays finite.
    y = np.clip(pos.value.astype(np.float64), -0.999, 0.999)
    target = np.arctanh(y)
    a = x.T @ x + l2 * np.eye(x.shape[1])
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
