"""Versioned compact Connect Four convolutional value-and-policy inference.

``C4CNN001`` stores a concrete two-plane 6x7 model: a 3x3 stem with 16
channels, two 16-channel residual blocks, then separate value and policy
heads.  It is deliberately inference-only here.  Production fitting remains
restricted to self-play outcomes and completed-Q policy targets.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

ROWS = 6
COLS = 7
CHANNELS = 16
BLOCKS = 2
MAGIC = b"C4CNN001"
VERSION = 1
# magic, version, rows, cols, input channels, channels, residual blocks,
# value hidden width, policy outputs, float count
HEADER = struct.Struct("<8s9I")
VALUE_HIDDEN = 32
POLICY_OUTPUTS = 7


def _count() -> int:
    stem = CHANNELS * 2 * 3 * 3 + CHANNELS
    blocks = BLOCKS * 2 * (CHANNELS * CHANNELS * 3 * 3 + CHANNELS)
    value = CHANNELS + 1 + ROWS * COLS * VALUE_HIDDEN + VALUE_HIDDEN + VALUE_HIDDEN + 1
    policy = CHANNELS + 1 + ROWS * COLS * POLICY_OUTPUTS + POLICY_OUTPUTS
    return stem + blocks + value + policy


N_WEIGHTS = _count()


def _unpack(w: np.ndarray) -> list[np.ndarray]:
    w = np.asarray(w, dtype=np.float32)
    if w.shape != (N_WEIGHTS,):
        raise ValueError(f"expected {N_WEIGHTS} C4CNN001 weights, got {w.shape}")
    at = 0

    def take(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal at
        n = int(np.prod(shape))
        out = w[at : at + n].reshape(shape)
        at += n
        return out

    out = [take((CHANNELS, 2, 3, 3)), take((CHANNELS,))]
    for _ in range(BLOCKS * 2):
        out.extend((take((CHANNELS, CHANNELS, 3, 3)), take((CHANNELS,))))
    out.extend(
        (
            take((1, CHANNELS, 1, 1)), take((1,)),
            take((ROWS * COLS, VALUE_HIDDEN)), take((VALUE_HIDDEN,)),
            take((VALUE_HIDDEN,)), take((1,)),
            take((1, CHANNELS, 1, 1)), take((1,)),
            take((ROWS * COLS, POLICY_OUTPUTS)), take((POLICY_OUTPUTS,)),
        )
    )
    assert at == N_WEIGHTS
    return out


def _conv(x: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int) -> np.ndarray:
    n, _, rows, cols = x.shape
    out = np.broadcast_to(b, (n, w.shape[0])).copy()[:, :, None, None]
    out = np.broadcast_to(out, (n, w.shape[0], rows, cols)).copy()
    for kr in range(w.shape[2]):
        for kc in range(w.shape[3]):
            src_r = slice(max(0, kr - padding), min(rows, rows + kr - padding))
            src_c = slice(max(0, kc - padding), min(cols, cols + kc - padding))
            dst_r = slice(max(0, padding - kr), min(rows, rows + padding - kr))
            dst_c = slice(max(0, padding - kc), min(cols, cols + padding - kc))
            out[:, :, dst_r, dst_c] += np.einsum("nirc,oi->norc", x[:, :, src_r, src_c], w[:, :, kr, kc])
    return out


def _predict_literal(weights: np.ndarray, me: np.ndarray, opp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if me.shape != opp.shape or me.ndim != 2 or me.shape[1] != ROWS * COLS:
        raise ValueError(f"expected matching (N, 42) planes, got {me.shape} and {opp.shape}")
    p = _unpack(weights)
    x = np.stack((me, opp), axis=1).reshape((-1, 2, ROWS, COLS))
    x = np.maximum(_conv(x, p[0], p[1], 1), 0.0)
    at = 2
    for _ in range(BLOCKS):
        residual = x
        x = np.maximum(_conv(x, p[at], p[at + 1], 1), 0.0)
        x = np.maximum(_conv(x, p[at + 2], p[at + 3], 1) + residual, 0.0)
        at += 4
    value = np.maximum(_conv(x, p[at], p[at + 1], 0), 0.0).reshape((-1, ROWS * COLS))
    value = np.maximum(value @ p[at + 2] + p[at + 3], 0.0)
    value = np.tanh(value @ p[at + 4] + p[at + 5][0]).astype(np.float32)
    at += 6
    policy = np.maximum(_conv(x, p[at], p[at + 1], 0), 0.0).reshape((-1, ROWS * COLS))
    return value, (policy @ p[at + 2] + p[at + 3]).astype(np.float32)


def predict(weights: np.ndarray, me: np.ndarray, opp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror-averaged value and absolute-column logits."""
    value, logits = _predict_literal(weights, me, opp)
    mirror_me = me.reshape((-1, ROWS, COLS))[:, :, ::-1].reshape((-1, ROWS * COLS))
    mirror_opp = opp.reshape((-1, ROWS, COLS))[:, :, ::-1].reshape((-1, ROWS * COLS))
    reflected_value, reflected_logits = _predict_literal(weights, mirror_me, mirror_opp)
    return (0.5 * (value + reflected_value)).astype(np.float32), (
        0.5 * (logits + reflected_logits[:, ::-1])
    ).astype(np.float32)


def write_weights(path: str, weights: np.ndarray) -> None:
    weights = np.asarray(weights, dtype="<f4")
    _unpack(weights)
    Path(path).write_bytes(
        HEADER.pack(MAGIC, VERSION, ROWS, COLS, 2, CHANNELS, BLOCKS, VALUE_HIDDEN, POLICY_OUTPUTS, N_WEIGHTS)
        + weights.tobytes()
    )


def read_weights(path: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: missing C4CNN001 header")
    header = HEADER.unpack(raw[: HEADER.size])
    expected = (MAGIC, VERSION, ROWS, COLS, 2, CHANNELS, BLOCKS, VALUE_HIDDEN, POLICY_OUTPUTS, N_WEIGHTS)
    if header != expected or len(raw) != HEADER.size + N_WEIGHTS * 4:
        raise ValueError(f"{path}: unsupported C4CNN001 layout")
    weights = np.frombuffer(raw[HEADER.size :], dtype="<f4").copy()
    _unpack(weights)
    return weights
