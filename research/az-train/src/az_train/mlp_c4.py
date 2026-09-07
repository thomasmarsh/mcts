# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""A compact, deterministic CPU value MLP for Connect Four.

The file format has an eight-byte magic followed by six little-endian u32
dimensions and little-endian f32 tensors in row-major order.  It is separate
from the legacy raw n-tuple arrays, whose byte lengths remain their layout
identifier.
"""

from __future__ import annotations

import resource
import struct
import sys
import time
from pathlib import Path

import numpy as np

INPUTS = 84
HIDDEN_1 = 128
HIDDEN_2 = 128
OUTPUTS = 1
MAGIC = b"C4MLP001"
VERSION = 1
HEADER = struct.Struct("<8s6I")
N_WEIGHTS = INPUTS * HIDDEN_1 + HIDDEN_1 + HIDDEN_1 * HIDDEN_2 + HIDDEN_2 + HIDDEN_2 + 1


def inputs(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    if me.shape != opp.shape or me.ndim != 2 or me.shape[1] != 42:
        raise ValueError(f"expected matching (N, 42) planes, got {me.shape} and {opp.shape}")
    return np.concatenate((me, opp), axis=1, dtype=np.float32)


def _unpack(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if weights.shape != (N_WEIGHTS,):
        raise ValueError(f"expected {N_WEIGHTS} MLP weights, got {weights.shape}")
    at = 0
    w1 = weights[at : at + INPUTS * HIDDEN_1].reshape(INPUTS, HIDDEN_1)
    at += INPUTS * HIDDEN_1
    b1 = weights[at : at + HIDDEN_1]
    at += HIDDEN_1
    w2 = weights[at : at + HIDDEN_1 * HIDDEN_2].reshape(HIDDEN_1, HIDDEN_2)
    at += HIDDEN_1 * HIDDEN_2
    b2 = weights[at : at + HIDDEN_2]
    at += HIDDEN_2
    w3 = weights[at : at + HIDDEN_2]
    at += HIDDEN_2
    return w1, b1, w2, b2, w3, weights[at : at + 1]


def predict(weights: np.ndarray, me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    w1, b1, w2, b2, w3, b3 = _unpack(np.asarray(weights, dtype=np.float32))
    h1 = np.maximum(inputs(me, opp) @ w1 + b1, 0.0)
    h2 = np.maximum(h1 @ w2 + b2, 0.0)
    return np.tanh(h2 @ w3 + b3[0]).astype(np.float32)


def loss_and_gradient(
    weights: np.ndarray, x: np.ndarray, value: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    """Mean tanh-squared loss and dense gradient for a deterministic batch."""
    w1, b1, w2, b2, w3, b3 = _unpack(np.asarray(weights, dtype=np.float32))
    z1 = x @ w1 + b1
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ w2 + b2
    h2 = np.maximum(z2, 0.0)
    score = h2 @ w3 + b3[0]
    prediction = np.tanh(score)
    error = prediction - value
    delta = (2.0 / len(x)) * error * (1.0 - prediction * prediction)
    gw3 = h2.T @ delta + 2.0 * l2 * w3
    gb3 = np.array([delta.sum()], dtype=np.float32)
    dz2 = (delta[:, None] * w3) * (z2 > 0.0)
    gw2 = h1.T @ dz2 + 2.0 * l2 * w2
    gb2 = dz2.sum(axis=0)
    dz1 = (dz2 @ w2.T) * (z1 > 0.0)
    gw1 = x.T @ dz1 + 2.0 * l2 * w1
    gb1 = dz1.sum(axis=0)
    return float(
        np.mean(error * error)
        + l2 * (np.dot(w1.ravel(), w1.ravel()) + np.dot(w2.ravel(), w2.ravel()) + np.dot(w3, w3))
    ), np.concatenate([g.reshape(-1) for g in [gw1, gb1, gw2, gb2, gw3, gb3]]).astype(np.float32)


def fit_value_head_with_diagnostics(
    me: np.ndarray,
    opp: np.ndarray,
    value: np.ndarray,
    l2: float = 1e-4,
    *,
    seed: int = 0,
    batch_size: int = 512,
    epochs: int = 80,
    learning_rate: float = 1e-3,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Adam training with one bounded minibatch and deterministic permutations."""
    x = inputs(me, opp).astype(np.float32, copy=False)
    y = np.asarray(value, dtype=np.float32)
    if len(x) == 0:
        raise ValueError("cannot fit an empty position set")
    rng = np.random.default_rng(seed)
    # He scaling avoids a biased all-positive initial ReLU layer.
    w1 = (rng.standard_normal((INPUTS, HIDDEN_1)) * np.sqrt(2.0 / INPUTS)).astype(np.float32)
    b1 = np.zeros(HIDDEN_1, dtype=np.float32)
    w2 = (rng.standard_normal((HIDDEN_1, HIDDEN_2)) * np.sqrt(2.0 / HIDDEN_1)).astype(np.float32)
    b2 = np.zeros(HIDDEN_2, dtype=np.float32)
    w3 = (rng.standard_normal(HIDDEN_2) * np.sqrt(2.0 / HIDDEN_2)).astype(np.float32)
    b3 = np.zeros(1, dtype=np.float32)
    params = [w1, b1, w2, b2, w3, b3]
    moments = [np.zeros_like(p) for p in params]
    velocities = [np.zeros_like(p) for p in params]
    beta1, beta2 = 0.9, 0.999
    step = 0
    started = time.perf_counter()
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch = order[start : start + batch_size]
            xb, yb = x[batch], y[batch]
            z1 = xb @ w1 + b1
            h1 = np.maximum(z1, 0.0)
            z2 = h1 @ w2 + b2
            h2 = np.maximum(z2, 0.0)
            score = h2 @ w3 + b3[0]
            prediction = np.tanh(score)
            delta = (2.0 / len(xb)) * (prediction - yb) * (1.0 - prediction * prediction)
            gw3 = h2.T @ delta + 2.0 * l2 * w3
            gb3 = np.array([delta.sum()], dtype=np.float32)
            dz2 = (delta[:, None] * w3) * (z2 > 0.0)
            gw2 = h1.T @ dz2 + 2.0 * l2 * w2
            gb2 = dz2.sum(axis=0)
            dz1 = (dz2 @ w2.T) * (z1 > 0.0)
            gw1 = xb.T @ dz1 + 2.0 * l2 * w1
            gb1 = dz1.sum(axis=0)
            step += 1
            for p, g, m, v in zip(
                params, [gw1, gb1, gw2, gb2, gw3, gb3], moments, velocities, strict=True
            ):
                m *= beta1
                m += (1.0 - beta1) * g
                v *= beta2
                v += (1.0 - beta2) * g * g
                p -= (
                    learning_rate
                    * (m / (1.0 - beta1**step))
                    / (np.sqrt(v / (1.0 - beta2**step)) + 1e-8)
                )
    weights = np.concatenate([p.reshape(-1) for p in params]).astype(np.float32)
    return weights, {
        "optimizer": "adam_tanh_mse",
        "optimizer_seed": seed,
        "optimizer_batch_size": batch_size,
        "optimizer_epochs": epochs,
        "optimizer_learning_rate": learning_rate,
        "optimizer_steps": step,
        "fit_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
        "peak_working_set_estimate_bytes": int(
            x.nbytes + sum(p.nbytes * 3 for p in params) + batch_size * (84 + 128 + 128) * 4
        ),
    }


def write_weights(path: str, weights: np.ndarray) -> None:
    weights = np.asarray(weights, dtype="<f4")
    _unpack(weights)
    Path(path).write_bytes(
        HEADER.pack(MAGIC, VERSION, INPUTS, HIDDEN_1, HIDDEN_2, OUTPUTS, N_WEIGHTS)
        + weights.tobytes()
    )


def read_weights(path: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: missing Connect Four MLP header")
    magic, version, inputs_, h1, h2, outputs, count = HEADER.unpack(raw[: HEADER.size])
    if (magic, version, inputs_, h1, h2, outputs, count) != (
        MAGIC,
        VERSION,
        INPUTS,
        HIDDEN_1,
        HIDDEN_2,
        OUTPUTS,
        N_WEIGHTS,
    ):
        raise ValueError(f"{path}: unsupported Connect Four MLP layout")
    weights = np.frombuffer(raw[HEADER.size :], dtype="<f4").copy()
    _unpack(weights)
    return weights
