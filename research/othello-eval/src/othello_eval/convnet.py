# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnusedVariable=false
# ruff: noqa: E501, E702
"""Versioned compact Othello convolutional value network.

``OTCNN001`` stores a concrete two-plane 8x8 model: a 3x3 stem with 16
channels, two 16-channel residual blocks, then a value head -- the direct
8x8 generalization of Connect Four's ``C4CNN001``
(``research/az-train/src/az_train/convnet_c4.py`` / ``games/connect4/src/
convnet.rs``). Value-only for now: the corpora available to fit a
from-scratch architecture against are outcome-labelled only, with no
completed-Q policy targets, so a policy head is deferred to whenever the CNN
is actually wired into a self-play loop that produces those targets -- the
same value-then-policy sequencing the n-tuple port used.

Training uses the board's single natural (literal) orientation only, the
same choice ``convnet_c4``'s main fit path makes; equivariance is instead a
property of :func:`predict`, which averages the literal network's output
over all 8 D4-transformed copies of the input board. This is cheaper than
folding D4-averaging into the training loss (as the linear n-tuple/policy
sidecar does, where it is nearly free) and matches the already-accepted C4
precedent for a network with real per-orientation compute cost.
"""

from __future__ import annotations

import resource
import struct
import sys
import time
from pathlib import Path

import numpy as np

from othello_eval.ntuple import D4

BOARD = 8
CHANNELS = 16
BLOCKS = 2
VALUE_HIDDEN = 32
MAGIC = b"OTCNN001"
VERSION = 1
# magic, version, rows, cols, input channels, channels, residual blocks,
# value hidden width, float count
HEADER = struct.Struct("<8s8I")


def _count() -> int:
    stem = CHANNELS * 2 * 3 * 3 + CHANNELS
    blocks = BLOCKS * 2 * (CHANNELS * CHANNELS * 3 * 3 + CHANNELS)
    value = CHANNELS + 1 + BOARD * BOARD * VALUE_HIDDEN + VALUE_HIDDEN + VALUE_HIDDEN + 1
    return stem + blocks + value


N_WEIGHTS = _count()
_BIAS_PARAMETER_INDICES = (1, 3, 5, 7, 9, 11)


def _unpack(w: np.ndarray) -> list[np.ndarray]:
    w = np.asarray(w, dtype=np.float32)
    if w.shape != (N_WEIGHTS,):
        raise ValueError(f"expected {N_WEIGHTS} OTCNN001 weights, got {w.shape}")
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
            take((BOARD * BOARD, VALUE_HIDDEN)), take((VALUE_HIDDEN,)),
            take((VALUE_HIDDEN,)), take((1,)),
        )
    )
    assert at == N_WEIGHTS
    return out


def initial_weights(seed: int) -> np.ndarray:
    """Deterministic initializer with constants only on actual biases."""
    rng = np.random.default_rng(seed)
    weights = (rng.standard_normal(N_WEIGHTS) * 0.03).astype(np.float32)
    parameters = _unpack(weights)
    for index in _BIAS_PARAMETER_INDICES:
        parameters[index].fill(0.05)
    return weights


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


def _conv_backward(
    x: np.ndarray, w: np.ndarray, grad: np.ndarray, padding: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _n, _channels, rows, cols = x.shape
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    for kr in range(w.shape[2]):
        for kc in range(w.shape[3]):
            src_r = slice(max(0, kr - padding), min(rows, rows + kr - padding))
            src_c = slice(max(0, kc - padding), min(cols, cols + kc - padding))
            dst_r = slice(max(0, padding - kr), min(rows, rows + padding - kr))
            dst_c = slice(max(0, padding - kc), min(cols, cols + padding - kc))
            source, output = x[:, :, src_r, src_c], grad[:, :, dst_r, dst_c]
            dw[:, :, kr, kc] = np.einsum("norc,nirc->oi", output, source)
            dx[:, :, src_r, src_c] += np.einsum("norc,oi->nirc", output, w[:, :, kr, kc])
    return dx, dw, grad.sum(axis=(0, 2, 3))


def _planes(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """``(N, 2, 8, 8)`` occupancy planes from ``(N, 64)`` 0/1 arrays."""
    if me.shape != opp.shape or me.ndim != 2 or me.shape[1] != BOARD * BOARD:
        raise ValueError(f"expected matching (N, 64) planes, got {me.shape} and {opp.shape}")
    return np.stack((me, opp), axis=1).reshape((-1, 2, BOARD, BOARD))


def _predict_literal(weights: np.ndarray, me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Value only, single (literal) orientation -- no D4 averaging."""
    p = _unpack(weights)
    x = np.maximum(_conv(_planes(me, opp), p[0], p[1], 1), 0.0)
    at = 2
    for _ in range(BLOCKS):
        residual = x
        x = np.maximum(_conv(x, p[at], p[at + 1], 1), 0.0)
        x = np.maximum(_conv(x, p[at + 2], p[at + 3], 1) + residual, 0.0)
        at += 4
    value = np.maximum(_conv(x, p[at], p[at + 1], 0), 0.0).reshape((-1, BOARD * BOARD))
    value = np.maximum(value @ p[at + 2] + p[at + 3], 0.0)
    return np.tanh(value @ p[at + 4] + p[at + 5][0]).astype(np.float32)


def me_opp_planes(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 64)`` float32 0/1 occupancy planes from dump position records."""
    black = positions["black"].astype(np.uint64)
    white = positions["white"].astype(np.uint64)
    side0 = positions["side"] == 0
    me_bits = np.where(side0, black, white)
    opp_bits = np.where(side0, white, black)
    squares = np.arange(BOARD * BOARD, dtype=np.uint64)
    me = ((me_bits[:, None] >> squares[None, :]) & np.uint64(1)).astype(np.float32)
    opp = ((opp_bits[:, None] >> squares[None, :]) & np.uint64(1)).astype(np.float32)
    return me, opp


def predict(weights: np.ndarray, me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """D4-averaged value: run the literal network on all 8 D4-transformed
    copies of the board and average -- see the module docstring for why
    this, not a symmetrized training loss, carries the equivariance."""
    total = np.zeros(me.shape[0], dtype=np.float64)
    for sym in range(8):
        cols = D4[sym]
        total += _predict_literal(weights, me[:, cols], opp[:, cols]).astype(np.float64)
    return (total / 8.0).astype(np.float32)


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(prediction) < 2 or np.std(prediction) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def validation_metrics(weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray) -> dict[str, float]:
    prediction = predict(weights, me, opp)
    nonzero = value != 0.0
    metrics = {
        "value_mse": float(np.mean((prediction - value) ** 2)),
        "value_pearson": _pearson(prediction, value),
        "value_sign_agreement": float(
            np.mean(np.sign(prediction[nonzero]) == np.sign(value[nonzero]))
        ) if np.any(nonzero) else 0.0,
    }
    return {name: m if np.isfinite(m) else 0.0 for name, m in metrics.items()}


def _literal_loss_gradient(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray, l2: float,
) -> tuple[float, np.ndarray]:
    """Value MSE (literal orientation only) plus its dense gradient."""
    p = _unpack(weights)
    x0 = _planes(me, opp)
    z0 = _conv(x0, p[0], p[1], 1); x = np.maximum(z0, 0.0)
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    at = 2
    for _ in range(BLOCKS):
        residual = x
        z1 = _conv(x, p[at], p[at + 1], 1); h1 = np.maximum(z1, 0.0)
        z2 = _conv(h1, p[at + 2], p[at + 3], 1); x = np.maximum(z2 + residual, 0.0)
        blocks.append((residual, z1, h1, z2)); at += 4
    z_value = _conv(x, p[at], p[at + 1], 0)
    value_features = np.maximum(z_value, 0.0).reshape((-1, BOARD * BOARD))
    value_hidden_z = value_features @ p[at + 2] + p[at + 3]
    value_hidden = np.maximum(value_hidden_z, 0.0)
    value_score = value_hidden @ p[at + 4] + p[at + 5][0]
    value_prediction = np.tanh(value_score)
    n = len(me)
    value_loss = np.mean((value_prediction - value) ** 2)

    gradient = np.zeros_like(weights)
    gp = _unpack(gradient)
    dv = (2.0 / n) * (value_prediction - value) * (1.0 - value_prediction**2)
    gp[at + 4][:] = value_hidden.T @ dv
    gp[at + 5][0] = dv.sum()
    d_hidden = (dv[:, None] * p[at + 4]) * (value_hidden_z > 0.0)
    gp[at + 2][:] = value_features.T @ d_hidden
    gp[at + 3][:] = d_hidden.sum(axis=0)
    d_value_features = d_hidden @ p[at + 2].T
    d_z_value = d_value_features.reshape(z_value.shape) * (z_value > 0.0)
    dx, d_value_weight, d_value_bias = _conv_backward(x, p[at], d_z_value, 0)
    gp[at][:] = d_value_weight
    gp[at + 1][:] = d_value_bias
    for block in range(BLOCKS - 1, -1, -1):
        residual, z1, h1, z2 = blocks[block]
        d_z2 = dx * (z2 + residual > 0.0)
        block_at = 2 + block * 4
        d_h1, d_second_weight, d_second_bias = _conv_backward(h1, p[block_at + 2], d_z2, 1)
        gp[block_at + 2][:] = d_second_weight
        gp[block_at + 3][:] = d_second_bias
        d_z1 = d_h1 * (z1 > 0.0)
        dx_branch, d_first_weight, d_first_bias = _conv_backward(residual, p[block_at], d_z1, 1)
        gp[block_at][:] = d_first_weight
        gp[block_at + 1][:] = d_first_bias
        dx = d_z2 + dx_branch
    d_z0 = dx * (z0 > 0.0)
    _, d_stem_weight, d_stem_bias = _conv_backward(x0, p[0], d_z0, 1)
    gp[0][:] = d_stem_weight
    gp[1][:] = d_stem_bias
    regularized = [0, 2, 4, 6, 8, at, at + 2, at + 4]
    reg = sum(float(np.dot(p[i].ravel(), p[i].ravel())) for i in regularized)
    for i in regularized:
        gp[i][:] += 2.0 * l2 * p[i]
    return float(value_loss + l2 * reg), gradient


def _epoch_batches(rng: np.random.Generator, row_count: int, batch_size: int) -> tuple[np.ndarray, ...]:
    order = rng.permutation(row_count)
    return tuple(order[start : start + batch_size] for start in range(0, row_count, batch_size))


def fit(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray], l2: float = 1e-4,
    *, seed: int = 0, batch_size: int = 256, epochs: int = 24, learning_rate: float = 2e-3,
    validate_every: int = 1, report_every: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Deterministic Adam fit of the literal-orientation value network.

    ``validate_every`` skips the (D4-averaged, 8x-cost) validation pass on
    epochs not a multiple of it, purely to cut wall time on a large
    validation set; it never affects the weights fit stops with.

    ``report_every``, when non-zero, prints one line every that many epochs
    (and on the last epoch) -- this loop is slow enough on a real corpus
    that it must not run silent for tens of minutes with no sign of life.
    """
    if not len(me):
        raise ValueError("CNN fitting requires non-empty rows")
    rng = np.random.default_rng(seed)
    weights = initial_weights(seed)
    moment, velocity = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, step = 0.9, 0.999, 0
    started = time.perf_counter()
    vm, vo, vv = validation
    validation_epoch_trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        for batch in _epoch_batches(rng, len(me), batch_size):
            _, gradient = _literal_loss_gradient(weights, me[batch], opp[batch], value[batch], l2)
            step += 1
            moment = beta1 * moment + (1.0 - beta1) * gradient
            velocity = beta2 * velocity + (1.0 - beta2) * gradient * gradient
            weights -= learning_rate * (moment / (1.0 - beta1**step)) / (np.sqrt(velocity / (1.0 - beta2**step)) + 1e-8)
        if epoch % validate_every == 0 or epoch == epochs:
            validation_epoch_trace.append(validation_metrics(weights, vm, vo, vv))
            if report_every and (epoch % report_every == 0 or epoch == epochs):
                m = validation_epoch_trace[-1]
                elapsed = time.perf_counter() - started
                print(
                    f"  epoch {epoch:4d}  val mse {m['value_mse']:.4f}  "
                    f"pearson {m['value_pearson']:.4f}  sign-acc {m['value_sign_agreement']:.4f}  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
    train_metrics = validation_metrics(weights, me, opp, value)
    metadata: dict[str, object] = {
        "optimizer": "adam_literal_value_mse",
        "optimizer_seed": seed, "optimizer_batch_size": batch_size, "optimizer_epochs": epochs,
        "optimizer_learning_rate": learning_rate, "optimizer_steps": step,
        "fit_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
        "train_metrics": train_metrics,
        "validation_epoch_trace": validation_epoch_trace,
        "final_validation_metrics": validation_epoch_trace[-1],
    }
    return weights, metadata


def write_weights(path: str, weights: np.ndarray) -> None:
    weights = np.asarray(weights, dtype="<f4")
    _unpack(weights)
    Path(path).write_bytes(
        HEADER.pack(MAGIC, VERSION, BOARD, BOARD, 2, CHANNELS, BLOCKS, VALUE_HIDDEN, N_WEIGHTS)
        + weights.tobytes()
    )


def read_weights(path: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: missing OTCNN001 header")
    header = HEADER.unpack(raw[: HEADER.size])
    expected = (MAGIC, VERSION, BOARD, BOARD, 2, CHANNELS, BLOCKS, VALUE_HIDDEN, N_WEIGHTS)
    if header != expected or len(raw) != HEADER.size + N_WEIGHTS * 4:
        raise ValueError(f"{path}: unsupported OTCNN001 layout")
    weights = np.frombuffer(raw[HEADER.size :], dtype="<f4").copy()
    _unpack(weights)
    return weights
