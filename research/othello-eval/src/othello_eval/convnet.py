# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnusedVariable=false
# ruff: noqa: E501, E702
"""Versioned compact Othello convolutional value+policy network.

``OTCNN001`` (version 2) stores a concrete two-plane 8x8 model: a 3x3 stem
with 16 channels, two 16-channel residual blocks, then separate value and
policy heads sharing that trunk -- the direct 8x8 generalization of Connect
Four's ``C4CNN001`` (``research/az-train/src/az_train/convnet_c4.py`` /
``games/connect4/src/convnet.rs``), which also shares one trunk between both
heads. Version 1 was value-only; the policy head was deferred until a
self-play loop existed to produce completed-Q targets to train it against
(the same value-then-policy sequencing the n-tuple port used) -- see
``games/othello/src/policy.rs``'s D4-equivariant linear policy sidecar for
the interface this head's D4-averaging and pass-as-mean-logit convention
matches.

Training uses the board's single natural (literal) orientation only for
*both* heads, the same choice ``convnet_c4``'s main fit path and this
module's own value-only predecessor make; equivariance is instead a property
of :func:`predict`, which averages the literal network's output over all 8
D4-transformed copies of the input board and, for the policy head, maps each
orientation's canonical-frame logits back to real board squares via the
inverse permutation (mirroring ``othello_eval.policy``'s ``INV`` table). This
is cheaper than folding D4-averaging into the training loss and matches the
already-accepted C4/value precedent for a network with real per-orientation
compute cost.

``Move::PASS`` has no board square. Following ``games/othello/src/
policy.rs``/``az_train.policy_othello``'s convention, the *training* loss
treats PASS as logit-space column 64 (``COLUMNS = 65``) whose logit is the
mean of the 64 real-square logits, so its gradient is spread evenly back
across all 64 columns during backprop. Inference-time callers that only need
per-square logits (e.g. cross-language fixtures) get the 64-column D4-averaged
array from :func:`predict` and can take the mean themselves for PASS, exactly
as ``games/othello/src/policy.rs::NTuplePolicyNet::logits`` does.
"""

from __future__ import annotations

import math
import resource
import struct
import sys
import time
from pathlib import Path

import numpy as np

from othello_eval.ntuple import D4
from othello_eval.policy import INV

BOARD = 8
CHANNELS = 16
BLOCKS = 2
VALUE_HIDDEN = 32
POLICY_OUTPUTS = 64
SQUARES = 64
COLUMNS = SQUARES + 1  # 64 real squares + PASS
MAGIC = b"OTCNN001"
VERSION = 2
# magic, version, rows, cols, input channels, channels, residual blocks,
# value hidden width, policy outputs, float count
HEADER = struct.Struct("<8s9I")


def _count() -> int:
    stem = CHANNELS * 2 * 3 * 3 + CHANNELS
    blocks = BLOCKS * 2 * (CHANNELS * CHANNELS * 3 * 3 + CHANNELS)
    value = CHANNELS + 1 + BOARD * BOARD * VALUE_HIDDEN + VALUE_HIDDEN + VALUE_HIDDEN + 1
    policy = CHANNELS + 1 + BOARD * BOARD * POLICY_OUTPUTS + POLICY_OUTPUTS
    return stem + blocks + value + policy


N_WEIGHTS = _count()
_BIAS_PARAMETER_INDICES = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


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
            take((1, CHANNELS, 1, 1)), take((1,)),
            take((BOARD * BOARD, POLICY_OUTPUTS)), take((POLICY_OUTPUTS,)),
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


def _predict_literal(weights: np.ndarray, me: np.ndarray, opp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Value and 64-column policy logits, single (literal) orientation -- no
    D4 averaging, no PASS column."""
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
    value = np.tanh(value @ p[at + 4] + p[at + 5][0]).astype(np.float32)
    at += 6
    policy_features = np.maximum(_conv(x, p[at], p[at + 1], 0), 0.0).reshape((-1, BOARD * BOARD))
    policy = (policy_features @ p[at + 2] + p[at + 3]).astype(np.float32)
    return value, policy


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


def predict(weights: np.ndarray, me: np.ndarray, opp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D4-averaged value and 64-column policy logits: run the literal network
    on all 8 D4-transformed copies of the board, average the value directly,
    and map each orientation's canonical-frame policy logits back to real
    board squares via ``INV`` before averaging -- see the module docstring
    for why this, not a symmetrized training loss, carries the equivariance.
    """
    value_total = np.zeros(me.shape[0], dtype=np.float64)
    policy_total = np.zeros((me.shape[0], SQUARES), dtype=np.float64)
    for sym in range(8):
        cols = D4[sym]
        v, logits = _predict_literal(weights, me[:, cols], opp[:, cols])
        value_total += v.astype(np.float64)
        policy_total += logits[:, INV[sym]].astype(np.float64)
    return (value_total / 8.0).astype(np.float32), (policy_total / 8.0).astype(np.float32)


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(prediction) < 2 or np.std(prediction) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def _with_pass(policy64: np.ndarray) -> np.ndarray:
    """Append the PASS column (mean of the 64 real-square columns)."""
    return np.concatenate([policy64, policy64.mean(axis=1, keepdims=True)], axis=1)


def _masked_policy_cross_entropy(policy64: np.ndarray, target: np.ndarray, legal: np.ndarray) -> float:
    """Finite legal-column (65 = 64 squares + PASS) cross entropy."""
    logits = _with_pass(policy64)
    masked = np.where(legal, logits, -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    probability = np.exp(shifted) * legal
    probability /= probability.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(target * np.log(np.maximum(probability, 1e-30)), axis=1)))


def validation_metrics(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    policy: np.ndarray | None = None, legal: np.ndarray | None = None,
) -> dict[str, float]:
    prediction, logits = predict(weights, me, opp)
    nonzero = value != 0.0
    metrics = {
        "value_mse": float(np.mean((prediction - value) ** 2)),
        "value_pearson": _pearson(prediction, value),
        "value_sign_agreement": float(
            np.mean(np.sign(prediction[nonzero]) == np.sign(value[nonzero]))
        ) if np.any(nonzero) else 0.0,
    }
    if policy is not None and legal is not None:
        metrics["masked_policy_cross_entropy"] = _masked_policy_cross_entropy(logits, policy, legal)
    return {name: m if np.isfinite(m) else 0.0 for name, m in metrics.items()}


def _literal_loss_gradient(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray, l2: float,
    policy: np.ndarray | None = None, legal: np.ndarray | None = None,
    *, value_loss_weight: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Value MSE (literal orientation only), plus legal-column policy cross
    entropy when ``policy``/``legal`` (dense ``(N, 65)`` arrays, PASS as
    column 64) are given, plus L2, and the dense gradient of all of it.

    Passing ``policy=None`` (the value-only v1 path) skips the policy head's
    contribution entirely -- it still gets an L2 gradient like any other
    parameter group, so an all-zero policy head stays exactly zero under
    value-only fitting rather than drifting from regularization alone.
    """
    if not np.isfinite(value_loss_weight) or value_loss_weight < 0.0:
        raise ValueError("value_loss_weight must be finite and non-negative")
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
    value_at = at; at += 6
    n = len(me)
    value_loss = np.mean((value_prediction - value) ** 2)

    gradient = np.zeros_like(weights)
    gp = _unpack(gradient)
    dv = value_loss_weight * (2.0 / n) * (value_prediction - value) * (1.0 - value_prediction**2)
    gp[value_at + 4][:] = value_hidden.T @ dv
    gp[value_at + 5][0] = dv.sum()
    d_hidden = (dv[:, None] * p[value_at + 4]) * (value_hidden_z > 0.0)
    gp[value_at + 2][:] = value_features.T @ d_hidden
    gp[value_at + 3][:] = d_hidden.sum(axis=0)
    d_value_features = d_hidden @ p[value_at + 2].T
    d_z_value = d_value_features.reshape(z_value.shape) * (z_value > 0.0)
    dx_value, d_value_weight, d_value_bias = _conv_backward(x, p[value_at], d_z_value, 0)
    gp[value_at][:] = d_value_weight
    gp[value_at + 1][:] = d_value_bias

    z_policy = _conv(x, p[at], p[at + 1], 0)
    policy_features = np.maximum(z_policy, 0.0).reshape((-1, BOARD * BOARD))
    policy_logits = policy_features @ p[at + 2] + p[at + 3]
    dx_policy = np.zeros_like(x)
    policy_loss = 0.0
    if policy is not None and legal is not None:
        logits65 = _with_pass(policy_logits)
        masked = np.where(legal, logits65, -np.inf)
        shifted = masked - np.max(masked, axis=1, keepdims=True)
        probability = np.exp(shifted) * legal
        probability /= probability.sum(axis=1, keepdims=True)
        row_log_likelihood = np.sum(policy * np.log(np.maximum(probability, 1e-30)), axis=1)
        policy_loss = float(-np.mean(row_log_likelihood))
        dp65 = (probability - policy) / n
        # PASS (column 64) has no board square; its gradient is the mean of
        # the 64 real-square logits, so its adjoint spreads dp65's PASS
        # column evenly back across all 64 policy_logits columns.
        dp = dp65[:, :SQUARES] + dp65[:, SQUARES : SQUARES + 1] / SQUARES
        gp[at + 2][:] = policy_features.T @ dp
        gp[at + 3][:] = dp.sum(axis=0)
        d_policy_features = dp @ p[at + 2].T
        d_z_policy = d_policy_features.reshape(z_policy.shape) * (z_policy > 0.0)
        dx_policy, d_policy_weight, d_policy_bias = _conv_backward(x, p[at], d_z_policy, 0)
        gp[at][:] = d_policy_weight
        gp[at + 1][:] = d_policy_bias

    dx = dx_value + dx_policy
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
    regularized = [0, 2, 4, 6, 8, value_at, value_at + 2, value_at + 4, at, at + 2]
    reg = sum(float(np.dot(p[i].ravel(), p[i].ravel())) for i in regularized)
    for i in regularized:
        gp[i][:] += 2.0 * l2 * p[i]
    return float(value_loss_weight * value_loss + policy_loss + l2 * reg), gradient


def _epoch_batches(rng: np.random.Generator, row_count: int, batch_size: int) -> tuple[np.ndarray, ...]:
    order = rng.permutation(row_count)
    return tuple(order[start : start + batch_size] for start in range(0, row_count, batch_size))


def fit(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray], l2: float = 1e-4,
    *, seed: int = 0, batch_size: int = 256, epochs: int = 24, learning_rate: float = 2e-3,
    validate_every: int = 1, report_every: int = 0,
    policy: np.ndarray | None = None, legal: np.ndarray | None = None,
    validation_policy: np.ndarray | None = None, validation_legal: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Deterministic Adam fit of the literal-orientation value(+policy) network.

    ``policy``/``legal`` (dense ``(N, 65)`` arrays, PASS as column 64), when
    both given, add the policy cross-entropy term to the loss alongside value
    MSE; omitting them fits the value head only (the v1 behaviour), leaving
    the policy head's weights at their (zero-drifting-under-L2) initializer.

    ``validate_every`` skips the (D4-averaged, 8x-cost) validation pass on
    epochs not a multiple of it, purely to cut wall time on a large
    validation set; it never affects the weights fit stops with.

    ``report_every``, when non-zero, prints one line every that many epochs
    (and on the last epoch) -- this loop is slow enough on a real corpus
    that it must not run silent for tens of minutes with no sign of life.
    """
    if not len(me):
        raise ValueError("CNN fitting requires non-empty rows")
    has_policy = policy is not None and legal is not None
    rng = np.random.default_rng(seed)
    weights = initial_weights(seed)
    moment, velocity = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, step = 0.9, 0.999, 0
    started = time.perf_counter()
    vm, vo, vv = validation
    validation_epoch_trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        for batch in _epoch_batches(rng, len(me), batch_size):
            batch_policy = policy[batch] if has_policy and policy is not None else None
            batch_legal = legal[batch] if has_policy and legal is not None else None
            _, gradient = _literal_loss_gradient(
                weights, me[batch], opp[batch], value[batch], l2, batch_policy, batch_legal,
            )
            step += 1
            moment = beta1 * moment + (1.0 - beta1) * gradient
            velocity = beta2 * velocity + (1.0 - beta2) * gradient * gradient
            weights -= learning_rate * (moment / (1.0 - beta1**step)) / (np.sqrt(velocity / (1.0 - beta2**step)) + 1e-8)
        if epoch % validate_every == 0 or epoch == epochs:
            validation_epoch_trace.append(
                validation_metrics(weights, vm, vo, vv, validation_policy, validation_legal)
            )
            if report_every and (epoch % report_every == 0 or epoch == epochs):
                m = validation_epoch_trace[-1]
                elapsed = time.perf_counter() - started
                extra = f"  policy ce {m['masked_policy_cross_entropy']:.4f}" if "masked_policy_cross_entropy" in m else ""
                print(
                    f"  epoch {epoch:4d}  val mse {m['value_mse']:.4f}  "
                    f"pearson {m['value_pearson']:.4f}  sign-acc {m['value_sign_agreement']:.4f}"
                    f"{extra}  ({elapsed:.1f}s)",
                    flush=True,
                )
    train_metrics = validation_metrics(weights, me, opp, value, policy, legal)
    metadata: dict[str, object] = {
        "optimizer": "adam_literal_value_mse" + ("_policy_ce" if has_policy else ""),
        "optimizer_seed": seed, "optimizer_batch_size": batch_size, "optimizer_epochs": epochs,
        "optimizer_learning_rate": learning_rate, "optimizer_steps": step,
        "fit_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
        "train_metrics": train_metrics,
        "validation_epoch_trace": validation_epoch_trace,
        "final_validation_metrics": validation_epoch_trace[-1],
    }
    return weights, metadata


def _block_param_count(channels: int = CHANNELS) -> int:
    return 2 * (channels * channels * 3 * 3 + channels)


def n_weights_for(
    blocks: int, tied: bool, channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> int:
    """Weight count for a ``blocks``-residual-block trunk, either
    independently-parameterized (``tied=False``, ``blocks`` distinct block
    weight sets) or weight-tied (``tied=True``, one block's weights reused
    ``blocks`` times), with a ``channels``-wide trunk and a ``value_hidden``-
    wide value dense layer -- otherwise the same stem/value/policy head
    shapes as ``OTCNN001``. Policy output width stays fixed at
    ``POLICY_OUTPUTS`` (one column per board square) regardless of trunk
    capacity. ``n_weights_for(BLOCKS, False) == N_WEIGHTS``."""
    stem = channels * 2 * 3 * 3 + channels
    block_total = _block_param_count(channels) if tied else blocks * _block_param_count(channels)
    value = channels + 1 + BOARD * BOARD * value_hidden + value_hidden + value_hidden + 1
    policy = channels + 1 + BOARD * BOARD * POLICY_OUTPUTS + POLICY_OUTPUTS
    return stem + block_total + value + policy


def _unpack_k(
    w: np.ndarray, blocks: int, tied: bool, channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> list[np.ndarray]:
    """Same layout ``_unpack`` uses, generalized to ``blocks`` residual
    blocks that are either each independently parameterized or all sharing
    one block's weights (``tied``). Every tensor list produced this way
    alternates (weight, bias) pairs start to finish, the same property
    ``_unpack``'s own ``_BIAS_PARAMETER_INDICES``/L2 list relies on, so the
    generic gradient/L2 code below can use ``range(1, len(p), 2)`` /
    ``range(0, len(p), 2)`` instead of a hardcoded index list."""
    w = np.asarray(w, dtype=np.float32)
    expected = n_weights_for(blocks, tied, channels, value_hidden)
    if w.shape != (expected,):
        raise ValueError(
            f"expected {expected} weights for blocks={blocks} tied={tied} "
            f"channels={channels} value_hidden={value_hidden}, got {w.shape}"
        )
    at = 0

    def take(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal at
        n = int(np.prod(shape))
        out = w[at : at + n].reshape(shape)
        at += n
        return out

    out = [take((channels, 2, 3, 3)), take((channels,))]
    n_block_sets = 1 if tied else blocks
    for _ in range(n_block_sets * 2):
        out.extend((take((channels, channels, 3, 3)), take((channels,))))
    out.extend(
        (
            take((1, channels, 1, 1)), take((1,)),
            take((BOARD * BOARD, value_hidden)), take((value_hidden,)),
            take((value_hidden,)), take((1,)),
            take((1, channels, 1, 1)), take((1,)),
            take((BOARD * BOARD, POLICY_OUTPUTS)), take((POLICY_OUTPUTS,)),
        )
    )
    assert at == expected
    return out


def initial_weights_k(
    seed: int, blocks: int, tied: bool, channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> np.ndarray:
    """Deterministic initializer, generalized from :func:`initial_weights`:
    constants only on actual biases, found generically as the odd-indexed
    tensors of :func:`_unpack_k`'s output."""
    rng = np.random.default_rng(seed)
    weights = (rng.standard_normal(n_weights_for(blocks, tied, channels, value_hidden)) * 0.03).astype(np.float32)
    parameters = _unpack_k(weights, blocks, tied, channels, value_hidden)
    for index in range(1, len(parameters), 2):
        parameters[index].fill(0.05)
    return weights


def _fan_in(tensor: np.ndarray) -> int:
    """Fan-in of a weight tensor in :func:`_unpack_k`'s layout: conv weights
    are ``(out_channels, in_channels, kh, kw)`` (fan-in = the trailing dims'
    product); dense weights are stored ``(in_features, out_features)`` --
    or, when ``out_features == 1``, squeezed to a 1-D ``(in_features,)``
    vector -- so fan-in is the leading dimension either way."""
    return int(np.prod(tensor.shape[1:])) if tensor.ndim == 4 else int(tensor.shape[0])


def kaiming_weights_k(
    seed: int, blocks: int, tied: bool, channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
    bias: float = 0.0,
) -> np.ndarray:
    """Kaiming/He-normal alternative to :func:`initial_weights_k`: unlike
    that function's single fixed 0.03 standard deviation applied to every
    weight tensor regardless of size, each tensor here is drawn with std
    ``sqrt(2 / fan_in)`` -- the standard scaling for ReLU networks, derived
    to keep forward-pass activation variance roughly constant layer to layer
    regardless of a layer's width or the stack's depth, unlike a fixed std
    applied uniformly to every tensor (whose fit to a ReLU stack's actual
    signal propagation only holds for one particular width/depth
    combination and drifts arbitrarily far from that for any other).
    ``bias`` (default ``0.0``, the standard Kaiming pairing) sets every
    bias tensor to that fixed value -- exposed so a caller can isolate the
    weight-scaling change from the bias-init change (e.g. ``bias=0.05`` to
    match ``initial_weights_k``'s own bias constant while keeping fan-in
    scaling), which the default reproduces neither silently nor by
    accident: it stays a second, deliberate deviation from
    ``initial_weights_k``, not conflated with the init-scale change unless a
    caller explicitly asks for that combination. A wholly separate
    function: ``initial_weights_k`` and every existing caller of it are
    untouched."""
    rng = np.random.default_rng(seed)
    weights = np.zeros(n_weights_for(blocks, tied, channels, value_hidden), dtype=np.float32)
    parameters = _unpack_k(weights, blocks, tied, channels, value_hidden)
    for index in range(0, len(parameters), 2):
        tensor = parameters[index]
        std = math.sqrt(2.0 / _fan_in(tensor))
        tensor[...] = rng.standard_normal(tensor.shape).astype(np.float32) * std
    for index in range(1, len(parameters), 2):
        parameters[index].fill(bias)
    return weights


def _orthogonal(rng: np.random.Generator, shape: tuple[int, ...], gain: float) -> np.ndarray:
    """Saxe/Sussillo/Ganguli orthogonal init, matching ``torch.nn.init.
    orthogonal_``'s construction exactly: draw a random Gaussian matrix of
    the tensor's flattened ``(rows, cols)`` shape (``rows`` the leading/
    fan-out dimension, ``cols`` the flattened remainder), take the ``Q``
    factor of its QR decomposition, sign-correct against ``R``'s diagonal so
    the distribution isn't biased toward a particular orientation, scale by
    ``gain``, and reshape back to the tensor's real shape. Only
    ``min(rows, cols)`` vectors can be exactly mutually orthonormal in a
    space of the other dimension's size, so whichever side is smaller is the
    one QR is asked to orthogonalize -- built by transposing before the QR
    (and back after) whenever ``rows < cols``, the standard trick for that
    case."""
    rows = shape[0]
    cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    transpose = rows < cols
    a = rng.standard_normal((cols, rows) if transpose else (rows, cols))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if transpose:
        q = q.T
    return (gain * q).reshape(shape).astype(np.float32)


def orthogonal_weights_k(
    seed: int, blocks: int, tied: bool, channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
    gain: float = math.sqrt(2.0), bias: float = 0.0,
) -> np.ndarray:
    """Orthogonal alternative to :func:`initial_weights_k`/
    :func:`kaiming_weights_k`: every weight tensor's rows (or columns, for a
    "tall" tensor) are exactly orthogonal rather than independently
    Gaussian, which can stabilize deep/tied stacks better than fan-in
    scaling alone since it preserves a transform's singular-value spectrum
    exactly rather than only in expectation -- a repeated (tied) transform's
    spectrum compounds across every application, so exact orthogonality
    matters more for those stacks than for an untied one. Default ``gain``
    is ``sqrt(2)``, the standard ReLU
    correction (Saxe et al.'s original construction targets a linear/tanh
    network; ReLU halves the signal on average, so the same ``sqrt(2)``
    correction Kaiming uses applies here too). ``bias`` (default ``0.0``,
    same pairing as :func:`kaiming_weights_k`) sets every bias tensor to
    that fixed value -- see :func:`kaiming_weights_k`'s own ``bias``
    parameter for why this is exposed rather than hardcoded. A wholly
    separate function -- ``initial_weights_k`` is untouched."""
    rng = np.random.default_rng(seed)
    weights = np.zeros(n_weights_for(blocks, tied, channels, value_hidden), dtype=np.float32)
    parameters = _unpack_k(weights, blocks, tied, channels, value_hidden)
    for index in range(0, len(parameters), 2):
        tensor = parameters[index]
        tensor[...] = _orthogonal(rng, tensor.shape, gain)
    for index in range(1, len(parameters), 2):
        parameters[index].fill(bias)
    return weights


def kaiming_bias05_weights_k(
    seed: int, blocks: int, tied: bool, channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> np.ndarray:
    """:func:`kaiming_weights_k` with ``bias=0.05`` -- isolates the
    fan-in-scaled weight std from the zero-bias pairing by keeping
    ``initial_weights_k``'s own bias constant instead. Same 5-positional-arg
    signature as every other ``*_weights_k`` initializer so it plugs into
    ``az_train.convnet_othello_torch.INIT_FUNCTIONS`` unchanged."""
    return kaiming_weights_k(seed, blocks, tied, channels, value_hidden, bias=0.05)


def orthogonal_bias05_weights_k(
    seed: int, blocks: int, tied: bool, channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> np.ndarray:
    """:func:`orthogonal_weights_k` with ``bias=0.05`` -- isolates exact
    orthogonality from the zero-bias pairing, same reasoning as
    :func:`kaiming_bias05_weights_k`."""
    return orthogonal_weights_k(seed, blocks, tied, channels, value_hidden, bias=0.05)


def _predict_literal_k(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, blocks: int, tied: bool,
    channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_predict_literal`, generalized to ``blocks`` residual blocks,
    tied or not. When ``tied``, every block iteration reads the same weight
    slice (``bi`` is constant across the loop) instead of advancing to a new
    slice each time."""
    p = _unpack_k(weights, blocks, tied, channels, value_hidden)
    x = np.maximum(_conv(_planes(me, opp), p[0], p[1], 1), 0.0)
    at = 2
    for i in range(blocks):
        bi = at if tied else at + i * 4
        residual = x
        x = np.maximum(_conv(x, p[bi], p[bi + 1], 1), 0.0)
        x = np.maximum(_conv(x, p[bi + 2], p[bi + 3], 1) + residual, 0.0)
    n_block_sets = 1 if tied else blocks
    at = 2 + n_block_sets * 4
    value = np.maximum(_conv(x, p[at], p[at + 1], 0), 0.0).reshape((-1, BOARD * BOARD))
    value = np.maximum(value @ p[at + 2] + p[at + 3], 0.0)
    value = np.tanh(value @ p[at + 4] + p[at + 5][0]).astype(np.float32)
    at += 6
    policy_features = np.maximum(_conv(x, p[at], p[at + 1], 0), 0.0).reshape((-1, BOARD * BOARD))
    policy = (policy_features @ p[at + 2] + p[at + 3]).astype(np.float32)
    return value, policy


def predict_k(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, blocks: int, tied: bool,
    channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`predict`, generalized to ``blocks``/``tied``: same D4-averaging
    scheme, over :func:`_predict_literal_k` instead of the fixed-``BLOCKS``
    literal forward pass."""
    value_total = np.zeros(me.shape[0], dtype=np.float64)
    policy_total = np.zeros((me.shape[0], SQUARES), dtype=np.float64)
    for sym in range(8):
        cols = D4[sym]
        v, logits = _predict_literal_k(weights, me[:, cols], opp[:, cols], blocks, tied, channels, value_hidden)
        value_total += v.astype(np.float64)
        policy_total += logits[:, INV[sym]].astype(np.float64)
    return (value_total / 8.0).astype(np.float32), (policy_total / 8.0).astype(np.float32)


def validation_metrics_k(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray, blocks: int, tied: bool,
    policy: np.ndarray | None = None, legal: np.ndarray | None = None,
    channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> dict[str, float]:
    prediction, logits = predict_k(weights, me, opp, blocks, tied, channels, value_hidden)
    nonzero = value != 0.0
    metrics = {
        "value_mse": float(np.mean((prediction - value) ** 2)),
        "value_pearson": _pearson(prediction, value),
        "value_sign_agreement": float(
            np.mean(np.sign(prediction[nonzero]) == np.sign(value[nonzero]))
        ) if np.any(nonzero) else 0.0,
    }
    if policy is not None and legal is not None:
        metrics["masked_policy_cross_entropy"] = _masked_policy_cross_entropy(logits, policy, legal)
    return {name: m if np.isfinite(m) else 0.0 for name, m in metrics.items()}


def _literal_loss_gradient_k(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray, l2: float,
    blocks: int, tied: bool,
    policy: np.ndarray | None = None, legal: np.ndarray | None = None,
    *, channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN, value_loss_weight: float = 1.0,
) -> tuple[float, np.ndarray]:
    """:func:`_literal_loss_gradient`, generalized to ``blocks``/``tied``.

    When ``tied``, every block iteration's forward pass reads the *same*
    weight slice ``p[bi:bi+4]`` (``bi`` does not advance with the loop
    index), the standard weight-tied/backprop-through-time forward. Each
    iteration's own activations (``residual``, ``z1``, ``h1``, ``z2``) are
    still cached separately in ``iterations``, so the backward pass below
    can replay the single shared block's existing backward computation once
    per iteration and accumulate (``+=``, not overwrite) each iteration's
    weight-gradient contribution into the one shared ``gp[bi:bi+4]`` slot --
    summing, not averaging, matching how a shared weight's total gradient is
    the sum of every place it was used. When not tied, ``bi`` is distinct
    per iteration, so the same ``+=`` accumulation degenerates to a single
    assignment per block, identical to :func:`_literal_loss_gradient`'s own
    per-block loop.
    """
    if not np.isfinite(value_loss_weight) or value_loss_weight < 0.0:
        raise ValueError("value_loss_weight must be finite and non-negative")
    p = _unpack_k(weights, blocks, tied, channels, value_hidden)
    n_block_sets = 1 if tied else blocks
    x0 = _planes(me, opp)
    z0 = _conv(x0, p[0], p[1], 1); x = np.maximum(z0, 0.0)
    iterations: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    at = 2
    for i in range(blocks):
        bi = at if tied else at + i * 4
        residual = x
        z1 = _conv(x, p[bi], p[bi + 1], 1); h1 = np.maximum(z1, 0.0)
        z2 = _conv(h1, p[bi + 2], p[bi + 3], 1); x = np.maximum(z2 + residual, 0.0)
        iterations.append((bi, residual, z1, h1, z2))
    at = 2 + n_block_sets * 4
    z_value = _conv(x, p[at], p[at + 1], 0)
    value_features = np.maximum(z_value, 0.0).reshape((-1, BOARD * BOARD))
    hidden_z = value_features @ p[at + 2] + p[at + 3]
    hidden_activation = np.maximum(hidden_z, 0.0)
    value_score = hidden_activation @ p[at + 4] + p[at + 5][0]
    value_prediction = np.tanh(value_score)
    value_at = at; at += 6
    n = len(me)
    value_loss = np.mean((value_prediction - value) ** 2)

    gradient = np.zeros_like(weights)
    gp = _unpack_k(gradient, blocks, tied, channels, value_hidden)
    dv = value_loss_weight * (2.0 / n) * (value_prediction - value) * (1.0 - value_prediction**2)
    gp[value_at + 4][:] = hidden_activation.T @ dv
    gp[value_at + 5][0] = dv.sum()
    d_hidden = (dv[:, None] * p[value_at + 4]) * (hidden_z > 0.0)
    gp[value_at + 2][:] = value_features.T @ d_hidden
    gp[value_at + 3][:] = d_hidden.sum(axis=0)
    d_value_features = d_hidden @ p[value_at + 2].T
    d_z_value = d_value_features.reshape(z_value.shape) * (z_value > 0.0)
    dx_value, d_value_weight, d_value_bias = _conv_backward(x, p[value_at], d_z_value, 0)
    gp[value_at][:] = d_value_weight
    gp[value_at + 1][:] = d_value_bias

    z_policy = _conv(x, p[at], p[at + 1], 0)
    policy_features = np.maximum(z_policy, 0.0).reshape((-1, BOARD * BOARD))
    policy_logits = policy_features @ p[at + 2] + p[at + 3]
    dx_policy = np.zeros_like(x)
    policy_loss = 0.0
    if policy is not None and legal is not None:
        logits65 = _with_pass(policy_logits)
        masked = np.where(legal, logits65, -np.inf)
        shifted = masked - np.max(masked, axis=1, keepdims=True)
        probability = np.exp(shifted) * legal
        probability /= probability.sum(axis=1, keepdims=True)
        row_log_likelihood = np.sum(policy * np.log(np.maximum(probability, 1e-30)), axis=1)
        policy_loss = float(-np.mean(row_log_likelihood))
        dp65 = (probability - policy) / n
        dp = dp65[:, :SQUARES] + dp65[:, SQUARES : SQUARES + 1] / SQUARES
        gp[at + 2][:] = policy_features.T @ dp
        gp[at + 3][:] = dp.sum(axis=0)
        d_policy_features = dp @ p[at + 2].T
        d_z_policy = d_policy_features.reshape(z_policy.shape) * (z_policy > 0.0)
        dx_policy, d_policy_weight, d_policy_bias = _conv_backward(x, p[at], d_z_policy, 0)
        gp[at][:] = d_policy_weight
        gp[at + 1][:] = d_policy_bias

    dx = dx_value + dx_policy
    for bi, residual, z1, h1, z2 in reversed(iterations):
        d_z2 = dx * (z2 + residual > 0.0)
        d_h1, d_second_weight, d_second_bias = _conv_backward(h1, p[bi + 2], d_z2, 1)
        gp[bi + 2][:] += d_second_weight
        gp[bi + 3][:] += d_second_bias
        d_z1 = d_h1 * (z1 > 0.0)
        dx_branch, d_first_weight, d_first_bias = _conv_backward(residual, p[bi], d_z1, 1)
        gp[bi][:] += d_first_weight
        gp[bi + 1][:] += d_first_bias
        dx = d_z2 + dx_branch
    d_z0 = dx * (z0 > 0.0)
    _, d_stem_weight, d_stem_bias = _conv_backward(x0, p[0], d_z0, 1)
    gp[0][:] = d_stem_weight
    gp[1][:] = d_stem_bias
    regularized = list(range(0, len(p), 2))
    reg = sum(float(np.dot(p[i].ravel(), p[i].ravel())) for i in regularized)
    for i in regularized:
        gp[i][:] += 2.0 * l2 * p[i]
    return float(value_loss_weight * value_loss + policy_loss + l2 * reg), gradient


def fit_k(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray], blocks: int, tied: bool,
    l2: float = 1e-4,
    *, seed: int = 0, batch_size: int = 256, epochs: int = 24, learning_rate: float = 2e-3,
    validate_every: int = 1, report_every: int = 0,
    policy: np.ndarray | None = None, legal: np.ndarray | None = None,
    validation_policy: np.ndarray | None = None, validation_legal: np.ndarray | None = None,
    channels: int = CHANNELS, value_hidden: int = VALUE_HIDDEN,
) -> tuple[np.ndarray, dict[str, object]]:
    """:func:`fit`, generalized to ``blocks``/``tied`` -- same Adam loop and
    metadata shape, over :func:`_literal_loss_gradient_k`/
    :func:`validation_metrics_k` instead of the fixed-``BLOCKS`` versions."""
    if not len(me):
        raise ValueError("CNN fitting requires non-empty rows")
    has_policy = policy is not None and legal is not None
    rng = np.random.default_rng(seed)
    weights = initial_weights_k(seed, blocks, tied, channels, value_hidden)
    moment, velocity = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, step = 0.9, 0.999, 0
    started = time.perf_counter()
    vm, vo, vv = validation
    validation_epoch_trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        for batch in _epoch_batches(rng, len(me), batch_size):
            batch_policy = policy[batch] if has_policy and policy is not None else None
            batch_legal = legal[batch] if has_policy and legal is not None else None
            _, gradient = _literal_loss_gradient_k(
                weights, me[batch], opp[batch], value[batch], l2, blocks, tied, batch_policy, batch_legal,
                channels=channels, value_hidden=value_hidden,
            )
            step += 1
            moment = beta1 * moment + (1.0 - beta1) * gradient
            velocity = beta2 * velocity + (1.0 - beta2) * gradient * gradient
            weights -= learning_rate * (moment / (1.0 - beta1**step)) / (np.sqrt(velocity / (1.0 - beta2**step)) + 1e-8)
        if epoch % validate_every == 0 or epoch == epochs:
            validation_epoch_trace.append(
                validation_metrics_k(
                    weights, vm, vo, vv, blocks, tied, validation_policy, validation_legal, channels, value_hidden,
                )
            )
            if report_every and (epoch % report_every == 0 or epoch == epochs):
                m = validation_epoch_trace[-1]
                elapsed = time.perf_counter() - started
                extra = f"  policy ce {m['masked_policy_cross_entropy']:.4f}" if "masked_policy_cross_entropy" in m else ""
                print(
                    f"  epoch {epoch:4d}  val mse {m['value_mse']:.4f}  "
                    f"pearson {m['value_pearson']:.4f}  sign-acc {m['value_sign_agreement']:.4f}"
                    f"{extra}  ({elapsed:.1f}s)",
                    flush=True,
                )
    train_metrics = validation_metrics_k(weights, me, opp, value, blocks, tied, policy, legal, channels, value_hidden)
    metadata: dict[str, object] = {
        "optimizer": "adam_literal_value_mse" + ("_policy_ce" if has_policy else ""),
        "optimizer_seed": seed, "optimizer_batch_size": batch_size, "optimizer_epochs": epochs,
        "optimizer_learning_rate": learning_rate, "optimizer_steps": step,
        "blocks": blocks, "tied": tied, "channels": channels, "value_hidden": value_hidden,
        "n_weights": int(n_weights_for(blocks, tied, channels, value_hidden)),
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
        HEADER.pack(MAGIC, VERSION, BOARD, BOARD, 2, CHANNELS, BLOCKS, VALUE_HIDDEN, POLICY_OUTPUTS, N_WEIGHTS)
        + weights.tobytes()
    )


def read_weights(path: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: missing OTCNN001 header")
    header = HEADER.unpack(raw[: HEADER.size])
    expected = (MAGIC, VERSION, BOARD, BOARD, 2, CHANNELS, BLOCKS, VALUE_HIDDEN, POLICY_OUTPUTS, N_WEIGHTS)
    if header != expected or len(raw) != HEADER.size + N_WEIGHTS * 4:
        raise ValueError(f"{path}: unsupported OTCNN001 layout")
    weights = np.frombuffer(raw[HEADER.size :], dtype="<f4").copy()
    _unpack(weights)
    return weights
