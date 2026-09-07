# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnusedVariable=false
# ruff: noqa: E501, E702
"""Versioned compact Connect Four convolutional value-and-policy inference.

``C4CNN001`` stores a concrete two-plane 6x7 model: a 3x3 stem with 16
channels, two 16-channel residual blocks, then separate value and policy
heads.  It is deliberately inference-only here.  Production fitting remains
restricted to self-play outcomes and completed-Q policy targets.
"""

from __future__ import annotations

import resource
import struct
import sys
import time
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


def _parameter_groups() -> dict[str, tuple[int, int]]:
    """Flat parameter spans for stable training diagnostics."""
    offsets = np.cumsum([0, *[tensor.size for tensor in _unpack(np.zeros(N_WEIGHTS, dtype=np.float32))]])
    return {
        "stem": (int(offsets[0]), int(offsets[2])),
        "residual_block_1": (int(offsets[2]), int(offsets[6])),
        "residual_block_2": (int(offsets[6]), int(offsets[10])),
        "value_head": (int(offsets[10]), int(offsets[16])),
        "policy_head": (int(offsets[16]), int(offsets[20])),
    }


def _group_l2(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.linalg.norm(values[start:end], ord=2))
        for name, (start, end) in _parameter_groups().items()
    }


def _gradient_conflict_metrics(value_gradient: np.ndarray, policy_gradient: np.ndarray) -> dict[str, float]:
    """Return finite shared-gradient norms and cosine, using zero for a zero norm."""
    value_l2 = float(np.linalg.norm(value_gradient, ord=2))
    policy_l2 = float(np.linalg.norm(policy_gradient, ord=2))
    value_l2 = value_l2 if np.isfinite(value_l2) else 0.0
    policy_l2 = policy_l2 if np.isfinite(policy_l2) else 0.0
    denominator = value_l2 * policy_l2
    cosine = float(np.dot(value_gradient, policy_gradient) / denominator) if denominator else 0.0
    return {
        "value_gradient_l2": value_l2,
        "policy_gradient_l2": policy_l2,
        "cosine_similarity": cosine if np.isfinite(cosine) else 0.0,
    }


def _aggregate_gradient_conflict(
    cosines: dict[str, list[float]],
) -> dict[str, dict[str, float | int]]:
    """Summarize every finite-neutral shared-gradient cosine by parameter group."""
    return {
        name: {
            "batch_count": len(values),
            "mean_cosine_similarity": float(np.mean(values)) if values else 0.0,
            "min_cosine_similarity": float(np.min(values)) if values else 0.0,
            "max_cosine_similarity": float(np.max(values)) if values else 0.0,
            "negative_cosine_batch_count": sum(value < 0.0 for value in values),
        }
        for name, values in cosines.items()
    }


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


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation, using 0.0 when either input lacks variation."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(prediction) < 2 or np.std(prediction) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def _masked_policy_cross_entropy(logits: np.ndarray, target: np.ndarray, legal: np.ndarray) -> float:
    """Finite legal-action cross entropy for rows with at least one legal action."""
    masked = np.where(legal, logits, -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    probability = np.exp(shifted) * legal
    probability /= probability.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(target * np.log(np.maximum(probability, 1e-30)), axis=1)))


def orientation_diagnostics(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    policy: np.ndarray, legal: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compare literal training predictions with mandatory mirror-averaged inference.

    Correlations are 0.0 for singleton or constant vectors, and sign agreement is
    0.0 when every value target is zero.  These are reporting metrics only.
    """
    literal_value, literal_logits = _predict_literal(weights, me, opp)
    mirror_me = me.reshape((-1, ROWS, COLS))[:, :, ::-1].reshape((-1, ROWS * COLS))
    mirror_opp = opp.reshape((-1, ROWS, COLS))[:, :, ::-1].reshape((-1, ROWS * COLS))
    reflected_value, reflected_logits = _predict_literal(weights, mirror_me, mirror_opp)
    reflected_logits = reflected_logits[:, ::-1]
    averaged_value = 0.5 * (literal_value + reflected_value)
    averaged_logits = 0.5 * (literal_logits + reflected_logits)
    nonzero = value != 0.0

    def value_metrics(prediction: np.ndarray) -> dict[str, float]:
        return {
            "mse": float(np.mean((prediction - value) ** 2)),
            "pearson": _pearson(prediction, value),
            "sign_agreement": float(np.mean(np.sign(prediction[nonzero]) == np.sign(value[nonzero])))
            if np.any(nonzero)
            else 0.0,
        }

    return {
        "literal": {
            **value_metrics(literal_value),
            "masked_policy_cross_entropy": _masked_policy_cross_entropy(literal_logits, policy, legal),
        },
        "mirror_averaged": {
            **value_metrics(averaged_value),
            "masked_policy_cross_entropy": _masked_policy_cross_entropy(averaged_logits, policy, legal),
        },
        "literal_vs_reflected_remapped": {
            "value_mae": float(np.mean(np.abs(literal_value - reflected_value))),
            "value_pearson": _pearson(literal_value, reflected_value),
            "policy_logit_mae": float(np.mean(np.abs(literal_logits - reflected_logits))),
        },
    }


def _conv_backward(
    x: np.ndarray, w: np.ndarray, grad: np.ndarray, padding: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient of `_conv`, retaining the deliberately small direct kernel loops."""
    n, channels, rows, cols = x.shape
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


def _literal_loss_gradient_terms(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    policy: np.ndarray, legal: np.ndarray, l2: float,
    *, value_weight: float, policy_weight: float, include_regularization: bool,
) -> tuple[float, np.ndarray]:
    """Value MSE plus legal-column policy cross entropy and its dense gradient."""
    p = _unpack(weights)
    x0 = np.stack((me, opp), axis=1).reshape((-1, 2, ROWS, COLS))
    z0 = _conv(x0, p[0], p[1], 1); x = np.maximum(z0, 0.0)
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    at = 2
    for _ in range(BLOCKS):
        residual = x
        z1 = _conv(x, p[at], p[at + 1], 1); h1 = np.maximum(z1, 0.0)
        z2 = _conv(h1, p[at + 2], p[at + 3], 1); x = np.maximum(z2 + residual, 0.0)
        blocks.append((residual, z1, h1, z2)); at += 4
    z_value = _conv(x, p[at], p[at + 1], 0); value_features = np.maximum(z_value, 0.0).reshape((-1, 42))
    value_hidden_z = value_features @ p[at + 2] + p[at + 3]
    value_hidden = np.maximum(value_hidden_z, 0.0)
    value_score = value_hidden @ p[at + 4] + p[at + 5][0]
    value_prediction = np.tanh(value_score)
    value_at = at; at += 6
    z_policy = _conv(x, p[at], p[at + 1], 0); policy_features = np.maximum(z_policy, 0.0).reshape((-1, 42))
    logits = policy_features @ p[at + 2] + p[at + 3]
    masked = np.where(legal, logits, -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    probability = np.exp(shifted) * legal
    probability /= probability.sum(axis=1, keepdims=True)
    n = len(me)
    value_loss = np.mean((value_prediction - value) ** 2)
    policy_loss = -np.mean(np.sum(policy * np.log(np.maximum(probability, 1e-30)), axis=1))
    gradient = np.zeros_like(weights)
    gp = _unpack(gradient)
    dv = value_weight * (2.0 / n) * (value_prediction - value) * (1.0 - value_prediction**2)
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
    dp = policy_weight * (probability - policy) / n
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
        dx = dx + dx_branch
    d_z0 = dx * (z0 > 0.0)
    _, d_stem_weight, d_stem_bias = _conv_backward(x0, p[0], d_z0, 1)
    gp[0][:] = d_stem_weight
    gp[1][:] = d_stem_bias
    regularized = [0, 2, 4, 6, 8, value_at, value_at + 2, value_at + 4, at, at + 2]
    reg = sum(float(np.dot(p[i].ravel(), p[i].ravel())) for i in regularized)
    if include_regularization:
        for i in regularized:
            gp[i][:] += 2.0 * l2 * p[i]
    return float(value_weight * value_loss + policy_weight * policy_loss + (l2 * reg if include_regularization else 0.0)), gradient


def _literal_loss_gradient(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    policy: np.ndarray, legal: np.ndarray, l2: float,
) -> tuple[float, np.ndarray]:
    """Value MSE plus legal-column policy cross entropy and its dense gradient."""
    return _literal_loss_gradient_terms(
        weights, me, opp, value, policy, legal, l2,
        value_weight=1.0, policy_weight=1.0, include_regularization=True,
    )


def _head_gradient_components(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    policy: np.ndarray, legal: np.ndarray, l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate value, policy, and regularization gradients without changing fitting."""
    _, value_gradient = _literal_loss_gradient_terms(
        weights, me, opp, value, policy, legal, l2,
        value_weight=1.0, policy_weight=0.0, include_regularization=False,
    )
    _, policy_gradient = _literal_loss_gradient_terms(
        weights, me, opp, value, policy, legal, l2,
        value_weight=0.0, policy_weight=1.0, include_regularization=False,
    )
    _, regularization_gradient = _literal_loss_gradient_terms(
        weights, me, opp, value, policy, legal, l2,
        value_weight=0.0, policy_weight=0.0, include_regularization=True,
    )
    return value_gradient, policy_gradient, regularization_gradient


def fit_value_policy_with_diagnostics(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray, policy: np.ndarray, legal: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], l2: float = 1e-4,
    *, seed: int = 0, batch_size: int = 64, epochs: int = 24, learning_rate: float = 2e-3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Deterministic Adam fit using only production self-play targets."""
    if not len(me) or not np.all(legal.any(axis=1)):
        raise ValueError("CNN fitting requires non-empty rows with legal policy targets")
    rng = np.random.default_rng(seed)
    weights = (rng.standard_normal(N_WEIGHTS) * 0.03).astype(np.float32)
    for tensor in _unpack(weights):
        if tensor.ndim == 1:
            tensor.fill(0.05)
    initial_weights = weights.copy()
    moment, velocity = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, step = 0.9, 0.999, 0
    first_batch_gradient_l2: dict[str, float] | None = None
    first_batch_head_gradient_conflict: dict[str, dict[str, float]] | None = None
    shared_group_spans = {
        name: span for name, span in _parameter_groups().items()
        if name in {"stem", "residual_block_1", "residual_block_2"}
    }
    shared_head_gradient_cosines = {name: [] for name in shared_group_spans}
    started = time.perf_counter()
    for _ in range(epochs):
        for start in range(0, len(me), batch_size):
            batch = rng.permutation(len(me))[start : start + batch_size]
            _, gradient = _literal_loss_gradient(weights, me[batch], opp[batch], value[batch], policy[batch], legal[batch], l2)
            value_gradient, policy_gradient, regularization_gradient = _head_gradient_components(
                weights, me[batch], opp[batch], value[batch], policy[batch], legal[batch], l2,
            )
            batch_head_gradient_conflict = {
                name: _gradient_conflict_metrics(value_gradient[start:end], policy_gradient[start:end])
                for name, (start, end) in shared_group_spans.items()
            }
            for name, conflict_metrics in batch_head_gradient_conflict.items():
                shared_head_gradient_cosines[name].append(conflict_metrics["cosine_similarity"])
            if first_batch_gradient_l2 is None:
                if not np.allclose(value_gradient + policy_gradient + regularization_gradient, gradient, rtol=0.0, atol=1e-8):
                    raise AssertionError("loss-gradient decomposition changed the joint gradient")
                first_batch_gradient_l2 = _group_l2(gradient)
                first_batch_head_gradient_conflict = batch_head_gradient_conflict
            step += 1; moment = beta1 * moment + (1.0 - beta1) * gradient; velocity = beta2 * velocity + (1.0 - beta2) * gradient * gradient
            weights -= learning_rate * (moment / (1.0 - beta1**step)) / (np.sqrt(velocity / (1.0 - beta2**step)) + 1e-8)
    vm, vo, vv, vp, vl = validation
    def metrics(a: np.ndarray, b: np.ndarray, y: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        pv, logits = predict(weights, a, b)
        return float(np.mean((pv - y) ** 2)), _masked_policy_cross_entropy(logits, target, mask)
    train_value_mse, train_policy_ce = metrics(me, opp, value, policy, legal)
    validation_value_mse, validation_policy_ce = metrics(vm, vo, vv, vp, vl)
    assert first_batch_gradient_l2 is not None
    assert first_batch_head_gradient_conflict is not None
    parameter_groups = {
        name: {
            "first_batch_gradient_l2": first_batch_gradient_l2[name],
            "initial_to_final_delta_l2": delta,
        }
        for name, delta in _group_l2(weights - initial_weights).items()
    }
    return weights, {"optimizer": "adam_value_mse_policy_ce", "optimizer_seed": seed, "optimizer_batch_size": batch_size, "optimizer_epochs": epochs, "optimizer_learning_rate": learning_rate, "optimizer_steps": step, "fit_wall_seconds": time.perf_counter() - started, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)), "parameter_groups": parameter_groups, "first_batch_shared_head_gradient_conflict": first_batch_head_gradient_conflict, "shared_head_gradient_conflict_timeline": _aggregate_gradient_conflict(shared_head_gradient_cosines), "train_value_mse": train_value_mse, "validation_value_mse": validation_value_mse, "train_policy_cross_entropy": train_policy_ce, "validation_policy_cross_entropy": validation_policy_ce, "orientation_diagnostics": {"train": orientation_diagnostics(weights, me, opp, value, policy, legal), "validation": orientation_diagnostics(weights, vm, vo, vv, vp, vl)}}


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
