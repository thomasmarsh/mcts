# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportPossiblyUnboundVariable=false
"""Sparse seven-column n-tuple policy sidecar for Connect Four."""

from __future__ import annotations

import numpy as np

from az_train.ntuple_c4 import COLS, N_WEIGHTS, active_indices

POLICY_WEIGHTS = N_WEIGHTS * COLS


def logits(weights: np.ndarray, me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Symmetry-averaged `(N, 7)` absolute-column logits."""
    w = np.asarray(weights, dtype=np.float32).reshape(N_WEIGHTS, COLS)
    direct = w[active_indices(me, opp)].sum(axis=1)
    mirror_me = np.asarray(me).reshape(-1, 6, 7)[:, :, ::-1].reshape(-1, 42)
    mirror_opp = np.asarray(opp).reshape(-1, 6, 7)[:, :, ::-1].reshape(-1, 42)
    mirrored = w[active_indices(mirror_me, mirror_opp)].sum(axis=1)[:, ::-1]
    return ((direct + mirrored) * 0.5).astype(np.float32)


def targets(policy: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((len(policy), COLS), dtype=np.float32)
    legal = np.zeros_like(target, dtype=bool)
    for row, entries in enumerate(policy):
        for col, probability in entries:
            target[row, col] = probability
            legal[row, col] = True
    if np.any(~legal.any(axis=1)):
        raise ValueError("policy training requires a target for every position")
    return target, legal


def metrics(raw: np.ndarray, target: np.ndarray, legal: np.ndarray) -> dict[str, float]:
    masked = np.where(legal, raw, -np.inf)
    log_z = np.logaddexp.reduce(masked, axis=1)
    log_prob = masked - log_z[:, None]
    cross_entropy = float(-np.sum(target[target > 0] * log_prob[target > 0]) / len(target))
    target_log = np.zeros_like(target)
    np.log(target, out=target_log, where=target > 0)
    entropy = float(-np.mean(np.sum(np.where(target > 0, target * target_log, 0.0), axis=1)))
    kl = cross_entropy - entropy
    top1 = float(np.mean(np.argmax(masked, axis=1) == np.argmax(target, axis=1)))
    return {
        "cross_entropy": cross_entropy,
        "target_entropy": entropy,
        "kl_divergence": kl,
        "top1_agreement": top1,
    }


def fit(
    me: np.ndarray,
    opp: np.ndarray,
    policy: list[list[tuple[int, float]]],
    validation: tuple[np.ndarray, np.ndarray, list[list[tuple[int, float]]]],
    l2: float = 1e-4,
    seed: int = 0,
    epochs: int = 60,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict[str, object]]:
    """Deterministic minibatch SGD with validation early stopping."""
    target, legal = targets(policy)
    va_me, va_opp, va_policy = validation
    va_target, va_legal = targets(va_policy)
    active = active_indices(me, opp)
    mirror_me = me.reshape(-1, 6, 7)[:, :, ::-1].reshape(-1, 42)
    mirror_opp = opp.reshape(-1, 6, 7)[:, :, ::-1].reshape(-1, 42)
    active_mirror = active_indices(mirror_me, mirror_opp)
    rng = np.random.default_rng(seed)
    weights = np.zeros((N_WEIGHTS, COLS), dtype=np.float64)
    best = weights.copy()
    best_loss = float("inf")
    stale = 0
    completed_epochs = 0
    for _epoch in range(epochs):
        completed_epochs = _epoch + 1
        order = rng.permutation(len(me))
        for start in range(0, len(me), batch_size):
            rows = order[start : start + batch_size]
            a, am = active[rows], active_mirror[rows]
            raw = (weights[a].sum(axis=1) + weights[am].sum(axis=1)[:, ::-1]) * 0.5
            masked = np.where(legal[rows], raw, -np.inf)
            prob = np.exp(masked - np.logaddexp.reduce(masked, axis=1)[:, None])
            delta = (prob - target[rows]) / len(rows)
            grad = np.zeros_like(weights)
            np.add.at(grad, a.ravel(), np.repeat(delta * 0.5, a.shape[1], axis=0))
            mirrored_delta = delta[:, ::-1]
            np.add.at(grad, am.ravel(), np.repeat(mirrored_delta * 0.5, am.shape[1], axis=0))
            grad += l2 * weights
            grad[0] -= l2 * weights[0]
            weights -= 0.15 * grad
        val = metrics(logits(weights.ravel(), va_me, va_opp), va_target, va_legal)["cross_entropy"]
        if val < best_loss - 1e-7:
            best_loss, best, stale = val, weights.copy(), 0
        else:
            stale += 1
        if stale >= 8:
            break
    train = metrics(logits(best.ravel(), me, opp), target, legal)
    valid = metrics(logits(best.ravel(), va_me, va_opp), va_target, va_legal)
    return best.astype(np.float32).ravel(), {
        "train": train,
        "validation": valid,
        "epochs": completed_epochs,
        "l2": l2,
        "seed": seed,
    }


def write_weights(path: str, weights: np.ndarray) -> None:
    if weights.shape != (POLICY_WEIGHTS,):
        raise ValueError(f"expected {POLICY_WEIGHTS} weights")
    weights.astype("<f4").tofile(path)
