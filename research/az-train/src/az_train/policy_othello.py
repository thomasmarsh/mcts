# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""D4-equivariant n-tuple policy sidecar trainer for Othello.

Same active tuple features as the value head (``az_train.ntuple_othello``),
but with one weight *row* of 64 columns (one per board square) per feature
instead of a scalar -- the direct generalization of
``az_train.policy_c4``'s two-orientation (identity + mirror) sidecar to
Othello's full D4 group of 8 orientations, matching the forward pass
``games/othello/src/policy.rs``/``othello_eval.policy`` already implement
and pin against each other with a cross-language fixture. This module adds
the *fit* (those two only do inference); the completed-Q improved-policy
target from Gumbel self-play (``az_train.records_othello``) is the label.

``Move::PASS`` has no board square, so it is column index 64 in the
65-column target/legal/logit arrays this module works with; its Rust-side
logit (``games/othello/src/policy.rs::logits``) is the mean of the 64
real-square logits, so its gradient is spread evenly back across all 64
D4-averaged columns during backprop (see :func:`_backward`).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from az_train.ntuple_othello import D4, ModelGeometry, featurize
from az_train.records_othello import PASS, Positions

SQUARES = 64
COLUMNS = SQUARES + 1  # 64 real squares + PASS


def _inv_table() -> np.ndarray:
    """``(8, 64)``: ``INV[k, D4[k, i]] == i`` for every orientation `k`."""
    inv = np.zeros((8, SQUARES), dtype=np.int64)
    for k in range(8):
        inv[k, D4[k]] = np.arange(SQUARES)
    return inv


INV = _inv_table()


def _forward(weights: np.ndarray, feat_idx: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(N, COLUMNS)`` logits (last column is PASS) plus the per-orientation
    canonical-frame indices used, so :func:`_backward` doesn't recompute
    them. ``weights`` is ``(n_weights, 64)``."""
    n = feat_idx.shape[0]
    out = np.zeros((n, SQUARES), dtype=np.float64)
    idx_syms: list[np.ndarray] = []
    for sym in range(8):
        idx_sym = feat_idx[:, sym::8]  # (N, n_tuples), canonical-frame global indices
        idx_syms.append(idx_sym)
        raw = weights[idx_sym].sum(axis=1)  # (N, 64) canonical frame
        out += raw[:, INV[sym]]
    out /= 8.0
    pass_logit = out.mean(axis=1, keepdims=True)
    return np.concatenate([out, pass_logit], axis=1), idx_syms


def _backward(
    d_logits: np.ndarray, idx_syms: list[np.ndarray], n_weights: int
) -> np.ndarray:
    """Gradient of the loss w.r.t. the flat ``(n_weights, 64)`` weight table,
    given ``d_logits`` (``dL/dlogit``, ``(N, COLUMNS)``)."""
    d_out = d_logits[:, :SQUARES] + d_logits[:, SQUARES:SQUARES + 1] / SQUARES
    grad = np.zeros((n_weights, SQUARES), dtype=np.float64)
    n_tuples = idx_syms[0].shape[1]
    for sym in range(8):
        # d_raw[:, c] = d_out[:, D4[sym][c]] / 8 -- the exact transpose of the
        # forward gather `out += raw[:, INV[sym]]`, since D4 and INV are
        # mutual inverse permutations.
        d_raw = d_out[:, D4[sym]] / 8.0  # (N, 64) canonical frame
        rows = idx_syms[sym]  # (N, n_tuples)
        rows_flat = rows.reshape(-1)
        vals_flat = np.repeat(d_raw, n_tuples, axis=0)
        # `np.add.at` accumulates directly into `grad`'s selected rows -- a
        # `bincount` over the full `(n_weights * SQUARES,)` flat table (as
        # `az_train.policy_c4`'s much smaller weight table can afford) would
        # reallocate and zero a multi-million-entry array on every minibatch
        # here, which dominated the wall clock at Othello's ~113k-feature
        # geometry (>100x Connect Four's).
        np.add.at(grad, rows_flat, vals_flat)
    return grad


def targets(policy: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    """``(N, COLUMNS)`` dense target probabilities and legal-move mask."""
    target = np.zeros((len(policy), COLUMNS), dtype=np.float64)
    legal = np.zeros_like(target, dtype=bool)
    for row, entries in enumerate(policy):
        for square, probability in entries:
            target[row, square] = probability
            legal[row, square] = True
    if np.any(~legal.any(axis=1)):
        raise ValueError("policy training requires a target for every position")
    return target, legal


def logits(weights: np.ndarray, pos: Positions, geom: ModelGeometry) -> np.ndarray:
    """``(N, COLUMNS)`` D4-symmetrized logits, for inference/diagnostics."""
    feat_idx = featurize(pos, geom)
    w = np.asarray(weights, dtype=np.float64).reshape(geom.n_weights, SQUARES)
    out, _ = _forward(w, feat_idx)
    return out


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
    uniform_ce = float(np.mean(np.log(legal.sum(axis=1))))
    return {
        "cross_entropy": cross_entropy,
        "target_entropy": entropy,
        "kl_divergence": kl,
        "top1_agreement": top1,
        "uniform_baseline_cross_entropy": uniform_ce,
    }


def fit(
    train: Positions,
    geom: ModelGeometry,
    validation: Positions,
    l2: float = 1e-4,
    seed: int = 0,
    epochs: int = 60,
    batch_size: int = 256,
    patience: int = 8,
    report_every: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic minibatch SGD with validation early stopping."""
    feat_idx = featurize(train, geom)
    va_feat_idx = featurize(validation, geom)
    target, legal = targets(train.policy)
    va_target, va_legal = targets(validation.policy)
    n_weights = geom.n_weights

    def eval_cross_entropy(
        w: np.ndarray, feat: np.ndarray, tgt: np.ndarray, lgl: np.ndarray
    ) -> float:
        raw, _ = _forward(w, feat)
        return metrics(raw, tgt, lgl)["cross_entropy"]

    rng = np.random.default_rng(seed)
    weights = np.zeros((n_weights, SQUARES), dtype=np.float64)
    best = weights.copy()
    best_loss = float("inf")
    stale = 0
    completed_epochs = 0
    n = feat_idx.shape[0]
    start_time = time.time()
    for _epoch in range(epochs):
        completed_epochs = _epoch + 1
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            rows = order[start : start + batch_size]
            bf, bt, bl = feat_idx[rows], target[rows], legal[rows]
            raw, idx_syms = _forward(weights, bf)
            masked = np.where(bl, raw, -np.inf)
            prob = np.exp(masked - np.logaddexp.reduce(masked, axis=1)[:, None])
            d_logits = (prob - bt) / len(rows)
            grad = _backward(d_logits, idx_syms, n_weights)
            grad += l2 * weights
            weights -= 0.15 * grad
        val = eval_cross_entropy(weights, va_feat_idx, va_target, va_legal)
        if val < best_loss - 1e-7:
            best_loss, best, stale = val, weights.copy(), 0
        else:
            stale += 1
        if completed_epochs % report_every == 0 or completed_epochs == epochs:
            elapsed = time.time() - start_time
            print(
                f"  policy epoch {completed_epochs:4d}  val ce {val:.4f} "
                f"(best {best_loss:.4f})  ({elapsed:.1f}s)",
                flush=True,
            )
        if stale >= patience:
            break
    best_flat = best.ravel().astype(np.float32)
    train_metrics = metrics(logits(best_flat, train, geom), target, legal)
    valid_metrics = metrics(logits(best_flat, validation, geom), va_target, va_legal)
    return best.astype(np.float32).ravel(), {
        "train": train_metrics,
        "validation": valid_metrics,
        "epochs": completed_epochs,
        "l2": l2,
        "seed": seed,
    }


def write_weights(out_dir: str, weights: np.ndarray, n_weights: int, meta: dict) -> None:
    import json
    from pathlib import Path

    if weights.shape != (n_weights * SQUARES,):
        raise ValueError(f"expected {n_weights * SQUARES} weights, got {weights.shape}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(out / "policy.bin")
    (out / "policy.meta.json").write_text(json.dumps(meta, indent=2) + "\n")


assert PASS == SQUARES, "Move::PASS must be the column just past the 64 real squares"
