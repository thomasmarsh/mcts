"""Self-play diagnostics that need no oracle: what the games, the search targets and the network's
own predictions say about the health of a run. Pure numpy over decoded positions, so it is unit
testable without torch or a GPU; ``gonnect_cnn.diagnose`` feeds it a generation's shard and the
net's predictions on that shard's held-out games.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from az_train.gonnect_records import Positions

CALIBRATION_BINS = 10


def _entropy(p: np.ndarray) -> np.ndarray:
    """Row entropy in nats of probability rows (zeros contribute nothing)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)


def masked_softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    z = np.where(legal, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.where(legal, np.exp(z), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def calibration(predicted: np.ndarray, outcome: np.ndarray, bins: int = CALIBRATION_BINS) -> dict:
    """Predicted value against realised outcome. ``predicted`` in ``[-1, 1]`` is read as a win
    probability ``(v + 1) / 2`` and ``outcome`` is ``+1`` or ``-1`` for the mover.

    ``ece`` is the expected calibration error over equal-width bins of predicted value (the
    count-weighted gap between mean predicted and realised win probability), ``brier`` the mean
    squared probability error. Positions of one game share an outcome, so with a few dozen held-out
    games these are noisy; they show a drift, not a fine-grained calibration curve.
    """
    p = (np.clip(predicted, -1, 1) + 1) / 2
    o = (outcome > 0).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.minimum((p * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        sel = which == b
        n = int(sel.sum())
        if n == 0:
            rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": 0})
            continue
        mp, mo = float(p[sel].mean()), float(o[sel].mean())
        ece += n / len(p) * abs(mp - mo)
        rows.append(
            {"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": n,
             "mean_pred": mp, "mean_outcome": mo}
        )  # fmt: skip
    return {"ece": float(ece), "brier": float(np.mean((p - o) ** 2)), "bins": rows}


def diagnostics(
    shard: Positions,
    stats: dict[str, Any],
    held: Positions,
    predicted_value: np.ndarray,
    logits: np.ndarray,
) -> dict[str, Any]:
    """One generation's oracle-free diagnostics.

    ``shard`` is every position the generation's self-play produced and ``stats`` the self-play
    driver's counters; ``held`` are the held-out games with the net's ``predicted_value`` and
    ``logits`` on them (after the generation's fit).
    """
    started = max(1, stats.get("games_started", 1))
    finished = max(1, stats.get("games_finished", 1))
    n_legal = shard.legal.sum(axis=1)
    target_entropy = _entropy(shard.policy)
    normalised = target_entropy / np.log(np.maximum(n_legal, 2))
    ko_cells = shard.planes[:, 2].reshape(len(shard), -1).any(axis=1)

    prior = masked_softmax(logits, held.legal)
    prior_entropy = _entropy(prior)
    held_target_entropy = _entropy(held.policy)
    cross_entropy = -(held.policy * np.log(np.maximum(prior, 1e-30))).sum(axis=1)
    return {
        "selfplay": {
            "mean_plies": stats.get("mean_plies"),
            "black_win_rate": stats.get("black_wins", 0) / finished,
            "swap_rate": stats.get("swaps", 0) / finished,
            "capped_rate": stats.get("games_capped", 0) / started,
            "ko_position_rate": float(ko_cells.mean()),
            "mean_legal_moves": float(n_legal.mean()),
        },
        "policy": {
            "target_entropy": float(target_entropy.mean()),
            "target_entropy_normalised": float(normalised.mean()),
            "prior_entropy": float(prior_entropy.mean()),
            "kl_target_to_prior": float((cross_entropy - held_target_entropy).mean()),
        },
        "calibration": calibration(predicted_value, held.value),
    }
