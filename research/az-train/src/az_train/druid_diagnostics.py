"""Self-play diagnostics for the Druid CNN track that need no oracle: what the games, the search
targets and the network's own predictions say about the health of a run. Pure numpy over decoded
positions, so it is unit testable without torch or a GPU; ``druid_cnn.diagnose`` feeds it a
generation's shard and the net's predictions on that shard's held-out games.

Every sub-decision of a turn is its own ply, so policy metrics are also split by the phase a
position is in: choosing the piece kind, choosing a lintel's orientation, or choosing the cell.
The phase is read off the legal-action mask (the four non-cell entries are only ever legal in the
first two), so no shard field is needed.

The value diagnostics exist to make one failure visible generation by generation: a value head
that fits the near-coin-flip outcomes of a weak net's games (held-out Pearson turning negative,
held-out MSE above the 1.0 of a constant zero) while its training error keeps falling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from az_train.druid_records import Positions
from az_train.gonnect_diagnostics import CALIBRATION_BINS, _entropy, calibration, masked_softmax

PHASES = ("piece", "orientation", "cell")


def phase_of(legal: np.ndarray, size: int) -> np.ndarray:
    """``(N,)`` int8: 0 while a piece kind is chosen, 1 for a lintel's orientation, 2 for a cell."""
    cells = size * size
    piece = legal[:, cells] | legal[:, cells + 1]
    orientation = legal[:, cells + 2] | legal[:, cells + 3]
    return np.where(piece, 0, np.where(orientation, 1, 2)).astype(np.int8)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def value_fit(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """How the value head does against outcomes. ``mse_vs_constant`` is the MSE divided by the
    1.0 a constant-zero predictor scores on +-1 targets: above 1 the head is worse than knowing
    nothing."""
    mse = float(np.mean((predicted - target) ** 2)) if len(target) else None
    return {
        "mse": mse,
        "mse_vs_constant": None if mse is None else mse / max(float(np.mean(target**2)), 1e-12),
        "pearson": _pearson(predicted, target),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(target)))
        if len(target)
        else None,
        "pred_std": float(predicted.std()) if len(predicted) else None,
    }


PROGRESS_BINS = 5


def game_progress(game: np.ndarray) -> np.ndarray:
    """``(N,)`` float in [0, 1): each position's ply index over its game's length, assuming a
    game's positions are stored in ply order (the shard writer's order)."""
    order = np.argsort(game, kind="stable")
    _, inverse, counts = np.unique(game[order], return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    within = np.arange(len(game)) - starts[inverse]
    progress = np.empty(len(game))
    progress[order] = within / counts[inverse]
    return progress


def value_by_progress(
    predicted: np.ndarray, target: np.ndarray, game: np.ndarray, bins: int = PROGRESS_BINS
) -> list[dict[str, Any]]:
    """Value fit per equal slice of game progress. Early positions of a game are near coin flips,
    so a high early-slice MSE is a noise floor, not a head that has failed to learn; a late slice
    that stays high is the head genuinely underfitting."""
    slot = np.minimum((game_progress(game) * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        sel = slot == b
        out.append({"bin": b, "n": int(sel.sum()), **value_fit(predicted[sel], target[sel])})
    return out


def policy_by_phase(
    held: Positions, logits: np.ndarray, size: int
) -> dict[str, dict[str, float | int | None]]:
    """Held-out policy cross-entropy (with the uniform-policy baseline), top-1 agreement and target
    entropy per phase. Positions with a single legal action are counted apart (``forced``): they
    carry no information either way."""
    prior = masked_softmax(logits, held.legal)
    ce = -(held.policy * np.log(np.maximum(prior, 1e-30))).sum(axis=1)
    top1 = prior.argmax(axis=1) == held.policy.argmax(axis=1)
    target_entropy = _entropy(held.policy)
    n_legal = held.legal.sum(axis=1)
    phase = phase_of(held.legal, size)
    out: dict[str, dict[str, float | int | None]] = {}
    for i, name in enumerate(PHASES):
        sel = (phase == i) & (n_legal > 1)
        n = int(sel.sum())
        out[name] = {
            "positions": n,
            "forced": int(((phase == i) & (n_legal <= 1)).sum()),
            "share": float(n / max(1, int((n_legal > 1).sum()))),
            "policy_ce": float(ce[sel].mean()) if n else None,
            # What a uniform policy over the legal actions scores: a net at chance sits here.
            "uniform_ce": float(np.log(n_legal[sel]).mean()) if n else None,
            "policy_top1": float(top1[sel].mean()) if n else None,
            "target_entropy": float(target_entropy[sel].mean()) if n else None,
            "mean_legal": float(n_legal[sel].mean()) if n else None,
        }
    return out


def diagnostics(
    shard: Positions,
    stats: dict[str, Any],
    held: Positions,
    predicted_value: np.ndarray,
    logits: np.ndarray,
    size: int,
    train_predicted: np.ndarray | None = None,
    train_target: np.ndarray | None = None,
) -> dict[str, Any]:
    """One generation's oracle-free diagnostics.

    ``shard`` is every position the generation's self-play produced and ``stats`` the self-play
    driver's counters; ``held`` are the held-out games with the net's ``predicted_value`` and
    ``logits`` on them (after the generation's fit). ``train_predicted`` / ``train_target`` are the
    net's predictions and the outcomes on a sample of the generation's own training positions, for
    the train-versus-held-out value gap.
    """
    started = max(1, stats.get("games_started", 1))
    finished = max(1, stats.get("games_finished", 1))
    n_legal = shard.legal.sum(axis=1)
    target_entropy = _entropy(shard.policy)
    normalised = target_entropy / np.log(np.maximum(n_legal, 2))
    shard_phase = phase_of(shard.legal, size)

    prior = masked_softmax(logits, held.legal)
    prior_entropy = _entropy(prior)
    held_target_entropy = _entropy(held.policy)
    cross_entropy = -(held.policy * np.log(np.maximum(prior, 1e-30))).sum(axis=1)

    value = {
        "held": {
            **value_fit(predicted_value, held.value),
            "by_progress": value_by_progress(predicted_value, held.value, held.game),
        }
    }
    if train_predicted is not None and train_target is not None:
        value["train"] = value_fit(train_predicted, train_target)
        value["mse_gap"] = value["held"]["mse"] - value["train"]["mse"]
    return {
        "selfplay": {
            "mean_plies": stats.get("mean_plies"),
            "black_win_rate": stats.get("black_wins", 0) / finished,
            "draw_rate": stats.get("draws", 0) / finished,
            "capped_rate": stats.get("games_capped", 0) / started,
            "mean_legal_moves": float(n_legal.mean()),
            "phase_share": {
                name: float((shard_phase == i).mean()) for i, name in enumerate(PHASES)
            },
        },
        "policy": {
            "target_entropy": float(target_entropy.mean()),
            "target_entropy_normalised": float(normalised.mean()),
            "prior_entropy": float(prior_entropy.mean()),
            "kl_target_to_prior": float((cross_entropy - held_target_entropy).mean()),
            "by_phase": policy_by_phase(held, logits, size),
        },
        "value": value,
        "calibration": calibration(predicted_value, held.value, CALIBRATION_BINS),
    }
