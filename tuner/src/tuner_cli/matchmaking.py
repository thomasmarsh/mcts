"""Per-trial matchmaking -- plays a candidate config against the opponent pool.

Unlike the tuner's fixed-instance cost aggregation, a trial's rating is built up
one game at a time against whichever pool anchor is currently closest to the
candidate's own live OpenSkill rating ("ladder of trash"): a brand-new
strategy starts near an unrated 25.0 mu, gets matched against `"default"`
first, and if it loses badly enough its rating drops toward `"random"`'s
anchor -- producing a usable gradient instead of a flat loss against a fixed
baseline. Anchors are frozen: the candidate's rating updates every game, the
opponent's never does.

Uses the Thurstone-Mosteller Partial model (`ThurstoneMostellerPart`) which is
the closest OpenSkill counterpart to the TrueSkill algorithm
"""

from __future__ import annotations

from pathlib import Path

from openskill.models import ThurstoneMostellerPart

from .config import SearchConfig
from .pool import OpponentPool
from .target import play_game

# Shared model instance for per-trial matchmaking.
_MODEL = ThurstoneMostellerPart()

_MIN_GAMES = 5
_MAX_GAMES = 15
_SIGMA_THRESHOLD = 2.0


def evaluate_trial_worker(
    cfg: SearchConfig,
    binary: Path,
    candidate_config: dict,
    pool: OpponentPool,
    seed: int,
    trace_path: str | None = None,
) -> tuple[float, float, list[dict]]:
    """Standalone wrapper around ``play_trial`` for ``ProcessPoolExecutor`` workers.

    Defined at module level so it can be pickled across process boundaries.
    Each worker re-imports this module and gets its own ``_MODEL`` instance,
    so there is no shared-memory footgun.
    """
    return play_trial(cfg, binary, candidate_config, pool, seed_base=seed, trace_path=trace_path)


def play_trial(
    cfg: SearchConfig,
    binary: Path,
    candidate_config: dict,
    pool: OpponentPool,
    *,
    seed_base: int,
    trace_path: str | None = None,
    min_games: int = _MIN_GAMES,
    max_games: int = _MAX_GAMES,
    sigma_threshold: float = _SIGMA_THRESHOLD,
) -> tuple[float, float, list[dict]]:
    """Rate ``candidate_config`` by iteratively matching it against the pool.

    Each step plays one ``play_game`` match (``cfg.target.rounds``
    round-robin pairs) against ``pool.closest(rating.mu)`` and folds every
    individual round's win/loss/draw outcome into the candidate's own rating
    via ``_MODEL.rate``, discarding the opponent's updated rating
    each time (pool anchors never mutate from matchmaking). Stops once at
    least ``min_games`` steps (``play_game`` matches, not individual rounds)
    have been played and either ``max_games`` steps have run or the
    candidate's ``sigma`` has dropped below ``sigma_threshold``, whichever
    comes first. A step whose match crashes or times out still counts toward
    both bounds -- otherwise a persistently broken config would spin forever
    -- but contributes no rating update or logged game.

    Returns ``(mu, sigma, games)`` where ``games`` is a list of
    ``{"opponent": anchor_id, "outcome": "win"|"loss"|"draw"}`` dicts, one per
    individual round across every step, for the trial JSONL ``extra``.
    """
    rating = _MODEL.rating()
    games: list[dict] = []

    step = 0
    while step < min_games or (step < max_games and rating.sigma >= sigma_threshold):
        anchor = pool.closest(rating.mu)
        opponent_rating = _MODEL.rating(mu=anchor.mu, sigma=anchor.sigma)

        wins, losses, draws, status = play_game(
            cfg,
            binary,
            candidate_config,
            anchor.config,
            seed=seed_base + step,
            trace_path=trace_path,
        )
        step += 1

        if status is not None:
            continue

        for _ in range(wins):
            result = _MODEL.rate([[rating], [opponent_rating]])
            rating = result[0][0]
            games.append({"opponent": anchor.id, "outcome": "win"})
        for _ in range(losses):
            result = _MODEL.rate([[opponent_rating], [rating]])
            rating = result[1][0]
            games.append({"opponent": anchor.id, "outcome": "loss"})
        for _ in range(draws):
            result = _MODEL.rate([[rating], [opponent_rating]], scores=[0, 0])
            rating = result[0][0]
            games.append({"opponent": anchor.id, "outcome": "draw"})

    return rating.mu, rating.sigma, games
