"""Per-trial matchmaking -- plays a candidate config against the opponent pool.

Unlike SMAC3's fixed-instance cost aggregation, a trial's rating is built up
one game at a time against whichever pool anchor is currently closest to the
candidate's own live TrueSkill rating ("ladder of trash"): a brand-new
strategy starts near an unrated 25.0 mu, gets matched against `"default"`
first, and if it loses badly enough its rating drops toward `"random"`'s
anchor -- producing a usable gradient instead of a flat loss against a fixed
baseline. Anchors are frozen: the candidate's rating updates every game, the
opponent's never does.
"""

from __future__ import annotations

from pathlib import Path

import trueskill

from .config import SearchConfig
from .pool import OpponentPool
from .target import play_game

_MIN_GAMES = 5
_MAX_GAMES = 15
_SIGMA_THRESHOLD = 2.0


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
    via ``trueskill.rate_1vs1``, discarding the opponent's updated rating
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
    rating = trueskill.Rating()
    games: list[dict] = []

    step = 0
    while step < min_games or (step < max_games and rating.sigma >= sigma_threshold):
        anchor = pool.closest(rating.mu)
        opponent_rating = trueskill.Rating(mu=anchor.mu, sigma=anchor.sigma)

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
            rating, _ = trueskill.rate_1vs1(rating, opponent_rating)
            games.append({"opponent": anchor.id, "outcome": "win"})
        for _ in range(losses):
            _, rating = trueskill.rate_1vs1(opponent_rating, rating)
            games.append({"opponent": anchor.id, "outcome": "loss"})
        for _ in range(draws):
            rating, _ = trueskill.rate_1vs1(rating, opponent_rating, drawn=True)
            games.append({"opponent": anchor.id, "outcome": "draw"})

    return rating.mu, rating.sigma, games
