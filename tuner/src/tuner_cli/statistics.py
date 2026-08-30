"""Pure pair-level scoring and conservative comparison rules."""

from __future__ import annotations

import math

from .domain import Estimate, GameResult, PairResult

ALPHA = 0.05


def game_utility(game: GameResult) -> float:
    return {"candidate_win": 1.0, "draw": 0.5, "baseline_win": 0.0}[game.outcome]


def pair_utility(pair: PairResult) -> float:
    return sum(game_utility(game) for game in pair.games) / 2


def marginal_interval(values: tuple[float, ...]) -> Estimate:
    if not values:
        raise ValueError("an interval needs at least one pair")
    mean = sum(values) / len(values)
    half = math.sqrt(math.log(2 / ALPHA) / (2 * len(values)))
    return Estimate(mean, max(0.0, mean - half), min(1.0, mean + half))


def paired_difference(left: tuple[float, ...], right: tuple[float, ...]) -> Estimate:
    if not left or len(left) != len(right):
        raise ValueError("paired differences need equal non-empty inputs")
    mean = sum(a - b for a, b in zip(left, right, strict=True)) / len(left)
    half = math.sqrt(2 * math.log(2 / ALPHA) / len(left))
    return Estimate(mean, max(-1.0, mean - half), min(1.0, mean + half))


def tie_relation(difference: Estimate) -> str:
    if difference.lower > 0:
        return "better"
    if difference.upper < 0:
        return "worse"
    return "tie"
