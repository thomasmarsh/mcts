"""Offline, deterministic calibration of scripted evaluation pairs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .config import RatingPolicy, ResourcePolicy, SearchConfig
from .evaluation import (
    GameResult,
    OpponentSnapshot,
    Outcome,
    PairResult,
    PairTask,
    Rating,
    StrategyMetrics,
    TrialEvaluationState,
    TrialReportDecision,
    game_id_for,
    pair_id_for,
)
from .lifecycle import SessionId, TrialId


_OUTCOME_ALIASES: dict[str, Outcome] = {
    "win": "candidate_win",
    "loss": "baseline_win",
    "draw": "draw",
    "candidate_win": "candidate_win",
    "baseline_win": "baseline_win",
}
_SIDES = ("first", "second")
_CALIBRATION_SESSION = SessionId("rating-calibration")
_CALIBRATION_TRIAL = TrialId("scripted-outcomes")
_METRICS = StrategyMetrics(0, 0, 0)


@dataclass(frozen=True)
class ScriptedPair:
    """The candidate's outcomes in the first- and second-seat games."""

    first: Outcome
    second: Outcome

    @property
    def outcomes(self) -> tuple[Outcome, Outcome]:
        return (self.first, self.second)


@dataclass(frozen=True)
class CalibrationRow:
    """Production evaluation state after one scripted pair."""

    pair_index: int
    pair: ScriptedPair
    rating: Rating
    score: float
    decision: TrialReportDecision


def parse_scripted_pair(value: str) -> ScriptedPair:
    """Parse ``first:win,second:loss`` into one explicit seat-swapped pair."""
    games = value.split(",")
    if len(games) != 2:
        raise ValueError("a pair needs exactly two games: first:OUTCOME,second:OUTCOME")

    parsed: list[Outcome] = []
    for expected_side, game in zip(_SIDES, games, strict=True):
        side, separator, raw_outcome = game.partition(":")
        if separator != ":" or side != expected_side:
            raise ValueError(
                "pair seats must be explicit and ordered as first:OUTCOME,second:OUTCOME"
            )
        try:
            parsed.append(_OUTCOME_ALIASES[raw_outcome])
        except KeyError as error:
            choices = ", ".join(sorted(_OUTCOME_ALIASES))
            raise ValueError(
                f"unknown outcome {raw_outcome!r}; use one of {choices}"
            ) from error
    return ScriptedPair(*parsed)


def parse_sigma_stop(value: str) -> float | None:
    """Accept a positive confidence bound or ``none`` to disable confidence stops."""
    if value == "none":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError("sigma stop must be a number or 'none'") from error
    return parsed


def resolve_policy(
    config: SearchConfig,
    *,
    min_pairs: int | None = None,
    max_pairs: int | None = None,
    sigma_stop: float | None | object = ...,  # Ellipsis means no override.
    conservative_k: float | None = None,
) -> tuple[ResourcePolicy, RatingPolicy]:
    """Return a validated resource/rating policy after optional CLI overrides."""
    resource = replace(
        config.optimizer.resource,
        min_pairs=config.optimizer.resource.min_pairs
        if min_pairs is None
        else min_pairs,
        max_pairs=config.optimizer.resource.max_pairs
        if max_pairs is None
        else max_pairs,
    )
    rating = replace(
        config.optimizer.rating,
        sigma_stop=config.optimizer.rating.sigma_stop
        if sigma_stop is ...
        else sigma_stop,
        conservative_k=(
            config.optimizer.rating.conservative_k
            if conservative_k is None
            else conservative_k
        ),
    )
    resolved = replace(
        config, optimizer=replace(config.optimizer, resource=resource, rating=rating)
    )
    resolved.validate()
    return resource, rating


def calibrate(
    pairs: Sequence[ScriptedPair],
    resource_policy: ResourcePolicy,
    rating_policy: RatingPolicy,
    opponent: OpponentSnapshot,
    initial_rating: Rating | None = None,
) -> list[CalibrationRow]:
    """Apply scripted pairs through the production rating and stopping state."""
    state = TrialEvaluationState(resource_policy, rating_policy)
    if initial_rating is not None:
        state.rating = initial_rating

    rows: list[CalibrationRow] = []
    for pair_index, pair in enumerate(pairs, start=1):
        task = _pair_task(pair_index - 1, opponent, state.rating)
        games = tuple(
            GameResult(
                game_id_for(task.pair_id, side),
                side,  # type: ignore[arg-type]
                outcome,
                seed=pair_index,
                round=0,
                trace_game_seq=None,
                plies=0,
                elapsed_ms=0,
                candidate=_METRICS,
                baseline=_METRICS,
            )
            for side, outcome in zip(_SIDES, pair.outcomes, strict=True)
        )
        state.apply_pair(PairResult(task, games))  # type: ignore[arg-type]
        rows.append(
            CalibrationRow(
                pair_index,
                pair,
                state.rating,
                state.score(),
                state.decision(),
            )
        )
    return rows


def render_calibration(rows: Sequence[CalibrationRow]) -> str:
    """Render values with Python's lossless float representation."""
    return "\n".join(
        "pair={index} first={first} second={second} mu={mu!r} sigma={sigma!r} "
        "conservative_score={score!r} decision={outcome}/{reason}".format(
            index=row.pair_index,
            first=row.pair.first,
            second=row.pair.second,
            mu=row.rating.mu,
            sigma=row.rating.sigma,
            score=row.score,
            outcome=row.decision.outcome,
            reason=row.decision.reason,
        )
        for row in rows
    )


def _pair_task(
    pair_index: int, opponent: OpponentSnapshot, rating_before: Rating
) -> PairTask:
    pair_id = pair_id_for(_CALIBRATION_SESSION, _CALIBRATION_TRIAL, pair_index)
    return PairTask(
        _CALIBRATION_SESSION,
        _CALIBRATION_TRIAL,
        pair_id,
        pair_index,
        pair_index + 1,
        {"source": "rating-calibration"},
        opponent,
        "rating-calibration",
        rating_before,
    )
