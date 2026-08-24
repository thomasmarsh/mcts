"""Typed evidence and state for one seat-swapped evaluation pair."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NewType, Sequence
from uuid import NAMESPACE_URL, uuid5

from openskill.models import ThurstoneMostellerPart

from .lifecycle import (
    SessionId,
    TrialId,
    pool_snapshot_fingerprint as _pool_snapshot_fingerprint,
)

if TYPE_CHECKING:
    from .config import RatingPolicy, ResourcePolicy

if TYPE_CHECKING:
    from .pool import Anchor

PairId = NewType("PairId", str)
GameId = NewType("GameId", str)
CandidateSide = Literal["first", "second"]
Outcome = Literal["candidate_win", "baseline_win", "draw"]

_MODEL = ThurstoneMostellerPart()
SCORE_FORMULA_VERSION = 1


@dataclass(frozen=True)
class Rating:
    mu: float
    sigma: float


@dataclass(frozen=True)
class StrategyMetrics:
    iterations_total: int
    iterations_first_half: int
    move_time_ms: int


@dataclass(frozen=True)
class OpponentSnapshot:
    anchor_id: str
    config: dict
    mu: float
    sigma: float

    @classmethod
    def from_anchor(cls, anchor: Anchor) -> OpponentSnapshot:
        return cls(anchor.id, deepcopy(anchor.config), anchor.mu, anchor.sigma)


@dataclass(frozen=True)
class PairTask:
    session_id: SessionId
    trial_id: TrialId
    pair_id: PairId
    pair_index: int
    seed: int
    candidate_config: dict
    opponent: OpponentSnapshot
    pool_snapshot_fingerprint: str
    rating_before: Rating
    trace_game_sequence_start: int | None = None


@dataclass(frozen=True)
class GameResult:
    game_id: GameId
    candidate_side: CandidateSide
    outcome: Outcome
    seed: int
    round: int
    trace_game_seq: int | None
    plies: int
    elapsed_ms: int
    candidate: StrategyMetrics
    baseline: StrategyMetrics


@dataclass(frozen=True)
class PairResult:
    task: PairTask
    games: tuple[GameResult, GameResult]


TrialReportOutcome = Literal["continue", "complete", "prune"]
TrialReportReason = Literal[
    "below_min_pairs",
    "pruning_disabled",
    "confidence",
    "max_pairs",
    "startup_exempt",
    "hyperband_keep",
    "hyperband_prune",
]


@dataclass(frozen=True)
class TrialReportDecision:
    outcome: TrialReportOutcome
    reason: TrialReportReason
    pruning_exempt: bool = False
    bracket_id: str | None = None
    rung_resource: int | None = None


@dataclass
class TrialEvaluationState:
    resource_policy: ResourcePolicy
    rating_policy: RatingPolicy
    rating: Rating = field(default_factory=lambda: Rating(25.0, _MODEL.rating().sigma))
    completed_pairs: int = 0
    games: list[tuple[OpponentSnapshot, GameResult]] = field(default_factory=list)

    def should_continue(self) -> bool:
        return self.decision().outcome == "continue"

    def score(self) -> float:
        return conservative_score(self.rating, self.rating_policy.conservative_k)

    def decision(self) -> TrialReportDecision:
        if (
            self.completed_pairs >= self.resource_policy.min_pairs
            and self.rating_policy.sigma_stop is not None
            and self.rating.sigma <= self.rating_policy.sigma_stop
        ):
            return TrialReportDecision("complete", "confidence")
        if self.completed_pairs >= self.resource_policy.max_pairs:
            return TrialReportDecision("complete", "max_pairs")
        if self.completed_pairs < self.resource_policy.min_pairs:
            return TrialReportDecision("continue", "below_min_pairs")
        return TrialReportDecision("continue", "pruning_disabled")

    def apply_pair(self, result: PairResult) -> Rating:
        if len(result.games) != 2:
            raise ValueError("an evaluation pair must contain exactly two games")
        for game in result.games:
            self.rating = _rate_game(self.rating, result.task.opponent, game.outcome)
            self.games.append((result.task.opponent, game))
        self.completed_pairs += 1
        return self.rating

    def legacy_games(self) -> list[dict]:
        return [
            {"opponent": opponent.anchor_id, "outcome": _legacy_outcome(game.outcome)}
            for opponent, game in self.games
        ]


def pair_id_for(session_id: SessionId, trial_id: TrialId, pair_index: int) -> PairId:
    """Return the deterministic identity for one trial evaluation pair."""
    value = uuid5(
        NAMESPACE_URL, f"mcts-tuner:{session_id}:{trial_id}:pair:{pair_index}"
    )
    return PairId(f"pair-{value.hex}")


def game_id_for(pair_id: PairId, candidate_side: CandidateSide) -> GameId:
    """Return the deterministic identity for one physical seat assignment."""
    value = uuid5(NAMESPACE_URL, f"mcts-tuner:{pair_id}:game:{candidate_side}")
    return GameId(f"game-{value.hex}")


def pool_snapshot_fingerprint(anchors: Sequence[Anchor]) -> str:
    """Fingerprint the complete frozen pool snapshot available to a pair."""
    return _pool_snapshot_fingerprint(anchors)


def configured_game_seed(seed: int) -> int:
    """Mirror game-host's SplitMix64 seed derivation for comparison round zero."""
    value = seed & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & 9_007_199_254_740_991


def conservative_score(rating: Rating, conservative_k: float) -> float:
    """Return the version-1 maximization objective for a candidate rating."""
    return rating.mu - conservative_k * rating.sigma


def _rate_game(rating: Rating, opponent: OpponentSnapshot, outcome: Outcome) -> Rating:
    candidate = _MODEL.rating(mu=rating.mu, sigma=rating.sigma)
    anchor = _MODEL.rating(mu=opponent.mu, sigma=opponent.sigma)
    if outcome == "candidate_win":
        return _to_rating(_MODEL.rate([[candidate], [anchor]])[0][0])
    if outcome == "baseline_win":
        return _to_rating(_MODEL.rate([[anchor], [candidate]])[1][0])
    return _to_rating(_MODEL.rate([[candidate], [anchor]], scores=[0, 0])[0][0])


def _to_rating(value: object) -> Rating:
    return Rating(mu=value.mu, sigma=value.sigma)  # type: ignore[attr-defined]


def _legacy_outcome(outcome: Outcome) -> str:
    return {"candidate_win": "win", "baseline_win": "loss", "draw": "draw"}[outcome]
