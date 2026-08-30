"""Immutable values independent of subprocess and ConfigSpace objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    fingerprint: str
    canonical_config: str


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_index: int
    source: Literal["schema_default", "configspace_random"]
    proposer_version: Literal["configspace-random-v1"]
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class IterationBudget:
    max_iterations: int


@dataclass(frozen=True, slots=True)
class TaskCase:
    task_id: str
    phase: Literal["tuning", "validation"]
    ordinal: int
    seed: int
    opponent_id: str
    opponent_fingerprint: str
    game_config_fingerprint: str
    start: Literal["default"] = "default"


@dataclass(frozen=True, slots=True)
class TaskBlock:
    block_id: str
    phase: Literal["tuning", "validation"]
    cases: tuple[TaskCase, ...]


@dataclass(frozen=True, slots=True)
class PairTask:
    pair_id: str
    candidate_id: str
    task_case: TaskCase
    budget: IterationBudget


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    iterations_total: int
    iterations_first_half: int
    move_time_ms: int


@dataclass(frozen=True, slots=True)
class GameResult:
    game_id: str
    candidate_side: Literal["first", "second"]
    outcome: Literal["candidate_win", "baseline_win", "draw"]
    derived_seed: int
    round: int
    seq: int
    trace_game_seq: int | None
    plies: int
    elapsed_ms: int
    candidate_metrics: StrategyMetrics
    opponent_metrics: StrategyMetrics
    raw_record: str


@dataclass(frozen=True, slots=True)
class PairResult:
    task: PairTask
    games: tuple[GameResult, GameResult]

    def __post_init__(self) -> None:
        if tuple(game.candidate_side for game in self.games) != ("first", "second"):
            raise ValueError("a pair needs candidate-first then candidate-second games")


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class Observation:
    candidate_id: str
    phase: Literal["tuning", "validation"]
    block_id: str
    prefix_length: int
    budget: IterationBudget
    pair_utilities: tuple[float, ...]
    estimate: Estimate


@dataclass(frozen=True, slots=True)
class ValidationError:
    field: str
    message: str
    candidate_index: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    errors: tuple[ValidationError, ...]
