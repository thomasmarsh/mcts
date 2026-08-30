"""Immutable values used by tuner policy, artifacts, and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Phase = Literal["tuning", "validation"]
OpponentRole = Literal["default", "historical_reference"]


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    fingerprint: str
    canonical_config: str


@dataclass(frozen=True, slots=True)
class Opponent:
    opponent_id: str
    source_id: Literal["schema_default", "inline"]
    label: str
    role: OpponentRole
    weight: int
    canonical_config: str
    configuration_fingerprint: str


@dataclass(frozen=True, slots=True)
class OpponentPanel:
    panel_id: str
    fingerprint: str
    opponents: tuple[Opponent, ...]
    total_weight: int


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_index: int
    source: Literal["schema_default", "configspace_random"]
    proposer_version: Literal["configspace-random-v1"]
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class SearchEffort:
    max_iterations: int


IterationBudget = SearchEffort


@dataclass(frozen=True, slots=True)
class TaskCase:
    task_id: str
    phase: Phase
    ordinal: int
    seed: int
    stratum_id: str
    opponent_id: str
    opponent_fingerprint: str
    panel_fingerprint: str
    game_config_fingerprint: str
    start: Literal["default"] = "default"


@dataclass(frozen=True, slots=True)
class TaskCorpus:
    corpus_id: str
    fingerprint: str
    phase: Phase
    task_policy_version: Literal["weighted-fair-prefix-v1"]
    cases: tuple[TaskCase, ...]

    @property
    def block_id(self) -> str:
        return self.corpus_id


TaskBlock = TaskCorpus


@dataclass(frozen=True, slots=True)
class TaskPrefix:
    prefix_id: str
    corpus_id: str
    length: int
    task_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskCountFidelity:
    task_prefix: TaskPrefix


@dataclass(frozen=True, slots=True)
class ObjectiveEpoch:
    epoch_id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ObservationContext:
    objective_epoch_id: str
    phase: Phase
    task_prefix: TaskPrefix
    search_effort: SearchEffort


@dataclass(frozen=True, slots=True)
class PairTask:
    pair_id: str
    candidate_id: str
    task_case: TaskCase
    budget: SearchEffort


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
    context: ObservationContext
    pair_utilities: tuple[float, ...]
    estimate: Estimate

    @property
    def phase(self) -> Phase:
        return self.context.phase

    @property
    def block_id(self) -> str:
        return self.context.task_prefix.corpus_id

    @property
    def prefix_length(self) -> int:
        return self.context.task_prefix.length

    @property
    def budget(self) -> SearchEffort:
        return self.context.search_effort


@dataclass(frozen=True, slots=True)
class ValidationError:
    field: str
    message: str
    candidate_index: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    errors: tuple[ValidationError, ...]


@dataclass(frozen=True, slots=True)
class ReplayState:
    proposals: tuple[Proposal, ...]
    dispositions: tuple[tuple[int, Literal["accepted", "rejected"]], ...]
    cohort: tuple[Candidate, ...] | None
    completed_pairs: tuple[PairResult, ...]
    observations: tuple[Observation, ...]
    finalists: tuple[Candidate, ...] | None
    terminal_status: Literal["open", "configuration_failed", "complete"]
    next_pair_id: str | None
