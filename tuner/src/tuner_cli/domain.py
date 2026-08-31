"""Immutable values used by tuner policy, artifacts, and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Phase = Literal["tuning", "validation"]
EffortKind = Literal["iterations", "time_ms"]
OpponentRole = Literal["default", "historical_reference"]
ProposalSource = Literal["schema_default", "bootstrap_random", "smac_model", "random_reserve"]
ShadowDisposition = Literal["continue", "eliminate", "protected"]


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
    cohort_index: int
    cohort_slot: int
    candidate: Candidate
    frontier: ObservationFrontier
    provenance: ProposalProvenance

    @property
    def source(self) -> ProposalSource:
        return self.provenance.source

    @property
    def proposer_version(self) -> str:
        return self.provenance.proposer_version


@dataclass(frozen=True, slots=True)
class ObservationReference:
    observation_id: str
    candidate_id: str
    objective_epoch_id: str
    prefix_id: str
    task_ids: tuple[str, ...]
    search_effort: SearchEffort


@dataclass(frozen=True, slots=True)
class ObservationFrontier:
    frontier_id: str
    objective_epoch_id: str
    prefix_id: str
    task_ids: tuple[str, ...]
    search_effort: SearchEffort
    observation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProposalProvenance:
    source: ProposalSource
    proposer_version: str
    source_attempt: int
    origin: str | None
    acquisition: float | None
    prediction: float | None
    uncertainty: float | None
    parent_candidate_id: str | None


@dataclass(frozen=True, slots=True)
class ModelAttempt:
    source_attempt: int
    seed: int


@dataclass(frozen=True, slots=True)
class ModelObservation:
    candidate: Candidate
    reference: ObservationReference
    cost: float


@dataclass(frozen=True, slots=True)
class ProposedConfiguration:
    candidate: Candidate
    origin: str | None
    acquisition: float | None = None
    prediction: float | None = None
    uncertainty: float | None = None
    parent_candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class SearchEffort:
    kind: EffortKind
    value: int

    def __post_init__(self) -> None:
        raw_kind: object = self.kind
        if raw_kind not in {"iterations", "time_ms"}:
            raise ValueError("search effort kind must be 'iterations' or 'time_ms'")
        if type(self.value) is not int or self.value <= 0:
            raise ValueError("search effort value must be a positive integer")


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
    observation_id: str
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
class ComputeBudget:
    tuning_pair_attempts: int
    validation_pair_attempts: int


@dataclass(frozen=True, slots=True)
class PhaseCompute:
    pair_attempts: int = 0
    completed_pairs: int = 0
    failed_attempts: int = 0
    censored_attempts: int = 0
    physical_games: int = 0
    search_iterations: int = 0
    wall_time_ms: int = 0


@dataclass(frozen=True, slots=True)
class ComputeLedger:
    tuning: PhaseCompute = PhaseCompute()
    validation: PhaseCompute = PhaseCompute()


@dataclass(frozen=True, slots=True)
class ShadowCandidateDecision:
    candidate_id: str
    favorable_resamples: int
    total_resamples: int
    disposition: ShadowDisposition


@dataclass(frozen=True, slots=True)
class ShadowRaceDecision:
    cohort_index: int
    prefix_id: str
    observation_ids: tuple[str, ...]
    boundary_candidate_id: str
    decisions: tuple[ShadowCandidateDecision, ...]
    policy_version: Literal["stratified-paired-bootstrap-v1"]


@dataclass(frozen=True, slots=True)
class PairAttemptFacts:
    started_attempts: int = 0
    failed_attempts: int = 0
    censored_attempts: int = 0
    completed_attempts: int = 0


@dataclass(frozen=True, slots=True)
class CandidateFailure:
    cohort_index: int
    candidate_id: str
    triggering_pair_id: str
    started_attempts: int
    failed_attempts: int
    censored_attempts: int
    completed_tuning_pair_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplayState:
    proposals: tuple[Proposal, ...]
    dispositions: tuple[tuple[int, Literal["accepted", "rejected"]], ...]
    completed_cohorts: tuple[CohortRecord, ...]
    active_elites: tuple[Candidate, ...]
    completed_pairs: tuple[PairResult, ...]
    observations: tuple[Observation, ...]
    finalists: tuple[Candidate, ...] | None
    terminal_status: Literal["open", "configuration_failed", "complete"]
    tuning_block_index: int
    pending_resource_allocation: ResourceAllocation | None
    compute: ComputeLedger = ComputeLedger()
    shadow_races: tuple[ShadowRaceDecision, ...] = ()
    candidate_failures: tuple[CandidateFailure, ...] = ()
    pair_attempts: tuple[tuple[str, PairAttemptFacts], ...] = ()
    refill_attempts: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ResolveProposal:
    """A pending proposal awaits accept/reject disposition."""

    proposal_index: int


@dataclass(frozen=True, slots=True)
class ExecutePair:
    """The next pair of the frozen task plan is ready to run."""

    task: PairTask


@dataclass(frozen=True, slots=True)
class EmitObservation:
    """A candidate has a complete pair prefix and no observation for this phase."""

    candidate_id: str
    phase: Phase


@dataclass(frozen=True, slots=True)
class EmitShadowRace:
    """Record the evidence-only race decision for a completed tuning prefix."""

    cohort_index: int
    prefix_id: str


@dataclass(frozen=True, slots=True)
class CompleteCohort:
    """Every accepted candidate has a tuning observation; close the cohort."""


@dataclass(frozen=True, slots=True)
class StartNextCohort:
    """Retain the latest cohort's elites before introducing challengers."""


@dataclass(frozen=True, slots=True)
class DeepenCohort:
    """Advance the complete cohort to its next cumulative tuning prefix."""

    block_index: int
    prefix_id: str


@dataclass(frozen=True, slots=True)
class SelectFinalists:
    """The cohort is closed and finalists have not been chosen."""


@dataclass(frozen=True, slots=True)
class IntroduceProposal:
    """Fewer than the target number of candidates are accepted and no other work is pending."""


@dataclass(frozen=True, slots=True)
class FailCandidate:
    failure: CandidateFailure


@dataclass(frozen=True, slots=True)
class CompleteRun:
    """Every finalist has a validation observation; write the run completion."""


@dataclass(frozen=True, slots=True)
class NoDecision:
    """No fixed-cohort operation applies; the caller raises."""


@dataclass(frozen=True, slots=True)
class IntroduceCandidate:
    cohort_slot: int
    source: ProposalSource


@dataclass(frozen=True, slots=True)
class RefillCandidate:
    cohort_slot: int
    source: ProposalSource
    failed_candidate_id: str


@dataclass(frozen=True, slots=True)
class DeepenCohortAllocation:
    block_index: int
    prefix_id: str


@dataclass(frozen=True, slots=True)
class BeginValidation:
    tuning_prefix_id: str


@dataclass(frozen=True, slots=True)
class RetainElites:
    cohort_index: int
    candidate_ids: tuple[str, ...]
    prefix_id: str


@dataclass(frozen=True, slots=True)
class CohortRecord:
    cohort_index: int
    candidates: tuple[Candidate, ...]
    retained_candidate_ids: tuple[str, ...]


ResourceAllocation = (
    IntroduceCandidate | RefillCandidate | DeepenCohortAllocation | BeginValidation | RetainElites
)


AllocationDecision = (
    ResolveProposal
    | ExecutePair
    | EmitObservation
    | EmitShadowRace
    | CompleteCohort
    | StartNextCohort
    | DeepenCohort
    | SelectFinalists
    | IntroduceProposal
    | FailCandidate
    | CompleteRun
    | NoDecision
)
