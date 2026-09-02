"""Pure typed row builders: one function per table, no I/O.

Each builder maps already-decoded values -- ``(run_id, Manifest, ReplayState,
report object)`` -- to a list of frozen row dataclasses. The builders invent no
identity and compute no scientific value; they read what replay and the report
already expose.
"""

from __future__ import annotations

from dataclasses import dataclass

from tuner_cli.artifacts import Manifest, manifest_json
from tuner_cli.codec import JsonObject, elements, integer, json_object, number, string
from tuner_cli.domain import PairResult, PhaseCompute, ReplayState
from tuner_cli.identity import canonical_json
from tuner_cli.statistics import pair_utility


@dataclass(frozen=True, slots=True)
class RunRow:
    run_id: str
    manifest_run_id: str | None
    manifest_fingerprint: str | None
    terminal_status: str | None
    report_available: int
    ingest_error: str | None


@dataclass(frozen=True, slots=True)
class RunManifestRow:
    run_id: str
    manifest_json: str
    game_kind: str
    objective_id: str
    cohort_size: int
    finalists: int
    seed: int
    task_seed: int
    shadow_policy_kind: str
    active_elimination: int


@dataclass(frozen=True, slots=True)
class RunReportRow:
    run_id: str
    report_json: str
    schema_version: int
    status: str
    validation_claim: str


@dataclass(frozen=True, slots=True)
class CohortRow:
    run_id: str
    cohort_index: int
    candidate_ids: str
    retained_candidate_ids: str


@dataclass(frozen=True, slots=True)
class CandidateRow:
    run_id: str
    candidate_id: str
    fingerprint: str
    canonical_config: str
    cohort_index: int
    cohort_slot: int
    source: str
    parent_candidate_id: str | None


@dataclass(frozen=True, slots=True)
class ProposalRow:
    run_id: str
    proposal_index: int
    cohort_index: int
    cohort_slot: int
    candidate_id: str
    source: str
    source_attempt: int
    disposition: str | None
    frontier_id: str
    origin: str | None
    acquisition: float | None
    prediction: float | None
    uncertainty: float | None
    parent_candidate_id: str | None
    refill_of_candidate_id: str | None


@dataclass(frozen=True, slots=True)
class PairRow:
    run_id: str
    pair_id: str
    phase: str
    candidate_id: str
    task_id: str
    opponent_id: str
    pair_utility: float


@dataclass(frozen=True, slots=True)
class GameRow:
    run_id: str
    game_id: str
    pair_id: str
    candidate_side: str
    outcome: str
    plies: int
    elapsed_ms: int
    candidate_iterations_total: int
    opponent_iterations_total: int


@dataclass(frozen=True, slots=True)
class ObservationRow:
    run_id: str
    observation_id: str
    candidate_id: str
    phase: str
    prefix_id: str
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ShadowDecisionRow:
    run_id: str
    race_index: int
    cohort_index: int
    prefix_id: str
    candidate_id: str
    boundary_candidate_id: str
    disposition: str
    policy_kind: str
    policy_version: str


@dataclass(frozen=True, slots=True)
class ActiveEliminationDecisionRow:
    run_id: str
    batch_index: int
    cohort_index: int
    prefix_id: str
    candidate_id: str
    action: str
    margin_kind: str


@dataclass(frozen=True, slots=True)
class ValidationRow:
    run_id: str
    candidate_id: str
    rank: int
    estimate: float
    lower: float
    upper: float
    wins: int
    draws: int
    losses: int


@dataclass(frozen=True, slots=True)
class ComputePhaseRow:
    run_id: str
    phase: str
    pair_attempts: int
    completed_pairs: int
    failed_attempts: int
    censored_attempts: int
    physical_games: int
    search_iterations: int
    wall_time_ms: int


def run_manifest_row(run_id: str, manifest: Manifest) -> RunManifestRow:
    return RunManifestRow(
        run_id,
        canonical_json(manifest_json(manifest)),
        manifest.spec.kind,
        manifest.objective_id,
        manifest.cohort_size,
        manifest.finalists,
        manifest.seed,
        manifest.task_seed,
        manifest.shadow_policy.kind,
        0 if manifest.active_elimination is None else 1,
    )


def run_report_row(run_id: str, report: JsonObject) -> RunReportRow:
    claim = json_object(report["validation_claim"], "validation claim")
    return RunReportRow(
        run_id,
        canonical_json(report),
        integer(report["schema_version"], "report schema version"),
        string(report["status"], "report status"),
        string(claim["claim"], "validation claim value"),
    )


def cohort_rows(run_id: str, state: ReplayState) -> list[CohortRow]:
    return [
        CohortRow(
            run_id,
            cohort.cohort_index,
            canonical_json(list(candidate.candidate_id for candidate in cohort.candidates)),
            canonical_json(list(cohort.retained_candidate_ids)),
        )
        for cohort in state.completed_cohorts
    ]


def candidate_rows(run_id: str, state: ReplayState) -> list[CandidateRow]:
    accepted = {index for index, disposition in state.dispositions if disposition == "accepted"}
    seen: set[str] = set()
    rows: list[CandidateRow] = []
    for proposal in state.proposals:
        candidate_id = proposal.candidate.candidate_id
        if proposal.proposal_index not in accepted or candidate_id in seen:
            continue
        seen.add(candidate_id)
        rows.append(
            CandidateRow(
                run_id,
                candidate_id,
                proposal.candidate.fingerprint,
                proposal.candidate.canonical_config,
                proposal.cohort_index,
                proposal.cohort_slot,
                proposal.source,
                proposal.provenance.parent_candidate_id,
            )
        )
    return rows


def proposal_rows(run_id: str, state: ReplayState) -> list[ProposalRow]:
    disposition = dict(state.dispositions)
    refill = dict(state.refill_attempts)
    return [
        ProposalRow(
            run_id,
            proposal.proposal_index,
            proposal.cohort_index,
            proposal.cohort_slot,
            proposal.candidate.candidate_id,
            proposal.source,
            proposal.provenance.source_attempt,
            disposition.get(proposal.proposal_index),
            proposal.frontier.frontier_id,
            proposal.provenance.origin,
            proposal.provenance.acquisition,
            proposal.provenance.prediction,
            proposal.provenance.uncertainty,
            proposal.provenance.parent_candidate_id,
            refill.get(proposal.proposal_index),
        )
        for proposal in state.proposals
    ]


def _pair_row(run_id: str, pair: PairResult) -> PairRow:
    task = pair.task
    return PairRow(
        run_id,
        task.pair_id,
        task.task_case.phase,
        task.candidate_id,
        task.task_case.task_id,
        task.task_case.opponent_id,
        pair_utility(pair),
    )


def pair_rows(run_id: str, state: ReplayState) -> list[PairRow]:
    return [_pair_row(run_id, pair) for pair in state.completed_pairs]


def game_rows(run_id: str, state: ReplayState) -> list[GameRow]:
    return [
        GameRow(
            run_id,
            game.game_id,
            pair.task.pair_id,
            game.candidate_side,
            game.outcome,
            game.plies,
            game.elapsed_ms,
            game.candidate_metrics.iterations_total,
            game.opponent_metrics.iterations_total,
        )
        for pair in state.completed_pairs
        for game in pair.games
    ]


def observation_rows(run_id: str, state: ReplayState) -> list[ObservationRow]:
    return [
        ObservationRow(
            run_id,
            observation.observation_id,
            observation.candidate_id,
            observation.phase,
            observation.context.task_prefix.prefix_id,
            observation.estimate.mean,
            observation.estimate.lower,
            observation.estimate.upper,
        )
        for observation in state.observations
    ]


def shadow_decision_rows(run_id: str, state: ReplayState) -> list[ShadowDecisionRow]:
    return [
        ShadowDecisionRow(
            run_id,
            race_index,
            race.cohort_index,
            race.prefix_id,
            decision.candidate_id,
            race.boundary_candidate_id,
            decision.disposition,
            race.policy_kind,
            race.policy_version,
        )
        for race_index, race in enumerate(state.shadow_races)
        for decision in race.decisions
    ]


def active_elimination_decision_rows(
    run_id: str, state: ReplayState
) -> list[ActiveEliminationDecisionRow]:
    return [
        ActiveEliminationDecisionRow(
            run_id,
            batch_index,
            batch.cohort_index,
            batch.prefix_id,
            action.candidate_id,
            action.action,
            type(action.margin).__name__,
        )
        for batch_index, batch in enumerate(state.elimination_allocations)
        for action in batch.actions
    ]


def _validation_row(run_id: str, rank: int, entry: JsonObject) -> ValidationRow:
    marginal = json_object(entry["weighted_marginal"], "weighted marginal")
    interval = json_object(marginal["interval"], "weighted marginal interval")
    return ValidationRow(
        run_id,
        string(entry["candidate_id"], "validation candidate id"),
        rank,
        number(marginal["estimate"], "validation estimate"),
        number(interval["lower"], "validation interval lower"),
        number(interval["upper"], "validation interval upper"),
        integer(entry["wins"], "validation wins"),
        integer(entry["draws"], "validation draws"),
        integer(entry["losses"], "validation losses"),
    )


def validation_rows(run_id: str, report: JsonObject | None) -> list[ValidationRow]:
    if report is None:
        return []
    order = elements(report["validation_order"], "validation order")
    return [
        _validation_row(run_id, rank, json_object(entry, "validation order entry"))
        for rank, entry in enumerate(order)
    ]


def _compute_phase_row(run_id: str, phase: str, bucket: PhaseCompute) -> ComputePhaseRow:
    return ComputePhaseRow(
        run_id,
        phase,
        bucket.pair_attempts,
        bucket.completed_pairs,
        bucket.failed_attempts,
        bucket.censored_attempts,
        bucket.physical_games,
        bucket.search_iterations,
        bucket.wall_time_ms,
    )


def compute_phase_rows(run_id: str, state: ReplayState) -> list[ComputePhaseRow]:
    ledger = state.compute
    return [
        _compute_phase_row(run_id, "tuning", ledger.tuning),
        _compute_phase_row(run_id, "validation", ledger.validation),
        _compute_phase_row(run_id, "diagnostic", ledger.diagnostic),
    ]
