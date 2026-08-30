"""Small fixed-stage handlers for a foreground tuning run."""

from __future__ import annotations

from .allocator import (
    decide_allocation,
    matching_pairs,
    observation_candidate,
    pair_candidates,
    proposal_at,
)
from .artifacts import Manifest, production_claim
from .codec import JsonObject, is_json_object, strict_json
from .cohort import (
    accepted_candidates,
    create_proposal,
    proposal_disposition,
    proposal_payload,
)
from .domain import (
    Candidate,
    CompleteCohort,
    CompleteRun,
    EmitObservation,
    ExecutePair,
    IntroduceProposal,
    NoDecision,
    ObservationContext,
    PairTask,
    ReplayState,
    ResolveProposal,
    SelectFinalists,
)
from .event_payloads import (
    CohortCompletedPayload,
    FinalistsSelectedPayload,
    PairFailedPayload,
    PairIdentity,
    PairStartedPayload,
    RunCompletedPayload,
    RunInterruptedPayload,
)
from .evidence import SCIENTIFIC, EvidenceWriter, pair_payload, read_events
from .identity import canonical_json
from .observations import contextual_observation
from .proposer import POLICY_VERSION, ModelProposer, tuning_frontier
from .replay import fold_events, observation_payload
from .schema import GameSpec
from .selection import select_finalists as choose_finalists
from .target import PairExecutionError, Target


def continue_run(
    manifest: Manifest,
    writer: EvidenceWriter,
    target: Target,
    default: Candidate,
    spec: GameSpec,
    model: ModelProposer,
    timeout: int,
) -> None:
    while True:
        state = fold_events(manifest, read_events(writer.path))
        if state.terminal_status != "open":
            return
        advance_one(manifest, writer, target, default, spec, model, timeout, state)


def advance_one(
    manifest: Manifest,
    writer: EvidenceWriter,
    target: Target,
    default: Candidate,
    spec: GameSpec,
    model: ModelProposer,
    timeout: int,
    state: ReplayState,
) -> None:
    match decide_allocation(manifest, state):
        case ResolveProposal(proposal_index):
            proposal = proposal_at(state, proposal_index)
            writer.append(proposal_disposition(target, manifest, state, proposal))
        case ExecutePair(task):
            execute_pair(manifest, writer, target, state, task, timeout)
        case EmitObservation():
            emit_observation(manifest, writer, state)
        case CompleteCohort():
            complete_cohort(manifest, writer, state)
        case SelectFinalists():
            select_finalists(manifest, writer, state)
        case CompleteRun():
            complete_run(manifest, writer, state)
        case IntroduceProposal():
            writer.append(proposal_payload(create_proposal(manifest, state, default, spec, model)))
        case NoDecision():
            raise RuntimeError("no fixed-cohort continuation operation is available")


def execute_pair(
    manifest: Manifest,
    writer: EvidenceWriter,
    target: Target,
    state: ReplayState,
    task: PairTask,
    timeout: int,
) -> None:
    candidate = next(
        item for item in pair_candidates(state) if item.candidate_id == task.candidate_id
    )
    opponent = next(
        item for item in manifest.panel.opponents if item.opponent_id == task.task_case.opponent_id
    )
    writer.append(_pair_started_payload(task))
    try:
        result = target.evaluate(
            task, candidate, opponent, manifest.spec.default_game_config, timeout
        )
    except PairExecutionError as error:
        writer.append(failure_payload(task, error))
        raise
    except KeyboardInterrupt as error:
        writer.append(RunInterruptedPayload("pair_execution", task.pair_id))
        raise KeyboardInterrupt from error
    writer.append(pair_payload(result))


def _pair_identity(task: PairTask) -> PairIdentity:
    return PairIdentity(
        task.task_case.phase,
        task.candidate_id,
        task.task_case.task_id,
        task.pair_id,
        task.task_case.opponent_id,
        task.budget.max_iterations,
    )


def _pair_started_payload(task: PairTask) -> PairStartedPayload:
    return PairStartedPayload(_pair_identity(task), task.task_case.seed)


def failure_payload(task: PairTask, error: PairExecutionError) -> PairFailedPayload:
    partial: tuple[str, ...] = tuple(
        canonical_json(record)
        for line in error.stdout.splitlines()
        if (record := json_record(line))
    )
    return PairFailedPayload(
        _pair_identity(task),
        error.kind,
        tuple(error.command),
        error.returncode,
        error.stderr,
        error.stdout,
        partial,
    )


def json_record(line: str) -> JsonObject | None:
    try:
        value = strict_json(line, "partial game output")
    except ValueError:
        return None
    return (
        value if is_json_object(value) and value.get("type") == "configured_match_result" else None
    )


def emit_observation(manifest: Manifest, writer: EvidenceWriter, state: ReplayState) -> None:
    candidate, phase = observation_candidate(manifest, state)
    if candidate is None:
        return
    pairs = matching_pairs(state, candidate, phase)
    context = manifest.tuning_prefix if phase == "tuning" else manifest.validation_prefix
    value = contextual_observation(
        candidate,
        ObservationContext(manifest.epoch.epoch_id, phase, context, manifest.efforts[phase]),
        pairs,
    )
    opponent_count = len({pair.task.task_case.opponent_id for pair in pairs})
    writer.append(observation_payload(value, opponent_count))


def complete_cohort(manifest: Manifest, writer: EvidenceWriter, state: ReplayState) -> bool:
    accepted = accepted_candidates(state)
    tuning = tuple(item for item in state.observations if item.phase == "tuning")
    if (
        state.cohort is not None
        or len(accepted) != manifest.cohort_size
        or len(tuning) != len(accepted)
    ):
        return False
    writer.append(
        CohortCompletedPayload(
            tuple(item.candidate_id for item in accepted),
            tuple(manifest.source_schedule),
            POLICY_VERSION,
            tuning_frontier(tuning).frontier_id,
        )
    )
    return True


def select_finalists(manifest: Manifest, writer: EvidenceWriter, state: ReplayState) -> bool:
    if state.cohort is None or state.finalists is not None:
        return False
    tuning = tuple(item for item in state.observations if item.phase == "tuning")
    finalists = choose_finalists(state.cohort, tuning, manifest.finalists)
    context = tuning[0].context
    writer.append(
        FinalistsSelectedPayload(
            tuple(item.candidate_id for item in finalists),
            {item.candidate_id: item.estimate.mean for item in tuning},
            context.objective_epoch_id,
            context.task_prefix.corpus_id,
            context.task_prefix.prefix_id,
            context.task_prefix.task_ids,
            context.search_effort.max_iterations,
            "tuning_point_estimate_fingerprint_v1",
        )
    )
    return True


def complete_run(manifest: Manifest, writer: EvidenceWriter, state: ReplayState) -> bool:
    if state.finalists is None or state.terminal_status != "open":
        return False
    validation = [item for item in state.observations if item.phase == "validation"]
    if len(validation) != len(state.finalists):
        return False
    claim, missing = production_claim(
        manifest.validation_prefix,
        manifest.production_validation_corpus,
        manifest.efforts["validation"],
        manifest.efforts["production"],
    )
    count = sum(event.type in SCIENTIFIC for event in read_events(writer.path)) + 1
    writer.append(
        RunCompletedPayload(
            manifest.fingerprint,
            tuple(item.candidate_id for item in state.cohort or ()),
            tuple(item.candidate_id for item in state.finalists),
            {"events": count},
            claim,
            manifest.epoch.epoch_id,
            manifest.validation_prefix.prefix_id,
            manifest.efforts["validation"].max_iterations,
            tuple(missing),
            tuning_frontier(
                tuple(item for item in state.observations if item.phase == "tuning")
            ).frontier_id,
        )
    )
    return True
