"""Small fixed-stage handlers for a foreground tuning run."""

from __future__ import annotations

import json

from .artifacts import Manifest, production_claim
from .cohort import (
    accepted_candidates,
    create_proposal,
    pending_proposal,
    proposal_disposition,
    proposal_payload,
)
from .domain import Candidate, PairTask, ReplayState
from .evidence import SCIENTIFIC, EvidenceWriter, pair_payload, read_events
from .identity import canonical_json, pair_task
from .proposer import POLICY_VERSION, ModelProposer, tuning_frontier
from .replay import _observation, _selection, fold_events, observation_payload
from .schema import GameSpec
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
        if advance_one(manifest, writer, target, default, spec, model, timeout, state):
            continue
        raise RuntimeError("no fixed-cohort continuation operation is available")


def advance_one(
    manifest: Manifest,
    writer: EvidenceWriter,
    target: Target,
    default: Candidate,
    spec: GameSpec,
    model: ModelProposer,
    timeout: int,
    state: ReplayState,
) -> bool:
    if proposal := pending_proposal(state):
        event_type, payload = proposal_disposition(target, manifest, state, proposal)
        writer.append(event_type, payload)
        return True
    if task := pending_pair(manifest, state):
        execute_pair(manifest, writer, target, state, task, timeout)
        return True
    if emit_observation(manifest, writer, state):
        return True
    if complete_cohort(manifest, writer, state):
        return True
    if select_finalists(manifest, writer, state):
        return True
    if complete_run(manifest, writer, state):
        return True
    if len(accepted_candidates(state)) < manifest.cohort_size:
        writer.append(
            "proposal_created",
            proposal_payload(create_proposal(manifest, state, default, spec, model)),
        )
        return True
    return False


def pending_pair(manifest: Manifest, state: ReplayState) -> PairTask | None:
    if state.next_pair_id is None:
        return None
    for candidate in _pair_candidates(state):
        for case in manifest.prefix_cases(_pair_phase(state)):
            effort = manifest.efforts[_pair_phase(state)]
            task = pair_task(candidate, case, effort)
            if task.pair_id == state.next_pair_id:
                return task
    raise ValueError("replay pending pair is not part of the frozen task plan")


def _pair_candidates(state: ReplayState) -> tuple[Candidate, ...]:
    if state.finalists is not None:
        return state.finalists
    return accepted_candidates(state)


def _pair_phase(state: ReplayState) -> str:
    return "validation" if state.finalists is not None else "tuning"


def execute_pair(
    manifest: Manifest,
    writer: EvidenceWriter,
    target: Target,
    state: ReplayState,
    task: PairTask,
    timeout: int,
) -> None:
    candidate = next(
        item for item in _pair_candidates(state) if item.candidate_id == task.candidate_id
    )
    opponent = next(
        item for item in manifest.panel.opponents if item.opponent_id == task.task_case.opponent_id
    )
    writer.append("pair_started", _pair_started_payload(task))
    try:
        result = target.evaluate(
            task, candidate, opponent, manifest.spec.default_game_config, timeout
        )
    except PairExecutionError as error:
        writer.append("pair_failed", failure_payload(task, error))
        raise
    except KeyboardInterrupt as error:
        writer.append("run_interrupted", {"stage": "pair_execution", "pair_id": task.pair_id})
        raise KeyboardInterrupt from error
    writer.append("pair_completed", pair_payload(result))


def _pair_started_payload(task: PairTask) -> dict[str, object]:
    return {
        "phase": task.task_case.phase,
        "candidate_id": task.candidate_id,
        "task_id": task.task_case.task_id,
        "pair_id": task.pair_id,
        "opponent_id": task.task_case.opponent_id,
        "budget": task.budget.max_iterations,
        "task_seed": task.task_case.seed,
    }


def failure_payload(task: PairTask, error: PairExecutionError) -> dict[str, object]:
    partial = [
        canonical_json(record)
        for line in error.stdout.splitlines()
        if (record := json_record(line))
    ]
    identity = _pair_started_payload(task)
    identity.pop("task_seed")
    return {
        **identity,
        "kind": error.kind,
        "command": error.command,
        "returncode": error.returncode,
        "stderr": error.stderr,
        "stdout": error.stdout,
        "partial_output": partial,
    }


def json_record(line: str) -> dict[str, object] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return (
        value
        if isinstance(value, dict) and value.get("type") == "configured_match_result"
        else None
    )


def emit_observation(manifest: Manifest, writer: EvidenceWriter, state: ReplayState) -> bool:
    candidate, phase = observation_candidate(manifest, state)
    if candidate is None:
        return False
    pairs = matching_pairs(state, candidate, phase)
    prefix = manifest.tuning_prefix if phase == "tuning" else manifest.validation_prefix
    if len(pairs) != prefix.length:
        return False
    value = _observation(candidate, phase, manifest, pairs)
    opponent_count = len({pair.task.task_case.opponent_id for pair in pairs})
    writer.append("observation_completed", observation_payload(value, opponent_count))
    return True


def observation_candidate(manifest: Manifest, state: ReplayState) -> tuple[Candidate | None, str]:
    if state.finalists is not None:
        observed = {item.candidate_id for item in state.observations if item.phase == "validation"}
        return next(
            (item for item in state.finalists if item.candidate_id not in observed), None
        ), "validation"
    if len(accepted_candidates(state)) < manifest.bootstrap_candidates:
        return None, "tuning"
    observed = {item.candidate_id for item in state.observations if item.phase == "tuning"}
    return next(
        (item for item in accepted_candidates(state) if item.candidate_id not in observed), None
    ), "tuning"


def matching_pairs(state: ReplayState, candidate: Candidate, phase: str):
    return [
        pair
        for pair in state.completed_pairs
        if pair.task.candidate_id == candidate.candidate_id and pair.task.task_case.phase == phase
    ]


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
        "cohort_completed",
        {
            "candidate_ids": [item.candidate_id for item in accepted],
            "sources": list(manifest.source_schedule),
            "schedule_version": POLICY_VERSION,
            "final_frontier_id": tuning_frontier(tuning).frontier_id,
        },
    )
    return True


def select_finalists(manifest: Manifest, writer: EvidenceWriter, state: ReplayState) -> bool:
    if state.cohort is None or state.finalists is not None:
        return False
    tuning = tuple(item for item in state.observations if item.phase == "tuning")
    finalists = _selection(state.cohort, tuning, manifest)
    context = tuning[0].context
    writer.append(
        "finalists_selected",
        {
            "finalist_ids": [item.candidate_id for item in finalists],
            "tuning_estimates": {item.candidate_id: item.estimate.mean for item in tuning},
            "objective_epoch_id": context.objective_epoch_id,
            "corpus_id": context.task_prefix.corpus_id,
            "prefix_id": context.task_prefix.prefix_id,
            "prefix_task_ids": list(context.task_prefix.task_ids),
            "search_effort": context.search_effort.max_iterations,
            "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
        },
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
        "run_completed",
        {
            "manifest_fingerprint": manifest.fingerprint,
            "accepted_ids": [item.candidate_id for item in state.cohort or ()],
            "finalist_ids": [item.candidate_id for item in state.finalists],
            "evidence_counts": {"events": count},
            "validation_claim": claim,
            "objective_epoch_id": manifest.epoch.epoch_id,
            "validation_prefix_id": manifest.validation_prefix.prefix_id,
            "validation_search_effort": manifest.efforts["validation"].max_iterations,
            "missing_production_axes": list(missing),
            "cohort_frontier_id": tuning_frontier(
                tuple(item for item in state.observations if item.phase == "tuning")
            ).frontier_id,
        },
    )
    return True
