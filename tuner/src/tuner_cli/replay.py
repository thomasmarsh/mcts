"""Finite, strict evidence transitions for a panel-aware tuning run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .artifacts import Manifest, production_claim
from .domain import Candidate, Observation, ObservationContext, PairResult, Proposal, ReplayState
from .evidence import EvidenceEvent, decode_pair_payload
from .identity import candidate_from_canonical_config, pair_task
from .observations import comparable, observation
from .statistics import pair_utility


def expected_pairs(
    manifest: Manifest,
    cohort: tuple[Candidate, ...],
    finalists: tuple[Candidate, ...] | None = None,
) -> tuple:
    tuning = tuple(
        pair_task(candidate, case, manifest.efforts["tuning"])
        for case in manifest.prefix_cases("tuning")
        for candidate in cohort
    )
    if finalists is None:
        return tuning
    validation = tuple(
        pair_task(candidate, case, manifest.efforts["validation"])
        for case in manifest.prefix_cases("validation")
        for candidate in finalists
    )
    return tuning + validation


def _context(manifest: Manifest, phase: str) -> ObservationContext:
    prefix = manifest.tuning_prefix if phase == "tuning" else manifest.validation_prefix
    return ObservationContext(manifest.epoch.epoch_id, phase, prefix, manifest.efforts[phase])


def _observation(
    candidate: Candidate, phase: str, manifest: Manifest, pairs: list[PairResult]
) -> Observation:
    cases = manifest.prefix_cases(phase)
    by_task = {pair.task.task_case.task_id: pair for pair in pairs}
    if len(by_task) != len(cases) or tuple(by_task) != tuple(case.task_id for case in cases):
        if set(by_task) != {case.task_id for case in cases}:
            raise ValueError("observation needs a complete common task prefix")
    utilities = tuple(pair_utility(by_task[case.task_id]) for case in cases)
    return observation(candidate.candidate_id, _context(manifest, phase), utilities)


def observation_payload(value: Observation, opponent_count: int) -> dict[str, object]:
    context = value.context
    return {
        "candidate_id": value.candidate_id,
        "phase": context.phase,
        "objective_epoch_id": context.objective_epoch_id,
        "corpus_id": context.task_prefix.corpus_id,
        "prefix_id": context.task_prefix.prefix_id,
        "prefix_task_ids": list(context.task_prefix.task_ids),
        "prefix_length": context.task_prefix.length,
        "search_effort": context.search_effort.max_iterations,
        "pair_utilities": list(value.pair_utilities),
        "estimate": asdict(value.estimate),
        "counts": {
            "pairs": context.task_prefix.length,
            "games": context.task_prefix.length * 2,
            "opponents": opponent_count,
        },
    }


def _selection(
    cohort: tuple[Candidate, ...], values: list[Observation], manifest: Manifest
) -> tuple[Candidate, ...]:
    if len(values) != len(cohort) or {item.candidate_id for item in values} != {
        item.candidate_id for item in cohort
    }:
        raise ValueError("finalist selection needs all tuning observations")
    for value in values[1:]:
        comparable(values[0], value)
    means = {item.candidate_id: item.estimate.mean for item in values}
    return tuple(
        sorted(
            cohort, key=lambda candidate: (-means[candidate.candidate_id], candidate.fingerprint)
        )[: manifest.finalists]
    )


@dataclass(slots=True)
class _Replay:
    manifest: Manifest
    proposals: list[Proposal] = field(default_factory=list)
    dispositions: dict[int, str] = field(default_factory=dict)
    cohort: tuple[Candidate, ...] | None = None
    completed: list[PairResult] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    finalists: tuple[Candidate, ...] | None = None
    terminal: str = "open"

    def accepted(self) -> tuple[Candidate, ...]:
        return tuple(
            item.candidate
            for item in self.proposals
            if self.dispositions.get(item.proposal_index) == "accepted"
        )

    def plan(self) -> tuple:
        return (
            ()
            if self.cohort is None
            else expected_pairs(self.manifest, self.cohort, self.finalists)
        )


def _proposal(payload: dict[str, object]) -> Proposal:
    candidate = candidate_from_canonical_config(
        payload["canonical_config"] if isinstance(payload["canonical_config"], str) else ""
    )
    if (
        candidate.candidate_id != payload["candidate_id"]
        or candidate.fingerprint != payload["fingerprint"]
    ):
        raise ValueError("proposal candidate identity is invalid")
    return Proposal(
        payload["proposal_index"], payload["source"], payload["proposer_version"], candidate
    )


def _apply_proposal_created(state: _Replay, payload: dict[str, object]) -> None:
    proposal = _proposal(payload)
    if proposal.proposal_index != len(state.proposals):
        raise ValueError("proposal indices must be contiguous")
    if proposal.proposal_index == 0:
        default = candidate_from_canonical_config(state.manifest.opponent.canonical_config)
        if proposal.source != "schema_default" or proposal.candidate != default:
            raise ValueError("proposal zero must be the schema default")
    elif proposal.source != "configspace_random":
        raise ValueError("non-default proposal must use ConfigSpace")
    state.proposals.append(proposal)


def _apply_disposition(state: _Replay, event: EvidenceEvent) -> None:
    payload, index = event.payload, event.payload["proposal_index"]
    if (
        not isinstance(index, int)
        or index not in range(len(state.proposals))
        or index in state.dispositions
    ):
        raise ValueError("invalid or repeated proposal disposition")
    candidate = state.proposals[index].candidate
    if any(
        payload[key] != getattr(candidate, key)
        for key in ("candidate_id", "fingerprint", "canonical_config")
    ):
        raise ValueError("proposal disposition does not reference its proposal")
    if event.type == "proposal_rejected" and payload["reason"] == "semantic_validation":
        panel_ids = [item.opponent_id for item in state.manifest.panel.opponents]
        rejected_ids = [item.get("opponent_id") for item in payload["errors"]]
        if any(item not in panel_ids for item in rejected_ids) or rejected_ids != sorted(
            rejected_ids, key=panel_ids.index
        ):
            raise ValueError("proposal rejection does not preserve panel order")
    state.dispositions[index] = "accepted" if event.type == "proposal_accepted" else "rejected"
    if len(state.accepted()) > state.manifest.cohort_size or len(
        {item.fingerprint for item in state.accepted()}
    ) != len(state.accepted()):
        raise ValueError("accepted cohort has duplicate or excessive candidates")


def _apply_cohort(state: _Replay, payload: dict[str, object]) -> None:
    if state.cohort is not None or len(state.accepted()) != state.manifest.cohort_size:
        raise ValueError("cohort is incomplete or already frozen")
    if payload["candidate_ids"] != [item.candidate_id for item in state.accepted()] or len(
        payload["validation_response_fingerprints"]
    ) != len(state.manifest.panel.opponents):
        raise ValueError("cohort validation does not bind the frozen panel")
    state.cohort = state.accepted()


def _apply_pair(state: _Replay, payload: dict[str, object]) -> None:
    plan = state.plan()
    if len(state.completed) >= len(plan):
        raise ValueError("pair completion is not expected")
    state.completed.append(decode_pair_payload(payload, plan[len(state.completed)]))


def _apply_observation(state: _Replay, payload: dict[str, object]) -> None:
    phase = payload["phase"]
    candidates = state.cohort if phase == "tuning" else state.finalists
    if candidates is None:
        raise ValueError("observation precedes cohort or finalist selection")
    candidate = next(
        (item for item in candidates if item.candidate_id == payload["candidate_id"]), None
    )
    if candidate is None or any(
        item.phase == phase and item.candidate_id == candidate.candidate_id
        for item in state.observations
    ):
        raise ValueError("invalid or repeated observation")
    pairs = [
        item
        for item in state.completed
        if item.task.candidate_id == candidate.candidate_id and item.task.task_case.phase == phase
    ]
    value = _observation(candidate, phase, state.manifest, pairs)
    opponent_count = len({pair.task.task_case.opponent_id for pair in pairs})
    if payload != observation_payload(value, opponent_count):
        raise ValueError("observation does not match completed raw pairs")
    state.observations.append(value)


def _apply_finalists(state: _Replay, payload: dict[str, object]) -> None:
    if (
        state.cohort is None
        or state.finalists is not None
        or len(state.completed) != len(state.plan())
    ):
        raise ValueError("finalist selection is premature")
    tuning = [item for item in state.observations if item.phase == "tuning"]
    finalists = _selection(state.cohort, tuning, state.manifest)
    context = _context(state.manifest, "tuning")
    expected = {
        "finalist_ids": [item.candidate_id for item in finalists],
        "tuning_estimates": {item.candidate_id: item.estimate.mean for item in tuning},
        "objective_epoch_id": context.objective_epoch_id,
        "corpus_id": context.task_prefix.corpus_id,
        "prefix_id": context.task_prefix.prefix_id,
        "prefix_task_ids": list(context.task_prefix.task_ids),
        "search_effort": context.search_effort.max_iterations,
        "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
    }
    if payload != expected:
        raise ValueError("finalist selection does not match tuning evidence")
    state.finalists = finalists


def _apply_completion(state: _Replay, event: EvidenceEvent) -> None:
    if (
        state.cohort is None
        or state.finalists is None
        or len(state.completed) != len(state.plan())
        or len([item for item in state.observations if item.phase == "validation"])
        != len(state.finalists)
    ):
        raise ValueError("run completion is premature")
    claim, missing = production_claim(
        state.manifest.validation_prefix,
        state.manifest.production_validation_corpus,
        state.manifest.efforts["validation"],
        state.manifest.efforts["production"],
    )
    scientific = (
        len(state.proposals)
        + len(state.dispositions)
        + 1
        + len(state.completed)
        + len(state.observations)
        + 1
        + 1
    )
    expected = {
        "manifest_fingerprint": state.manifest.fingerprint,
        "accepted_ids": [item.candidate_id for item in state.cohort],
        "finalist_ids": [item.candidate_id for item in state.finalists],
        "evidence_counts": {"events": scientific},
        "validation_claim": claim,
        "objective_epoch_id": state.manifest.epoch.epoch_id,
        "validation_prefix_id": state.manifest.validation_prefix.prefix_id,
        "validation_search_effort": state.manifest.efforts["validation"].max_iterations,
        "missing_production_axes": list(missing),
    }
    if event.payload != expected:
        raise ValueError("run completion does not bind replay state")
    state.terminal = "complete"


def _operational_pair(state: _Replay, event: EvidenceEvent) -> None:
    plan = state.plan()
    if len(state.completed) >= len(plan):
        raise ValueError("operational pair record is not expected")
    task, payload = plan[len(state.completed)], event.payload
    expected = {
        "phase": task.task_case.phase,
        "candidate_id": task.candidate_id,
        "task_id": task.task_case.task_id,
        "pair_id": task.pair_id,
        "opponent_id": task.task_case.opponent_id,
        "budget": task.budget.max_iterations,
    }
    if any(payload[key] != value for key, value in expected.items()):
        raise ValueError("operational pair record does not match pending pair")
    if event.type == "pair_started" and payload["task_seed"] != task.task_case.seed:
        raise ValueError("pair start does not match pending task seed")


def _apply(state: _Replay, event: EvidenceEvent) -> None:
    if state.terminal != "open":
        raise ValueError("event follows terminal run state")
    if event.type == "proposal_created":
        _apply_proposal_created(state, event.payload)
    elif event.type in {"proposal_accepted", "proposal_rejected"}:
        _apply_disposition(state, event)
    elif event.type == "cohort_accepted":
        _apply_cohort(state, event.payload)
    elif event.type == "pair_completed":
        _apply_pair(state, event.payload)
    elif event.type == "observation_completed":
        _apply_observation(state, event.payload)
    elif event.type == "finalists_selected":
        _apply_finalists(state, event.payload)
    elif event.type == "run_completed":
        _apply_completion(state, event)
    elif event.type == "run_failed":
        state.terminal = "configuration_failed"
    elif event.type in {"pair_started", "pair_failed"}:
        _operational_pair(state, event)
    elif event.type == "run_interrupted":
        if (
            event.payload["pair_id"] is not None
            and event.payload["pair_id"] != state.plan()[len(state.completed)].pair_id
        ):
            raise ValueError("interruption does not identify pending pair")
    else:
        raise ValueError(f"unhandled evidence event {event.type}")


def fold_events(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    state = _Replay(manifest)
    for event in events:
        _apply(state, event)
    next_pair = None
    if (
        state.terminal == "open"
        and state.cohort is not None
        and len(state.completed) < len(state.plan())
    ):
        next_pair = state.plan()[len(state.completed)].pair_id
    return ReplayState(
        tuple(state.proposals),
        tuple(sorted(state.dispositions.items())),
        state.cohort,
        tuple(state.completed),
        tuple(state.observations),
        state.finalists,
        state.terminal,
        next_pair,
    )


def replay(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    return fold_events(manifest, events)
