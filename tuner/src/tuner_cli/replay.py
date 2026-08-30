"""Strict staged replay for fixed bootstrap, model, and reserve proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .artifacts import Manifest, production_claim
from .domain import (
    Candidate,
    Observation,
    ObservationContext,
    PairResult,
    Proposal,
    ProposalProvenance,
    ReplayState,
)
from .evidence import EvidenceEvent, decode_pair_payload
from .identity import candidate_from_canonical_config, pair_task
from .observations import contextual_observation
from .proposer import POLICY_VERSION, empty_frontier, tuning_frontier
from .selection import select_finalists


def _context(manifest: Manifest, phase: str) -> ObservationContext:
    prefix = manifest.tuning_prefix if phase == "tuning" else manifest.validation_prefix
    return ObservationContext(manifest.epoch.epoch_id, phase, prefix, manifest.efforts[phase])


def observation_payload(value: Observation, opponent_count: int) -> dict[str, object]:
    context = value.context
    return {
        "observation_id": value.observation_id,
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


@dataclass(slots=True)
class _Replay:
    manifest: Manifest
    proposals: list[Proposal] = field(default_factory=list)
    dispositions: dict[int, str] = field(default_factory=dict)
    completed: list[PairResult] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    cohort: tuple[Candidate, ...] | None = None
    finalists: tuple[Candidate, ...] | None = None
    terminal: str = "open"

    def accepted(self) -> tuple[Candidate, ...]:
        return tuple(
            item.candidate
            for item in self.proposals
            if self.dispositions.get(item.proposal_index) == "accepted"
        )

    def tuning_observations(self) -> tuple[Observation, ...]:
        return tuple(item for item in self.observations if item.phase == "tuning")

    def visible_frontier(self):
        values = self.tuning_observations()
        return (
            tuning_frontier(values) if values else empty_frontier(_context(self.manifest, "tuning"))
        )

    def next_tuning_candidate(self) -> Candidate | None:
        if len(self.accepted()) < self.manifest.bootstrap_candidates:
            return None
        observed = {item.candidate_id for item in self.tuning_observations()}
        return next((item for item in self.accepted() if item.candidate_id not in observed), None)

    def pair_plan(self) -> tuple:
        candidate = self.next_tuning_candidate()
        if candidate is not None:
            return tuple(
                pair_task(candidate, case, self.manifest.efforts["tuning"])
                for case in self.manifest.prefix_cases("tuning")
            )
        if self.finalists is None:
            return ()
        return tuple(
            pair_task(candidate, case, self.manifest.efforts["validation"])
            for case in self.manifest.prefix_cases("validation")
            for candidate in self.finalists
        )

    def completed_in_plan(self) -> tuple[PairResult, ...]:
        ids = {item.pair_id for item in self.pair_plan()}
        return tuple(item for item in self.completed if item.task.pair_id in ids)


def _proposal(payload: dict[str, object]) -> Proposal:
    candidate = candidate_from_canonical_config(payload["canonical_config"])
    if (
        candidate.candidate_id != payload["candidate_id"]
        or candidate.fingerprint != payload["fingerprint"]
    ):
        raise ValueError("proposal candidate identity is invalid")
    provenance = ProposalProvenance(
        payload["source"],
        payload["proposer_version"],
        payload["source_attempt"],
        payload["origin"],
        payload["acquisition"],
        payload["prediction"],
        payload["uncertainty"],
        payload["parent_candidate_id"],
    )
    frontier = _frontier_from_payload(payload)
    return Proposal(
        payload["proposal_index"], payload["cohort_slot"], candidate, frontier, provenance
    )


def _frontier_from_payload(payload: dict[str, object]):
    from .domain import ObservationFrontier, SearchEffort

    return ObservationFrontier(
        payload["frontier_id"],
        "",
        "",
        (),
        SearchEffort(1),
        tuple(payload["frontier_observation_ids"]),
    )


def _apply_proposal_created(state: _Replay, payload: dict[str, object]) -> None:
    if len(state.accepted()) == state.manifest.cohort_size:
        raise ValueError("proposal follows a complete cohort")
    if state.proposals and state.proposals[-1].proposal_index not in state.dispositions:
        raise ValueError("proposal follows an undisposed proposal")
    if (
        len(state.accepted()) >= state.manifest.bootstrap_candidates
        and state.next_tuning_candidate() is not None
    ):
        raise ValueError("proposal precedes the accepted candidate's tuning observation")
    proposal = _proposal(payload)
    if proposal.proposal_index != len(state.proposals):
        raise ValueError("proposal indices must be contiguous")
    slot = len(state.accepted())
    if proposal.cohort_slot != slot or proposal.source != state.manifest.source_schedule[slot]:
        raise ValueError("proposal does not match the frozen source schedule")
    attempt = 1 + sum(item.source == proposal.source for item in state.proposals)
    if proposal.provenance.source_attempt != attempt:
        raise ValueError("proposal source attempt is not contiguous")
    frontier = state.visible_frontier()
    if (
        proposal.frontier.frontier_id != frontier.frontier_id
        or proposal.frontier.observation_ids != frontier.observation_ids
    ):
        raise ValueError("proposal does not bind the visible observation frontier")
    if slot < state.manifest.bootstrap_candidates and proposal.frontier.observation_ids:
        raise ValueError("bootstrap proposal has a nonempty frontier")
    if (
        slot >= state.manifest.bootstrap_candidates
        and len(proposal.frontier.observation_ids) != slot
    ):
        raise ValueError("guided proposal lacks a complete comparable frontier")
    if slot == 0:
        default = candidate_from_canonical_config(state.manifest.opponent.canonical_config)
        if proposal.source != "schema_default" or proposal.candidate != default:
            raise ValueError("proposal zero must be the schema default")
    state.proposals.append(proposal)


def _apply_disposition(state: _Replay, event: EvidenceEvent) -> None:
    payload, index = event.payload, event.payload["proposal_index"]
    if (
        not isinstance(index, int)
        or index not in range(len(state.proposals))
        or index in state.dispositions
    ):
        raise ValueError("invalid or repeated proposal disposition")
    proposal = state.proposals[index]
    expected = {
        "cohort_slot": proposal.cohort_slot,
        "source": proposal.source,
        "source_attempt": proposal.provenance.source_attempt,
        "candidate_id": proposal.candidate.candidate_id,
        "fingerprint": proposal.candidate.fingerprint,
        "canonical_config": proposal.candidate.canonical_config,
    }
    if any(payload[key] != value for key, value in expected.items()):
        raise ValueError("proposal disposition does not reference its proposal")
    if event.type == "proposal_accepted":
        if len(payload["panel_response_fingerprints"]) != len(state.manifest.panel.opponents):
            raise ValueError("proposal acceptance does not bind the panel")
    elif payload["reason"] == "semantic_validation":
        ordered = [item.opponent_id for item in state.manifest.panel.opponents]
        ids = [item["opponent_id"] for item in payload["errors"]]
        if ids != sorted(ids, key=ordered.index):
            raise ValueError("proposal rejection does not preserve panel order")
    state.dispositions[index] = "accepted" if event.type == "proposal_accepted" else "rejected"
    accepted = state.accepted()
    if len({item.fingerprint for item in accepted}) != len(accepted):
        raise ValueError("accepted cohort contains a duplicate")


def _apply_pair(state: _Replay, payload: dict[str, object]) -> None:
    plan, completed = state.pair_plan(), state.completed_in_plan()
    if len(completed) >= len(plan):
        raise ValueError("pair completion is not expected")
    state.completed.append(decode_pair_payload(payload, plan[len(completed)]))


def _apply_observation(state: _Replay, payload: dict[str, object]) -> None:
    phase = payload["phase"]
    candidates = state.accepted() if phase == "tuning" else state.finalists
    if candidates is None:
        raise ValueError("observation precedes finalist selection")
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
    value = contextual_observation(candidate, _context(state.manifest, phase), pairs)
    if payload != observation_payload(
        value, len({pair.task.task_case.opponent_id for pair in pairs})
    ):
        raise ValueError("observation does not match completed raw pairs")
    state.observations.append(value)


def _apply_cohort(state: _Replay, payload: dict[str, object]) -> None:
    accepted, observations = state.accepted(), state.tuning_observations()
    if (
        state.cohort is not None
        or len(accepted) != state.manifest.cohort_size
        or len(observations) != len(accepted)
    ):
        raise ValueError("cohort is incomplete or already frozen")
    frontier = tuning_frontier(observations)
    expected = {
        "candidate_ids": [item.candidate_id for item in accepted],
        "sources": list(state.manifest.source_schedule),
        "schedule_version": POLICY_VERSION,
        "final_frontier_id": frontier.frontier_id,
    }
    if payload != expected:
        raise ValueError("cohort completion does not bind its observations")
    state.cohort = accepted


def _apply_finalists(state: _Replay, payload: dict[str, object]) -> None:
    if state.cohort is None or state.finalists is not None:
        raise ValueError("finalist selection is premature")
    tuning = state.tuning_observations()
    finalists = select_finalists(state.cohort, tuning, state.manifest.finalists)
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
    expected = {
        "manifest_fingerprint": state.manifest.fingerprint,
        "accepted_ids": [item.candidate_id for item in state.cohort],
        "finalist_ids": [item.candidate_id for item in state.finalists],
        "evidence_counts": {"events": _scientific_count(state)},
        "validation_claim": claim,
        "objective_epoch_id": state.manifest.epoch.epoch_id,
        "validation_prefix_id": state.manifest.validation_prefix.prefix_id,
        "validation_search_effort": state.manifest.efforts["validation"].max_iterations,
        "missing_production_axes": list(missing),
        "cohort_frontier_id": tuning_frontier(state.tuning_observations()).frontier_id,
    }
    if event.payload != expected:
        raise ValueError("run completion does not bind replay state")
    state.terminal = "complete"


def _scientific_count(state: _Replay) -> int:
    return (
        len(state.proposals)
        + len(state.dispositions)
        + len(state.completed)
        + len(state.observations)
        + 3
    )


def _operational_pair(state: _Replay, event: EvidenceEvent) -> None:
    plan, completed = state.pair_plan(), state.completed_in_plan()
    if len(completed) >= len(plan):
        raise ValueError("operational pair record is not expected")
    task, payload = plan[len(completed)], event.payload
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
    elif event.type == "pair_completed":
        _apply_pair(state, event.payload)
    elif event.type == "observation_completed":
        _apply_observation(state, event.payload)
    elif event.type == "cohort_completed":
        _apply_cohort(state, event.payload)
    elif event.type == "finalists_selected":
        _apply_finalists(state, event.payload)
    elif event.type == "run_completed":
        _apply_completion(state, event)
    elif event.type == "run_failed":
        state.terminal = "configuration_failed"
    elif event.type in {"pair_started", "pair_failed"}:
        _operational_pair(state, event)
    elif event.type == "run_interrupted":
        return
    else:
        raise ValueError(f"unhandled evidence event {event.type}")


def fold_events(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    state = _Replay(manifest)
    for event in events:
        _apply(state, event)
    plan, completed = state.pair_plan(), state.completed_in_plan()
    next_pair = (
        plan[len(completed)].pair_id
        if state.terminal == "open" and len(completed) < len(plan)
        else None
    )
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
