"""Finite, explicit evidence transitions for a version-2 tuning run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .artifacts import Manifest
from .domain import Candidate, IterationBudget, Observation, PairResult, Proposal, ReplayState
from .evidence import EvidenceEvent, decode_pair_payload
from .identity import candidate_from_canonical_config, pair_task
from .statistics import marginal_interval, pair_utility


def expected_pairs(
    manifest: Manifest,
    cohort: tuple[Candidate, ...],
    finalists: tuple[Candidate, ...] | None = None,
) -> tuple:
    """Reconstruct the deterministic pair plan; no event supplies task identity."""
    tuning = tuple(
        pair_task(candidate, case, IterationBudget(manifest.budgets["tuning"]))
        for case in manifest.tuning.cases
        for candidate in cohort
    )
    if finalists is None:
        return tuning
    validation = tuple(
        pair_task(candidate, case, IterationBudget(manifest.budgets["validation"]))
        for case in manifest.validation.cases
        for candidate in finalists
    )
    return tuning + validation


def _observation(
    candidate: Candidate, phase: str, manifest: Manifest, pairs: list[PairResult]
) -> Observation:
    block = manifest.tuning if phase == "tuning" else manifest.validation
    budget = IterationBudget(manifest.budgets[phase])
    by_task = {pair.task.task_case.task_id: pair for pair in pairs}
    if len(by_task) != len(block.cases) or set(by_task) != {case.task_id for case in block.cases}:
        raise ValueError("observation needs a complete common task block")
    utilities = tuple(pair_utility(by_task[case.task_id]) for case in block.cases)
    return Observation(
        candidate.candidate_id,
        phase,
        block.block_id,
        len(utilities),
        budget,
        utilities,
        marginal_interval(utilities),
    )  # type: ignore[arg-type]


def observation_payload(observation: Observation) -> dict[str, object]:
    return {
        "candidate_id": observation.candidate_id,
        "phase": observation.phase,
        "block_id": observation.block_id,
        "prefix_length": observation.prefix_length,
        "budget": observation.budget.max_iterations,
        "pair_utilities": list(observation.pair_utilities),
        "estimate": asdict(observation.estimate),
        "counts": {"pairs": observation.prefix_length, "games": observation.prefix_length * 2},
    }


def _selection(
    cohort: tuple[Candidate, ...], observations: list[Observation], manifest: Manifest
) -> tuple[Candidate, ...]:
    means = {item.candidate_id: item.estimate.mean for item in observations}
    if set(means) != {candidate.candidate_id for candidate in cohort}:
        raise ValueError("finalist selection needs all tuning observations")
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
    if not isinstance(payload["canonical_config"], str):
        raise ValueError("proposal configuration must be a string")
    candidate = candidate_from_canonical_config(payload["canonical_config"])
    if (
        candidate.candidate_id != payload["candidate_id"]
        or candidate.fingerprint != payload["fingerprint"]
    ):
        raise ValueError("proposal candidate identity is invalid")
    return Proposal(
        payload["proposal_index"], payload["source"], payload["proposer_version"], candidate
    )  # type: ignore[arg-type]


def _apply_proposal_created(state: _Replay, payload: dict[str, object]) -> None:
    proposal = _proposal(payload)
    if proposal.proposal_index != len(state.proposals):
        raise ValueError("proposal indices must be contiguous")
    if proposal.proposal_index == 0:
        if proposal.source != "schema_default" or proposal.candidate != state.manifest.opponent:
            raise ValueError("proposal zero must be the schema default")
    elif proposal.source != "configspace_random":
        raise ValueError("non-default proposal must use ConfigSpace")
    state.proposals.append(proposal)


def _apply_disposition(state: _Replay, event: EvidenceEvent) -> None:
    payload = event.payload
    index = payload["proposal_index"]
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
    state.dispositions[index] = "accepted" if event.type == "proposal_accepted" else "rejected"
    accepted = state.accepted()
    if len(accepted) > state.manifest.cohort_size or len(
        {item.fingerprint for item in accepted}
    ) != len(accepted):
        raise ValueError("accepted cohort has duplicate or excessive candidates")


def _apply_cohort(state: _Replay, payload: dict[str, object]) -> None:
    if state.cohort is not None or len(state.accepted()) != state.manifest.cohort_size:
        raise ValueError("cohort is incomplete or already frozen")
    if payload["candidate_ids"] != [item.candidate_id for item in state.accepted()]:
        raise ValueError("cohort order is invalid")
    state.cohort = state.accepted()


def _apply_pair(state: _Replay, payload: dict[str, object]) -> None:
    plan = state.plan()
    if len(state.completed) >= len(plan):
        raise ValueError("pair completion is not expected")
    state.completed.append(decode_pair_payload(payload, plan[len(state.completed)]))


def _apply_observation(state: _Replay, payload: dict[str, object]) -> None:
    if state.cohort is None:
        raise ValueError("observation precedes cohort")
    phase = payload["phase"]
    candidates = state.cohort if phase == "tuning" else state.finalists
    if candidates is None:
        raise ValueError("validation observation precedes finalist selection")
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
    observation = _observation(candidate, phase, state.manifest, pairs)
    if payload != observation_payload(observation):
        raise ValueError("observation does not match completed raw pairs")
    state.observations.append(observation)


def _apply_finalists(state: _Replay, payload: dict[str, object]) -> None:
    if (
        state.cohort is None
        or state.finalists is not None
        or len(state.completed) != len(state.plan())
    ):
        raise ValueError("finalist selection is premature")
    tuning = [item for item in state.observations if item.phase == "tuning"]
    finalists = _selection(state.cohort, tuning, state.manifest)
    expected = {
        "finalist_ids": [item.candidate_id for item in finalists],
        "tuning_estimates": {item.candidate_id: item.estimate.mean for item in tuning},
        "source_block": state.manifest.tuning.block_id,
        "budget": state.manifest.budgets["tuning"],
        "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
    }
    if payload != expected:
        raise ValueError("finalist selection does not match tuning evidence")
    state.finalists = finalists


def _apply_completion(state: _Replay, event: EvidenceEvent) -> None:
    if state.cohort is None or state.finalists is None or len(state.completed) != len(state.plan()):
        raise ValueError("run completion is premature")
    if len([item for item in state.observations if item.phase == "validation"]) != len(
        state.finalists
    ):
        raise ValueError("run completion needs all validation observations")
    claim = (
        "production"
        if state.manifest.budgets["validation"] == state.manifest.budgets["production"]
        else "mechanics_smoke"
    )
    scientific_events = (
        len(state.proposals)
        + len(state.dispositions)
        + 1  # cohort_accepted
        + len(state.completed)
        + len(state.observations)
        + 1  # finalists_selected
        + 1  # this run_completed event
    )
    expected = {
        "manifest_fingerprint": state.manifest.fingerprint,
        "accepted_ids": [item.candidate_id for item in state.cohort],
        "finalist_ids": [item.candidate_id for item in state.finalists],
        "evidence_counts": {"events": scientific_events},
        "validation_claim": claim,
    }
    if event.payload != expected:
        raise ValueError("run completion does not bind replay state")
    state.terminal = "complete"


def _expected_operational_pair(state: _Replay) -> object:
    plan = state.plan()
    if not plan or len(state.completed) >= len(plan):
        raise ValueError("operational pair record is not expected")
    return plan[len(state.completed)]


def _apply_pair_operational(state: _Replay, event: EvidenceEvent) -> None:
    task = _expected_operational_pair(state)
    payload = event.payload
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


def _apply_interruption(state: _Replay, payload: dict[str, object]) -> None:
    pair_id = payload["pair_id"]
    if pair_id is None:
        if state.cohort is not None:
            raise ValueError("pair interruption must identify the pending pair")
        return
    task = _expected_operational_pair(state)
    if pair_id != task.pair_id:
        raise ValueError("interruption does not identify the pending pair")


def _apply(state: _Replay, event: EvidenceEvent) -> None:
    """Dispatch one already-decoded event; each transition owns one invariant set."""
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
        if state.cohort is not None or state.completed:
            raise ValueError("configuration failure cannot follow game evidence")
        state.terminal = "configuration_failed"
    elif event.type in {"pair_started", "pair_failed"}:
        _apply_pair_operational(state, event)
    elif event.type == "run_interrupted":
        _apply_interruption(state, event.payload)
    else:
        raise ValueError(f"unhandled evidence event {event.type}")


def fold_events(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    """Fold events in order; the dispatcher intentionally contains no policy logic."""
    state = _Replay(manifest)
    for event in events:
        _apply(state, event)
    next_pair = None
    if state.terminal == "open" and state.cohort is not None:
        plan = state.plan()
        if len(state.completed) < len(plan):
            next_pair = plan[len(state.completed)].pair_id
    return ReplayState(
        tuple(state.proposals),
        tuple(sorted(state.dispositions.items())),
        state.cohort,
        tuple(state.completed),
        tuple(state.observations),
        state.finalists,
        state.terminal,
        next_pair,
    )  # type: ignore[arg-type]


def replay(manifest: Manifest, events: list[EvidenceEvent]) -> ReplayState:
    return fold_events(manifest, events)
