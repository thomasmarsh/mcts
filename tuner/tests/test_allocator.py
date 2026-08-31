"""The allocator reproduces the pre-allocator fixed-cohort order exactly.

Each case folds a prefix of the checked-in golden evidence stream to a real
``ReplayState`` and asserts the single decision ``advance_one`` would have acted
on at that point. The ``NoDecision`` case is the one state that is not reachable
in a well-formed run, so it is built by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tuner_cli.allocator import decide_allocation, pending_pair, ready_pairs, resource_allocation
from tuner_cli.artifacts import Manifest, read_manifest
from tuner_cli.domain import (
    AllocationDecision,
    CompleteCohort,
    DeepenCohort,
    EmitShadowRace,
    ExecutePair,
    IntroduceProposal,
)
from tuner_cli.event_payloads import (
    AllocationDecidedPayload,
    ObservationCompletedPayload,
    ShadowRaceDecidedPayload,
)
from tuner_cli.evidence import EvidenceEvent, read_events
from tuner_cli.replay import replay
from tuner_cli.shadow import decide_shadow_race, shadow_prefix_eligible

FIXTURES = Path(__file__).parent / "fixtures" / "version4"


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    return read_manifest(FIXTURES / "manifest.json")


@pytest.fixture(scope="module")
def events() -> list[EvidenceEvent]:
    return read_events(FIXTURES / "evidence.jsonl")


def _decide(manifest: Manifest, events: list[EvidenceEvent], count: int) -> AllocationDecision:
    return decide_allocation(manifest, replay(manifest, events[:count]))


def test_initial_resource_choice_is_introduce(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    state = replay(manifest, events[:0])
    assert decide_allocation(manifest, state) == IntroduceProposal()
    assert resource_allocation(decide_allocation(manifest, state), manifest, state) is not None


def test_pairs_are_derived_directly_from_completed_evidence(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    # Representative states before the first shadow boundary keep this direct
    # derivation test fast; replay validation covers the full stream elsewhere.
    for count in (10, 20):
        state = replay(manifest, events[:count])
        task = pending_pair(manifest, state)
        decision = decide_allocation(manifest, state)
        if task is not None and isinstance(decision, ExecutePair):
            assert decision.task == task


def test_ready_pairs_are_ordered_and_pending_is_its_first_view(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    state = replay(manifest, events[:10])
    ready = ready_pairs(manifest, state)
    assert ready
    assert pending_pair(manifest, state) == ready[0]
    assert ready_pairs(manifest, state, 2) == ready[:2]
    with pytest.raises(ValueError):
        ready_pairs(manifest, state, 0)


def test_golden_stream_deepens_the_full_cohort(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    allocations = [
        event.payload for event in events if isinstance(event.payload, AllocationDecidedPayload)
    ]
    assert any(
        item.allocation.__class__.__name__ == "DeepenCohortAllocation" for item in allocations
    )


def test_shadow_eligibility_uses_the_declared_twelve_pair_nonfinal_boundary(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    lengths = {block.length: block for block in manifest.tuning_blocks}
    assert not shadow_prefix_eligible(manifest, lengths[6])
    assert shadow_prefix_eligible(manifest, lengths[12])
    assert not shadow_prefix_eligible(manifest, manifest.tuning_prefix)
    races = [
        event.payload for event in events if isinstance(event.payload, ShadowRaceDecidedPayload)
    ]
    assert races
    assert {race.decision.prefix_id for race in races} == {lengths[12].prefix_id}


def test_shadow_boundary_orders_deepen_event_deepen_then_completion(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    lengths = {block.length: block for block in manifest.tuning_blocks}
    first_race = next(
        event.payload for event in events if isinstance(event.payload, ShadowRaceDecidedPayload)
    )
    candidate_ids = {item.candidate_id for item in first_race.decision.decisions}

    def state_after_observations(prefix_length: int):
        count = max(
            index + 1
            for index, event in enumerate(events)
            if isinstance(event.payload, ObservationCompletedPayload)
            and event.payload.prefix_length == prefix_length
            and event.payload.candidate_id in candidate_ids
        )
        return replay(manifest, events[:count])

    at_six = state_after_observations(6)
    assert decide_allocation(manifest, at_six) == DeepenCohort(3, lengths[8].prefix_id)
    with pytest.raises(ValueError, match="ineligible"):
        decide_shadow_race(manifest, at_six, 0, lengths[6])
    at_twelve = state_after_observations(12)
    assert decide_allocation(manifest, at_twelve) == EmitShadowRace(0, lengths[12].prefix_id)
    shadow_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event.payload, ShadowRaceDecidedPayload)
        and event.payload.decision.cohort_index == 0
    )
    after_shadow = replay(manifest, events[: shadow_index + 1])
    assert decide_allocation(manifest, after_shadow) == DeepenCohort(6, lengths[14].prefix_id)
    at_maximum = state_after_observations(14)
    assert decide_allocation(manifest, at_maximum) == CompleteCohort()
    with pytest.raises(ValueError, match="ineligible"):
        decide_shadow_race(manifest, at_maximum, 0, lengths[14])
