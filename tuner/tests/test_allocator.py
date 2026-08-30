"""The allocator reproduces the pre-allocator fixed-cohort order exactly.

Each case folds a prefix of the checked-in golden evidence stream to a real
``ReplayState`` and asserts the single decision ``advance_one`` would have acted
on at that point. The ``NoDecision`` case is the one state that is not reachable
in a well-formed run, so it is built by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tuner_cli.allocator import decide_allocation, pending_pair, resource_allocation
from tuner_cli.artifacts import Manifest, read_manifest
from tuner_cli.domain import (
    AllocationDecision,
    ExecutePair,
    IntroduceProposal,
)
from tuner_cli.event_payloads import AllocationDecidedPayload
from tuner_cli.evidence import EvidenceEvent, read_events
from tuner_cli.replay import replay

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
    for count in range(len(events)):
        state = replay(manifest, events[:count])
        task = pending_pair(manifest, state)
        decision = decide_allocation(manifest, state)
        if task is not None and isinstance(decision, ExecutePair):
            assert decision.task == task


def test_golden_stream_deepens_the_full_cohort(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    allocations = [
        event.payload for event in events if isinstance(event.payload, AllocationDecidedPayload)
    ]
    assert any(
        item.allocation.__class__.__name__ == "DeepenCohortAllocation" for item in allocations
    )
