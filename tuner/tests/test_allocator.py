"""The allocator reproduces the pre-allocator fixed-cohort order exactly.

Each case folds a prefix of the checked-in golden evidence stream to a real
``ReplayState`` and asserts the single decision ``advance_one`` would have acted
on at that point. The ``NoDecision`` case is the one state that is not reachable
in a well-formed run, so it is built by hand.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tuner_cli.allocator import decide_allocation
from tuner_cli.artifacts import Manifest, read_manifest
from tuner_cli.domain import (
    AllocationDecision,
    CompleteCohort,
    CompleteRun,
    EmitObservation,
    ExecutePair,
    IntroduceProposal,
    NoDecision,
    ResolveProposal,
    SelectFinalists,
)
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


# (prefix length, exact decision) for every stage whose decision carries no
# state-derived payload.
_EXACT: list[tuple[int, AllocationDecision]] = [
    (0, IntroduceProposal()),
    (1, ResolveProposal(0)),
    (2, IntroduceProposal()),
    (36, CompleteCohort()),
    (37, SelectFinalists()),
    (48, CompleteRun()),
]


@pytest.mark.parametrize(("count", "expected"), _EXACT)
def test_exact_stage_decisions(
    manifest: Manifest, events: list[EvidenceEvent], count: int, expected: AllocationDecision
) -> None:
    assert _decide(manifest, events, count) == expected


@pytest.mark.parametrize(("count", "phase"), [(8, "tuning"), (10, "tuning"), (38, "validation")])
def test_pending_pair_is_executed(
    manifest: Manifest, events: list[EvidenceEvent], count: int, phase: str
) -> None:
    state = replay(manifest, events[:count])
    decision = decide_allocation(manifest, state)
    assert isinstance(decision, ExecutePair)
    assert decision.task.pair_id == state.next_pair_id
    assert decision.task.task_case.phase == phase


@pytest.mark.parametrize(("count", "phase"), [(12, "tuning"), (46, "validation")])
def test_complete_prefix_emits_an_observation(
    manifest: Manifest, events: list[EvidenceEvent], count: int, phase: str
) -> None:
    decision = _decide(manifest, events, count)
    assert isinstance(decision, EmitObservation)
    assert decision.phase == phase


def test_no_available_operation_yields_no_decision(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    # Finalists chosen, no validation work recorded, and nothing points at the
    # next pair: the state that falls through to ``advance_one``'s raise.
    stranded = replace(replay(manifest, events[:38]), next_pair_id=None, completed_pairs=())
    assert decide_allocation(manifest, stranded) == NoDecision()
