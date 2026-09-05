"""Checkpoint-then-tail replay must equal a full from-scratch fold.

`ReplayCheckpoint` compresses `_Replay`'s bookkeeping (allocations,
budget-extension/superseded-evidence counts) that `ReplayState` alone does not
carry -- see the docstring on `ReplayCheckpoint` in `replay.py`. These tests
pin the load-bearing invariant: resuming from a checkpoint and folding only
the unread tail must produce a `ReplayState` identical, field for field, to
folding every event from scratch.
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import pytest
from test_run import FakeModel, FakeTarget, _budgeted_options

from tuner_cli.artifacts import Manifest, read_manifest
from tuner_cli.evidence import EvidenceEvent, read_events
from tuner_cli.replay import ReplayCheckpoint, fold_checkpoint, replay
from tuner_cli.run import run_foreground

FIXTURES = Path(__file__).parent / "fixtures" / "projection-root" / "version4-active-halving"


def _assert_checkpoint_then_tail_matches_full_replay(
    manifest: Manifest, events: list[EvidenceEvent]
) -> None:
    expected = replay(manifest, events)
    for split in range(len(events) + 1):
        prefix, tail = events[:split], events[split:]
        checkpoint = fold_checkpoint(manifest, prefix)
        resumed = fold_checkpoint(manifest, tail, resume_from=checkpoint)
        assert resumed.state == expected, f"split at {split} diverged from a full replay"


def test_active_halving_fixture_resumes_identically_at_every_split() -> None:
    """A real run exercising diagnostics, shadow races, and eliminations."""
    manifest = read_manifest(FIXTURES / "manifest.json")
    events = read_events(FIXTURES / "evidence.jsonl")
    _assert_checkpoint_then_tail_matches_full_replay(manifest, events)


def test_budget_extension_reopen_resumes_identically_at_every_split(tmp_path: Path) -> None:
    """Covers the trickiest bookkeeping: a reopened, budget-extended run whose
    `allocations`/`superseded_*` counts a bare `ReplayState` cannot carry."""
    options = _budgeted_options(tmp_path, 19)
    run_foreground(options, FakeTarget(), model_proposer=FakeModel())
    manifest = read_manifest(options.run_dir / "manifest.json")
    assert replay(manifest, read_events(options.run_dir / "evidence.jsonl")).terminal_status == (
        "complete"
    )

    run_foreground(
        replace(
            options,
            resume=True,
            extend_tuning_pairs=6,
            extend_reason="fund another cohort",
            extend_requested_at="2026-09-02T00:00:00+00:00",
        ),
        FakeTarget(),
        model_proposer=FakeModel(),
    )
    events = read_events(options.run_dir / "evidence.jsonl")
    assert any(event.type == "budget_extended" for event in events)
    _assert_checkpoint_then_tail_matches_full_replay(manifest, events)


def test_checkpoint_round_trips_through_pickle() -> None:
    """The projection store persists a checkpoint as a pickled blob -- confirm
    the dataclass (and everything it nests) actually survives that."""
    manifest = read_manifest(FIXTURES / "manifest.json")
    events = read_events(FIXTURES / "evidence.jsonl")
    checkpoint = fold_checkpoint(manifest, events)
    restored = pickle.loads(pickle.dumps(checkpoint))
    assert restored == checkpoint
    assert isinstance(restored, ReplayCheckpoint)


def test_invalid_extension_still_raises_when_resumed() -> None:
    """Resuming must not paper over a genuine replay violation: an extension
    that doesn't divide finalists is just as invalid mid-stream as it is from
    scratch."""
    from tuner_cli.event_payloads import BudgetExtendedPayload

    manifest = read_manifest(FIXTURES / "manifest.json")
    events = read_events(FIXTURES / "evidence.jsonl")
    checkpoint = fold_checkpoint(manifest, events)
    bad = EvidenceEvent(
        events[-1].sequence + 1,
        "budget_extended",
        BudgetExtendedPayload(0, 1, 0, "x", "2026-09-02T00:00:00+00:00"),
    )
    with pytest.raises(ValueError, match="divide finalists"):
        fold_checkpoint(manifest, [bad], resume_from=checkpoint)
