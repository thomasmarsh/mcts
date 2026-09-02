"""Continuation and budget extension (``budget_extended`` evidence event).

An extension is append-only evidence: it never edits ``manifest.compute_budget``.
Replay folds the ordered deltas into ``ReplayState.effective_budget`` and, when
the run had already completed, re-opens it at the last cohort boundary so the
allocator funds a fresh challenger cohort and a fresh finalist validation from
the raised budget.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_run import (
    FakeModel,
    FakeTarget,
    _budgeted_options,
    _completed_cohorts,
)

from tuner_cli.artifacts import read_manifest
from tuner_cli.event_payloads import BudgetExtendedPayload
from tuner_cli.evidence import EvidenceWriter, read_events
from tuner_cli.replay import replay
from tuner_cli.run import RunOptions, run_foreground

_EXTENSION = BudgetExtendedPayload(6, 0, 0, "fund another cohort", "2026-09-02T00:00:00+00:00")


def _complete_run(options: RunOptions) -> Path:
    run_foreground(options, FakeTarget(), model_proposer=FakeModel())
    return options.run_dir


def _events(run_dir: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()]


def test_payload_round_trip() -> None:
    encoded = _EXTENSION.encode()
    assert BudgetExtendedPayload.decode(encoded) == _EXTENSION
    with pytest.raises(ValueError):
        BudgetExtendedPayload.decode({**encoded, "tuning_pair_attempts_delta": -1})
    with pytest.raises(ValueError):
        BudgetExtendedPayload.decode(
            {
                **encoded,
                "tuning_pair_attempts_delta": 0,
                "validation_pair_attempts_delta": 0,
                "diagnostic_pair_attempts_delta": 0,
            }
        )


def _append(run_dir: Path, payload: BudgetExtendedPayload) -> None:
    EvidenceWriter.open(run_dir / "evidence.jsonl").append(payload)


def test_effective_budget_folds_extensions(tmp_path: Path) -> None:
    run_dir = _complete_run(_budgeted_options(tmp_path, 19))
    manifest = read_manifest(run_dir / "manifest.json")
    base = manifest.compute_budget
    _append(run_dir, _EXTENSION)
    _append(run_dir, replace(_EXTENSION, tuning_pair_attempts_delta=4))
    state = replay(manifest, read_events(run_dir / "evidence.jsonl"))
    assert state.effective_budget.tuning_pair_attempts == base.tuning_pair_attempts + 10
    assert state.effective_budget.validation_pair_attempts == base.validation_pair_attempts
    assert manifest.compute_budget == base


def test_over_corpus_extension_rejected(tmp_path: Path) -> None:
    run_dir = _complete_run(_budgeted_options(tmp_path, 19, finalists=2, validation_pair_budget=4))
    manifest = read_manifest(run_dir / "manifest.json")
    corpus = len(manifest.production_validation_corpus.cases)
    room = corpus * manifest.finalists - manifest.compute_budget.validation_pair_attempts
    _append(
        run_dir,
        replace(
            _EXTENSION,
            tuning_pair_attempts_delta=0,
            validation_pair_attempts_delta=room + manifest.finalists,
        ),
    )
    with pytest.raises(ValueError, match="frozen validation corpus"):
        replay(manifest, read_events(run_dir / "evidence.jsonl"))


def test_validation_delta_must_divide_finalists(tmp_path: Path) -> None:
    run_dir = _complete_run(_budgeted_options(tmp_path, 19, finalists=2, validation_pair_budget=4))
    manifest = read_manifest(run_dir / "manifest.json")
    _append(
        run_dir, replace(_EXTENSION, tuning_pair_attempts_delta=0, validation_pair_attempts_delta=1)
    )
    with pytest.raises(ValueError, match="divide finalists"):
        replay(manifest, read_events(run_dir / "evidence.jsonl"))


def test_extension_reopens_completed_run(tmp_path: Path) -> None:
    run_dir = _complete_run(_budgeted_options(tmp_path, 19))
    manifest = read_manifest(run_dir / "manifest.json")
    assert replay(manifest, read_events(run_dir / "evidence.jsonl")).terminal_status == "complete"
    _append(run_dir, _EXTENSION)
    state = replay(manifest, read_events(run_dir / "evidence.jsonl"))
    assert state.terminal_status == "open"
    assert state.finalists is None
    assert len(state.superseded_finalists) == 1


def test_allocator_funds_extended_cohort(tmp_path: Path) -> None:
    options = _budgeted_options(tmp_path, 19)
    run_dir = _complete_run(options)
    prefix = (run_dir / "evidence.jsonl").read_text()
    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    assert [c["cohort_index"] for c in _completed_cohorts(_events(run_dir))] == [0, 1]

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

    events = _events(run_dir)
    assert [c["cohort_index"] for c in _completed_cohorts(events)] == [0, 1, 2]
    assert sum(e["type"] == "budget_extended" for e in events) == 1
    assert sum(e["type"] == "run_completed" for e in events) == 2
    # The pre-extension evidence prefix is byte-stable.
    assert (run_dir / "evidence.jsonl").read_text().startswith(prefix)
    # The manifest and its fingerprint never change across an extension.
    assert (run_dir / "manifest.json").read_bytes() == manifest_bytes
    # The extended run still replays cleanly and rebuilds its report.
    replay(read_manifest(run_dir / "manifest.json"), read_events(run_dir / "evidence.jsonl"))
    report = json.loads((run_dir / "report.json").read_text())
    assert report["status"] == "complete"


def test_extension_flags_require_resume(tmp_path: Path) -> None:
    options = _budgeted_options(tmp_path, 19)
    with pytest.raises(ValueError, match="only with --resume"):
        run_foreground(
            replace(options, extend_tuning_pairs=6, extend_reason="x"),
            FakeTarget(),
            model_proposer=FakeModel(),
        )
    run_dir = _complete_run(replace(options, run_dir=tmp_path / "seed"))
    with pytest.raises(ValueError, match="--extend-reason"):
        run_foreground(
            replace(options, run_dir=run_dir, resume=True, extend_tuning_pairs=6),
            FakeTarget(),
            model_proposer=FakeModel(),
        )
