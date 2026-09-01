"""Public shadow-audit invariants over the strict version-4 fixture."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tuner_cli.artifacts import (
    Manifest,
    SuccessiveHalvingPolicySpecification,
    read_manifest,
)
from tuner_cli.evidence import read_events
from tuner_cli.race_policy import decide_shadow_race
from tuner_cli.replay import replay
from tuner_cli.shadow_audit import build_shadow_audit

FIXTURE = Path(__file__).parent / "fixtures" / "version4"


def _as_successive_halving(manifest: Manifest) -> Manifest:
    policy = SuccessiveHalvingPolicySpecification(
        "successive_halving",
        "successive-halving-common-prefix-eta2-v1",
        2,
        0.0,
        12,
        manifest.finalists,
        "tuning-point-estimate-fingerprint-v1",
    )
    return replace(manifest, shadow_policy=policy)


def test_fixture_shadow_audit_keeps_protected_paths_out_of_active_metrics() -> None:
    manifest = read_manifest(FIXTURE / "manifest.json")
    events = read_events(FIXTURE / "evidence.jsonl")
    audit = build_shadow_audit(manifest, replay(manifest, events), events)

    assert [path.cohort_index for path in audit.paths] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert audit.counterfactual_eliminations == 0
    assert audit.top_set_false_eliminations == 0
    assert audit.true_trash_eliminations == 0
    assert audit.brier_score == 0.0
    assert all(path.avoided_unique_pairs == 0 for path in audit.paths)
    assert all(not path.looks or path.looks[0].favorable_resamples >= 0 for path in audit.paths)


def test_successive_halving_audit_labels_looks_without_paired_evidence() -> None:
    """The max-prefix audit computes per-stratum reversal labels for the
    geometric policy too -- it must not demand paired-bootstrap evidence."""
    manifest = read_manifest(FIXTURE / "manifest.json")
    events = read_events(FIXTURE / "evidence.jsonl")
    halving = _as_successive_halving(manifest)

    final = replay(manifest, events)
    prefix = next(block for block in manifest.tuning_blocks if block.length == 12)
    # Recompute the geometric decision from the state as it stood at each
    # recorded shadow look (the policy only accepts the active complete cohort).
    look_indexes = [
        index
        for index, event in enumerate(events)
        if event.payload.__class__.__name__ == "ShadowRaceDecidedPayload"
    ]
    races = tuple(
        decide_shadow_race(halving, replay(manifest, events[:index]), cohort_index, prefix)
        for cohort_index, index in enumerate(look_indexes)
    )
    assert all(race.policy_kind == "successive_halving" for race in races)

    audit = build_shadow_audit(halving, replace(final, shadow_races=races), events)

    assert any(path.looks for path in audit.paths), "the halving audit reached _strata_for_look"
    assert audit.brier_score is None
    for path in audit.paths:
        for look in path.looks:
            assert look.favorable_resamples is None
            assert look.rank is not None
            assert look.strata, "per-stratum reversal labels are policy-neutral"
