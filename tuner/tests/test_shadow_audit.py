"""Public shadow-audit invariants over the strict version-4 fixture."""

from __future__ import annotations

from pathlib import Path

from tuner_cli.artifacts import read_manifest
from tuner_cli.evidence import read_events
from tuner_cli.replay import replay
from tuner_cli.shadow_audit import build_shadow_audit

FIXTURE = Path(__file__).parent / "fixtures" / "version4"


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
