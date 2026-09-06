from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tuner_cli.active_audit import build_active_audit
from tuner_cli.artifacts import Manifest, _active_elimination, _shadow_policy, read_manifest
from tuner_cli.domain import (
    ApplyElimination,
    CandidateEliminationAction,
    ComputeLedger,
    PhaseCompute,
    ReplayState,
    SuccessiveHalvingRankMargin,
)

FIXTURE = Path(__file__).parent / "fixtures" / "version4" / "manifest.json"


def _halving_manifest() -> Manifest:
    base = read_manifest(FIXTURE)
    halving = _shadow_policy(0.0, 0.05, "successive_halving", base.finalists, 0.1)
    return replace(
        base,
        shadow_policy=halving,
        active_elimination=_active_elimination(0.25, halving),
    )


def test_planned_unique_pair_savings_are_prefix_arithmetic_not_ledger_facts() -> None:
    manifest = _halving_manifest()
    # The single eligible shadow prefix has 12 pairs; the maximum tuning prefix
    # has 14, so every candidate cut there omits exactly 2 unique suffix pairs.
    cut_prefix = next(item for item in manifest.tuning_blocks if item.length == 12)
    batch = ApplyElimination(
        0,
        cut_prefix.prefix_id,
        (
            CandidateEliminationAction("p", "prune", SuccessiveHalvingRankMargin(5, 3, 2, 0)),
            CandidateEliminationAction(
                "a", "audit_continue", SuccessiveHalvingRankMargin(4, 3, 1, 1)
            ),
        ),
    )
    # A ledger with retries and censored attempts must not move the projection.
    state = ReplayState(
        (),
        (),
        (),
        (),
        (),
        (),
        None,
        "open",
        0,
        None,
        compute=ComputeLedger(
            tuning=PhaseCompute(
                pair_attempts=97, completed_pairs=80, failed_attempts=12, censored_attempts=5
            )
        ),
        elimination_allocations=(batch,),
    )

    audit = build_active_audit(manifest, state)
    summary = audit["summary"]
    assert summary["nominal_eliminations"] == 2
    assert summary["gross_nominal_suffix_unique_pairs"] == 4
    assert summary["audit_continuation_suffix_unique_pairs"] == 2
    assert summary["planned_unique_pair_savings"] == 2
    assert audit["actual_compute"]["tuning_pair_attempts"] == 97
    assert audit["policy"]["policy_kind"] == "successive_halving"
    assert audit["policy"]["policy_version"] == "successive-halving-spare-near-tie-v1"
