from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tuner_cli.artifacts import ActiveEliminationSpecification, read_manifest
from tuner_cli.domain import (
    ApplyElimination,
    Candidate,
    CandidateEliminationAction,
    CohortRecord,
    ObservationContext,
    ReplayState,
    ShadowCandidateDecision,
    ShadowRaceDecision,
)
from tuner_cli.elimination import active_elimination_allocation, audited_boundary_reversals
from tuner_cli.observations import observation


def test_active_elimination_sampling_is_deterministic_and_ignores_non_eliminations() -> None:
    manifest = replace(
        read_manifest(Path(__file__).parent / "fixtures" / "version4" / "manifest.json"),
        active_elimination=ActiveEliminationSpecification(0.5),
    )
    race = ShadowRaceDecision(
        0,
        manifest.tuning_blocks[0].prefix_id,
        (),
        "boundary",
        (
            ShadowCandidateDecision("candidate-a", 0, 4096, "eliminate"),
            ShadowCandidateDecision("candidate-b", 0, 4096, "continue"),
            ShadowCandidateDecision("boundary", 0, 4096, "eliminate"),
        ),
        manifest.shadow_policy.method_version,
    )
    state = ReplayState((), (), (), (), (), (), None, "open", 0, None)

    first = active_elimination_allocation(manifest, state, race)
    second = active_elimination_allocation(manifest, state, race)

    assert first == second
    assert [item.candidate_id for item in first.actions] == ["candidate-a"]
    assert first.actions[0].decision_margin == 0.05


def test_audited_boundary_reversal_uses_the_maximum_prefix_and_margin() -> None:
    manifest = replace(
        read_manifest(Path(__file__).parent / "fixtures" / "version4" / "manifest.json"),
        active_elimination=ActiveEliminationSpecification(0.5),
    )
    candidate = Candidate("candidate-a", "a", "{}")
    boundary = Candidate("candidate-b", "b", "{}")
    cohort = CohortRecord(0, (candidate, boundary), ())
    context = ObservationContext(
        manifest.epoch.epoch_id, "tuning", manifest.tuning_prefix, manifest.efforts["tuning"]
    )
    race = ShadowRaceDecision(
        0,
        manifest.tuning_blocks[0].prefix_id,
        (),
        boundary.candidate_id,
        (),
        manifest.shadow_policy.method_version,
    )
    state = ReplayState(
        (),
        (),
        (cohort,),
        (),
        (),
        (
            observation(candidate.candidate_id, context, (0.5,) * manifest.tuning_prefix.length),
            observation(boundary.candidate_id, context, (0.5,) * manifest.tuning_prefix.length),
        ),
        None,
        "open",
        0,
        None,
        shadow_races=(race,),
        elimination_allocations=(
            ApplyElimination(
                0,
                race.prefix_id,
                (CandidateEliminationAction(candidate.candidate_id, "audit_continue", -0.1),),
            ),
        ),
    )

    assert (
        audited_boundary_reversals(manifest, state, cohort)[0].candidate_id
        == candidate.candidate_id
    )
