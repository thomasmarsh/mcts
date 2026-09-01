from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tuner_cli.artifacts import (
    ActiveEliminationSpecification,
    _active_elimination,
    _decode_active_elimination,
    _shadow_policy,
    read_manifest,
)
from tuner_cli.domain import (
    ApplyElimination,
    Candidate,
    CandidateEliminationAction,
    CohortRecord,
    ObservationContext,
    PairedBootstrapEvidence,
    PairedProbabilityMargin,
    ReplayState,
    ShadowCandidateDecision,
    ShadowRaceDecision,
    SuccessiveHalvingEvidence,
    SuccessiveHalvingRankMargin,
)
from tuner_cli.elimination import active_elimination_allocation, audited_boundary_reversals
from tuner_cli.observations import observation


def _halving_manifest() -> object:
    base = read_manifest(Path(__file__).parent / "fixtures" / "version4" / "manifest.json")
    halving = _shadow_policy(0.0, 0.05, "successive_halving", base.finalists, 0.1)
    return replace(
        base,
        shadow_policy=halving,
        active_elimination=_active_elimination(0.25, halving),
    )


def _halving_race(
    manifest: object, rows: tuple[tuple[str, str, int | None, bool], ...]
) -> ShadowRaceDecision:
    return ShadowRaceDecision(
        0,
        manifest.tuning_blocks[0].prefix_id,  # type: ignore[attr-defined]
        (),
        "boundary",
        tuple(
            ShadowCandidateDecision(
                candidate_id,
                disposition,
                SuccessiveHalvingEvidence(rank, 6, 3, disposition == "eliminate" and newly),
            )
            for candidate_id, disposition, rank, newly in rows
        ),
        "successive_halving",
        "successive-halving-spare-near-tie-v1",
    )


def _empty_state() -> ReplayState:
    return ReplayState((), (), (), (), (), (), None, "open", 0, None)


def test_active_halving_allocation_tags_a_rank_margin_with_the_spared_count() -> None:
    manifest = _halving_manifest()
    race = _halving_race(
        manifest,
        (
            ("keep", "continue", 1, False),
            ("spared", "continue", 4, False),
            ("cut", "eliminate", 5, True),
        ),
    )
    allocation = active_elimination_allocation(manifest, _empty_state(), race)
    assert [item.candidate_id for item in allocation.actions] == ["cut"]
    margin = allocation.actions[0].margin
    assert margin == SuccessiveHalvingRankMargin(5, 3, 2, 1)


def test_active_halving_allocation_skips_candidates_eliminated_on_an_earlier_look() -> None:
    manifest = _halving_manifest()
    race = _halving_race(
        manifest,
        (
            ("kept", "continue", 1, False),
            ("earlier", "eliminate", None, False),
            ("fresh", "eliminate", 4, True),
        ),
    )
    allocation = active_elimination_allocation(manifest, _empty_state(), race)
    assert [item.candidate_id for item in allocation.actions] == ["fresh"]


def test_active_halving_allocation_rejects_a_race_from_a_different_policy_version() -> None:
    manifest = _halving_manifest()
    race = replace(
        _halving_race(manifest, (("cut", "eliminate", 4, True),)),
        policy_version="successive-halving-common-prefix-eta2-v1",
    )
    with pytest.raises(ValueError, match="policy version"):
        active_elimination_allocation(manifest, _empty_state(), race)


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
            ShadowCandidateDecision("candidate-a", "eliminate", PairedBootstrapEvidence(0, 4096)),
            ShadowCandidateDecision("candidate-b", "continue", PairedBootstrapEvidence(0, 4096)),
            ShadowCandidateDecision("boundary", "eliminate", PairedBootstrapEvidence(0, 4096)),
        ),
        "paired_bootstrap",
        manifest.shadow_policy.method_version,
    )
    state = ReplayState((), (), (), (), (), (), None, "open", 0, None)

    first = active_elimination_allocation(manifest, state, race)
    second = active_elimination_allocation(manifest, state, race)

    assert first == second
    assert [item.candidate_id for item in first.actions] == ["candidate-a"]
    margin = first.actions[0].margin
    assert isinstance(margin, PairedProbabilityMargin)
    assert margin == PairedProbabilityMargin(0.05, 0.0, 0.05)


def test_active_specification_binds_the_paired_shadow_policy_and_round_trips() -> None:
    paired = _shadow_policy(0.0, 0.05, "paired_bootstrap")
    spec = _active_elimination(0.25, paired)
    assert spec == ActiveEliminationSpecification(
        0.25, "paired_bootstrap", "stratified-paired-bootstrap-all-strata-v2", 0.0
    )
    assert _decode_active_elimination(spec.encoded(), paired) == spec


def test_active_specification_binds_the_spare_near_tie_halving_policy() -> None:
    halving = _shadow_policy(0.0, 0.05, "successive_halving", 4, 0.1)
    spec = _active_elimination(0.25, halving)
    assert spec == ActiveEliminationSpecification(
        0.25, "successive_halving", "successive-halving-spare-near-tie-v1", 0.1
    )
    assert _decode_active_elimination(spec.encoded(), halving) == spec


def test_active_elimination_rejects_the_ungated_eta2_halving_cut() -> None:
    eta2 = _shadow_policy(0.0, 0.05, "successive_halving", 4, 0.0)
    with pytest.raises(ValueError, match="gate-approved"):
        _active_elimination(0.25, eta2)


def test_decode_rejects_an_active_specification_bound_to_a_different_policy() -> None:
    paired = _shadow_policy(0.0, 0.05, "paired_bootstrap")
    halving = _shadow_policy(0.0, 0.05, "successive_halving", 4, 0.1)
    bound_to_halving = _active_elimination(0.25, halving)
    assert bound_to_halving is not None
    with pytest.raises(ValueError):
        _decode_active_elimination(bound_to_halving.encoded(), paired)


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
        "paired_bootstrap",
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
                (
                    CandidateEliminationAction(
                        candidate.candidate_id,
                        "audit_continue",
                        PairedProbabilityMargin(0.05, 0.15, -0.1),
                    ),
                ),
            ),
        ),
    )

    assert (
        audited_boundary_reversals(manifest, state, cohort)[0].candidate_id
        == candidate.candidate_id
    )
