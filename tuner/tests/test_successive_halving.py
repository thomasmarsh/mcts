"""Hand-verifiable checks for the frozen eta-2 successive-halving shadow policy.

Each case folds the checked-in version-4 golden evidence to a real cohort-0
``ReplayState`` at its single eligible 12-pair tuning prefix, swaps in a
successive-halving shadow policy, and asserts the pure geometric survivor cut.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tuner_cli.artifacts import Manifest, SuccessiveHalvingPolicySpecification, read_manifest
from tuner_cli.cohort import current_active_candidates
from tuner_cli.domain import (
    ReplayState,
    ShadowCandidateDecision,
    ShadowRaceDecision,
    SuccessiveHalvingEvidence,
    TaskPrefix,
)
from tuner_cli.event_payloads import ShadowRaceDecidedPayload
from tuner_cli.evidence import read_events
from tuner_cli.replay import replay
from tuner_cli.successive_halving import decide_successive_halving_shadow_race

FIXTURES = Path(__file__).parent / "fixtures" / "version4"
_METHOD = "successive-halving-common-prefix-eta2-v1"
_SPARE_METHOD = "successive-halving-spare-near-tie-v1"


def _halving_policy(
    survivor_floor: int, spare_margin: float = 0.0
) -> SuccessiveHalvingPolicySpecification:
    return SuccessiveHalvingPolicySpecification(
        "successive_halving",
        _SPARE_METHOD if spare_margin > 0.0 else _METHOD,
        2,
        0.0,
        12,
        survivor_floor,
        "tuning-point-estimate-fingerprint-v1",
        spare_margin,
    )


@pytest.fixture
def manifest() -> Manifest:
    return read_manifest(FIXTURES / "manifest.json")


@pytest.fixture
def prefix(manifest: Manifest) -> TaskPrefix:
    return next(block for block in manifest.tuning_blocks if block.length == 12)


@pytest.fixture
def state() -> ReplayState:
    events = read_events(FIXTURES / "evidence.jsonl")
    return replay(read_manifest(FIXTURES / "manifest.json"), events[:137])


@pytest.fixture
def elite_state() -> ReplayState:
    """Cohort 1, whose two retained elites are its weakest-ranked candidates."""
    events = read_events(FIXTURES / "evidence.jsonl")
    return replay(read_manifest(FIXTURES / "manifest.json"), events[:238])


def _dispositions(decision: ShadowRaceDecision) -> dict[str, str]:
    return {item.candidate_id: item.disposition for item in decision.decisions}


def test_eta_two_cut_halves_the_roster_and_marks_new_eliminations(
    manifest: Manifest, state: ReplayState, prefix: TaskPrefix
) -> None:
    halving = replace(manifest, shadow_policy=_halving_policy(manifest.finalists))
    roster = [item.candidate_id for item in current_active_candidates(state)]

    decision = decide_successive_halving_shadow_race(halving, state, 0, prefix)

    assert decision.policy_kind == "successive_halving"
    assert decision.policy_version == _METHOD
    # Every path is reported once, in cohort roster order, for the audit trace.
    assert [item.candidate_id for item in decision.decisions] == roster
    dispositions = _dispositions(decision)
    assert sorted(dispositions.values()) == ["continue", "continue", "eliminate", "eliminate"]
    evidence = {item.candidate_id: item.evidence for item in decision.decisions}
    for candidate_id, disposition in dispositions.items():
        entry = evidence[candidate_id]
        assert isinstance(entry, SuccessiveHalvingEvidence)
        assert entry.prior_survivor_count == 4
        assert entry.target_survivor_count == 2
        assert entry.newly_eliminated is (disposition == "eliminate")
    # The boundary is the weakest kept candidate: rank == target.
    kept_ranks = [evidence[c].rank for c, d in dispositions.items() if d == "continue"]
    assert decision.boundary_candidate_id in {c for c, d in dispositions.items() if d == "continue"}
    assert max(rank for rank in kept_ranks if rank is not None) == 2


def test_decision_survives_a_tagged_event_round_trip(
    manifest: Manifest, state: ReplayState, prefix: TaskPrefix
) -> None:
    halving = replace(manifest, shadow_policy=_halving_policy(manifest.finalists))
    decision = decide_successive_halving_shadow_race(halving, state, 0, prefix)

    restored = ShadowRaceDecidedPayload.decode(ShadowRaceDecidedPayload(decision).encode()).decision

    assert restored == decision


def test_target_never_falls_below_the_survivor_floor(
    manifest: Manifest, state: ReplayState, prefix: TaskPrefix
) -> None:
    halving = replace(manifest, shadow_policy=_halving_policy(4))

    decision = decide_successive_halving_shadow_race(halving, state, 0, prefix)

    # target == prior survivor count: a valid no-op batch, nobody newly eliminated.
    assert all(item.disposition == "continue" for item in decision.decisions)
    assert all(
        isinstance(item.evidence, SuccessiveHalvingEvidence)
        and item.evidence.target_survivor_count == 4
        and item.evidence.newly_eliminated is False
        for item in decision.decisions
    )


def test_retained_elites_are_always_protected(
    manifest: Manifest, elite_state: ReplayState, prefix: TaskPrefix
) -> None:
    halving = replace(manifest, shadow_policy=_halving_policy(manifest.finalists))
    elite_ids = {item.candidate_id for item in elite_state.active_elites}
    assert len(elite_ids) == manifest.finalists

    decision = decide_successive_halving_shadow_race(halving, elite_state, 1, prefix)

    dispositions = _dispositions(decision)
    for elite_id in elite_ids:
        assert dispositions[elite_id] == "protected"
    # The elites rank below the cut line, yet protection keeps every path alive.
    assert "eliminate" not in dispositions.values()


def test_zero_spare_margin_reproduces_the_eta_two_decision(
    manifest: Manifest, state: ReplayState, prefix: TaskPrefix
) -> None:
    eta2 = replace(manifest, shadow_policy=_halving_policy(manifest.finalists))
    spare_zero = replace(
        manifest,
        shadow_policy=SuccessiveHalvingPolicySpecification(
            "successive_halving",
            _SPARE_METHOD,
            2,
            0.0,
            12,
            manifest.finalists,
            "tuning-point-estimate-fingerprint-v1",
            0.0,
        ),
    )
    base = decide_successive_halving_shadow_race(eta2, state, 0, prefix)
    softened = decide_successive_halving_shadow_race(spare_zero, state, 0, prefix)

    assert softened == replace(base, policy_version=_SPARE_METHOD)


def test_near_tie_candidates_are_spared_at_a_positive_margin(
    manifest: Manifest, state: ReplayState, prefix: TaskPrefix
) -> None:
    from tuner_cli.observations import comparable_prefix_observations, paired_difference

    cut = replace(manifest, shadow_policy=_halving_policy(manifest.finalists))
    base = decide_successive_halving_shadow_race(cut, state, 0, prefix)
    eliminated = {item.candidate_id for item in base.decisions if item.disposition == "eliminate"}
    assert len(eliminated) == 2

    cohort = current_active_candidates(state)
    by_id = {
        item.candidate_id: item
        for item in comparable_prefix_observations(state.observations, cohort, prefix)
    }
    boundary = by_id[base.boundary_candidate_id]
    # Both cut candidates sit within 0.05 of the boundary in paired mean, so a
    # spare margin of 0.05 must carry both rather than resolve the boundary.
    assert all(paired_difference(by_id[cid], boundary).mean >= -0.05 for cid in eliminated)

    softened = replace(manifest, shadow_policy=_halving_policy(manifest.finalists, 0.05))
    decision = decide_successive_halving_shadow_race(softened, state, 0, prefix)

    assert decision.boundary_candidate_id == base.boundary_candidate_id
    assert all(item.disposition == "continue" for item in decision.decisions)
    for item in decision.decisions:
        assert isinstance(item.evidence, SuccessiveHalvingEvidence)
        assert item.evidence.newly_eliminated is False
        assert item.evidence.target_survivor_count == 2  # the rank cut is unchanged


def test_prior_hypothetical_eliminations_do_not_re_enter_a_later_rung(
    manifest: Manifest, state: ReplayState, prefix: TaskPrefix
) -> None:
    cohort = current_active_candidates(state)
    already_gone = cohort[-1].candidate_id
    earlier_prefix = next(block for block in manifest.tuning_blocks if block.length == 10)
    prior = ShadowRaceDecision(
        0,
        earlier_prefix.prefix_id,
        (),
        cohort[0].candidate_id,
        tuple(
            ShadowCandidateDecision(
                item.candidate_id,
                "eliminate" if item.candidate_id == already_gone else "continue",
                SuccessiveHalvingEvidence(None, 4, 3, item.candidate_id == already_gone),
            )
            for item in cohort
        ),
        "successive_halving",
        _METHOD,
    )
    seeded = replace(state, shadow_races=(prior,))
    halving = replace(manifest, shadow_policy=_halving_policy(manifest.finalists))

    decision = decide_successive_halving_shadow_race(halving, seeded, 0, prefix)

    by_id = {item.candidate_id: item for item in decision.decisions}
    gone = by_id[already_gone]
    assert gone.disposition == "eliminate"
    assert isinstance(gone.evidence, SuccessiveHalvingEvidence)
    assert gone.evidence.newly_eliminated is False
    assert gone.evidence.rank is None
    # Three survivors, eta-2 target of two, so exactly one fresh elimination.
    fresh = [
        item
        for item in decision.decisions
        if isinstance(item.evidence, SuccessiveHalvingEvidence) and item.evidence.newly_eliminated
    ]
    assert len(fresh) == 1
    assert all(
        isinstance(item.evidence, SuccessiveHalvingEvidence)
        and item.evidence.prior_survivor_count == 3
        and item.evidence.target_survivor_count == 2
        for item in decision.decisions
        if item.candidate_id != already_gone
    )
