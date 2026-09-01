"""Task-11 mechanism sweep: generator determinism + hand-verified aggregation."""

from __future__ import annotations

import random
from pathlib import Path

from tuner_cli.artifacts import read_manifest
from tuner_cli.domain import (
    Candidate,
    ObservationContext,
    ShadowCandidateDecision,
    ShadowRaceDecision,
    SuccessiveHalvingEvidence,
)
from tuner_cli.mechanism_calibration import (
    UTILITY_ALPHABET,
    _deviation_correlation,
    _quantiles,
    _utility_cdf,
    load_calibration,
)
from tuner_cli.mechanism_sim import as_halving, classify_trial, draw_pair, run_trial
from tuner_cli.mechanism_sweep import (
    CellResult,
    PolicyTally,
    evaluate_gate,
    run_sweep,
    wilson_interval,
)
from tuner_cli.observations import observation

FIXTURE = Path(__file__).parent / "fixtures" / "version4"
CALIBRATION = Path(__file__).parents[1] / "calibrations" / "druid-recorded-v1.json"


def _manifest():
    return as_halving(read_manifest(FIXTURE / "manifest.json"))


# --------------------------------------------------------------------------- #
# Instrumentation math (the repo's script-correctness rule)
# --------------------------------------------------------------------------- #


def test_utility_cdf_is_a_normalised_empirical_cdf() -> None:
    cdf = _utility_cdf((0.0, 0.0, 0.5, 1.0))
    assert [value for value, _ in cdf] == list(UTILITY_ALPHABET)
    assert dict(cdf) == {0.0: 0.5, 0.25: 0.5, 0.5: 0.75, 0.75: 0.75, 1.0: 1.0}


def test_quantiles_interpolate_linearly() -> None:
    result = _quantiles([0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.0])
    assert result == [[0.0, 0.0], [0.5, 1.5], [1.0, 3.0]]


def test_deviation_correlation_is_one_for_parallel_columns() -> None:
    vectors = [{"a": 0.1, "b": 0.6}, {"a": 0.2, "b": 0.7}, {"a": 0.3, "b": 0.8}]
    assert abs(_deviation_correlation(vectors) - 1.0) < 1e-9


def test_wilson_interval_hand_values() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(0, 100)
    assert abs(lo) < 1e-12 and 0.03 < hi < 0.04
    lo, hi = wilson_interval(50, 100)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9


def test_policy_tally_rates() -> None:
    tally = PolicyTally()
    tally.add(_classification("halving", eliminated=4, reversals=1, top_false=0, saved=24))
    tally.add(_classification("halving", eliminated=2, reversals=0, top_false=1, saved=12))
    rates = tally.rates()
    assert rates.eliminated == 6
    assert rates.mean_eliminated_per_trial == 3.0
    assert rates.mean_unique_pairs_saved == 18.0
    assert rates.boundary_reversal_rate == 1 / 6
    assert rates.top_set_false_eviction_rate == 1 / 6


# --------------------------------------------------------------------------- #
# Eviction classification
# --------------------------------------------------------------------------- #


def _classification(policy, *, eliminated, reversals, top_false, saved):
    from tuner_cli.mechanism_sim import TrialClassification

    return TrialClassification(policy, eliminated, top_false, reversals, 0, 0, saved)


def _obs(candidate_id: str, manifest, block, utilities: tuple[float, ...]):
    context = ObservationContext(
        manifest.epoch.epoch_id, "tuning", block, manifest.efforts["tuning"]
    )
    return observation(candidate_id, context, utilities)


def test_classify_trial_flags_a_top_set_boundary_reversal() -> None:
    manifest = _manifest()
    early_block = next(b for b in manifest.tuning_blocks if b.length == 12)
    maximum_block = manifest.tuning_prefix
    cohort = tuple(Candidate(f"candidate-{name}", name, "{}") for name in ("aaa", "bbb", "ccc"))
    # Boundary is bbb. At the maximum prefix ccc (eliminated) beats bbb outright
    # and tops the cohort; aaa sits in between.
    early = (
        _obs("candidate-aaa", manifest, early_block, (0.5,) * 12),
        _obs("candidate-bbb", manifest, early_block, (0.4,) * 12),
        _obs("candidate-ccc", manifest, early_block, (0.3,) * 12),
    )
    maximum = (
        _obs("candidate-aaa", manifest, maximum_block, (0.5,) * 14),
        _obs("candidate-bbb", manifest, maximum_block, (0.2,) * 14),
        _obs("candidate-ccc", manifest, maximum_block, (0.9,) * 14),
    )
    decision = ShadowRaceDecision(
        0,
        early_block.prefix_id,
        tuple(item.observation_id for item in early),
        "candidate-bbb",
        (
            ShadowCandidateDecision("candidate-aaa", "continue", _halving_evidence()),
            ShadowCandidateDecision("candidate-bbb", "continue", _halving_evidence()),
            ShadowCandidateDecision("candidate-ccc", "eliminate", _halving_evidence()),
        ),
        "successive_halving",
        "successive-halving-common-prefix-eta2-v1",
    )
    result = classify_trial(manifest, decision, early, maximum, cohort, early_block, maximum_block)
    assert result.eliminated == 1
    assert result.boundary_reversals == 1
    assert result.rule_tie_evictions == 0
    assert result.top_set_false_evictions == 1
    assert result.unique_pairs_saved == 14 - 12


def test_classify_trial_scores_a_clean_elimination_as_zero() -> None:
    manifest = _manifest()
    early_block = next(b for b in manifest.tuning_blocks if b.length == 12)
    maximum_block = manifest.tuning_prefix
    cohort = tuple(Candidate(f"candidate-{n}", n, "{}") for n in ("aaa", "bbb", "ccc"))
    early = tuple(
        _obs(f"candidate-{n}", manifest, early_block, (v,) * 12)
        for n, v in (("aaa", 0.6), ("bbb", 0.5), ("ccc", 0.1))
    )
    maximum = tuple(
        _obs(f"candidate-{n}", manifest, maximum_block, (v,) * 14)
        for n, v in (("aaa", 0.6), ("bbb", 0.5), ("ccc", 0.1))
    )
    decision = ShadowRaceDecision(
        0,
        early_block.prefix_id,
        tuple(item.observation_id for item in early),
        "candidate-bbb",
        (
            ShadowCandidateDecision("candidate-aaa", "continue", _halving_evidence()),
            ShadowCandidateDecision("candidate-bbb", "continue", _halving_evidence()),
            ShadowCandidateDecision("candidate-ccc", "eliminate", _halving_evidence()),
        ),
        "successive_halving",
        "successive-halving-common-prefix-eta2-v1",
    )
    result = classify_trial(manifest, decision, early, maximum, cohort, early_block, maximum_block)
    assert result.boundary_reversals == 0
    assert result.rule_tie_evictions == 0
    assert result.top_set_false_evictions == 0


def _halving_evidence() -> SuccessiveHalvingEvidence:
    return SuccessiveHalvingEvidence(1, 4, 2, False)


def test_as_halving_zero_spare_margin_matches_the_shipped_cut() -> None:
    """`spare_margin = 0.0` must leave the eta-2 rank cut untouched."""
    from tuner_cli.mechanism_sim import build_active_state, draw_trial, sample_cohort
    from tuner_cli.race_policy import decide_shadow_race

    calibration = load_calibration(CALIBRATION)
    manifest = _manifest()
    early = next(b for b in manifest.tuning_blocks if b.length == 12)
    for seed in ("a", "b", "c", "d", "e"):
        rng = random.Random(seed)
        cohort = sample_cohort(calibration, manifest, rng, 0.02, 1.0)
        state = build_active_state(manifest, cohort, draw_trial(cohort, calibration, manifest, rng))
        shipped = decide_shadow_race(manifest, state, 0, early)
        zero_spare = decide_shadow_race(as_halving(manifest, 0.0), state, 0, early)
        assert zero_spare == shipped
        spared = decide_shadow_race(as_halving(manifest, 0.10), state, 0, early)
        # A positive margin only ever keeps more candidates, never fewer.
        kept = {d.candidate_id for d in spared.decisions if d.disposition == "continue"}
        shipped_kept = {d.candidate_id for d in shipped.decisions if d.disposition == "continue"}
        assert shipped_kept <= kept


# --------------------------------------------------------------------------- #
# Generator determinism + a small asserted sweep slice
# --------------------------------------------------------------------------- #


def test_draw_pair_is_deterministic_and_in_alphabet() -> None:
    calibration = load_calibration(CALIBRATION)
    a = [draw_pair(calibration, 0.4, random.Random("s")) for _ in range(20)]
    b = [draw_pair(calibration, 0.4, random.Random("s")) for _ in range(20)]
    assert a == b
    assert set(a) <= set(UTILITY_ALPHABET)


def test_run_trial_is_seed_reproducible() -> None:
    calibration = load_calibration(CALIBRATION)
    manifest = read_manifest(FIXTURE / "manifest.json")
    first = run_trial(calibration, manifest, random.Random("trial-7"), 0.0, 1.0, 64)
    second = run_trial(calibration, manifest, random.Random("trial-7"), 0.0, 1.0, 64)
    assert first == second


def test_sweep_slice_bounds() -> None:
    calibration = load_calibration(CALIBRATION)
    manifest = read_manifest(FIXTURE / "manifest.json")
    sweep = run_sweep(
        calibration,
        manifest,
        boundary_gaps=(0.1,),
        spread_scales=(1.0,),
        trials=120,
        seed=1,
        paired_resamples=64,
    )
    (cell,) = sweep.cells
    halving = cell.policies["halving"]
    paired = cell.policies["paired"]
    assert halving.eliminated > 0
    assert halving.mean_unique_pairs_saved >= paired.mean_unique_pairs_saved
    assert 0.0 <= halving.boundary_reversal_rate <= 0.5
    assert sweep.overall["halving"].top_set_false_eviction_rate < 0.1
    assert "paired" not in sweep.gates
    assert {"spare05", "spare10"} <= set(sweep.gates)
    # Re-running the same slice reproduces every aggregate exactly.
    again = run_sweep(
        calibration,
        manifest,
        boundary_gaps=(0.1,),
        spread_scales=(1.0,),
        trials=120,
        seed=1,
        paired_resamples=64,
    )
    assert again.to_json() == sweep.to_json()


def test_evaluate_gate_clause_logic() -> None:
    good = CellResult(
        0.1,
        1.0,
        {
            "halving": _rates(reversal=0.0, top_false=0.0, saved=6.0),
            "paired": _rates(reversal=0.0, top_false=0.0, saved=4.0),
        },
    )
    gate = evaluate_gate("halving", (good,), good.policies["halving"])
    assert gate.passed

    bad_saved = CellResult(0.1, 1.0, {"halving": _rates(saved=2.0), "paired": _rates(saved=6.0)})
    gate = evaluate_gate("halving", (bad_saved,), bad_saved.policies["halving"])
    assert not gate.clauses["4_saves_at_least_as_much_as_paired"]


def _rates(*, reversal: float = 0.0, top_false: float = 0.0, saved: float = 6.0):
    from tuner_cli.mechanism_sweep import PolicyRates

    return PolicyRates(
        trials=100,
        eliminated=200,
        mean_eliminated_per_trial=2.0,
        mean_unique_pairs_saved=saved,
        mean_boundary_reversals_per_trial=reversal * 2.0,
        mean_top_set_false_per_trial=top_false * 2.0,
        top_set_false_eviction_rate=top_false,
        top_set_false_eviction_upper=top_false,
        boundary_reversal_rate=reversal,
        boundary_reversal_upper=reversal,
        rule_tie_eviction_rate=0.0,
        per_stratum_dangerous_flip_rate=0.0,
    )
