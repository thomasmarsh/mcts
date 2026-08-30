"""Evidence-only counterfactual audit for recorded shadow race decisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .artifacts import Manifest
from .compute import fold_ledger
from .domain import Candidate, PhaseCompute, ReplayState
from .event_payloads import PairCompletedPayload, PairFailedPayload, PairStartedPayload
from .evidence import EvidenceEvent
from .identity import pair_task
from .observations import comparable_prefix_observations, paired_difference
from .selection import select_top_candidates
from .shadow import paired_stratum_differences


@dataclass(frozen=True, slots=True)
class StratumAudit:
    stratum_id: str
    early_mean_difference: float
    maximum_mean_difference: float
    reversal: bool


@dataclass(frozen=True, slots=True)
class ShadowLookAudit:
    cohort_index: int
    prefix_id: str
    candidate_id: str
    boundary_candidate_id: str
    favorable_resamples: int
    total_resamples: int
    disposition: str
    early_mean_difference: float
    maximum_mean_difference: float
    final_reaches_recorded_boundary: bool
    strata: tuple[StratumAudit, ...]


@dataclass(frozen=True, slots=True)
class CandidatePathAudit:
    cohort_index: int
    candidate_id: str
    protected: bool
    final_top_set: bool
    looks: tuple[ShadowLookAudit, ...]
    first_elimination_prefix_id: str | None
    avoided_unique_pairs: int
    avoided_compute: PhaseCompute


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_prediction: float
    observed_success_rate: float


@dataclass(frozen=True, slots=True)
class StratumSummary:
    stratum_id: str
    looks: int
    reversals: int
    elimination_reversals: int


@dataclass(frozen=True, slots=True)
class ShadowAudit:
    paths: tuple[CandidatePathAudit, ...]
    calibration_bins: tuple[CalibrationBin, ...]
    strata: tuple[StratumSummary, ...]
    counterfactual_eliminations: int
    eligible_top_set_paths: int
    top_set_false_eliminations: int
    true_trash_eliminations: int
    brier_score: float | None
    recorded_compute_after_first_elimination: PhaseCompute


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _phase_add(left: PhaseCompute, right: PhaseCompute) -> PhaseCompute:
    return PhaseCompute(
        left.pair_attempts + right.pair_attempts,
        left.completed_pairs + right.completed_pairs,
        left.failed_attempts + right.failed_attempts,
        left.censored_attempts + right.censored_attempts,
        left.physical_games + right.physical_games,
        left.search_iterations + right.search_iterations,
        left.wall_time_ms + right.wall_time_ms,
    )


def _pair_events(events: Sequence[EvidenceEvent], pair_ids: set[str]) -> list[EvidenceEvent]:
    result: list[EvidenceEvent] = []
    for event in events:
        payload = event.payload
        if isinstance(payload, (PairStartedPayload, PairCompletedPayload, PairFailedPayload)):
            if payload.identity.pair_id in pair_ids:
                result.append(event)
    return result


def _suffix_compute(
    manifest: Manifest, events: Sequence[EvidenceEvent], candidate: Candidate, prefix_id: str
) -> tuple[int, PhaseCompute]:
    prefix = next((item for item in manifest.tuning_blocks if item.prefix_id == prefix_id), None)
    if prefix is None:
        raise ValueError("shadow decision has unknown tuning prefix")
    suffix = tuple(
        case for case in manifest.prefix_cases("tuning") if case.ordinal >= prefix.length
    )
    pair_ids = {pair_task(candidate, case, manifest.efforts["tuning"]).pair_id for case in suffix}
    selected = _pair_events(events, pair_ids)
    observed = {
        payload.identity.pair_id
        for event in selected
        if isinstance(
            (payload := event.payload),
            (PairStartedPayload, PairCompletedPayload, PairFailedPayload),
        )
    }
    if observed != pair_ids:
        raise ValueError("counterfactual suffix lacks recorded task cases")
    return len(pair_ids), fold_ledger(selected).tuning


def build_shadow_audit(
    manifest: Manifest, state: ReplayState, events: Sequence[EvidenceEvent]
) -> ShadowAudit:
    """Label immutable shadow decisions with maximum-prefix tuning evidence."""
    candidates_by_id, paths, stratum_rows, calibration, total_compute = _audit_inputs(state)
    for cohort in state.completed_cohorts:
        races = sorted(
            (item for item in state.shadow_races if item.cohort_index == cohort.cohort_index),
            key=lambda item: next(
                index
                for index, block in enumerate(manifest.tuning_blocks)
                if block.prefix_id == item.prefix_id
            ),
        )
        if (
            any(item.cohort_index == cohort.cohort_index for item in state.shadow_races)
            and not races
        ):
            raise ValueError("shadow decision lacks completed cohort")
        maximum = comparable_prefix_observations(
            state.observations, cohort.candidates, manifest.tuning_prefix
        )
        maximum_by_id = {item.candidate_id: item for item in maximum}
        top_ids = {
            item.candidate_id
            for item in select_top_candidates(cohort.candidates, maximum, manifest.finalists)
        }
        per_candidate: defaultdict[str, list[ShadowLookAudit]] = defaultdict(list)
        for race in races:
            if race.boundary_candidate_id not in maximum_by_id:
                raise ValueError("shadow boundary candidate is outside its cohort")
            prefix = next(
                (item for item in manifest.tuning_blocks if item.prefix_id == race.prefix_id), None
            )
            if prefix is None:
                raise ValueError("shadow decision has unknown tuning prefix")
            early = comparable_prefix_observations(state.observations, cohort.candidates, prefix)
            early_by_id = {item.candidate_id: item for item in early}
            for decision in race.decisions:
                if (
                    decision.candidate_id not in candidates_by_id
                    or decision.candidate_id not in early_by_id
                ):
                    raise ValueError("shadow decision references candidate outside its cohort")
                early_difference = paired_difference(
                    early_by_id[decision.candidate_id], early_by_id[race.boundary_candidate_id]
                ).mean
                maximum_difference = paired_difference(
                    maximum_by_id[decision.candidate_id], maximum_by_id[race.boundary_candidate_id]
                ).mean
                strata = tuple(
                    StratumAudit(
                        item.stratum_id,
                        _mean(item.values),
                        _mean(
                            next(
                                final.values
                                for final in paired_stratum_differences(
                                    manifest,
                                    maximum_by_id[decision.candidate_id],
                                    maximum_by_id[race.boundary_candidate_id],
                                )
                                if final.stratum_id == item.stratum_id
                            )
                        ),
                        (_mean(item.values) >= -manifest.shadow_policy.practical_effect_margin)
                        != (
                            _mean(
                                next(
                                    final.values
                                    for final in paired_stratum_differences(
                                        manifest,
                                        maximum_by_id[decision.candidate_id],
                                        maximum_by_id[race.boundary_candidate_id],
                                    )
                                    if final.stratum_id == item.stratum_id
                                )
                            )
                            >= -manifest.shadow_policy.practical_effect_margin
                        ),
                    )
                    for item in paired_stratum_differences(
                        manifest,
                        early_by_id[decision.candidate_id],
                        early_by_id[race.boundary_candidate_id],
                    )
                )
                for item in strata:
                    stratum_rows[item.stratum_id].append(
                        (
                            item.reversal,
                            decision.disposition == "eliminate"
                            and item.maximum_mean_difference
                            >= -manifest.shadow_policy.practical_effect_margin,
                        )
                    )
                per_candidate[decision.candidate_id].append(
                    ShadowLookAudit(
                        cohort.cohort_index,
                        race.prefix_id,
                        decision.candidate_id,
                        race.boundary_candidate_id,
                        decision.favorable_resamples,
                        decision.total_resamples,
                        decision.disposition,
                        early_difference,
                        maximum_difference,
                        maximum_difference >= -manifest.shadow_policy.practical_effect_margin,
                        strata,
                    )
                )
        for candidate in cohort.candidates:
            looks = tuple(per_candidate[candidate.candidate_id])
            protected = any(item.disposition == "protected" for item in looks)
            first = (
                None
                if protected
                else next((item for item in looks if item.disposition == "eliminate"), None)
            )
            if not protected:
                for item in looks[: (looks.index(first) + 1 if first else len(looks))]:
                    calibration.append(
                        (
                            item.favorable_resamples / item.total_resamples,
                            float(item.final_reaches_recorded_boundary),
                        )
                    )
            unique, compute = (
                (0, PhaseCompute())
                if first is None
                else _suffix_compute(manifest, events, candidate, first.prefix_id)
            )
            total_compute = _phase_add(total_compute, compute)
            paths.append(
                CandidatePathAudit(
                    cohort.cohort_index,
                    candidate.candidate_id,
                    protected,
                    candidate.candidate_id in top_ids,
                    looks,
                    None if first is None else first.prefix_id,
                    unique,
                    compute,
                )
            )
    return _finish_audit(paths, stratum_rows, calibration, total_compute)


def _completed_candidates(state: ReplayState) -> set[str]:
    if state.terminal_status != "complete":
        raise ValueError("shadow audit requires completed evidence")
    return {
        candidate.candidate_id
        for cohort in state.completed_cohorts
        for candidate in cohort.candidates
    }


def _audit_inputs(
    state: ReplayState,
) -> tuple[
    set[str],
    list[CandidatePathAudit],
    defaultdict[str, list[tuple[bool, bool]]],
    list[tuple[float, float]],
    PhaseCompute,
]:
    return _completed_candidates(state), [], defaultdict(list), [], PhaseCompute()


def _finish_audit(
    paths: list[CandidatePathAudit],
    stratum_rows: defaultdict[str, list[tuple[bool, bool]]],
    calibration: list[tuple[float, float]],
    total_compute: PhaseCompute,
) -> ShadowAudit:
    eliminated = [
        item
        for item in paths
        if not item.protected and item.first_elimination_prefix_id is not None
    ]
    eligible = [item for item in paths if not item.protected and item.final_top_set and item.looks]
    brier = (
        None if not calibration else sum((p - y) ** 2 for p, y in calibration) / len(calibration)
    )
    bins: list[CalibrationBin] = []
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        values = [(p, y) for p, y in calibration if lower <= p and (p < upper or upper == 1.0)]
        if values:
            bins.append(
                CalibrationBin(
                    lower,
                    upper,
                    len(values),
                    sum(p for p, _ in values) / len(values),
                    sum(y for _, y in values) / len(values),
                )
            )
    return ShadowAudit(
        tuple(paths),
        tuple(bins),
        tuple(
            StratumSummary(
                key, len(value), sum(row[0] for row in value), sum(row[1] for row in value)
            )
            for key, value in sorted(stratum_rows.items())
        ),
        len(eliminated),
        len(eligible),
        sum(item.final_top_set for item in eliminated),
        sum(not item.final_top_set for item in eliminated),
        brier,
        total_compute,
    )
