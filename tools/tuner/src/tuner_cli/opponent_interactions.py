"""Read-only opponent-specific tuning response analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .domain import (
    CohortRecord,
    Estimate,
    Observation,
    ObservationContext,
    OpponentPanel,
    PairResult,
)
from .statistics import marginal_interval, pair_utility, paired_difference_values, tie_relation


@dataclass(frozen=True, slots=True)
class OpponentResponse:
    candidate_id: str
    opponent_id: str
    estimate: Estimate
    pair_count: int
    pairs: tuple[PairResult, ...]


@dataclass(frozen=True, slots=True)
class OpponentContrast:
    opponent_id: str
    paired_difference: Estimate
    relation: str


@dataclass(frozen=True, slots=True)
class OpponentRankingReversal:
    left_opponent_id: str
    right_opponent_id: str


@dataclass(frozen=True, slots=True)
class CandidateOpponentInteraction:
    left_candidate_id: str
    right_candidate_id: str
    contrasts: tuple[OpponentContrast, ...]
    ranking_reversals: tuple[OpponentRankingReversal, ...]


@dataclass(frozen=True, slots=True)
class OpponentResponseAnalysis:
    responses: tuple[OpponentResponse, ...]
    interactions: tuple[CandidateOpponentInteraction, ...]


def build_opponent_response_analysis(
    panel: OpponentPanel,
    cohort: CohortRecord,
    observations: tuple[Observation, ...],
    completed_tuning_pairs: tuple[PairResult, ...],
) -> OpponentResponseAnalysis:
    """Project complete maximum-prefix tuning evidence in frozen roster order."""
    ordered_observations = _validate_observations(cohort, observations)
    context = ordered_observations[0].context if ordered_observations else None
    if context is None:
        raise ValueError("opponent analysis needs a non-empty completed cohort")
    evidence = _candidate_opponent_evidence(panel, cohort, context, completed_tuning_pairs)
    responses = tuple(
        OpponentResponse(
            candidate.candidate_id,
            opponent.opponent_id,
            marginal_interval(
                tuple(
                    pair_utility(pair)
                    for pair in evidence[candidate.candidate_id][opponent.opponent_id]
                )
            ),
            len(evidence[candidate.candidate_id][opponent.opponent_id]),
            evidence[candidate.candidate_id][opponent.opponent_id],
        )
        for candidate in cohort.candidates
        for opponent in panel.opponents
    )
    interactions = tuple(
        _interaction(left.candidate_id, right.candidate_id, panel, evidence)
        for left, right in combinations(cohort.candidates, 2)
    )
    return OpponentResponseAnalysis(responses, interactions)


def _validate_observations(
    cohort: CohortRecord, observations: tuple[Observation, ...]
) -> tuple[Observation, ...]:
    by_candidate = {item.candidate_id: item for item in observations}
    candidate_ids = tuple(candidate.candidate_id for candidate in cohort.candidates)
    if len(by_candidate) != len(observations) or set(by_candidate) != set(candidate_ids):
        raise ValueError("opponent analysis needs one observation per cohort candidate")
    ordered = tuple(by_candidate[candidate_id] for candidate_id in candidate_ids)
    reference = ordered[0] if ordered else None
    if reference is None:
        return ordered
    if reference.phase != "tuning":
        raise ValueError("opponent analysis observations must be tuning evidence")
    for item in ordered:
        if item.context != reference.context:
            raise ValueError("opponent analysis observations are not comparable")
    return ordered


def _candidate_opponent_evidence(
    panel: OpponentPanel,
    cohort: CohortRecord,
    context: ObservationContext,
    pairs: tuple[PairResult, ...],
) -> dict[str, dict[str, tuple[PairResult, ...]]]:
    candidate_ids = {candidate.candidate_id for candidate in cohort.candidates}
    panel_by_id = {opponent.opponent_id: opponent for opponent in panel.opponents}
    selected = [
        pair
        for pair in pairs
        if pair.task.candidate_id in candidate_ids
        and pair.task.task_case.phase == "tuning"
        and pair.task.budget == context.search_effort
        and pair.task.task_case.task_id in context.task_prefix.task_ids
    ]
    grouped: dict[str, dict[str, list[PairResult]]] = {
        candidate.candidate_id: {opponent.opponent_id: [] for opponent in panel.opponents}
        for candidate in cohort.candidates
    }
    for pair in selected:
        task = pair.task.task_case
        opponent = panel_by_id.get(task.opponent_id)
        if opponent is None:
            raise ValueError("opponent analysis evidence has an unknown panel opponent")
        if (
            task.opponent_fingerprint != opponent.configuration_fingerprint
            or task.panel_fingerprint != panel.fingerprint
        ):
            raise ValueError("opponent analysis evidence does not match the frozen panel")
        grouped[pair.task.candidate_id][task.opponent_id].append(pair)
    result: dict[str, dict[str, tuple[PairResult, ...]]] = {}
    expected_by_opponent: dict[str, tuple[tuple[str, int], ...]] | None = None
    for candidate in cohort.candidates:
        candidate_rows: dict[str, tuple[PairResult, ...]] = {}
        actual_by_opponent: dict[str, tuple[tuple[str, int], ...]] = {}
        for opponent in panel.opponents:
            ordered = tuple(
                sorted(
                    grouped[candidate.candidate_id][opponent.opponent_id],
                    key=lambda pair: pair.task.task_case.ordinal,
                )
            )
            task_ids = tuple(pair.task.task_case.task_id for pair in ordered)
            if len(set(task_ids)) != len(task_ids):
                raise ValueError("opponent analysis evidence has duplicate task evidence")
            candidate_rows[opponent.opponent_id] = ordered
            actual_by_opponent[opponent.opponent_id] = tuple(
                (pair.task.task_case.task_id, pair.task.task_case.ordinal) for pair in ordered
            )
        all_task_ids = tuple(
            task_id for task_ids in actual_by_opponent.values() for task_id, _ in task_ids
        )
        if set(all_task_ids) != set(context.task_prefix.task_ids) or len(all_task_ids) != len(
            context.task_prefix.task_ids
        ):
            raise ValueError("opponent analysis evidence lacks the complete task prefix")
        if expected_by_opponent is None:
            expected_by_opponent = actual_by_opponent
        elif actual_by_opponent != expected_by_opponent:
            raise ValueError("opponent analysis evidence has mismatched opponent task prefixes")
        result[candidate.candidate_id] = candidate_rows
    return result


def _interaction(
    left_candidate_id: str,
    right_candidate_id: str,
    panel: OpponentPanel,
    evidence: dict[str, dict[str, tuple[PairResult, ...]]],
) -> CandidateOpponentInteraction:
    contrasts = tuple(
        _contrast(
            opponent.opponent_id,
            evidence[left_candidate_id][opponent.opponent_id],
            evidence[right_candidate_id][opponent.opponent_id],
        )
        for opponent in panel.opponents
    )
    reversals = tuple(
        OpponentRankingReversal(left.opponent_id, right.opponent_id)
        for left, right in combinations(contrasts, 2)
        if {left.relation, right.relation} == {"better", "worse"}
    )
    return CandidateOpponentInteraction(left_candidate_id, right_candidate_id, contrasts, reversals)


def _contrast(
    opponent_id: str, left: tuple[PairResult, ...], right: tuple[PairResult, ...]
) -> OpponentContrast:
    left_by_task = {pair.task.task_case.task_id: pair for pair in left}
    right_by_task = {pair.task.task_case.task_id: pair for pair in right}
    if tuple(left_by_task) != tuple(right_by_task):
        raise ValueError("opponent contrast needs exact common task IDs")
    difference = paired_difference_values(
        tuple(pair_utility(pair) for pair in left_by_task.values()),
        tuple(pair_utility(pair) for pair in right_by_task.values()),
    )
    return OpponentContrast(opponent_id, difference, tie_relation(difference))
