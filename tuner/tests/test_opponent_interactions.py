from __future__ import annotations

from dataclasses import replace

import pytest

from tuner_cli.domain import (
    Candidate,
    CohortRecord,
    GameResult,
    ObservationContext,
    Opponent,
    OpponentPanel,
    PairResult,
    PairTask,
    SearchEffort,
    StrategyMetrics,
    TaskCase,
    TaskPrefix,
)
from tuner_cli.observations import observation
from tuner_cli.opponent_interactions import build_opponent_response_analysis


def _analysis_fixture(count: int = 10):
    opponents = (
        Opponent("alpha", "inline", "Alpha", "default", 1, "{}", "alpha-fingerprint"),
        Opponent("beta", "inline", "Beta", "historical_reference", 1, "{}", "beta-fingerprint"),
    )
    panel = OpponentPanel("panel", "panel-fingerprint", opponents, 2)
    candidates = (
        Candidate("left", "left-fingerprint", "{}"),
        Candidate("right", "right-fingerprint", "{}"),
    )
    cases = tuple(
        TaskCase(
            f"{opponent.opponent_id}-{index}",
            "tuning",
            ordinal,
            ordinal,
            "default",
            opponent.opponent_id,
            opponent.configuration_fingerprint,
            panel.fingerprint,
            "game",
        )
        for ordinal, (opponent, index) in enumerate(
            (item, index) for item in opponents for index in range(count)
        )
    )
    prefix = TaskPrefix("maximum", "tuning", len(cases), tuple(item.task_id for item in cases))
    context = ObservationContext("epoch", "tuning", prefix, SearchEffort("iterations", 3))
    pairs = tuple(
        _pair(candidate.candidate_id, case, _outcome(candidate.candidate_id, case.opponent_id))
        for candidate in candidates
        for case in cases
    )
    observations = tuple(
        observation(
            candidate.candidate_id,
            context,
            tuple(_utility(_outcome(candidate.candidate_id, case.opponent_id)) for case in cases),
        )
        for candidate in candidates
    )
    return panel, CohortRecord(7, candidates, ("left",)), observations, pairs


def _outcome(candidate_id: str, opponent_id: str) -> str:
    return (
        "candidate_win"
        if (candidate_id, opponent_id) in {("left", "alpha"), ("right", "beta")}
        else "baseline_win"
    )


def _utility(outcome: str) -> float:
    return 1.0 if outcome == "candidate_win" else 0.0


def _pair(candidate_id: str, case: TaskCase, outcome: str) -> PairResult:
    metrics = StrategyMetrics(1, 1, 1)
    games = tuple(
        GameResult(
            f"{candidate_id}-{case.task_id}-{side}",
            side,
            outcome,
            1,
            1,
            sequence,
            None,
            1,
            1,
            metrics,
            metrics,
            "{}",
        )
        for sequence, side in ((1, "first"), (2, "second"))
    )
    return PairResult(
        PairTask(
            f"pair-{candidate_id}-{case.task_id}", candidate_id, case, SearchEffort("iterations", 3)
        ),
        games,
    )  # type: ignore[arg-type]


def test_opponent_responses_use_common_panel_tasks_in_order() -> None:
    panel, cohort, observations, pairs = _analysis_fixture()
    analysis = build_opponent_response_analysis(panel, cohort, observations, pairs)
    assert [(item.candidate_id, item.opponent_id) for item in analysis.responses] == [
        ("left", "alpha"),
        ("left", "beta"),
        ("right", "alpha"),
        ("right", "beta"),
    ]
    assert [pair.task.task_case.task_id for pair in analysis.responses[0].pairs] == [
        f"alpha-{index}" for index in range(10)
    ]


def test_opponent_interaction_detects_conservative_ranking_reversal() -> None:
    panel, cohort, observations, pairs = _analysis_fixture()
    interaction = build_opponent_response_analysis(panel, cohort, observations, pairs).interactions[
        0
    ]
    assert [item.relation for item in interaction.contrasts] == ["better", "worse"]
    assert [
        (item.left_opponent_id, item.right_opponent_id) for item in interaction.ranking_reversals
    ] == [("alpha", "beta")]


def test_opponent_interaction_does_not_promote_uncertain_sign_change() -> None:
    panel, cohort, observations, pairs = _analysis_fixture(count=2)
    interaction = build_opponent_response_analysis(panel, cohort, observations, pairs).interactions[
        0
    ]
    assert [item.relation for item in interaction.contrasts] == ["tie", "tie"]
    assert interaction.ranking_reversals == ()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pairs: pairs[:-1],
        lambda pairs: (*pairs, pairs[0]),
        lambda pairs: (
            *pairs[:-1],
            replace(pairs[-1], task=replace(pairs[-1].task, budget=SearchEffort("iterations", 4))),
        ),
        lambda pairs: (
            *pairs[:-1],
            replace(
                pairs[-1],
                task=replace(
                    pairs[-1].task, task_case=replace(pairs[-1].task.task_case, phase="validation")
                ),
            ),
        ),
    ],
)
def test_opponent_analysis_rejects_incomplete_or_mismatched_evidence(mutate) -> None:  # type: ignore[no-untyped-def]
    panel, cohort, observations, pairs = _analysis_fixture()
    with pytest.raises(ValueError):
        build_opponent_response_analysis(panel, cohort, observations, mutate(pairs))
