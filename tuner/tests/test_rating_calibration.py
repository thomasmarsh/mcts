"""Deterministic proof for the offline rating-calibration example."""

from __future__ import annotations

import pytest

from tuner_cli.config import RatingPolicy, ResourcePolicy, SearchConfig
from tuner_cli.evaluation import (
    GameResult,
    OpponentSnapshot,
    PairResult,
    PairTask,
    Rating,
    StrategyMetrics,
    TrialEvaluationState,
    conservative_score,
    game_id_for,
    pair_id_for,
)
from tuner_cli.lifecycle import SessionId, TrialId
from tuner_cli.rating_calibration import (
    calibrate,
    parse_scripted_pair,
    parse_sigma_stop,
    render_calibration,
    resolve_policy,
)

OPPONENT = OpponentSnapshot("baseline", {}, 25.0, 0.5)
INITIAL_RATING = Rating(25.0, 8.333)
METRICS = StrategyMetrics(0, 0, 0)


@pytest.mark.parametrize(
    ("script", "expected_outcomes"),
    [
        ("first:win,second:win", ("candidate_win", "candidate_win")),
        ("first:loss,second:loss", ("baseline_win", "baseline_win")),
        ("first:draw,second:draw", ("draw", "draw")),
        ("first:win,second:loss", ("candidate_win", "baseline_win")),
    ],
)
def test_calibration_covers_each_pair_outcome_shape(script, expected_outcomes):
    pair = parse_scripted_pair(script)
    assert pair.outcomes == expected_outcomes

    resource = ResourcePolicy(min_pairs=2, max_pairs=3)
    rating_policy = RatingPolicy(sigma_stop=None, conservative_k=2.5)
    displayed = calibrate([pair], resource, rating_policy, OPPONENT, INITIAL_RATING)[0]
    production = TrialEvaluationState(resource, rating_policy, rating=INITIAL_RATING)
    _apply_production_pair(production, pair, 0)

    assert displayed.rating == production.rating
    assert displayed.score == conservative_score(production.rating, 2.5)
    assert displayed.decision == production.decision()


@pytest.mark.parametrize(
    "script",
    [
        "win,second:loss",
        "second:loss,first:win",
        "first:win,second:loss,first:draw",
        "first:unknown,second:loss",
    ],
)
def test_parser_requires_explicit_ordered_seats(script):
    with pytest.raises(ValueError):
        parse_scripted_pair(script)


def test_split_pair_reversed_seat_order_uses_production_game_order():
    policy = ResourcePolicy(min_pairs=2, max_pairs=4)
    rating_policy = RatingPolicy(sigma_stop=None, conservative_k=2.5)
    first_seat_win = calibrate(
        [parse_scripted_pair("first:win,second:loss")],
        policy,
        rating_policy,
        OPPONENT,
        INITIAL_RATING,
    )[0]
    second_seat_win = calibrate(
        [parse_scripted_pair("first:loss,second:win")],
        policy,
        rating_policy,
        OPPONENT,
        INITIAL_RATING,
    )[0]

    assert first_seat_win.rating != second_seat_win.rating
    assert first_seat_win.pair.first == "candidate_win"
    assert second_seat_win.pair.second == "candidate_win"


def test_displayed_values_are_exact_production_rating_score_and_decision():
    resource = ResourcePolicy(min_pairs=2, max_pairs=3)
    rating_policy = RatingPolicy(sigma_stop=None, conservative_k=2.5)
    rows = calibrate(
        [
            parse_scripted_pair("first:win,second:draw"),
            parse_scripted_pair("first:loss,second:win"),
        ],
        resource,
        rating_policy,
        OPPONENT,
        INITIAL_RATING,
    )

    production = TrialEvaluationState(resource, rating_policy, rating=INITIAL_RATING)
    for pair, row in zip(
        [
            parse_scripted_pair("first:win,second:draw"),
            parse_scripted_pair("first:loss,second:win"),
        ],
        rows,
        strict=True,
    ):
        _apply_production_pair(production, pair, row.pair_index - 1)
        assert row.rating == production.rating
        assert row.score == conservative_score(production.rating, 2.5)
        assert row.decision == production.decision()

    output = render_calibration(rows)
    assert f"mu={rows[-1].rating.mu!r}" in output
    assert f"sigma={rows[-1].rating.sigma!r}" in output
    assert f"conservative_score={rows[-1].score!r}" in output
    assert "decision=continue/pruning_disabled" in output


def test_stop_policy_covers_minimum_confidence_disabled_and_maximum():
    confidence_blocked = calibrate(
        [parse_scripted_pair("first:draw,second:draw")],
        ResourcePolicy(min_pairs=2, max_pairs=3),
        RatingPolicy(sigma_stop=100.0, conservative_k=3.0),
        OPPONENT,
        INITIAL_RATING,
    )[0]
    assert confidence_blocked.decision.reason == "below_min_pairs"

    confidence_disabled = calibrate(
        [parse_scripted_pair("first:draw,second:draw")],
        ResourcePolicy(min_pairs=1, max_pairs=2),
        RatingPolicy(sigma_stop=None, conservative_k=3.0),
        OPPONENT,
        INITIAL_RATING,
    )[0]
    assert confidence_disabled.decision.reason == "pruning_disabled"

    maxed_out = calibrate(
        [
            parse_scripted_pair("first:draw,second:draw"),
            parse_scripted_pair("first:draw,second:draw"),
        ],
        ResourcePolicy(min_pairs=1, max_pairs=2),
        RatingPolicy(sigma_stop=None, conservative_k=3.0),
        OPPONENT,
        INITIAL_RATING,
    )[-1]
    assert maxed_out.decision.reason == "max_pairs"


def test_resolved_policy_supports_custom_limits_k_and_disabled_sigma():
    resource, rating = resolve_policy(
        SearchConfig.defaults(),
        min_pairs=3,
        max_pairs=9,
        sigma_stop=None,
        conservative_k=2.7,
    )
    assert resource == ResourcePolicy(min_pairs=3, max_pairs=9)
    assert rating == RatingPolicy(sigma_stop=None, conservative_k=2.7)
    assert parse_sigma_stop("none") is None
    assert parse_sigma_stop("1.5") == 1.5


def _apply_production_pair(state: TrialEvaluationState, pair, pair_index: int) -> None:
    pair_id = pair_id_for(SessionId("test"), TrialId("calibration"), pair_index)
    task = PairTask(
        SessionId("test"),
        TrialId("calibration"),
        pair_id,
        pair_index,
        pair_index + 1,
        {},
        OPPONENT,
        "test",
        state.rating,
    )
    state.apply_pair(
        PairResult(
            task,
            tuple(
                GameResult(
                    game_id_for(pair_id, side),
                    side,
                    outcome,
                    pair_index + 1,
                    0,
                    None,
                    0,
                    0,
                    METRICS,
                    METRICS,
                )
                for side, outcome in zip(("first", "second"), pair.outcomes, strict=True)
            ),
        )
    )
