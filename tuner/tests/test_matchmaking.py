"""Typed pair rating-state and pool-snapshot tests."""

from __future__ import annotations

from types import SimpleNamespace

from tuner_cli.config import RatingPolicy, ResourcePolicy
from tuner_cli.evaluation import (
    GameResult,
    OpponentSnapshot,
    PairResult,
    PairTask,
    Rating,
    StrategyMetrics,
    TrialEvaluationState,
    game_id_for,
    pair_id_for,
    pool_snapshot_fingerprint,
)
from tuner_cli.lifecycle import SessionId, TrialId
from tuner_cli.pool import Anchor
from tuner_cli.pair_orchestration import make_next_pair_task
from tuner_cli.pool import OpponentPool


def _state(**kwargs) -> TrialEvaluationState:
    return TrialEvaluationState(ResourcePolicy(), RatingPolicy(), **kwargs)


def _task(index: int = 0) -> PairTask:
    session, trial = SessionId("session"), TrialId("trial")
    pair_id = pair_id_for(session, trial, index)
    return PairTask(
        session,
        trial,
        pair_id,
        index,
        7 + index,
        {"family": "rave"},
        OpponentSnapshot("default", {"family": "ucb1"}, 25.0, 0.5),
        "pool",
        Rating(25.0, 8.3),
    )


def _pair(outcomes: tuple[str, str]) -> PairResult:
    task = _task()
    metrics = StrategyMetrics(1, 1, 1)
    games = tuple(
        GameResult(
            game_id_for(task.pair_id, side),
            side,
            outcome,
            7,
            1,
            None,
            2,
            3,
            metrics,
            metrics,
        )
        for side, outcome in zip(("first", "second"), outcomes, strict=True)
    )
    return PairResult(task, games)


def test_pair_and_game_ids_are_stable_and_side_specific():
    assert pair_id_for(SessionId("s"), TrialId("t"), 1) == pair_id_for(
        SessionId("s"), TrialId("t"), 1
    )
    pair_id = pair_id_for(SessionId("s"), TrialId("t"), 1)
    assert game_id_for(pair_id, "first") != game_id_for(pair_id, "second")


def test_full_pool_fingerprint_changes_with_any_anchor_snapshot_field():
    anchors = [Anchor("a", {"family": "random"}, 0.0, 0.5)]
    before = pool_snapshot_fingerprint(anchors)
    anchors[0].config["q_init"] = "Infinity"
    assert pool_snapshot_fingerprint(anchors) != before


def test_state_applies_two_physical_outcomes_in_order_and_preserves_legacy_shape():
    state = _state()
    before = state.rating
    state.apply_pair(_pair(("candidate_win", "baseline_win")))
    assert state.completed_pairs == 1
    assert state.rating != before
    assert state.legacy_games() == [
        {"opponent": "default", "outcome": "win"},
        {"opponent": "default", "outcome": "loss"},
    ]
    assert state.should_continue()


def test_game_order_changes_rating_deterministically():
    win_then_loss = _state()
    loss_then_win = _state()
    win_then_loss.apply_pair(_pair(("candidate_win", "baseline_win")))
    loss_then_win.apply_pair(_pair(("baseline_win", "candidate_win")))
    assert win_then_loss.rating != loss_then_win.rating


def test_pair_stopping_keeps_five_pair_floor_and_fifteen_pair_ceiling():
    state = _state(rating=Rating(25.0, 1.0), completed_pairs=4)
    assert state.should_continue()
    state.completed_pairs = 5
    assert not state.should_continue()

    state.rating = Rating(25.0, 8.0)
    state.completed_pairs = 14
    assert state.should_continue()
    state.completed_pairs = 15
    assert not state.should_continue()


def test_stopping_decision_uses_resolved_policy_precedence_and_disabled_sigma():
    resource = ResourcePolicy(min_pairs=3, max_pairs=5)
    state = TrialEvaluationState(
        resource,
        RatingPolicy(sigma_stop=2.0, conservative_k=2.5),
        rating=Rating(25.0, 1.0),
        completed_pairs=2,
    )
    assert state.decision().reason == "below_min_pairs"

    state.completed_pairs = 3
    assert state.decision().reason == "confidence"

    state.completed_pairs = 5
    assert state.decision().reason == "confidence"

    state.rating_policy = RatingPolicy(sigma_stop=None, conservative_k=2.5)
    assert state.decision().reason == "max_pairs"
    assert state.score() == 22.5


def test_next_pair_selects_against_updated_candidate_rating(tmp_path):
    state = _state()
    active = SimpleNamespace(
        evaluation=state,
        trial_id=TrialId("trial"),
        seed=7,
        config={"family": "rave"},
    )
    pool = OpponentPool([Anchor("initial", {"family": "ucb1"}, 25.0, 0.5)])

    state.apply_pair(_pair(("candidate_win", "candidate_win")))
    pool.anchors.append(
        Anchor("updated", {"family": "rave"}, state.rating.mu, 0.5)
    )
    lifecycle = SimpleNamespace(session_id=SessionId("session"))
    task = make_next_pair_task(active, pool, lifecycle, str(tmp_path / "trace.jsonl"))
    assert task.opponent.anchor_id == "updated"
    assert task.rating_before == state.rating
