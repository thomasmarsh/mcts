"""One-pair scheduling and failure ordering tests."""

from __future__ import annotations

import json
from pathlib import Path

import optuna

from tuner_cli import attempt
from tuner_cli.config import OptimizerConfig, SearchConfig, TargetConfig
from tuner_cli.evaluation import (
    GameResult,
    OpponentSnapshot,
    PairResult,
    PairTask,
    Rating,
    StrategyMetrics,
    configured_game_seed,
    game_id_for,
)
from tuner_cli.lifecycle import AttemptId, LifecycleWriter, SessionId, TrialId
from tuner_cli import pair_orchestration
from tuner_cli.pair_orchestration import ScheduledPair
from tuner_cli.pool import Anchor, OpponentPool


class _Future:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def cancelled(self):
        return False

    def result(self):
        if self.error:
            raise self.error
        raise AssertionError("result was not configured")

    def cancel(self):
        return True


class _Executor:
    def __init__(self):
        self.calls: list[tuple] = []
        self.futures: list[_Future] = []

    def submit(self, *args):
        self.calls.append(args)
        future = _Future()
        self.futures.append(future)
        return future


def _active(study) -> attempt._ActiveTrial:
    trial = study.ask()
    trial.set_user_attr("config", {"family": "rave"})
    return attempt._ActiveTrial(trial, TrialId("trial"), {"family": "rave"}, 7)


def _task(active: attempt._ActiveTrial) -> PairTask:
    return PairTask(
        SessionId("session"),
        active.trial_id,
        "pair",
        0,
        7,
        active.config,
        OpponentSnapshot("random", {"family": "random"}, 0.0, 0.5),
        "pool",
        Rating(25.0, 8.3),
    )


def test_submission_emits_pair_started_and_one_future_for_one_pair(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(
        optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("game-nim"))
    )
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        attempt.submit_next_pair(
            executor,
            futures,
            active,
            cfg,
            Path("game-nim"),
            pool,
            study,
            writer,
            None,
            attempt._terminalize_from_pair(writer),
        )
    assert len(futures) == 1
    assert executor.calls[0][0] is pair_orchestration.evaluate_pair
    task = executor.calls[0][3]
    assert task.pair_index == 0
    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert records[-1]["event_type"] == "pair_started"
    assert records[-1]["payload"]["seed"] == configured_game_seed(task.seed)
    assert records[-1]["payload"]["round"] == 1


def test_initial_scheduling_keeps_multiple_trials_active_with_one_pair_each(
    tmp_path: Path,
):
    study = optuna.create_study(direction="maximize")
    executor, futures, active = _Executor(), {}, {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(
        optimizer=OptimizerConfig(n_workers=2),
        target=TargetConfig(binary=Path("game-nim")),
    )
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        attempt.schedule_initial_trials(
            2,
            2,
            executor,
            futures,
            active,
            cfg,
            Path("game-nim"),
            pool,
            study,
            writer,
            None,
        )
    assert len(active) == len(futures) == len(executor.calls) == 2
    assert all(call[0] is pair_orchestration.evaluate_pair for call in executor.calls)
    assert {call[3].trial_id for call in executor.calls} == set(active)


def test_worker_failure_emits_pair_failure_before_trial_terminal(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    scheduled = ScheduledPair(active, _task(active))
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        assert (
            attempt.worker_result(
                _Future(RuntimeError("boom")),
                study,
                writer,
                scheduled,
                attempt._terminalize_from_pair(writer),
            )
            is None
        )
    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [record["event_type"] for record in records] == [
        "pair_failed",
        "trial_failed",
    ]


def test_coordinator_cancellation_fails_running_pair_before_trial(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    future = _Future()
    scheduled = ScheduledPair(active, _task(active))
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        attempt.cancel_active_trials(
            {future: scheduled}, {active.trial_id: active}, study, writer
        )
    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [record["event_type"] for record in records] == [
        "pair_failed",
        "trial_cancelled",
    ]


def test_pair_completion_emits_ordered_physical_evidence_before_pair_terminal(
    tmp_path: Path,
):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    task = _task(active)
    metrics = StrategyMetrics(1, 1, 1)
    result = PairResult(
        task,
        (
            GameResult(
                game_id_for(task.pair_id, "first"),
                "first",
                "candidate_win",
                7,
                1,
                4,
                2,
                3,
                metrics,
                metrics,
            ),
            GameResult(
                game_id_for(task.pair_id, "second"),
                "second",
                "draw",
                7,
                1,
                5,
                4,
                6,
                metrics,
                metrics,
            ),
        ),
    )
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        attempt.finish_pair(writer, active, result)
    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [record["event_type"] for record in records] == [
        "game_finished",
        "game_finished",
        "pair_finished",
    ]
    assert set(records[0]["payload"]) == {
        "trial_id",
        "pair_id",
        "game_id",
        "candidate_side",
        "outcome",
        "seed",
        "round",
        "trace_game_seq",
        "plies",
        "elapsed_ms",
        "candidate",
        "baseline",
    }
    assert records[2]["payload"]["pair_index"] == 0


def test_success_preserves_legacy_trial_pool_and_incumbent_order(
    monkeypatch, tmp_path: Path
):
    events: list[str] = []
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    active.evaluation.rating = Rating(25.0, 2.0)

    monkeypatch.setattr(
        attempt,
        "record_completed_trial",
        lambda *args: events.append("lifecycle trial_completed"),
    )
    monkeypatch.setattr(
        attempt, "emit_legacy_trial", lambda *args: events.append("legacy trial")
    )
    monkeypatch.setattr(
        attempt,
        "save_inserted_pool_anchor",
        lambda *args: events.append("pool maybe_insert/save"),
    )
    monkeypatch.setattr(
        attempt,
        "emit_legacy_incumbent",
        lambda *args: events.append("legacy incumbent"),
    )

    attempt.complete_trial(
        study, object(), active, object(), tmp_path / "pool.json", "sha"
    )
    assert events == [
        "lifecycle trial_completed",
        "legacy trial",
        "pool maybe_insert/save",
        "legacy incumbent",
    ]
