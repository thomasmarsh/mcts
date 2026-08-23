"""One-pair scheduling and failure ordering tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from tuner_cli import attempt
from tuner_cli import coordinator
from tuner_cli.config import (
    OptimizerConfig,
    RatingPolicy,
    ResourcePolicy,
    SearchConfig,
    TargetConfig,
)
from tuner_cli.evaluation import (
    GameResult,
    OpponentSnapshot,
    PairResult,
    PairTask,
    Rating,
    StrategyMetrics,
    configured_game_seed,
    game_id_for,
    TrialEvaluationState,
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


def test_open_study_configures_tpe_startup_trials_without_a_pruner(
    monkeypatch, tmp_path: Path
):
    sampler_kwargs: dict = {}
    study_kwargs: dict = {}

    def sampler(**kwargs):
        sampler_kwargs.update(kwargs)
        return "sampler"

    def create_study(**kwargs):
        study_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(coordinator.optuna.samplers, "TPESampler", sampler)
    monkeypatch.setattr(coordinator.optuna, "create_study", create_study)
    cfg = SearchConfig._from_dict({"optimizer": {"sampler": {"startup_trials": 17}}})

    coordinator._open_study(tmp_path, "study", cfg)

    assert sampler_kwargs == {"seed": 42, "n_startup_trials": 17}
    assert study_kwargs["sampler"] == "sampler"
    assert "pruner" not in study_kwargs


def _active(study) -> attempt._ActiveTrial:
    trial = study.ask()
    trial.set_user_attr("config", {"family": "rave"})
    return attempt._ActiveTrial(
        trial,
        TrialId(f"trial-{trial.number}"),
        {"family": "rave"},
        7,
        TrialEvaluationState(ResourcePolicy(), RatingPolicy()),
    )


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


def _result(task: PairTask) -> PairResult:
    metrics = StrategyMetrics(1, 1, 1)
    return PairResult(
        task,
        (
            GameResult(
                game_id_for(task.pair_id, "first"),
                "first",
                "candidate_win",
                task.seed,
                1,
                None,
                2,
                3,
                metrics,
                metrics,
            ),
            GameResult(
                game_id_for(task.pair_id, "second"),
                "second",
                "baseline_win",
                task.seed,
                1,
                None,
                2,
                3,
                metrics,
                metrics,
            ),
        ),
    )


def _context(
    cfg: SearchConfig,
    study: optuna.Study,
    writer: LifecycleWriter,
    pool: OpponentPool,
    pool_path: Path,
) -> attempt._AttemptContext:
    return attempt._AttemptContext(
        cfg, Path("game-nim"), pool, pool_path, study, writer, "sha", None
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
    assert all(record["event_type"] != "trial_reported" for record in records)


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


def test_partial_pair_cannot_emit_completion_or_trial_report(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    task = _task(active)
    metrics = StrategyMetrics(1, 1, 1)
    partial = PairResult(
        task,
        (
            GameResult(
                game_id_for(task.pair_id, "first"),
                "first",
                "candidate_win",
                7,
                1,
                None,
                2,
                3,
                metrics,
                metrics,
            ),
        ),
    )
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        with pytest.raises(ValueError, match="exactly two games"):
            attempt.finish_pair(writer, active, partial)
    assert (tmp_path / "events.jsonl").read_text() == ""


def test_complete_pairs_report_consecutive_resources_before_max_terminal(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    cfg = SearchConfig(
        optimizer=OptimizerConfig(
            n_workers=1,
            resource=ResourcePolicy(min_pairs=2, max_pairs=3),
            rating=RatingPolicy(sigma_stop=None, conservative_k=2.5),
        ),
        target=TargetConfig(binary=Path("game-nim")),
    )
    active = _active(study)
    active.evaluation = TrialEvaluationState(
        cfg.optimizer.resource, cfg.optimizer.rating
    )
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = _context(cfg, study, writer, pool, tmp_path / "pool.json")
        scheduled = ScheduledPair(active, _task(active))
        assert attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )
        scheduled = futures.popitem()[1]
        assert attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )
        scheduled = futures.popitem()[1]
        assert not attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    reports = [
        record["payload"]
        for record in records
        if record["event_type"] == "trial_reported"
    ]
    assert [report["completed_pairs"] for report in reports] == [1, 2, 3]
    assert [report["reason"] for report in reports] == [
        "below_min_pairs",
        "pruning_disabled",
        "max_pairs",
    ]
    assert [report["outcome"] for report in reports] == [
        "continue",
        "continue",
        "complete",
    ]
    assert all(report["score_formula_version"] == 1 for report in reports)
    assert all(report["conservative_k"] == 2.5 for report in reports)
    assert records[-2]["event_type"] == "trial_reported"
    assert records[-1]["event_type"] == "trial_completed"
    assert records[-1]["payload"]["reason"] == "max_pairs"
    assert {
        key: records[-1]["payload"][key]
        for key in (
            "completed_pairs",
            "score_formula_version",
            "conservative_k",
            "pruning_exempt",
            "bracket_id",
            "rung_resource",
        )
    } == {
        key: reports[-1][key]
        for key in (
            "completed_pairs",
            "score_formula_version",
            "conservative_k",
            "pruning_exempt",
            "bracket_id",
            "rung_resource",
        )
    }
    assert study.trials[0].intermediate_values.keys() == {1, 2, 3}
    assert study.trials[0].value == reports[-1]["score"]
    assert reports[-1]["score"] == pytest.approx(
        reports[-1]["mu"] - 2.5 * reports[-1]["sigma"]
    )


def test_confidence_completion_reports_before_its_terminal_evidence(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    cfg = SearchConfig(
        optimizer=OptimizerConfig(
            n_workers=1,
            resource=ResourcePolicy(min_pairs=1, max_pairs=3),
            rating=RatingPolicy(sigma_stop=100.0),
        ),
        target=TargetConfig(binary=Path("game-nim")),
    )
    active = _active(study)
    active.evaluation = TrialEvaluationState(
        cfg.optimizer.resource, cfg.optimizer.rating
    )
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = _context(cfg, study, writer, pool, tmp_path / "pool.json")
        scheduled = ScheduledPair(active, _task(active))
        assert not attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )
        attempt.complete_trial(
            study,
            writer,
            active,
            pool,
            tmp_path / "pool.json",
            "sha",
            active.evaluation.score(),
            "confidence",
        )

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [record["event_type"] for record in records[-2:]] == [
        "trial_reported",
        "trial_completed",
    ]
    assert records[-2]["payload"]["reason"] == "confidence"
    assert records[-1]["payload"]["reason"] == "confidence"


def test_two_active_trials_complete_once_each_with_one_pair_resource(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    cfg = SearchConfig(
        optimizer=OptimizerConfig(
            n_workers=2,
            resource=ResourcePolicy(min_pairs=1, max_pairs=1),
            rating=RatingPolicy(sigma_stop=None),
        ),
        target=TargetConfig(binary=Path("game-nim")),
    )
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = _context(cfg, study, writer, pool, tmp_path / "pool.json")
        for active in (_active(study), _active(study)):
            active.evaluation = TrialEvaluationState(
                cfg.optimizer.resource, cfg.optimizer.rating
            )
            scheduled = ScheduledPair(active, _task(active))
            assert not attempt.continue_trial(
                executor, futures, scheduled, _result(scheduled.task), context
            )

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [record["event_type"] for record in records].count("trial_reported") == 2
    terminals = [
        record for record in records if record["event_type"] == "trial_completed"
    ]
    assert len(terminals) == 2
    assert {record["payload"]["trial_id"] for record in terminals} == {
        "trial-0",
        "trial-1",
    }
    assert all(record["payload"]["reason"] == "max_pairs" for record in terminals)
    assert len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) == 2


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
        study,
        SimpleNamespace(has_trial_terminal=lambda _trial_id: False),
        active,
        object(),
        tmp_path / "pool.json",
        "sha",
        19.0,
        "max_pairs",
    )
    assert events == [
        "lifecycle trial_completed",
        "legacy trial",
        "pool maybe_insert/save",
        "legacy incumbent",
    ]
