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
    PruningPolicy,
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
from tuner_cli.hyperband import HyperbandDecision
from tuner_cli.lifecycle import AttemptId, LifecycleWriter, SessionId, TrialId
from tuner_cli import pair_orchestration
from tuner_cli.pair_orchestration import ScheduledPair
from tuner_cli.pool import Anchor, OpponentPool


class _Future:
    def __init__(
        self,
        error: Exception | None = None,
        result: PairResult | None = None,
        cancel_error: Exception | None = None,
    ):
        self.error = error
        self.value = result
        self.cancel_error = cancel_error
        self.cancel_calls = 0
        self.result_calls = 0

    def cancelled(self):
        return False

    def result(self):
        self.result_calls += 1
        if self.error:
            raise self.error
        if self.value is not None:
            return self.value
        raise AssertionError("result was not configured")

    def cancel(self):
        self.cancel_calls += 1
        if self.cancel_error:
            raise self.cancel_error
        return True


class _Executor:
    def __init__(self):
        self.calls: list[tuple] = []
        self.futures: list[_Future] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self._processes: dict[int, _Process] = {}

    def submit(self, *args):
        self.calls.append(args)
        future = _Future()
        self.futures.append(future)
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool):
        self.shutdown_calls.append((wait, cancel_futures))


class _Process:
    def __init__(self):
        self.terminated = 0

    def terminate(self):
        self.terminated += 1


class _ScriptedPruningAdapter:
    def __init__(self, decisions: list[HyperbandDecision]):
        self.decisions = iter(decisions)
        self.created = 0
        self.observed = 0

    def create_trial(self, study):
        self.created += 1
        return SimpleNamespace(trial=study.ask())

    def observe_after_report(self, _trial):
        self.observed += 1
        return next(self.decisions)


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


def test_open_study_constructs_the_configured_hyperband_pruner(tmp_path: Path):
    cfg = SearchConfig._from_dict({"optimizer": {"pruning": {"enabled": True}}})

    study, _storage = coordinator._open_study(tmp_path, "study", cfg)

    assert study.pruner.__class__.__name__ == "HyperbandPruner"
    assert study.pruner._min_resource == cfg.optimizer.resource.min_pairs
    assert study.pruner._max_resource == cfg.optimizer.resource.max_pairs


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


def _pruning_config(
    *, min_pairs: int = 1, max_pairs: int = 3, sigma_stop: float | None = None
) -> SearchConfig:
    return SearchConfig(
        optimizer=OptimizerConfig(
            resource=ResourcePolicy(min_pairs=min_pairs, max_pairs=max_pairs),
            rating=RatingPolicy(sigma_stop=sigma_stop),
            pruning=PruningPolicy(enabled=True),
        ),
        target=TargetConfig(binary=Path("game-nim")),
    )


def _event_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _tell_calls(monkeypatch, study: optuna.Study) -> list[tuple[tuple, dict]]:
    calls: list[tuple[tuple, dict]] = []
    original_tell = study.tell

    def tell(*args, **kwargs):
        calls.append((args, kwargs))
        return original_tell(*args, **kwargs)

    monkeypatch.setattr(study, "tell", tell)
    return calls


def _stopped_attempt(
    monkeypatch,
    tmp_path: Path,
    *,
    workers: int,
    stop_before_scheduling: bool = False,
    repeat_stop: bool = False,
    completed_future: bool = False,
    cancel_error: Exception | None = None,
) -> tuple[optuna.Study, OpponentPool, _Executor, Path]:
    study = optuna.create_study(direction="maximize")
    cfg = SearchConfig(
        optimizer=OptimizerConfig(n_trials=workers, n_workers=workers),
        target=TargetConfig(binary=Path("game-nim")),
    )
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    executor = _Executor()
    executor._processes = {index: _Process() for index in range(workers)}
    stop_request = coordinator._StopRequest()
    if stop_before_scheduling:
        stop_request.request()

    def stop_after_wait(futures, **_kwargs):
        if completed_future:
            for future in futures:
                task = next(
                    call[3] for call in executor.calls if future in executor.futures
                )
                future.value = _result(task)
        stop_request.request()
        if repeat_stop:
            stop_request.request()
        for future in futures:
            future.cancel_error = cancel_error
        return set(futures), set()

    monkeypatch.setattr(coordinator, "preflight_check", lambda *_args: None)
    monkeypatch.setattr(coordinator, "ProcessPoolExecutor", lambda **_kwargs: executor)
    monkeypatch.setattr(coordinator, "wait", stop_after_wait)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(
        event_path, SessionId("session"), AttemptId("attempt")
    ) as writer:
        assert not coordinator._run_attempt(
            cfg,
            binary=Path("game-nim"),
            pool=pool,
            pool_path=tmp_path / "pool.json",
            study=study,
            lifecycle=writer,
            resolved_sha="sha",
            trace_path=None,
            should_stop=stop_request.requested,
        )
    return study, pool, executor, event_path


def test_stop_before_scheduling_emits_only_attempt_stop_and_releases_lock(
    monkeypatch, tmp_path: Path
):
    study, _pool, executor, event_path = _stopped_attempt(
        monkeypatch, tmp_path, workers=1, stop_before_scheduling=True
    )

    records = _event_records(event_path)
    assert [record["event_type"] for record in records] == ["attempt_stopped"]
    assert records[0]["payload"] == {"reason": "coordinator interrupted"}
    assert not study.trials
    assert not executor.calls
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("next")):
        pass


@pytest.mark.parametrize("workers", [1, 2])
def test_signal_stop_cancels_each_active_pair_without_completion_side_effects(
    monkeypatch, tmp_path: Path, workers: int
):
    study, pool, executor, event_path = _stopped_attempt(
        monkeypatch, tmp_path, workers=workers
    )

    records = _event_records(event_path)
    event_types = [record["event_type"] for record in records]
    assert event_types == (
        ["trial_created", "trial_started", "pair_started"] * workers
        + ["pair_failed"] * workers
        + ["trial_cancelled"] * workers
        + ["attempt_stopped"]
    )
    assert event_types.count("pair_failed") == workers
    assert event_types.count("trial_cancelled") == workers
    assert not {"game_finished", "pair_finished", "trial_reported"} & set(event_types)
    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.FAIL
    ] * workers
    assert len(pool.anchors) == 1
    assert all(future.result_calls == 0 for future in executor.futures)
    assert all(process.terminated == 1 for process in executor._processes.values())
    assert executor.shutdown_calls == [(False, True)]


def test_stop_wins_a_completed_future_race_without_updating_its_rating(
    monkeypatch, tmp_path: Path
):
    study, pool, executor, event_path = _stopped_attempt(
        monkeypatch, tmp_path, workers=1, completed_future=True
    )

    records = _event_records(event_path)
    assert [record["event_type"] for record in records][-3:] == [
        "pair_failed",
        "trial_cancelled",
        "attempt_stopped",
    ]
    assert executor.futures[0].result_calls == 0
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert len(pool.anchors) == 1


def test_repeated_signal_and_worker_cancellation_failure_still_stop_once(
    monkeypatch, tmp_path: Path
):
    study, _pool, executor, event_path = _stopped_attempt(
        monkeypatch,
        tmp_path,
        workers=1,
        repeat_stop=True,
        cancel_error=RuntimeError("worker already exited"),
    )

    event_types = [record["event_type"] for record in _event_records(event_path)]
    assert event_types.count("pair_failed") == 1
    assert event_types.count("trial_cancelled") == 1
    assert event_types.count("attempt_stopped") == 1
    assert executor.futures[0].cancel_calls == 1
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


def test_sigint_and_sigterm_handlers_only_request_the_same_stop(monkeypatch):
    handlers: dict[int, object] = {}

    def install(signum, handler):
        previous = handlers.get(signum, object())
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(coordinator.signal, "signal", install)
    with coordinator._install_stop_handlers() as stop_request:
        handlers[coordinator.signal.SIGINT](coordinator.signal.SIGINT, None)
        handlers[coordinator.signal.SIGTERM](coordinator.signal.SIGTERM, None)
        assert stop_request.requested()


def test_pruning_uses_one_automatic_worker_and_snapshots_the_adapter_trial():
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config()
    adapter = _ScriptedPruningAdapter([])

    active = attempt.create_active_trial(study, cfg, SessionId("session"), adapter)

    assert attempt.worker_count(cfg) == 1
    assert adapter.created == 1
    assert active.hyperband_trial is not None


def test_disabled_pruning_reports_then_completes_and_only_completion_updates_pool(
    monkeypatch, tmp_path: Path
):
    study = optuna.create_study(direction="maximize")
    cfg = SearchConfig(
        optimizer=OptimizerConfig(
            n_workers=1,
            resource=ResourcePolicy(min_pairs=1, max_pairs=2),
            rating=RatingPolicy(sigma_stop=None),
        ),
        target=TargetConfig(binary=Path("game-nim")),
    )
    active = _active(study)
    active.evaluation = TrialEvaluationState(
        cfg.optimizer.resource, cfg.optimizer.rating
    )
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    inserted: list[object] = []
    monkeypatch.setattr(
        attempt, "save_inserted_pool_anchor", lambda *args: inserted.append(args)
    )
    tells = _tell_calls(monkeypatch, study)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(
        event_path, SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = _context(cfg, study, writer, pool, tmp_path / "pool.json")
        scheduled = ScheduledPair(active, _task(active))
        assert attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )
        scheduled = futures.popitem()[1]
        assert not attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )

    records = _event_records(event_path)
    reports = [r["payload"] for r in records if r["event_type"] == "trial_reported"]
    assert [report["completed_pairs"] for report in reports] == [1, 2]
    assert [report["reason"] for report in reports] == ["pruning_disabled", "max_pairs"]
    assert [r["event_type"] for r in records].count("trial_completed") == 1
    assert not futures
    assert len(executor.calls) == len(reports) - 1 == 1
    assert len(tells) == len(inserted) == 1


def test_below_minimum_precedes_an_adversarial_prune(monkeypatch, tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config(min_pairs=2)
    adapter = _ScriptedPruningAdapter([HyperbandDecision(True, False, "4", 2)])
    active = attempt.create_active_trial(study, cfg, SessionId("session"), adapter)
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    tells = _tell_calls(monkeypatch, study)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(
        event_path, SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = attempt._AttemptContext(
            cfg,
            Path("game-nim"),
            pool,
            tmp_path / "pool.json",
            study,
            writer,
            "sha",
            None,
            adapter,
        )
        scheduled = ScheduledPair(active, _task(active))
        assert attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )
        scheduled = futures.popitem()[1]
        assert not attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )

    records = _event_records(event_path)
    reports = [r["payload"] for r in records if r["event_type"] == "trial_reported"]
    pruned = [r["payload"] for r in records if r["event_type"] == "trial_pruned"]
    assert [report["completed_pairs"] for report in reports] == [1, 2]
    assert [report["reason"] for report in reports] == [
        "below_min_pairs",
        "hyperband_prune",
    ]
    assert adapter.observed == 1
    assert len(tells) == len(pruned) == 1
    assert pruned[0]["bracket_id"] == "4"
    assert pruned[0]["rung_resource"] == 2
    assert not futures
    assert len(executor.calls) == 1


def test_startup_exempt_trial_is_not_delegated_before_max_completion(
    monkeypatch, tmp_path: Path
):
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config(max_pairs=2)
    adapter = _ScriptedPruningAdapter([HyperbandDecision(True, True, None, None)])
    active = attempt.create_active_trial(study, cfg, SessionId("session"), adapter)
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    tells = _tell_calls(monkeypatch, study)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(
        event_path, SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = attempt._AttemptContext(
            cfg,
            Path("game-nim"),
            pool,
            tmp_path / "pool.json",
            study,
            writer,
            "sha",
            None,
            adapter,
        )
        scheduled = ScheduledPair(active, _task(active))
        assert attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )
        scheduled = futures.popitem()[1]
        assert not attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )

    reports = [
        r["payload"]
        for r in _event_records(event_path)
        if r["event_type"] == "trial_reported"
    ]
    assert [report["reason"] for report in reports] == ["startup_exempt", "max_pairs"]
    assert reports[0]["pruning_exempt"] is True
    assert adapter.observed == 1
    assert len(tells) == 1
    assert not futures
    assert len(executor.calls) == 1


def test_delegated_keep_then_prune_has_one_terminal_and_no_pool_or_legacy_output(
    monkeypatch, tmp_path: Path
):
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config()
    adapter = _ScriptedPruningAdapter(
        [
            HyperbandDecision(False, False, "0", 1),
            HyperbandDecision(True, False, "0", 2),
        ]
    )
    active = attempt.create_active_trial(study, cfg, SessionId("session"), adapter)
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    tells = _tell_calls(monkeypatch, study)
    legacy_or_pool: list[str] = []
    monkeypatch.setattr(
        attempt, "emit_legacy_trial", lambda *args: legacy_or_pool.append("trial")
    )
    monkeypatch.setattr(
        attempt,
        "emit_legacy_incumbent",
        lambda *args: legacy_or_pool.append("incumbent"),
    )
    monkeypatch.setattr(
        attempt,
        "save_inserted_pool_anchor",
        lambda *args: legacy_or_pool.append("pool"),
    )
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(
        event_path, SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = attempt._AttemptContext(
            cfg,
            Path("game-nim"),
            pool,
            tmp_path / "pool.json",
            study,
            writer,
            "sha",
            None,
            adapter,
        )
        scheduled = ScheduledPair(active, _task(active))
        assert attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )
        scheduled = futures.popitem()[1]
        assert not attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )

    records = _event_records(event_path)
    reports = [r["payload"] for r in records if r["event_type"] == "trial_reported"]
    terminals = [r for r in records if r["event_type"] == "trial_pruned"]
    assert [report["reason"] for report in reports] == [
        "hyperband_keep",
        "hyperband_prune",
    ]
    assert [report["completed_pairs"] for report in reports] == [1, 2]
    assert len(tells) == len(terminals) == 1
    assert terminals[0]["payload"]["score"] == reports[-1]["score"]
    assert not legacy_or_pool
    assert not futures
    assert len(executor.calls) == 1


@pytest.mark.parametrize(
    ("sigma_stop", "max_pairs", "expected_reason"),
    [(100.0, 3, "confidence"), (None, 1, "max_pairs")],
)
def test_completion_precedes_a_pending_prune(
    monkeypatch, tmp_path: Path, sigma_stop, max_pairs, expected_reason
):
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config(sigma_stop=sigma_stop, max_pairs=max_pairs)
    adapter = _ScriptedPruningAdapter([HyperbandDecision(True, False, "0", 1)])
    active = attempt.create_active_trial(study, cfg, SessionId("session"), adapter)
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    tells = _tell_calls(monkeypatch, study)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(
        event_path, SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = attempt._AttemptContext(
            cfg,
            Path("game-nim"),
            pool,
            tmp_path / "pool.json",
            study,
            writer,
            "sha",
            None,
            adapter,
        )
        scheduled = ScheduledPair(active, _task(active))
        assert not attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )

    records = _event_records(event_path)
    reports = [r["payload"] for r in records if r["event_type"] == "trial_reported"]
    assert reports[0]["reason"] == expected_reason
    assert adapter.observed == 0
    assert len(tells) == 1
    assert [r["event_type"] for r in records][-2:] == [
        "trial_reported",
        "trial_completed",
    ]
    assert not futures
    assert not executor.calls


def test_pruned_terminal_replenishes_the_sequential_scheduler(
    monkeypatch, tmp_path: Path
):
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config()
    adapter = _ScriptedPruningAdapter([HyperbandDecision(True, False, "0", 1)])
    executor, futures, active = _Executor(), {}, {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    tells = _tell_calls(monkeypatch, study)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(
        event_path, SessionId("session"), AttemptId("attempt")
    ) as writer:
        context = attempt._AttemptContext(
            cfg,
            Path("game-nim"),
            pool,
            tmp_path / "pool.json",
            study,
            writer,
            "sha",
            None,
            adapter,
        )
        attempt.schedule_trial(
            executor,
            futures,
            active,
            cfg,
            Path("game-nim"),
            pool,
            study,
            writer,
            None,
            adapter,
        )
        future, scheduled = futures.popitem()
        assert not attempt.continue_trial(
            executor, futures, scheduled, _result(scheduled.task), context
        )
        active.pop(scheduled.active_trial.trial_id)
        remaining = attempt.replenish_trial(2, executor, futures, active, context)

    assert remaining == 1
    assert len(tells) == 1
    assert adapter.created == 2
    assert len(active) == len(futures) == 1
    assert (
        next(iter(futures.values())).active_trial.trial_id
        != scheduled.active_trial.trial_id
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
        futures = {future: scheduled}
        active_trials = {active.trial_id: active}
        attempt.cancel_active_trials(futures, active_trials, study, writer)
        attempt.cancel_active_trials(futures, active_trials, study, writer)
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


def test_complete_pairs_report_consecutive_resources_before_max_terminal(
    tmp_path: Path,
):
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
