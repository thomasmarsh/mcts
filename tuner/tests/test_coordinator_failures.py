"""One-pair scheduling and failure ordering tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from tuner_cli import attempt, coordinator, pair_orchestration
from tuner_cli.artifact_layout import TASK_SEQUENCE_MAX
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
    TrialEvaluationState,
    configured_game_seed,
    game_id_for,
)
from tuner_cli.hyperband import HyperbandDecision
from tuner_cli.lifecycle import AttemptId, LifecycleWriter, SessionId, TrialId
from tuner_cli.pair_orchestration import ScheduledPair
from tuner_cli.pool import Anchor, OpponentPool
from tuner_cli.task_artifacts import (
    ArtifactIntegrityError,
    TaskDescriptorAllocator,
    sha256_digest,
)
from tuner_cli.task_execution import TaskArtifactReference, TaskResultError


class _Future:
    def __init__(
        self,
        error: Exception | None = None,
        result: PairResult | None = None,
        cancel_error: Exception | None = None,
        cancelled: bool = False,
    ):
        self.error = error
        self.value = result
        self.cancel_error = cancel_error
        self.cancel_calls = 0
        self.result_calls = 0
        self._cancelled = cancelled

    def cancelled(self):
        return self._cancelled

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

    def create_trial(self, study, pruning_exempt=False):
        self.created += 1
        return SimpleNamespace(trial=study.ask(), pruning_exempt=pruning_exempt)

    def observe_after_report(self, _trial):
        self.observed += 1
        return next(self.decisions)


def test_open_study_configures_tpe_startup_trials_without_a_pruner(monkeypatch, tmp_path: Path):
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
    descriptors = TaskDescriptorAllocator.start(
        pool_path.parent / "bench-runs" / "attempt" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id="attempt",
        manifest_fingerprint="manifest",
    )
    return attempt._AttemptContext(
        cfg,
        Path("game-nim"),
        pool,
        pool_path,
        study,
        writer,
        "sha",
        task_descriptors=descriptors,
    )


def _test_descriptors(tmp_path: Path) -> TaskDescriptorAllocator:
    return TaskDescriptorAllocator.start(
        tmp_path / "bench-runs" / "attempt" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id="attempt",
        manifest_fingerprint="manifest",
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
                task = futures[future].task
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
    descriptors = TaskDescriptorAllocator.start(
        tmp_path / "bench-runs" / "attempt" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id="attempt",
        manifest_fingerprint="manifest",
    )
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
        assert not coordinator._run_attempt(
            cfg,
            binary=Path("game-nim"),
            pool=pool,
            pool_path=tmp_path / "pool.json",
            study=study,
            lifecycle=writer,
            resolved_sha="sha",
            should_stop=stop_request.requested,
            task_descriptors=descriptors,
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
    with pytest.raises(ValueError, match="precedes attempt_started"):
        LifecycleWriter(event_path, SessionId("session"), AttemptId("next"))


@pytest.mark.parametrize("workers", [1, 2])
def test_signal_stop_cancels_each_active_pair_without_completion_side_effects(
    monkeypatch, tmp_path: Path, workers: int
):
    study, pool, executor, event_path = _stopped_attempt(monkeypatch, tmp_path, workers=workers)

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
    assert [trial.state for trial in study.trials] == [optuna.trial.TrialState.FAIL] * workers
    assert len(pool.anchors) == 1
    assert all(future.result_calls == 0 for future in executor.futures)
    assert all(process.terminated == 1 for process in executor._processes.values())
    assert executor.shutdown_calls == [(False, True)]


def test_stop_wins_a_completed_future_race_without_updating_its_rating(monkeypatch, tmp_path: Path):
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


def test_pruning_uses_automatic_evaluation_slots_and_snapshots_the_adapter_trial(
    monkeypatch,
):
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config()
    adapter = _ScriptedPruningAdapter([])
    monkeypatch.setattr(attempt.os, "cpu_count", lambda: 10)

    active = attempt.create_active_trial(study, cfg, SessionId("session"), adapter)

    assert attempt.worker_count(cfg) == 5
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
    active.evaluation = TrialEvaluationState(cfg.optimizer.resource, cfg.optimizer.rating)
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    inserted: list[object] = []
    monkeypatch.setattr(attempt, "save_inserted_pool_anchor", lambda *args: inserted.append(args))
    tells = _tell_calls(monkeypatch, study)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
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
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
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
            task_descriptors=_test_descriptors(tmp_path),
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


def test_startup_exempt_trial_is_not_delegated_before_max_completion(monkeypatch, tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config(max_pairs=2)
    adapter = _ScriptedPruningAdapter([HyperbandDecision(True, True, None, None)])
    active = attempt.create_active_trial(study, cfg, SessionId("session"), adapter)
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    tells = _tell_calls(monkeypatch, study)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
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
            task_descriptors=_test_descriptors(tmp_path),
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
        r["payload"] for r in _event_records(event_path) if r["event_type"] == "trial_reported"
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
    monkeypatch.setattr(attempt, "emit_legacy_trial", lambda *args: legacy_or_pool.append("trial"))
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
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
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
            task_descriptors=_test_descriptors(tmp_path),
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
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
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
            task_descriptors=_test_descriptors(tmp_path),
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
    assert [r["event_type"] for r in records][-3:] == [
        "trial_reported",
        "trial_completed",
        "pool_anchor_decided",
    ]
    assert not futures
    assert not executor.calls


def test_pruned_terminal_replenishes_the_sequential_scheduler(monkeypatch, tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    cfg = _pruning_config()
    adapter = _ScriptedPruningAdapter([HyperbandDecision(True, False, "0", 1)])
    executor, futures, active = _Executor(), {}, {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    tells = _tell_calls(monkeypatch, study)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
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
            task_descriptors=_test_descriptors(tmp_path),
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
            adapter,
            task_descriptors=context.task_descriptors,
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
    assert next(iter(futures.values())).active_trial.trial_id != scheduled.active_trial.trial_id


def test_submission_emits_pair_started_and_one_future_for_one_pair(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    executor, futures = _Executor(), {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("game-nim")))
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
            attempt._terminalize_from_pair(writer),
            TaskDescriptorAllocator.start(
                tmp_path / "bench-runs" / "submission" / "tuning-artifacts",
                session_id="session",
                optimizer_id="optimizer",
                attempt_id="attempt",
                bench_run_id="submission",
                manifest_fingerprint="manifest",
            ),
        )
    assert len(futures) == 1
    assert executor.calls[0][0] is pair_orchestration.execute_task_bundle
    task = next(iter(futures.values())).task
    assert task.pair_index == 0
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert records[-1]["event_type"] == "pair_started"
    assert records[-1]["payload"]["seed"] == configured_game_seed(task.seed)
    assert records[-1]["payload"]["round"] == 1


def test_descriptor_commit_precedes_pair_event_and_worker_submission(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("game-nim")))
    executor, futures = _Executor(), {}
    physical_root = tmp_path / "bench-runs" / "physical-attempt" / "tuning-artifacts"
    descriptors = TaskDescriptorAllocator.start(
        physical_root,
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id="bench-run",
        manifest_fingerprint="manifest",
    )
    order: list[str] = []
    original_commit = descriptors.commit_task

    def commit_task(*args, **kwargs):
        committed = original_commit(*args, **kwargs)
        order.append("descriptor")
        return committed

    descriptors.commit_task = commit_task  # type: ignore[method-assign]
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
        original_emit = writer.emit

        def emit(event_type, payload):
            if event_type == "pair_started":
                assert list((descriptors.layout.root / "descriptors").glob("*.json"))
                order.append("event")
            return original_emit(event_type, payload)

        writer.emit = emit  # type: ignore[method-assign]

        class OrderingExecutor(_Executor):
            def submit(self, *args):
                records = _event_records(event_path)
                assert records[-1]["event_type"] == "pair_started"
                order.append("submit")
                return super().submit(*args)

        executor = OrderingExecutor()
        attempt.submit_next_pair(
            executor,
            futures,
            active,
            cfg,
            Path("/resolved/game-nim"),
            pool,
            study,
            writer,
            attempt._terminalize_from_pair(writer),
            descriptors,
        )

    assert order == ["descriptor", "event", "submit"]
    assert executor.calls[0][0] is pair_orchestration.execute_task_bundle
    descriptor_path = executor.calls[0][1]
    assert executor.calls[0][2]
    task = next(iter(futures.values())).task
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor = json.loads(descriptor_bytes)
    records = _event_records(event_path)
    pair_started = records[-1]["payload"]

    assert sha256_digest(descriptor_bytes) == pair_started["descriptor_digest"]
    assert descriptor["task_id"] == pair_started["task_id"]
    assert descriptor["task_sequence"] == pair_started["task_sequence"] == 1
    assert descriptor["session_id"] == "session"
    assert descriptor["optimizer_id"] == "optimizer"
    assert descriptor["attempt_id"] == "attempt"
    assert descriptor["bench_run_id"] == "bench-run"
    assert descriptor["trial_id"] == task.trial_id
    assert descriptor["pair_id"] == task.pair_id
    assert descriptor["candidate_config"] == {"family": "rave"}
    assert descriptor["opponent"] == {
        "anchor_id": "random",
        "config": {"family": "random"},
        "mu": 0.0,
        "sigma": 0.5,
    }
    assert descriptor["pool_snapshot"] == [descriptor["opponent"]]
    assert descriptor["rating_before"] == {
        "mu": task.rating_before.mu,
        "sigma": task.rating_before.sigma,
    }
    assert descriptor["binary"] == {"path": "/resolved/game-nim"}
    assert descriptor["game_ids"] == {
        "candidate_first": game_id_for(task.pair_id, "first"),
        "candidate_second": game_id_for(task.pair_id, "second"),
    }
    assert descriptor["task_directory"] == f"tasks/{descriptor['task_id']}"
    assert descriptor["trace_game_sequences"] == {
        "candidate_first": 1,
        "candidate_second": 2,
    }
    assert str(descriptors.layout.root) not in descriptor_path.read_text()
    assert "legacy-trace.jsonl" not in descriptor_path.read_text()

    attempt_descriptor = json.loads(descriptors.layout.attempt.read_text())
    assert attempt_descriptor["attempt_id"] == "attempt"
    assert attempt_descriptor["manifest_fingerprint"] == "manifest"


def test_descriptor_allocation_is_monotonic_and_freezes_each_submission(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("game-nim")))
    executor, futures = _Executor(), {}
    descriptors = TaskDescriptorAllocator.start(
        tmp_path / "bench-runs" / "physical-attempt" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id=None,
        manifest_fingerprint="manifest",
    )
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        attempt.submit_next_pair(
            executor,
            futures,
            active,
            cfg,
            Path("/resolved/game-nim"),
            pool,
            study,
            writer,
            attempt._terminalize_from_pair(writer),
            descriptors,
        )
        active.config["family"] = "changed"
        pool.anchors[0].config["family"] = "changed"
        active.evaluation.completed_pairs = 1
        attempt.submit_next_pair(
            executor,
            futures,
            active,
            cfg,
            Path("/resolved/game-nim"),
            pool,
            study,
            writer,
            attempt._terminalize_from_pair(writer),
            descriptors,
        )

    files = sorted((descriptors.layout.root / "descriptors").glob("*.json"))
    payloads = [json.loads(path.read_text()) for path in files]
    assert [payload["task_sequence"] for payload in payloads] == [1, 2]
    assert len({payload["task_id"] for payload in payloads}) == 2
    assert [payload["pair_index"] for payload in payloads] == [0, 1]
    assert payloads[0]["candidate_config"] == {"family": "rave"}
    assert payloads[0]["pool_snapshot"][0]["config"] == {"family": "random"}
    assert payloads[1]["trace_game_sequences"] == {
        "candidate_first": 3,
        "candidate_second": 4,
    }


def test_descriptor_commit_failure_prevents_worker_submission(tmp_path: Path, monkeypatch):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("game-nim")))
    executor, futures = _Executor(), {}
    descriptors = TaskDescriptorAllocator.start(
        tmp_path / "bench-runs" / "physical-attempt" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id=None,
        manifest_fingerprint="manifest",
    )
    monkeypatch.setattr(
        descriptors,
        "commit_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("full disk")),
    )
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
        with pytest.raises(OSError, match="full disk"):
            attempt.submit_next_pair(
                executor,
                futures,
                active,
                cfg,
                Path("/resolved/game-nim"),
                pool,
                study,
                writer,
                attempt._terminalize_from_pair(writer),
                descriptors,
            )

    assert not executor.calls
    assert not futures
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert [record["event_type"] for record in _event_records(event_path)] == ["trial_failed"]


def test_exhausted_descriptor_sequence_prevents_worker_submission(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("game-nim")))
    executor, futures = _Executor(), {}
    descriptors = TaskDescriptorAllocator.start(
        tmp_path / "bench-runs" / "physical-attempt" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id=None,
        manifest_fingerprint="manifest",
    )
    descriptors._next_task_sequence = TASK_SEQUENCE_MAX + 1
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        with pytest.raises(ValueError, match="task_sequence"):
            attempt.submit_next_pair(
                executor,
                futures,
                active,
                cfg,
                Path("/resolved/game-nim"),
                pool,
                study,
                writer,
                attempt._terminalize_from_pair(writer),
                descriptors,
            )

    assert not executor.calls
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


def test_stop_before_scheduling_does_not_allocate_a_descriptor(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(
        optimizer=OptimizerConfig(n_workers=1),
        target=TargetConfig(binary=Path("game-nim")),
    )
    executor, futures, active = _Executor(), {}, {}
    descriptors = TaskDescriptorAllocator.start(
        tmp_path / "bench-runs" / "physical-attempt" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id=None,
        manifest_fingerprint="manifest",
    )
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        attempt.schedule_initial_trials(
            1,
            1,
            executor,
            futures,
            active,
            cfg,
            Path("/resolved/game-nim"),
            pool,
            study,
            writer,
            should_stop=lambda: True,
            task_descriptors=descriptors,
        )

    assert not executor.calls
    assert not futures
    assert not active
    assert not (descriptors.layout.root / "descriptors").exists()


def test_attempt_root_cannot_be_reused_by_a_new_physical_attempt(tmp_path: Path):
    root = tmp_path / "bench-runs" / "physical-attempt" / "tuning-artifacts"
    TaskDescriptorAllocator.start(
        root,
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt-one",
        bench_run_id=None,
        manifest_fingerprint="manifest",
    )
    with pytest.raises(ArtifactIntegrityError):
        TaskDescriptorAllocator.start(
            root,
            session_id="session",
            optimizer_id="optimizer",
            attempt_id="attempt-two",
            bench_run_id=None,
            manifest_fingerprint="manifest",
        )


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
            task_descriptors=TaskDescriptorAllocator.start(
                tmp_path / "bench-runs" / "initial" / "tuning-artifacts",
                session_id="session",
                optimizer_id="optimizer",
                attempt_id="attempt",
                bench_run_id="initial",
                manifest_fingerprint="manifest",
            ),
        )
    assert len(active) == len(futures) == len(executor.calls) == 2
    assert all(call[0] is pair_orchestration.execute_task_bundle for call in executor.calls)
    assert {call[1].parent.name for call in executor.calls} == {"descriptors"}


def test_parallel_pruning_allocates_distinct_startup_trials_and_task_artifacts(
    tmp_path: Path,
):
    study = optuna.create_study(direction="maximize")
    executor, futures, active = _Executor(), {}, {}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    cfg = SearchConfig(
        optimizer=OptimizerConfig(
            n_trials=3,
            n_workers=3,
            pruning=PruningPolicy(enabled=True, startup_trials=2),
        ),
        target=TargetConfig(binary=Path("game-nim")),
    )
    adapter = _ScriptedPruningAdapter([])
    descriptors = TaskDescriptorAllocator.start(
        tmp_path / "bench-runs" / "parallel" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id="parallel",
        manifest_fingerprint="manifest",
    )
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        remaining = attempt.schedule_initial_trials(
            3,
            3,
            executor,
            futures,
            active,
            cfg,
            Path("game-nim"),
            pool,
            study,
            writer,
            adapter,
            task_descriptors=descriptors,
            startup_allocator=attempt.StartupTrialAllocator.restore(study, 2),
        )

    assert remaining == 0
    assert len(active) == len(futures) == len(executor.calls) == 3
    assert [trial.hyperband_trial.pruning_exempt for trial in active.values()] == [
        True,
        True,
        False,
    ]
    assert [trial.user_attrs["pruning_startup_exempt"] for trial in study.trials] == [
        True,
        True,
        False,
    ]
    scheduled = list(futures.values())
    assert [item.descriptor.identity.task_sequence for item in scheduled] == [1, 2, 3]
    assert len({item.descriptor.identity.task_id for item in scheduled}) == 3
    assert len({item.descriptor_path for item in scheduled}) == 3
    created = [
        record["payload"]
        for record in _event_records(tmp_path / "events.jsonl")
        if record["event_type"] == "trial_created"
    ]
    assert [record["pruning_exempt"] for record in created] == [True, True, False]


def test_startup_allocator_restores_persisted_flags_before_new_trial():
    study = optuna.create_study(direction="maximize")
    first = study.ask()
    first.set_user_attr("pruning_startup_exempt", True)
    cfg = _pruning_config()
    adapter = _ScriptedPruningAdapter([])

    active = attempt.create_active_trial(
        study,
        cfg,
        SessionId("session"),
        adapter,
        attempt.StartupTrialAllocator.restore(study, 1),
    )

    assert not active.hyperband_trial.pruning_exempt
    assert study.trials[1].user_attrs["pruning_startup_exempt"] is False


def test_ready_batch_reports_in_descriptor_sequence_order(monkeypatch, tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    cfg = SearchConfig(
        optimizer=OptimizerConfig(
            resource=ResourcePolicy(min_pairs=1, max_pairs=1),
            rating=RatingPolicy(sigma_stop=None),
        ),
        target=TargetConfig(binary=Path("game-nim")),
    )
    first, second = _active(study), _active(study)
    first.evaluation = TrialEvaluationState(cfg.optimizer.resource, cfg.optimizer.rating)
    second.evaluation = TrialEvaluationState(cfg.optimizer.resource, cfg.optimizer.rating)
    first_descriptor = SimpleNamespace(
        identity=SimpleNamespace(task_id="task-1", task_sequence=1), digest="one"
    )
    second_descriptor = SimpleNamespace(
        identity=SimpleNamespace(task_id="task-2", task_sequence=2), digest="two"
    )
    first_future, second_future = _Future(), _Future()
    futures = {
        second_future: ScheduledPair(second, _task(second), second_descriptor),
        first_future: ScheduledPair(first, _task(first), first_descriptor),
    }
    active = {first.trial_id: first, second.trial_id: second}
    pool = OpponentPool([Anchor("random", {"family": "random"}, 0.0, 0.5)])
    monkeypatch.setattr(
        attempt,
        "worker_result",
        lambda _future, _study, _writer, scheduled, _terminal: _result(scheduled.task),
    )
    with LifecycleWriter(
        tmp_path / "events.jsonl", SessionId("session"), AttemptId("attempt")
    ) as writer:
        attempt.drain_scheduled_trials(
            0,
            _Executor(),
            futures,
            active,
            cfg,
            Path("game-nim"),
            pool,
            tmp_path / "pool.json",
            study,
            writer,
            "sha",
            lambda current, **_kwargs: (set(current), set()),
        )

    finished = [
        record["payload"]["task_id"]
        for record in _event_records(tmp_path / "events.jsonl")
        if record["event_type"] == "pair_finished"
    ]
    assert finished == ["task-1", "task-2"]


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
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [record["event_type"] for record in records] == [
        "pair_failed",
        "trial_failed",
    ]
    assert all(record["event_type"] != "trial_reported" for record in records)


def _artifact_scheduled(
    tmp_path: Path, active: attempt._ActiveTrial
) -> tuple[ScheduledPair, TaskDescriptorAllocator]:
    descriptors = TaskDescriptorAllocator.start(
        tmp_path / "bench-runs" / "physical-attempt" / "tuning-artifacts",
        session_id="session",
        optimizer_id="optimizer",
        attempt_id="attempt",
        bench_run_id=None,
        manifest_fingerprint="manifest",
    )
    cfg = SearchConfig(optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("game-nim")))
    task = _task(active)
    descriptor = descriptors.commit_task(
        task, cfg=cfg, binary=Path("/resolved/game-nim"), pool_snapshot=[]
    )
    return (
        ScheduledPair(
            active,
            task,
            descriptor,
            descriptors.layout.descriptor(descriptor.identity),
        ),
        descriptors,
    )


def test_artifact_result_is_read_once_before_lifecycle_and_rating_updates(
    monkeypatch, tmp_path: Path
):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    scheduled, _descriptors = _artifact_scheduled(tmp_path, active)
    reference = TaskArtifactReference(
        scheduled.descriptor.identity.task_id,
        "attempt",
        scheduled.descriptor.digest,
        "completed",
        "a" * 64,
    )
    reads: list[tuple] = []

    def read(*args):
        reads.append(args)
        return _result(scheduled.task)

    monkeypatch.setattr(pair_orchestration, "read_task_bundle", read)
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
        result = attempt.worker_result(
            _Future(result=reference),
            study,
            writer,
            scheduled,
            attempt._terminalize_from_pair(writer),
        )
        assert result == _result(scheduled.task)
        attempt.finish_pair(writer, active, result, scheduled.descriptor)

    assert reads == [
        (
            scheduled.descriptor_path,
            scheduled.descriptor.digest,
            reference,
            scheduled.task,
        )
    ]
    records = _event_records(event_path)
    for record in records:
        assert record["payload"]["task_id"] == scheduled.descriptor.identity.task_id
        assert record["payload"]["descriptor_digest"] == scheduled.descriptor.digest


@pytest.mark.parametrize(
    ("future", "reader_error", "event_type"),
    [
        (
            _Future(result=object()),
            TaskResultError("missing completion marker"),
            "trial_failed",
        ),
        (
            _Future(result=object()),
            TaskResultError("committed task failed"),
            "trial_failed",
        ),
        (_Future(cancelled=True), None, "trial_cancelled"),
    ],
)
def test_artifact_failure_paths_never_apply_a_partial_pair(
    monkeypatch, tmp_path: Path, future, reader_error, event_type
):
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    scheduled, _descriptors = _artifact_scheduled(tmp_path, active)
    if reader_error is not None:
        monkeypatch.setattr(
            pair_orchestration,
            "read_task_bundle",
            lambda *_args: (_ for _ in ()).throw(reader_error),
        )
    event_path = tmp_path / "events.jsonl"
    with LifecycleWriter(event_path, SessionId("session"), AttemptId("attempt")) as writer:
        assert (
            attempt.worker_result(
                future,
                study,
                writer,
                scheduled,
                attempt._terminalize_from_pair(writer),
            )
            is None
        )

    records = _event_records(event_path)
    assert [record["event_type"] for record in records] == ["pair_failed", event_type]
    assert records[0]["payload"]["task_id"] == scheduled.descriptor.identity.task_id
    assert records[0]["payload"]["descriptor_digest"] == scheduled.descriptor.digest
    assert active.evaluation.completed_pairs == 0


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
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
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
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
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
    active.evaluation = TrialEvaluationState(cfg.optimizer.resource, cfg.optimizer.rating)
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

    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    reports = [record["payload"] for record in records if record["event_type"] == "trial_reported"]
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
    assert [record["event_type"] for record in records[-3:]] == [
        "trial_reported",
        "trial_completed",
        "pool_anchor_decided",
    ]
    assert records[-2]["payload"]["reason"] == "max_pairs"
    assert records[-1]["payload"]["reason"] == "champion"
    assert {
        key: records[-2]["payload"][key]
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
    assert reports[-1]["score"] == pytest.approx(reports[-1]["mu"] - 2.5 * reports[-1]["sigma"])


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
    active.evaluation = TrialEvaluationState(cfg.optimizer.resource, cfg.optimizer.rating)
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

    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [record["event_type"] for record in records[-3:]] == [
        "trial_reported",
        "trial_completed",
        "pool_anchor_decided",
    ]
    assert records[-2]["payload"]["reason"] == "confidence"
    assert records[-1]["payload"]["reason"] == "champion"


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
            active.evaluation = TrialEvaluationState(cfg.optimizer.resource, cfg.optimizer.rating)
            scheduled = ScheduledPair(active, _task(active))
            assert not attempt.continue_trial(
                executor, futures, scheduled, _result(scheduled.task), context
            )

    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [record["event_type"] for record in records].count("trial_reported") == 2
    terminals = [record for record in records if record["event_type"] == "trial_completed"]
    assert len(terminals) == 2
    assert {record["payload"]["trial_id"] for record in terminals} == {
        "trial-0",
        "trial-1",
    }
    assert all(record["payload"]["reason"] == "max_pairs" for record in terminals)
    assert len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) == 2


def test_success_preserves_legacy_trial_pool_and_incumbent_order(monkeypatch, tmp_path: Path):
    events: list[str] = []
    study = optuna.create_study(direction="maximize")
    active = _active(study)
    active.evaluation.rating = Rating(25.0, 2.0)

    monkeypatch.setattr(
        attempt,
        "record_completed_trial",
        lambda *args: events.append("lifecycle trial_completed"),
    )
    monkeypatch.setattr(attempt, "emit_legacy_trial", lambda *args: events.append("legacy trial"))
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
