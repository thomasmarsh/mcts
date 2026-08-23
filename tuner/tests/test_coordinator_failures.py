from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest

from tuner_cli import coordinator
from tuner_cli.config import OptimizerConfig, SearchConfig, TargetConfig
from tuner_cli.lifecycle import AttemptId, LifecycleWriter, SessionId


class _Future:
    def __init__(self, *, cancelled: bool = False, error: Exception | None = None):
        self._cancelled, self._error = cancelled, error

    def cancelled(self):
        return self._cancelled

    def result(self):
        if self._error:
            raise self._error
        return (25.0, 2.0, [])

    def cancel(self):
        return True


class _Executor:
    future: _Future

    def __init__(self, *, future: _Future, **_kwargs):
        self.future = future

    def submit(self, *_args):
        return self.future

    def shutdown(self, **_kwargs):
        pass


class _Pool:
    def closest(self, _mu):
        return type("Anchor", (), {"config": {}})()

    def maybe_insert(self, *_args):
        return None


@pytest.mark.parametrize(
    ("future", "event_type"),
    [(_Future(error=RuntimeError("worker boom")), "trial_failed"), (_Future(cancelled=True), "trial_cancelled")],
)
def test_worker_failure_and_cancellation_leave_typed_terminal_evidence(
    monkeypatch, tmp_path: Path, future: _Future, event_type: str
):
    cfg = SearchConfig(
        optimizer=OptimizerConfig(n_trials=1, n_workers=1),
        target=TargetConfig(binary=Path("game-nim")),
    )
    cfg.parameters, cfg.conditions = [], []
    study = optuna.create_study(direction="maximize")
    with LifecycleWriter(tmp_path / "events.jsonl", SessionId("s"), AttemptId("a")) as writer:
        monkeypatch.setattr(coordinator, "ProcessPoolExecutor", lambda **kw: _Executor(future=future, **kw))
        monkeypatch.setattr(coordinator, "wait", lambda *_args, **_kwargs: ({future}, set()))
        monkeypatch.setattr(coordinator, "preflight_check", lambda *_args: None)
        with pytest.raises(RuntimeError):
            coordinator._run_attempt(
                cfg,
                binary=Path("game-nim"),
                pool=_Pool(),
                pool_path=tmp_path / "pool.json",
                study=study,
                lifecycle=writer,
                resolved_sha="sha",
                trace_path=None,
            )
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(record["event_type"] == event_type for record in records)
    assert records[-1]["event_type"] == "attempt_failed"


def test_success_preserves_legacy_trial_pool_and_incumbent_order(monkeypatch, tmp_path: Path):
    """Compatibility output remains ordered around the pool checkpoint."""
    from tuner_cli import attempt

    events: list[str] = []
    active_trial = type(
        "ActiveTrial",
        (),
        {
            "trial_id": "trial-1",
            "trial": type("Trial", (), {"number": 0})(),
            "config": {"c": 1},
            "seed": 7,
        },
    )()
    future = object()
    futures = {future: active_trial}
    active = {active_trial.trial_id: active_trial}

    monkeypatch.setattr(attempt, "worker_result", lambda *args: (25.0, 2.0, []))
    monkeypatch.setattr(
        attempt,
        "record_completed_trial",
        lambda *args: events.append("lifecycle trial_completed"),
    )
    monkeypatch.setattr(
        attempt,
        "emit_legacy_trial",
        lambda *args: events.append("legacy trial"),
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

    class _Executor:
        pass

    attempt.drain_scheduled_trials(
        1,
        _Executor(),
        futures,
        active,
        cfg=None,
        binary=Path("game-nim"),
        pool=object(),
        pool_path=tmp_path / "pool.json",
        study=object(),
        lifecycle=object(),
        resolved_sha="sha",
        trace_path=None,
        wait_for_completion=lambda *_args, **_kwargs: ({future}, set()),
    )

    assert events == [
        "lifecycle trial_completed",
        "legacy trial",
        "pool maybe_insert/save",
        "legacy incumbent",
    ]
