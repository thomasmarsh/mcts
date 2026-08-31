from __future__ import annotations

from threading import Barrier, Lock
from typing import cast

from tuner_cli.domain import Candidate, PairTask, SearchEffort, TaskCase
from tuner_cli.executor import (
    BoundedPairExecutor,
    PairFailed,
    PairJob,
    PairSucceeded,
    SequentialPairExecutor,
)
from tuner_cli.target import PairExecutionError, Target


def _job(index: int) -> PairJob:
    candidate = Candidate(f"candidate-{index}", f"fingerprint-{index}", "{}")
    task = PairTask(
        f"pair-{index}",
        candidate.candidate_id,
        TaskCase(f"task-{index}", "tuning", index, index, "s", "opponent", "f", "p", "g"),
        SearchEffort("iterations", 1),
    )
    return PairJob(task, candidate, candidate, "{}", 1)


class _Target:
    def __init__(self, fail: str | None = None, *, synchronize: bool = True) -> None:
        self.fail = fail
        self.active = self.maximum = 0
        self.lock = Lock()
        self.gate = Barrier(2) if synchronize else None

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        del candidate, opponent, game_config, timeout_seconds
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        if self.gate is not None:
            self.gate.wait(timeout=2)
        with self.lock:
            self.active -= 1
        if task.pair_id == self.fail:
            raise PairExecutionError("injected", "failure", ["game"])
        return cast(object, task)

    def cancel(self) -> None:
        return None


def test_bounded_executor_limits_work_and_returns_allocator_order() -> None:
    jobs = tuple(_job(index) for index in range(4))
    target = _Target()
    outcomes = BoundedPairExecutor(2).evaluate(cast(Target, target), jobs)
    assert target.maximum == 2
    assert [outcome.job.task.pair_id for outcome in outcomes] == [job.task.pair_id for job in jobs]
    assert all(isinstance(outcome, PairSucceeded) for outcome in outcomes)


def test_executors_return_typed_failure_at_the_original_job() -> None:
    jobs = tuple(_job(index) for index in range(2))
    target = _Target(fail="pair-1")
    outcomes = BoundedPairExecutor(2).evaluate(cast(Target, target), jobs)
    assert isinstance(outcomes[0], PairSucceeded)
    assert isinstance(outcomes[1], PairFailed)
    assert outcomes[1].job == jobs[1]


def test_sequential_executor_preserves_one_job_path() -> None:
    job = _job(0)
    target = _Target(synchronize=False)
    outcome = SequentialPairExecutor().evaluate(cast(Target, target), (job,))
    assert isinstance(outcome[0], PairSucceeded)
