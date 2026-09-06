from __future__ import annotations

from threading import Barrier, Lock
from typing import cast

from tuner_cli.domain import Candidate, PairTask, SearchEffort, TaskCase
from tuner_cli.executor import (
    BoundedPairPool,
    PairFailed,
    PairJob,
    PairOutcome,
    PairPool,
    PairSucceeded,
    SequentialPairPool,
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


def _drain(pool: PairPool, target: Target, jobs: list[PairJob]) -> list[PairOutcome]:
    """Mirror the run loop: keep the pool full, collect completions."""
    pending = list(jobs)
    outcomes: list[PairOutcome] = []
    try:
        while pending or pool.running():
            while pool.running() < pool.capacity and pending:
                pool.start(target, pending.pop(0))
            if pool.running():
                outcomes.append(pool.next_outcome(target))
    finally:
        pool.close()
    return outcomes


def test_bounded_pool_saturates_but_never_exceeds_capacity() -> None:
    jobs = [_job(index) for index in range(4)]
    target = _Target()
    outcomes = _drain(BoundedPairPool(2), cast(Target, target), jobs)
    assert target.maximum == 2
    assert {outcome.job.task.pair_id for outcome in outcomes} == {job.task.pair_id for job in jobs}
    assert all(isinstance(outcome, PairSucceeded) for outcome in outcomes)


def test_pools_return_typed_failure_at_the_original_job() -> None:
    jobs = [_job(index) for index in range(2)]
    target = _Target(fail="pair-1")
    outcomes = _drain(BoundedPairPool(2), cast(Target, target), jobs)
    failed = [outcome for outcome in outcomes if isinstance(outcome, PairFailed)]
    assert len(failed) == 1
    assert failed[0].job.task.pair_id == "pair-1"


def test_sequential_pool_runs_each_job_as_it_is_started() -> None:
    job = _job(0)
    target = _Target(synchronize=False)
    pool = SequentialPairPool()
    pool.start(cast(Target, target), job)
    assert pool.running() == 1
    outcome = pool.next_outcome(cast(Target, target))
    assert isinstance(outcome, PairSucceeded)
    pool.close()
