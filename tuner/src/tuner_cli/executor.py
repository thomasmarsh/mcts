"""Streaming, bounded execution of allocator-provided pair jobs.

The run loop keeps the worker pool saturated: it starts a pair as soon as a
worker is free and folds each completion as it arrives, instead of filling a
batch of ``capacity`` pairs and blocking on the slowest one. One long game no
longer stalls the workers that could be starting the next pairs.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Protocol

from .domain import Candidate, Opponent, PairTask
from .target import PairExecutionError, Target


@dataclass(frozen=True, slots=True)
class PairJob:
    task: PairTask
    candidate: Candidate
    opponent: Opponent
    game_config: str
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class PairSucceeded:
    job: PairJob
    result: object


@dataclass(frozen=True, slots=True)
class PairFailed:
    job: PairJob
    error: PairExecutionError


@dataclass(frozen=True, slots=True)
class PairInterrupted:
    job: PairJob


PairOutcome = PairSucceeded | PairFailed | PairInterrupted


def _evaluate(target: Target, job: PairJob) -> PairOutcome:
    try:
        result = target.evaluate(
            job.task, job.candidate, job.opponent, job.game_config, job.timeout_seconds
        )
        return PairSucceeded(job, result)
    except PairExecutionError as error:
        return PairFailed(job, error)
    except KeyboardInterrupt:
        target.cancel()
        return PairInterrupted(job)


class PairPool(Protocol):
    """A bounded set of in-flight pair evaluations for one run process.

    ``start`` begins a job; ``next_outcome`` blocks until one running job
    finishes and returns it -- in completion order, so the caller reorders
    into canonical evidence order itself. ``close`` releases the workers when
    the run loop exits.
    """

    capacity: int

    def running(self) -> int: ...

    def start(self, target: Target, job: PairJob) -> None: ...

    def next_outcome(self, target: Target) -> PairOutcome: ...

    def cancel(self, target: Target) -> None: ...

    def close(self) -> None: ...


class SequentialPairPool:
    """Runs each job to completion on the calling thread as it is started."""

    capacity = 1

    def __init__(self) -> None:
        self._ready: list[PairOutcome] = []

    def running(self) -> int:
        return len(self._ready)

    def start(self, target: Target, job: PairJob) -> None:
        self._ready.append(_evaluate(target, job))

    def next_outcome(self, target: Target) -> PairOutcome:
        del target
        return self._ready.pop(0)

    def cancel(self, target: Target) -> None:
        target.cancel()

    def close(self) -> None:
        return None


class BoundedPairPool:
    """Keeps up to ``capacity`` jobs running on a persistent thread pool."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or capacity <= 1:
            raise ValueError("bounded pool capacity must exceed one")
        self.capacity = capacity
        self._pool = ThreadPoolExecutor(max_workers=capacity)
        self._pending: set[Future[PairOutcome]] = set()

    def running(self) -> int:
        return len(self._pending)

    def start(self, target: Target, job: PairJob) -> None:
        self._pending.add(self._pool.submit(_evaluate, target, job))

    def next_outcome(self, target: Target) -> PairOutcome:
        del target
        done, self._pending = wait(self._pending, return_when=FIRST_COMPLETED)
        future = done.pop()
        self._pending |= done
        return future.result()

    def cancel(self, target: Target) -> None:
        target.cancel()
        for future in self._pending:
            future.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._pending.clear()

    def close(self) -> None:
        self._pool.shutdown(wait=True)
