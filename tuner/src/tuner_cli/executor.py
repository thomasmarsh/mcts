"""Bounded, ordered execution of allocator-provided pair jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


class PairExecutor(Protocol):
    capacity: int

    def evaluate(self, target: Target, jobs: tuple[PairJob, ...]) -> tuple[PairOutcome, ...]: ...

    def cancel(self, target: Target) -> None: ...


def _evaluate(target: Target, job: PairJob) -> PairOutcome:
    try:
        result = target.evaluate(
            job.task, job.candidate, job.opponent, job.game_config, job.timeout_seconds
        )
        return PairSucceeded(
            job,
            result,
        )
    except PairExecutionError as error:
        return PairFailed(job, error)
    except KeyboardInterrupt:
        target.cancel()
        return PairInterrupted(job)


class SequentialPairExecutor:
    capacity = 1

    def evaluate(self, target: Target, jobs: tuple[PairJob, ...]) -> tuple[PairOutcome, ...]:
        return tuple(_evaluate(target, job) for job in jobs)

    def cancel(self, target: Target) -> None:
        target.cancel()


class BoundedPairExecutor:
    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or capacity <= 1:
            raise ValueError("bounded executor capacity must exceed one")
        self.capacity = capacity

    def evaluate(self, target: Target, jobs: tuple[PairJob, ...]) -> tuple[PairOutcome, ...]:
        with ThreadPoolExecutor(max_workers=self.capacity) as pool:
            futures = tuple(pool.submit(_evaluate, target, job) for job in jobs)
            return tuple(future.result() for future in futures)

    def cancel(self, target: Target) -> None:
        target.cancel()
