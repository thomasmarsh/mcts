"""Coordinator-owned scheduling and lifecycle evidence for one evaluation pair."""

from __future__ import annotations

from concurrent.futures import CancelledError, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import optuna

from .config import SearchConfig
from .evaluation import (
    OpponentSnapshot,
    PairResult,
    PairTask,
    configured_game_seed,
    pair_id_for,
    pool_snapshot_fingerprint,
)
from .lifecycle import LifecycleWriter
from .pool import OpponentPool
from .task_artifacts import DescriptorCommit, TaskDescriptorAllocator
from .task_execution import execute_task_bundle, read_task_bundle
from .target import evaluate_pair


@dataclass(frozen=True)
class ScheduledPair:
    """One submitted worker future and its coordinator-owned pair context."""

    active_trial: Any
    task: PairTask
    descriptor: DescriptorCommit | None = None
    descriptor_path: Path | None = None


def make_next_pair_task(
    active_trial: Any,
    pool: OpponentPool,
    lifecycle: LifecycleWriter,
    trace_path: str | None,
) -> PairTask:
    """Select and snapshot the closest frozen anchor for the next pair."""
    pair_index = active_trial.evaluation.completed_pairs
    anchor = pool.closest(active_trial.evaluation.rating.mu)
    return PairTask(
        lifecycle.session_id,
        active_trial.trial_id,
        pair_id_for(lifecycle.session_id, active_trial.trial_id, pair_index),
        pair_index,
        active_trial.seed + pair_index,
        active_trial.config,
        OpponentSnapshot.from_anchor(anchor),
        pool_snapshot_fingerprint(pool.anchors),
        active_trial.evaluation.rating,
        trace_path,
    )


def submit_next_pair(
    executor: ProcessPoolExecutor,
    futures: dict[Any, ScheduledPair],
    active_trial: Any,
    cfg: SearchConfig,
    binary: Path,
    pool: OpponentPool,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    trace_path: str | None,
    terminalize_trial: Callable[[optuna.Study, Any, str, str], None],
    task_descriptors: TaskDescriptorAllocator | None = None,
) -> None:
    """Commit pair evidence, then emit its start event and submit one worker."""
    task = make_next_pair_task(active_trial, pool, lifecycle, trace_path)
    descriptor: DescriptorCommit | None = None
    descriptor_path: Path | None = None
    if task_descriptors is not None:
        try:
            descriptor = task_descriptors.commit_task(
                task,
                cfg=cfg,
                binary=binary,
                pool_snapshot=[
                    OpponentSnapshot.from_anchor(anchor) for anchor in pool.anchors
                ],
            )
            descriptor_path = task_descriptors.layout.descriptor(descriptor.identity)
        except Exception as error:
            message = f"task descriptor commit failed: {error}"
            terminalize_trial(study, active_trial, "trial_failed", message)
            raise
    lifecycle.emit("pair_started", pair_started_payload(task, descriptor))
    try:
        if descriptor is None:
            future = executor.submit(evaluate_pair, cfg, binary, task)
        else:
            future = executor.submit(
                execute_task_bundle,
                descriptor_path,
                descriptor.digest,
            )
    except Exception as error:
        message = f"worker submission failed: {error}"
        emit_pair_failed(lifecycle, task, message, descriptor)
        terminalize_trial(study, active_trial, "trial_failed", message)
        raise
    futures[future] = ScheduledPair(
        active_trial,
        task,
        descriptor,
        descriptor_path,
    )


def pair_started_payload(
    task: PairTask, descriptor: DescriptorCommit | None = None
) -> dict:
    """Build the stable pair-start payload without emitting it."""
    payload = {
        "trial_id": task.trial_id,
        "pair_id": task.pair_id,
        "pair_index": task.pair_index,
        "seed": configured_game_seed(task.seed),
        "round": 1,
        "opponent": opponent_payload(task.opponent),
        "pool_snapshot_fingerprint": task.pool_snapshot_fingerprint,
        "rating_before": rating_payload(task.rating_before),
    }
    if descriptor is not None:
        payload.update(
            {
                "descriptor_digest": descriptor.digest,
                "task_id": descriptor.identity.task_id,
                "task_sequence": descriptor.identity.task_sequence,
            }
        )
    return payload


def worker_result(
    future: Any,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    scheduled: ScheduledPair,
    terminalize_trial: Callable[[optuna.Study, Any, str, str], None],
) -> PairResult | None:
    """Return pair evidence or terminalize its trial after a pair failure."""
    if future.cancelled():
        return failed_pair(
            study,
            lifecycle,
            scheduled,
            terminalize_trial,
            "trial_cancelled",
            "worker future was cancelled",
        )
    try:
        result = future.result()
        if scheduled.descriptor is None:
            return result
        if scheduled.descriptor_path is None:
            raise RuntimeError("scheduled artifact task is missing its descriptor path")
        return read_task_bundle(
            scheduled.descriptor_path,
            scheduled.descriptor.digest,
            result,
            scheduled.task,
        )
    except CancelledError:
        return failed_pair(
            study,
            lifecycle,
            scheduled,
            terminalize_trial,
            "trial_cancelled",
            "worker future was cancelled",
        )
    except Exception as error:
        return failed_pair(
            study,
            lifecycle,
            scheduled,
            terminalize_trial,
            "trial_failed",
            f"worker failed: {error}",
        )


def failed_pair(
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    scheduled: ScheduledPair,
    terminalize_trial: Callable[[optuna.Study, Any, str, str], None],
    event_type: str,
    error: str,
) -> None:
    """Emit pair failure before terminalizing its containing trial."""
    emit_pair_failed(lifecycle, scheduled.task, error, scheduled.descriptor)
    terminalize_trial(study, scheduled.active_trial, event_type, error)
    return None


def finish_pair(
    lifecycle: LifecycleWriter,
    active_trial: Any,
    result: PairResult,
    descriptor: DescriptorCommit | None = None,
) -> None:
    """Emit physical games, update rating in their order, then finish the pair."""
    if len(result.games) != 2:
        raise ValueError("an evaluation pair must contain exactly two games")
    for game in result.games:
        lifecycle.emit(
            "game_finished", game_finished_payload(result.task, game, descriptor)
        )
    rating_after = active_trial.evaluation.apply_pair(result)
    lifecycle.emit(
        "pair_finished",
        _with_task_reference(
            {
                "trial_id": result.task.trial_id,
                "pair_id": result.task.pair_id,
                "pair_index": result.task.pair_index,
                "rating_before": rating_payload(result.task.rating_before),
                "rating_after": rating_payload(rating_after),
                "score": active_trial.evaluation.score(),
            },
            descriptor,
        ),
    )


def game_finished_payload(
    task: PairTask, game: Any, descriptor: DescriptorCommit | None = None
) -> dict:
    """Build one typed physical-game payload without emitting it."""
    return _with_task_reference(
        {
            "trial_id": task.trial_id,
            "pair_id": task.pair_id,
            "game_id": game.game_id,
            "candidate_side": game.candidate_side,
            "outcome": game.outcome,
            "seed": game.seed,
            "round": game.round,
            "trace_game_seq": game.trace_game_seq,
            "plies": game.plies,
            "elapsed_ms": game.elapsed_ms,
            "candidate": metrics_payload(game.candidate),
            "baseline": metrics_payload(game.baseline),
        },
        descriptor,
    )


def emit_pair_failed(
    lifecycle: LifecycleWriter,
    task: PairTask,
    error: str,
    descriptor: DescriptorCommit | None = None,
) -> None:
    """Record a pair terminal failure before its containing trial terminalizes."""
    lifecycle.emit(
        "pair_failed",
        _with_task_reference(
            {
                "trial_id": task.trial_id,
                "pair_id": task.pair_id,
                "pair_index": task.pair_index,
                "error": error,
            },
            descriptor,
        ),
    )


def opponent_payload(opponent: OpponentSnapshot) -> dict:
    return {
        "anchor_id": opponent.anchor_id,
        "config": opponent.config,
        "mu": opponent.mu,
        "sigma": opponent.sigma,
    }


def rating_payload(rating: Any) -> dict:
    return {"mu": rating.mu, "sigma": rating.sigma}


def metrics_payload(metrics: Any) -> dict:
    return {
        "iterations_total": metrics.iterations_total,
        "iterations_first_half": metrics.iterations_first_half,
        "move_time_ms": metrics.move_time_ms,
    }


def _with_task_reference(payload: dict, descriptor: DescriptorCommit | None) -> dict:
    if descriptor is not None:
        payload.update(
            {
                "task_id": descriptor.identity.task_id,
                "descriptor_digest": descriptor.digest,
            }
        )
    return payload
