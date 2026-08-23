"""Coordinator operations for one physical tuning attempt."""

from __future__ import annotations

import logging
import os
from concurrent.futures import FIRST_COMPLETED, CancelledError, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from .callback import emit_incumbent_record, emit_trial_record
from .config import SearchConfig
from .lifecycle import LifecycleWriter, SessionId, TrialId, trial_id_for
from .matchmaking import evaluate_trial_worker
from .pool import OpponentPool
from .space_optuna import suggest_config

logger = logging.getLogger("tuner_cli")


@dataclass
class _ActiveTrial:
    trial: optuna.Trial
    trial_id: TrialId
    config: dict
    seed: int


def worker_count(cfg: SearchConfig) -> int:
    """Resolve the existing worker default without changing parallelism policy."""
    if cfg.optimizer.n_workers is not None:
        return cfg.optimizer.n_workers
    return max(1, (os.cpu_count() or 2) // 2)


def schedule_initial_trials(
    remaining: int,
    workers: int,
    executor: ProcessPoolExecutor,
    futures: dict[Any, _ActiveTrial],
    active: dict[TrialId, _ActiveTrial],
    cfg: SearchConfig,
    binary: Path,
    pool: OpponentPool,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    trace_path: str | None,
) -> None:
    """Fill the worker pool with the first batch of unfinished trials."""
    for _ in range(min(workers, remaining)):
        schedule_trial(
            executor, futures, active, cfg, binary, pool, study, lifecycle, trace_path
        )


def schedule_trial(
    executor: ProcessPoolExecutor,
    futures: dict[Any, _ActiveTrial],
    active: dict[TrialId, _ActiveTrial],
    cfg: SearchConfig,
    binary: Path,
    pool: OpponentPool,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    trace_path: str | None,
) -> None:
    """Create lifecycle evidence and submit one trial to a worker."""
    active_trial = create_active_trial(study, cfg, lifecycle.session_id)
    active[active_trial.trial_id] = active_trial
    emit_trial_created_and_started(lifecycle, active_trial)
    try:
        future = executor.submit(
            evaluate_trial_worker,
            cfg,
            binary,
            active_trial.config,
            pool,
            active_trial.seed,
            trace_path,
        )
    except Exception as error:
        terminalize_trial(
            study, lifecycle, active_trial, "trial_failed", f"worker submission failed: {error}"
        )
        raise
    futures[future] = active_trial


def create_active_trial(
    study: optuna.Study, cfg: SearchConfig, session_id: SessionId
) -> _ActiveTrial:
    """Ask Optuna for one trial and attach its stable lifecycle identity."""
    trial = study.ask()
    config = suggest_config(trial, cfg)
    trial.set_user_attr("config", config)
    seed = cfg.optimizer.seed + trial.number
    return _ActiveTrial(trial, trial_id_for(session_id, trial.number), config, seed)


def emit_trial_created_and_started(lifecycle: LifecycleWriter, active_trial: _ActiveTrial) -> None:
    """Record the two non-terminal transitions before worker execution."""
    lifecycle.emit(
        "trial_created",
        {
            "trial_id": active_trial.trial_id,
            "trial_number": active_trial.trial.number,
            "config": active_trial.config,
            "seed": active_trial.seed,
        },
    )
    lifecycle.emit(
        "trial_started",
        {"trial_id": active_trial.trial_id, "trial_number": active_trial.trial.number},
    )


def drain_scheduled_trials(
    remaining: int,
    executor: ProcessPoolExecutor,
    futures: dict[Any, _ActiveTrial],
    active: dict[TrialId, _ActiveTrial],
    cfg: SearchConfig,
    binary: Path,
    pool: OpponentPool,
    pool_path: Path,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    resolved_sha: str,
    trace_path: str | None,
    wait_for_completion: Any,
) -> None:
    """Settle completed workers and replenish work until the target is met."""
    while futures:
        done, _ = wait_for_completion(futures, return_when=FIRST_COMPLETED)
        for future in done:
            active_trial = futures.pop(future)
            mu, sigma, games = worker_result(future, study, lifecycle, active_trial)
            record_completed_trial(study, lifecycle, active_trial, mu, sigma, games)
            active.pop(active_trial.trial_id, None)
            emit_legacy_trial(active_trial, mu, sigma, games, resolved_sha)
            save_inserted_pool_anchor(pool, pool_path, active_trial, mu, sigma)
            emit_legacy_incumbent(study, active_trial, mu, sigma)
            remaining -= 1
            if remaining > 0:
                schedule_trial(
                    executor, futures, active, cfg, binary, pool, study, lifecycle, trace_path
                )


def worker_result(
    future: Any,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    active_trial: _ActiveTrial,
) -> tuple[float, float, list[dict]]:
    """Return a worker result or record its typed terminal failure evidence."""
    if future.cancelled():
        terminalize_trial(
            study, lifecycle, active_trial, "trial_cancelled", "worker future was cancelled"
        )
        raise RuntimeError("worker future was cancelled")
    try:
        return future.result()
    except CancelledError:
        terminalize_trial(
            study, lifecycle, active_trial, "trial_cancelled", "worker future was cancelled"
        )
        raise RuntimeError("worker future was cancelled") from None
    except Exception as error:
        terminalize_trial(study, lifecycle, active_trial, "trial_failed", f"worker failed: {error}")
        raise


def record_completed_trial(
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    active_trial: _ActiveTrial,
    mu: float,
    sigma: float,
    games: list[dict],
) -> None:
    """Commit Optuna and lifecycle terminal evidence for a successful trial."""
    score = mu - 3 * sigma
    study.tell(active_trial.trial, score)
    lifecycle.emit_trial_terminal(
        "trial_completed",
        active_trial.trial_id,
        {
            "trial_number": active_trial.trial.number,
            "config": active_trial.config,
            "seed": active_trial.seed,
            "mu": mu,
            "sigma": sigma,
            "score": score,
            "games": games,
        },
    )


def emit_legacy_trial(
    active_trial: _ActiveTrial,
    mu: float,
    sigma: float,
    games: list[dict],
    resolved_sha: str,
) -> None:
    """Preserve the established terminal trial stdout record."""
    emit_trial_record(
        active_trial.trial.number,
        active_trial.config,
        active_trial.seed,
        mu,
        sigma,
        games,
        resolved_sha,
    )


def emit_legacy_incumbent(
    study: optuna.Study, active_trial: _ActiveTrial, mu: float, sigma: float
) -> None:
    """Preserve incumbent output after any pool checkpoint for this trial."""
    if study.best_trial.number == active_trial.trial.number:
        emit_incumbent_record(active_trial.config, mu, sigma)


def save_inserted_pool_anchor(
    pool: OpponentPool, pool_path: Path, active_trial: _ActiveTrial, mu: float, sigma: float
) -> None:
    """Persist the pool only when this completed trial adds an anchor."""
    if pool.maybe_insert(active_trial.config, mu, sigma) is not None:
        pool.save(pool_path)


def terminalize_trial(
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    active_trial: _ActiveTrial,
    event_type: str,
    error: str,
) -> None:
    """Mark a non-terminal active trial terminal and record its evidence."""
    if lifecycle.has_trial_terminal(active_trial.trial_id):
        return
    try:
        study.tell(active_trial.trial, state=TrialState.FAIL)
    except (RuntimeError, ValueError) as tell_error:
        logger.warning(
            "Could not mark Optuna trial %s failed: %s", active_trial.trial.number, tell_error
        )
    lifecycle.emit_trial_terminal(
        event_type,
        active_trial.trial_id,
        {
            "trial_number": active_trial.trial.number,
            "config": active_trial.config,
            "seed": active_trial.seed,
            "error": error,
        },
    )


def cancel_active_trials(
    futures: dict[Any, _ActiveTrial],
    active: dict[TrialId, _ActiveTrial],
    study: optuna.Study,
    lifecycle: LifecycleWriter,
) -> None:
    """Cancel submitted workers and record the coordinator interruption for each trial."""
    for future in futures:
        future.cancel()
    for active_trial in list(active.values()):
        terminalize_trial(
            study, lifecycle, active_trial, "trial_cancelled", "coordinator interrupted"
        )
