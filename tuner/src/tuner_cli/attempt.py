"""Coordinator operations for one physical tuning attempt."""

from __future__ import annotations

import logging
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from .callback import emit_incumbent_record, emit_trial_record
from .config import SearchConfig
from .evaluation import (
    PairResult,
    SCORE_FORMULA_VERSION,
    TrialReportDecision,
    TrialEvaluationState,
)
from .lifecycle import LifecycleWriter, SessionId, TrialId, trial_id_for
from .pair_orchestration import (
    ScheduledPair,
    emit_pair_failed,
    finish_pair,
    submit_next_pair,
    worker_result,
)
from .pool import OpponentPool
from .space_optuna import suggest_config

logger = logging.getLogger("tuner_cli")


@dataclass
class _ActiveTrial:
    trial: optuna.Trial
    trial_id: TrialId
    config: dict
    seed: int
    evaluation: TrialEvaluationState


@dataclass(frozen=True)
class _AttemptContext:
    cfg: SearchConfig
    binary: Path
    pool: OpponentPool
    pool_path: Path
    study: optuna.Study
    lifecycle: LifecycleWriter
    resolved_sha: str
    trace_path: str | None


def worker_count(cfg: SearchConfig) -> int:
    """Resolve the existing worker default without changing parallelism policy."""
    if cfg.optimizer.n_workers is not None:
        return cfg.optimizer.n_workers
    return max(1, (os.cpu_count() or 2) // 2)


def schedule_initial_trials(
    remaining: int,
    workers: int,
    executor: ProcessPoolExecutor,
    futures: dict[Any, ScheduledPair],
    active: dict[TrialId, _ActiveTrial],
    cfg: SearchConfig,
    binary: Path,
    pool: OpponentPool,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    trace_path: str | None,
) -> None:
    """Fill the configured worker limit with first pairs from distinct trials."""
    for _ in range(min(workers, remaining)):
        schedule_trial(
            executor, futures, active, cfg, binary, pool, study, lifecycle, trace_path
        )


def schedule_trial(
    executor: ProcessPoolExecutor,
    futures: dict[Any, ScheduledPair],
    active: dict[TrialId, _ActiveTrial],
    cfg: SearchConfig,
    binary: Path,
    pool: OpponentPool,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    trace_path: str | None,
) -> None:
    """Ask Optuna for one trial, then submit only its first evaluation pair."""
    active_trial = create_active_trial(study, cfg, lifecycle.session_id)
    active[active_trial.trial_id] = active_trial
    emit_trial_created_and_started(lifecycle, active_trial)
    submit_next_pair(
        executor,
        futures,
        active_trial,
        cfg,
        binary,
        pool,
        study,
        lifecycle,
        trace_path,
        _terminalize_from_pair(lifecycle),
    )


def create_active_trial(
    study: optuna.Study, cfg: SearchConfig, session_id: SessionId
) -> _ActiveTrial:
    """Ask Optuna for one trial and attach its stable lifecycle identity."""
    trial = study.ask()
    config = suggest_config(trial, cfg)
    trial.set_user_attr("config", config)
    seed = cfg.optimizer.seed + trial.number
    return _ActiveTrial(
        trial,
        trial_id_for(session_id, trial.number),
        config,
        seed,
        TrialEvaluationState(cfg.optimizer.resource, cfg.optimizer.rating),
    )


def emit_trial_created_and_started(
    lifecycle: LifecycleWriter, active_trial: _ActiveTrial
) -> None:
    """Record the two non-terminal transitions before pair execution."""
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
    futures: dict[Any, ScheduledPair],
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
    """Settle pair futures, continuing each live trial one pair at a time."""
    context = _AttemptContext(
        cfg, binary, pool, pool_path, study, lifecycle, resolved_sha, trace_path
    )
    while futures:
        done, _ = wait_for_completion(futures, return_when=FIRST_COMPLETED)
        for future in done:
            scheduled = futures.pop(future)
            result = worker_result(
                future, study, lifecycle, scheduled, _terminalize_from_pair(lifecycle)
            )
            if result is None:
                active.pop(scheduled.active_trial.trial_id, None)
            elif continue_trial(executor, futures, scheduled, result, context):
                continue
            else:
                active.pop(scheduled.active_trial.trial_id, None)
            remaining = replenish_trial(remaining, executor, futures, active, context)


def _terminalize_from_pair(lifecycle: LifecycleWriter):
    def terminalize(
        study: optuna.Study, active_trial: _ActiveTrial, event_type: str, error: str
    ) -> None:
        terminalize_trial(study, lifecycle, active_trial, event_type, error)

    return terminalize


def continue_trial(
    executor: ProcessPoolExecutor,
    futures: dict[Any, ScheduledPair],
    scheduled: ScheduledPair,
    result: PairResult,
    context: _AttemptContext,
) -> bool:
    """Finish a valid pair and report whether its trial received another pair."""
    active_trial = scheduled.active_trial
    finish_pair(context.lifecycle, active_trial, result)
    score, decision = report_trial(context.lifecycle, active_trial)
    if decision.outcome == "complete":
        complete_trial(
            context.study,
            context.lifecycle,
            active_trial,
            context.pool,
            context.pool_path,
            context.resolved_sha,
            score,
            decision.reason,
        )
        return False
    submit_next_pair(
        executor,
        futures,
        active_trial,
        context.cfg,
        context.binary,
        context.pool,
        context.study,
        context.lifecycle,
        context.trace_path,
        _terminalize_from_pair(context.lifecycle),
    )
    return True


def report_trial(
    lifecycle: LifecycleWriter, active_trial: _ActiveTrial
) -> tuple[float, TrialReportDecision]:
    """Report one completed evaluation resource and preserve its decision evidence."""
    evaluation = active_trial.evaluation
    score = evaluation.score()
    active_trial.trial.report(score, evaluation.completed_pairs)
    decision = evaluation.decision()
    lifecycle.emit(
        "trial_reported",
        {
            "trial_id": active_trial.trial_id,
            "trial_number": active_trial.trial.number,
            "completed_pairs": evaluation.completed_pairs,
            "mu": evaluation.rating.mu,
            "sigma": evaluation.rating.sigma,
            "score": score,
            "score_formula_version": SCORE_FORMULA_VERSION,
            "conservative_k": evaluation.rating_policy.conservative_k,
            "outcome": decision.outcome,
            "reason": decision.reason,
            "pruning_exempt": False,
            "bracket_id": None,
            "rung_resource": None,
        },
    )
    return score, decision


def replenish_trial(
    remaining: int,
    executor: ProcessPoolExecutor,
    futures: dict[Any, ScheduledPair],
    active: dict[TrialId, _ActiveTrial],
    context: _AttemptContext,
) -> int:
    """Count one terminal trial and replace it if target work remains."""
    remaining -= 1
    if remaining > 0:
        schedule_trial(
            executor,
            futures,
            active,
            context.cfg,
            context.binary,
            context.pool,
            context.study,
            context.lifecycle,
            context.trace_path,
        )
    return remaining


def complete_trial(
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    active_trial: _ActiveTrial,
    pool: OpponentPool,
    pool_path: Path,
    resolved_sha: str,
    score: float,
    stop_reason: str,
) -> None:
    """Preserve the successful Optuna, lifecycle, legacy, pool, incumbent order."""
    if lifecycle.has_trial_terminal(active_trial.trial_id):
        return
    rating = active_trial.evaluation.rating
    games = active_trial.evaluation.legacy_games()
    record_completed_trial(
        study, lifecycle, active_trial, rating.mu, rating.sigma, games, score, stop_reason
    )
    emit_legacy_trial(active_trial, rating.mu, rating.sigma, games, resolved_sha)
    save_inserted_pool_anchor(pool, pool_path, active_trial, rating.mu, rating.sigma)
    emit_legacy_incumbent(study, active_trial, rating.mu, rating.sigma)


def record_completed_trial(
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    active_trial: _ActiveTrial,
    mu: float,
    sigma: float,
    games: list[dict],
    score: float,
    stop_reason: str,
) -> None:
    """Commit Optuna and lifecycle terminal evidence for a successful trial."""
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
            "completed_pairs": active_trial.evaluation.completed_pairs,
            "score_formula_version": SCORE_FORMULA_VERSION,
            "conservative_k": active_trial.evaluation.rating_policy.conservative_k,
            "reason": stop_reason,
            "pruning_exempt": False,
            "bracket_id": None,
            "rung_resource": None,
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
    pool: OpponentPool,
    pool_path: Path,
    active_trial: _ActiveTrial,
    mu: float,
    sigma: float,
) -> None:
    """Persist the pool only when this completed trial adds an anchor."""
    if pool.maybe_insert(active_trial.config, mu, sigma) is not None:
        pool.save(pool_path)


def terminalize_trial(
    study: optuna.Study | None,
    lifecycle: LifecycleWriter,
    active_trial: _ActiveTrial,
    event_type: str,
    error: str,
) -> None:
    """Mark a non-terminal active trial terminal and record its evidence."""
    if lifecycle.has_trial_terminal(active_trial.trial_id):
        return
    if study is not None:
        try:
            study.tell(active_trial.trial, state=TrialState.FAIL)
        except (RuntimeError, ValueError) as tell_error:
            logger.warning(
                "Could not mark Optuna trial %s failed: %s",
                active_trial.trial.number,
                tell_error,
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
    futures: dict[Any, ScheduledPair],
    active: dict[TrialId, _ActiveTrial],
    study: optuna.Study,
    lifecycle: LifecycleWriter,
) -> None:
    """Cancel submitted workers and record coordinator interruption for each trial."""
    for future, scheduled in futures.items():
        future.cancel()
        emit_pair_failed(lifecycle, scheduled.task, "coordinator interrupted")
    for active_trial in list(active.values()):
        terminalize_trial(
            study, lifecycle, active_trial, "trial_cancelled", "coordinator interrupted"
        )
