"""Coordinator-owned Optuna lifecycle and worker orchestration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import optuna

from .attempt import (
    _ActiveTrial,
    cancel_active_trials,
    drain_scheduled_trials,
    schedule_initial_trials,
    terminalize_trial,
    worker_count,
)
from .callback import _resolve_git_sha
from .config import SearchConfig
from .lifecycle import AttemptId, LifecycleWriter, SessionId, TrialId, make_attempt_id
from .manifest import build_session_manifest, write_manifest_atomic
from .pool import Anchor, OpponentPool
from .target import preflight_check


def run_optimization(
    cfg: SearchConfig,
    *,
    run_id: str,
    git_sha: str | None = None,
    trace_path: str | None = None,
    session_id: str | None = None,
    attempt_id: str | None = None,
    lifecycle_path: str | Path | None = None,
    game_kind: str | None = None,
) -> tuple[optuna.Study, OpponentPool]:
    """Run unfinished study work while recording lifecycle evidence."""
    binary = cfg.resolve_binary()
    session = SessionId(session_id or run_id)
    attempt = AttemptId(attempt_id) if attempt_id is not None else make_attempt_id()
    output_dir = Path("optuna_output") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _resolve_search_space(cfg, binary)
    pool_path = output_dir / "pool.json"
    pool = _load_or_initialize_pool(cfg, pool_path)
    study, storage = _open_study(output_dir, run_id, cfg)
    resolved_sha = git_sha or _resolve_git_sha()
    manifest, manifest_path = _write_session_manifest(
        cfg, game_kind, binary, resolved_sha, run_id, storage, output_dir
    )
    event_path = (
        Path(lifecycle_path) if lifecycle_path is not None else output_dir / "lifecycle.jsonl"
    )

    with LifecycleWriter(event_path, session, attempt) as lifecycle:
        _emit_session_started(lifecycle, manifest, manifest_path, run_id, cfg.optimizer.n_trials)
        _emit_attempt_started(lifecycle, run_id, storage, cfg.optimizer.n_trials)
        _run_attempt(
            cfg,
            binary=binary,
            pool=pool,
            pool_path=pool_path,
            study=study,
            lifecycle=lifecycle,
            resolved_sha=resolved_sha,
            trace_path=trace_path,
        )
    return study, pool


def _resolve_search_space(cfg: SearchConfig, binary: Path) -> None:
    """Populate the configuration from the game binary's authoritative schema."""
    parameters, conditions, _advertised_baselines = SearchConfig.parameters_from_binary(binary)
    cfg.parameters = parameters
    cfg.conditions = conditions


def _load_or_initialize_pool(cfg: SearchConfig, pool_path: Path) -> OpponentPool:
    """Load the persistent pool, adding configured anchors once before saving."""
    pool = OpponentPool.load(pool_path) if pool_path.exists() else OpponentPool.bootstrap(cfg)
    for anchor_id, config in cfg.target.baseline_configs.items():
        if not any(anchor.id == anchor_id for anchor in pool.anchors):
            pool.anchors.append(Anchor(anchor_id, dict(config), mu=25.0, sigma=0.5))
    pool.save(pool_path)
    return pool


def _open_study(
    output_dir: Path, run_id: str, cfg: SearchConfig
) -> tuple[optuna.Study, str]:
    """Open the run-compatible persistent Optuna study."""
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(output_dir / 'study.db').resolve()}"
    study = optuna.create_study(
        direction="maximize",
        study_name=run_id,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.optimizer.seed),
    )
    return study, storage


def _write_session_manifest(
    cfg: SearchConfig,
    game_kind: str | None,
    binary: Path,
    git_sha: str,
    run_id: str,
    storage: str,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Build and persist the immutable manifest for this logical session."""
    manifest = build_session_manifest(
        cfg,
        game_kind=game_kind,
        binary=binary,
        git_sha=git_sha,
        study_name=run_id,
        storage=storage,
    )
    manifest_path = output_dir / "session-manifest.json"
    write_manifest_atomic(manifest_path, manifest)
    return manifest, manifest_path


def _emit_session_started(
    lifecycle: LifecycleWriter,
    manifest: dict[str, Any],
    manifest_path: Path,
    run_id: str,
    target_trial_count: int,
) -> None:
    """Record a session start when this lifecycle artifact has none yet."""
    if lifecycle.has_session_started:
        return
    lifecycle.emit(
        "session_started",
        {
            "manifest": manifest,
            "manifest_fingerprint": manifest["fingerprint"],
            "manifest_path": str(manifest_path),
            "study_name": run_id,
            "target_trial_count": target_trial_count,
        },
    )


def _emit_attempt_started(
    lifecycle: LifecycleWriter, run_id: str, storage: str, target_trial_count: int
) -> None:
    """Record the physical attempt's existing study and target."""
    lifecycle.emit(
        "attempt_started",
        {
            "run_id": run_id,
            "study_name": run_id,
            "storage": storage,
            "target_trial_count": target_trial_count,
        },
    )


def _run_attempt(
    cfg: SearchConfig,
    *,
    binary: Path,
    pool: OpponentPool,
    pool_path: Path,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    resolved_sha: str,
    trace_path: str | None,
) -> None:
    futures: dict[Any, _ActiveTrial] = {}
    active: dict[TrialId, _ActiveTrial] = {}
    executor: ProcessPoolExecutor | None = None
    interrupted = False
    try:
        preflight_check(cfg, pool.closest(25.0).config, pool.closest(0.0).config)
        remaining = max(0, cfg.optimizer.n_trials - len(study.trials))
        workers = worker_count(cfg)
        executor = ProcessPoolExecutor(max_workers=workers)
        schedule_initial_trials(
            remaining,
            workers,
            executor,
            futures,
            active,
            cfg,
            binary,
            pool,
            study,
            lifecycle,
            trace_path,
        )
        drain_scheduled_trials(
            remaining,
            executor,
            futures,
            active,
            cfg,
            binary,
            pool,
            pool_path,
            study,
            lifecycle,
            resolved_sha,
            trace_path,
            wait,
        )

        lifecycle.emit("attempt_completed", {"target_trial_count": cfg.optimizer.n_trials})
    except KeyboardInterrupt:
        interrupted = True
        cancel_active_trials(futures, active, study, lifecycle)
        lifecycle.emit("attempt_stopped", {"reason": "coordinator interrupted"})
        raise
    except Exception as error:
        for active_trial in list(active.values()):
            terminalize_trial(
                study, lifecycle, active_trial, "trial_failed", f"attempt failed: {error}"
            )
        lifecycle.emit("attempt_failed", {"error": str(error)})
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
