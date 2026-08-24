"""Coordinator-owned Optuna lifecycle and worker orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
import signal
from typing import Any, Callable, Iterator

import optuna
from optuna.trial import TrialState

from .attempt import (
    AttemptStopRequested,
    _ActiveTrial,
    StartupTrialAllocator,
    cancel_active_trials,
    drain_scheduled_trials,
    schedule_initial_trials,
    terminalize_trial,
    worker_count,
)
from .callback import _resolve_git_sha
from .config import SearchConfig
from .lifecycle import (
    AttemptId,
    LifecycleWriter,
    OrphanedAttempt,
    SessionId,
    TrialId,
    trial_id_for,
)
from .hyperband import OptunaHyperbandAdapter
from .manifest import SessionForkRequired, build_session_manifest, write_manifest_atomic
from .pool import OpponentPool, recover_pool
from .task_artifacts import TaskDescriptorAllocator
from .target import preflight_check


class _StopRequest:
    """A signal-safe stop flag observed by the coordinator at work boundaries."""

    def __init__(self) -> None:
        self._requested = False

    def request(self, *_: object) -> None:
        self._requested = True

    def requested(self) -> bool:
        return self._requested


@contextmanager
def _install_stop_handlers() -> Iterator[_StopRequest]:
    """Install handlers that defer all cleanup and lifecycle I/O to the coordinator."""
    stop_request = _StopRequest()
    previous = {
        signum: signal.signal(signum, stop_request.request)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield stop_request
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_optimization(
    cfg: SearchConfig,
    *,
    run_id: str | None = None,
    optimizer_id: str | None = None,
    bench_run_id: str | None = None,
    git_sha: str | None = None,
    session_id: str | None = None,
    attempt_id: str | None = None,
    lifecycle_path: str | Path | None = None,
    game_kind: str | None = None,
    artifact_root: str | Path,
) -> tuple[optuna.Study, OpponentPool]:
    """Run unfinished study work while recording lifecycle evidence."""
    cfg.validate()
    binary = cfg.resolve_binary()
    optimizer = optimizer_id or run_id
    if optimizer is None:
        raise ValueError("optimizer_id or legacy run_id is required")
    session = SessionId(session_id or optimizer)
    if attempt_id is None:
        raise ValueError("artifact_root requires a physical attempt_id")
    attempt = AttemptId(attempt_id)
    artifact_root = _validate_artifact_root(artifact_root, bench_run_id, attempt)
    output_dir = Path("optuna_output") / optimizer
    output_dir.mkdir(parents=True, exist_ok=True)
    _resolve_search_space(cfg, binary)
    pruning_adapter = _pruning_adapter(cfg)
    resolved_sha = git_sha or _resolve_git_sha()
    manifest, manifest_path = _write_session_manifest(
        cfg,
        game_kind,
        binary,
        resolved_sha,
        optimizer,
        _study_storage(output_dir),
        output_dir,
    )
    study, storage = _open_study(output_dir, optimizer, cfg, pruning_adapter)
    event_path = (
        Path(lifecycle_path)
        if lifecycle_path is not None
        else output_dir / "lifecycle.jsonl"
    )
    pool_path = output_dir / "pool.json"
    pool: OpponentPool

    with _install_stop_handlers() as stop_request:
        with LifecycleWriter(event_path, session, attempt) as lifecycle:
            _emit_session_started(
                lifecycle, manifest, manifest_path, optimizer, cfg.optimizer.n_trials
            )
            task_descriptors = TaskDescriptorAllocator.start(
                artifact_root,
                session_id=session,
                optimizer_id=optimizer,
                attempt_id=attempt,
                bench_run_id=bench_run_id,
                manifest_fingerprint=manifest["fingerprint"],
            )
            _emit_attempt_started(
                lifecycle, optimizer, bench_run_id, storage, cfg.optimizer.n_trials
            )
            _recover_orphaned_attempt(lifecycle, study)
            pool = recover_pool(
                cfg, pool_path, manifest["fingerprint"], lifecycle, study
            )
            _emit_pool_revised(lifecycle, pool)
            _run_attempt(
                cfg,
                binary=binary,
                pool=pool,
                pool_path=pool_path,
                study=study,
                lifecycle=lifecycle,
                resolved_sha=resolved_sha,
                manifest_fingerprint=manifest["fingerprint"],
                pruning_adapter=pruning_adapter,
                should_stop=stop_request.requested,
                task_descriptors=task_descriptors,
            )
    return study, pool


def _validate_artifact_root(
    artifact_root: str | Path, bench_run_id: str | None, attempt_id: AttemptId
) -> Path:
    """Accept only the server-owned root for this physical attempt."""
    if bench_run_id is None or not bench_run_id:
        raise ValueError("artifact_root requires a physical bench_run_id")
    root = Path(artifact_root)
    if not root.is_absolute():
        raise ValueError("artifact_root must be absolute")
    if root.name != "tuning-artifacts" or root.parent.name != bench_run_id:
        raise ValueError("artifact_root must belong to the physical bench_run_id")
    if root.parent.parent.name != "bench-runs":
        raise ValueError("artifact_root must be below a bench-runs directory")
    return root


def _resolve_search_space(cfg: SearchConfig, binary: Path) -> None:
    """Populate the configuration from the game binary's authoritative schema."""
    parameters, conditions, _advertised_baselines = SearchConfig.parameters_from_binary(
        binary
    )
    cfg.parameters = parameters
    cfg.conditions = conditions


def _open_study(
    output_dir: Path,
    run_id: str,
    cfg: SearchConfig,
    pruning_adapter: OptunaHyperbandAdapter | None = None,
) -> tuple[optuna.Study, str]:
    """Open the run-compatible persistent Optuna study."""
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = _study_storage(output_dir)
    if pruning_adapter is None:
        pruning_adapter = _pruning_adapter(cfg)
    kwargs: dict[str, Any] = {}
    if pruning_adapter is not None:
        kwargs["pruner"] = pruning_adapter.pruner
    study = optuna.create_study(
        direction="maximize",
        study_name=run_id,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=cfg.optimizer.seed,
            n_startup_trials=cfg.optimizer.sampler.startup_trials,
        ),
        **kwargs,
    )
    return study, storage


def _study_storage(output_dir: Path) -> str:
    """Return the stable storage address before opening or mutating a study."""
    return f"sqlite:///{(output_dir / 'study.db').resolve()}"


def _pruning_adapter(cfg: SearchConfig) -> OptunaHyperbandAdapter | None:
    """Construct the coordinator-owned Hyperband boundary when configured."""
    if not cfg.optimizer.pruning.enabled:
        return None
    return OptunaHyperbandAdapter(cfg.optimizer.resource, cfg.optimizer.pruning)


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
    optimizer_id: str,
    target_trial_count: int,
) -> None:
    """Record a session start when this lifecycle artifact has none yet."""
    if lifecycle.has_session_started:
        if lifecycle.manifest_fingerprint != manifest["fingerprint"]:
            raise SessionForkRequired(
                "fork required: lifecycle journal manifest fingerprint does not match"
            )
        return
    lifecycle.emit(
        "session_started",
        {
            "manifest": manifest,
            "manifest_fingerprint": manifest["fingerprint"],
            "manifest_path": str(manifest_path),
            "lifecycle_path": str(lifecycle.path),
            "optimizer_id": optimizer_id,
            "study_name": optimizer_id,
            "target_trial_count": target_trial_count,
        },
    )


def _emit_attempt_started(
    lifecycle: LifecycleWriter,
    optimizer_id: str,
    bench_run_id: str | None,
    storage: str,
    target_trial_count: int,
) -> None:
    """Record the physical attempt's existing study and target."""
    lifecycle.emit(
        "attempt_started",
        {
            "optimizer_id": optimizer_id,
            "bench_run_id": bench_run_id,
            "study_name": optimizer_id,
            "storage": storage,
            "target_trial_count": target_trial_count,
        },
    )


def _recover_orphaned_attempt(lifecycle: LifecycleWriter, study: optuna.Study) -> None:
    """Fail the one lock-free prior attempt before any new scheduling boundary."""
    orphan = lifecycle.journal_snapshot.orphaned_attempt
    if orphan is None:
        return
    trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
    recovery_trials: list[dict[str, object]] = []
    for recovered in orphan.trials:
        trial = trials.get(recovered.trial_number)
        if trial is None:
            raise ValueError(
                f"recovery identity conflict: Optuna is missing trial {recovered.trial_number}"
            )
        if recovered.trial_id != trial_id_for(
            lifecycle.session_id, recovered.trial_number
        ):
            raise ValueError(
                "recovery identity conflict: trial id is not deterministic"
            )
        reason = "abrupt_attempt_recovery"
        if trial.state == TrialState.RUNNING:
            study.tell(recovered.trial_number, state=TrialState.FAIL)
        else:
            reason = "recovery_evidence_gap"
        recovery_trials.append(
            {
                "trial_id": recovered.trial_id,
                "trial_number": recovered.trial_number,
                "reason": reason,
            }
        )
    _emit_attempt_recovered(lifecycle, orphan, recovery_trials)


def _emit_attempt_recovered(
    lifecycle: LifecycleWriter,
    orphan: OrphanedAttempt,
    trials: list[dict[str, object]],
) -> None:
    """Record exact recovery scope after Optuna has consumed its running slots."""
    lifecycle.emit(
        "attempt_recovered",
        {
            "prior_attempt_id": orphan.attempt_id,
            "prior_bench_run_id": orphan.bench_run_id,
            "trials": trials,
            "pair_ids": list(orphan.pair_ids),
            "reason": "abrupt_attempt_recovery",
        },
    )


def _emit_pool_revised(lifecycle: LifecycleWriter, pool: OpponentPool) -> None:
    """Record the loaded pool after the attempt can be associated with it."""
    lifecycle.emit("pool_revised", pool.revision_payload())


def _run_attempt(
    cfg: SearchConfig,
    *,
    binary: Path,
    pool: OpponentPool,
    pool_path: Path,
    study: optuna.Study,
    lifecycle: LifecycleWriter,
    resolved_sha: str,
    manifest_fingerprint: str = "legacy",
    pruning_adapter: OptunaHyperbandAdapter | None = None,
    should_stop: Callable[[], bool] | None = None,
    task_descriptors: TaskDescriptorAllocator,
) -> bool:
    futures: dict[Any, _ActiveTrial] = {}
    active: dict[TrialId, _ActiveTrial] = {}
    executor: ProcessPoolExecutor | None = None
    stopped = False
    try:
        _raise_if_stop_requested(should_stop)
        preflight_check(cfg, pool.closest(25.0).config, pool.closest(0.0).config)
        _raise_if_stop_requested(should_stop)
        remaining = max(0, cfg.optimizer.n_trials - len(study.trials))
        workers = worker_count(cfg)
        startup_allocator = (
            StartupTrialAllocator.restore(study, cfg.optimizer.pruning.startup_trials)
            if pruning_adapter is not None
            else None
        )
        executor = ProcessPoolExecutor(max_workers=workers)
        remaining = schedule_initial_trials(
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
            pruning_adapter,
            should_stop,
            task_descriptors,
            startup_allocator,
        )
        _raise_if_stop_requested(should_stop)
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
            wait,
            pruning_adapter,
            should_stop,
            manifest_fingerprint,
            task_descriptors,
            startup_allocator,
        )

        _raise_if_stop_requested(should_stop)

        lifecycle.emit(
            "attempt_completed", {"target_trial_count": cfg.optimizer.n_trials}
        )
        return True
    except (AttemptStopRequested, KeyboardInterrupt):
        stopped = True
        _terminate_workers(executor)
        cancel_active_trials(futures, active, study, lifecycle)
        lifecycle.emit("attempt_stopped", {"reason": "coordinator interrupted"})
        return False
    except Exception as error:
        for active_trial in list(active.values()):
            terminalize_trial(
                study,
                lifecycle,
                active_trial,
                "trial_failed",
                f"attempt failed: {error}",
            )
        lifecycle.emit("attempt_failed", {"error": str(error)})
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=not stopped, cancel_futures=stopped)


def _raise_if_stop_requested(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise AttemptStopRequested


def _terminate_workers(executor: ProcessPoolExecutor | None) -> None:
    """Terminate running pool workers before their non-blocking shutdown."""
    if executor is None:
        return
    processes = getattr(executor, "_processes", {})
    for process in list(processes.values()):
        try:
            if hasattr(process, "kill"):
                process.kill()
            else:
                process.terminate()
        except (AttributeError, OSError):
            continue
