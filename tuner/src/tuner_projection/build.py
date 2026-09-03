"""Discovery and orchestration: scan a runs root, decide skip/re-project per
run, drive the typed replay + report codecs, and commit rows through the store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tuner_cli.artifacts import read_manifest
from tuner_cli.evidence import read_events
from tuner_cli.replay import replay
from tuner_cli.report import build_report

from . import rows
from .rows import RunRow
from .store import ChangeFingerprint, RunProjection, Store, open_store


@dataclass(frozen=True, slots=True)
class ProjectionSummary:
    projected: int
    skipped: int
    ingest_errors: int
    pruned: int


def _fingerprint(run_dir: Path) -> ChangeFingerprint:
    evidence = run_dir / "evidence.jsonl"
    try:
        stat = evidence.stat()
        size, mtime_ns = stat.st_size, stat.st_mtime_ns
    except OSError:
        size, mtime_ns = -1, -1
    try:
        manifest_fingerprint = str(
            json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["fingerprint"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        manifest_fingerprint = ""
    return ChangeFingerprint(size, mtime_ns, manifest_fingerprint)


def _error_projection(run_id: str, fingerprint: ChangeFingerprint, error: str) -> RunProjection:
    manifest_fingerprint = fingerprint.manifest_fingerprint or None
    return RunProjection(
        run=RunRow(run_id, None, manifest_fingerprint, None, 0, error),
        fingerprint=fingerprint,
        manifest=None,
        report=None,
    )


def _project_run(run_dir: Path, run_id: str, fingerprint: ChangeFingerprint) -> RunProjection:
    try:
        manifest = read_manifest(run_dir / "manifest.json")
        events = read_events(run_dir / "evidence.jsonl")
        state = replay(manifest, events)
    except (OSError, ValueError) as error:
        return _error_projection(run_id, fingerprint, f"{type(error).__name__}: {error}")
    report_obj = None
    if state.terminal_status == "complete":
        try:
            report_obj = build_report(run_dir)
        except (OSError, ValueError) as error:
            return _error_projection(run_id, fingerprint, f"report: {error}")
    return RunProjection(
        run=RunRow(
            run_id,
            manifest.run_id,
            manifest.fingerprint,
            state.terminal_status,
            0 if report_obj is None else 1,
            None,
        ),
        fingerprint=fingerprint,
        manifest=rows.run_manifest_row(run_id, manifest),
        report=None if report_obj is None else rows.run_report_row(run_id, report_obj),
        cohorts=rows.cohort_rows(run_id, state),
        candidates=rows.candidate_rows(run_id, state),
        proposals=rows.proposal_rows(run_id, state),
        pairs=rows.pair_rows(run_id, state),
        games=rows.game_rows(run_id, state),
        observations=rows.observation_rows(run_id, state),
        shadow_decisions=rows.shadow_decision_rows(run_id, state),
        active_elimination_decisions=rows.active_elimination_decision_rows(run_id, state),
        validation_rows=rows.validation_rows(run_id, report_obj),
        compute_phases=rows.compute_phase_rows(run_id, state),
    )


def _discover(runs_root: Path) -> list[Path]:
    return sorted(
        (path.parent for path in runs_root.glob("*/manifest.json")),
        key=lambda directory: directory.name,
    )


def _launch_err_tail(run_dir: Path) -> str:
    try:
        text = (run_dir / "launch.err").read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    detail = text[-4000:] if text else "(launch.err was empty or absent)"
    return f"run exited during startup and never wrote manifest.json — {detail}"


def _orphan_launch_failures(runs_root: Path) -> dict[str, str]:
    """`run_id -> diagnostic` for every run the detached launcher journalled as
    terminated that never got far enough to write a ``manifest.json``.

    ``_discover`` finds runs only by their manifest, so without this a run that
    dies in argument/objective validation, on a missing binary, or on an
    import error would be silently absent from the projection -- the operator
    launches it and nothing ever appears. Here it surfaces as a failed row
    carrying its ``launch.err``.
    """
    journal = runs_root / "launches.jsonl"
    try:
        lines = journal.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    run_dirs: dict[str, Path] = {}
    terminated: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except ValueError:
            continue
        if event.get("event") == "launch":
            record = event.get("record", {})
            run_id = record.get("run_id")
            if isinstance(run_id, str):
                raw_dir = record.get("run_dir")
                run_dirs[run_id] = Path(raw_dir) if isinstance(raw_dir, str) else runs_root / run_id
        elif event.get("event") == "terminal":
            run_id = event.get("run_id")
            if isinstance(run_id, str):
                terminated.add(run_id)
    failures: dict[str, str] = {}
    for run_id in terminated:
        run_dir = run_dirs.get(run_id, runs_root / run_id)
        if not (run_dir / "manifest.json").exists():
            failures[run_id] = _launch_err_tail(run_dir)
    return failures


def _prune(store: Store, discovered: set[str]) -> int:
    stale = [run_id for run_id in store.projected_run_ids() if run_id not in discovered]
    for run_id in stale:
        store.delete_run(run_id)
    return len(stale)


def project_runs(runs_root: Path, db_path: Path, *, rebuild: bool) -> ProjectionSummary:
    if rebuild and db_path.exists():
        db_path.unlink()
    store = open_store(db_path)
    projected = skipped = errors = 0
    try:
        run_dirs = _discover(runs_root)
        discovered = {directory.name for directory in run_dirs}
        for run_dir in run_dirs:
            run_id = run_dir.name
            fingerprint = _fingerprint(run_dir)
            if not rebuild and store.fingerprint(run_id) == fingerprint:
                skipped += 1
                continue
            projection = _project_run(run_dir, run_id, fingerprint)
            store.replace_run(projection)
            projected += 1
            errors += 1 if projection.run.ingest_error is not None else 0
        # Runs the launcher started that died before writing a manifest never
        # reach `_discover`; surface them here so the fleet can show the
        # failure instead of the run just vanishing.
        startup_fingerprint = ChangeFingerprint(-1, -1, "")
        for run_id, detail in _orphan_launch_failures(runs_root).items():
            if run_id in discovered:
                continue
            discovered.add(run_id)
            if not rebuild and store.fingerprint(run_id) == startup_fingerprint:
                skipped += 1
                continue
            store.replace_run(_error_projection(run_id, startup_fingerprint, detail))
            projected += 1
            errors += 1
        pruned = _prune(store, discovered)
        store.vacuum()
        return ProjectionSummary(projected, skipped, errors, pruned)
    finally:
        store.close()
