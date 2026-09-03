"""Orchestrator behavior: incremental skip, rebuild equivalence, the
ingest-error path, idempotence, and the golden canonical dump."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tuner_projection.build import project_runs
from tuner_projection.store import open_store

FIXTURES = Path(__file__).parent / "fixtures"
PROJECTION_ROOT = FIXTURES / "projection-root"
GOLDEN = FIXTURES / "projection" / "version4.dump.sql"


def _dump(db_path: Path) -> str:
    store = open_store(db_path)
    try:
        return store.canonical_dump()
    finally:
        store.close()


def _copy_root(destination: Path) -> Path:
    root = destination / "runs"
    root.mkdir()
    for name in ("version4", "version4-active-halving"):
        shutil.copytree(FIXTURES / name, root / name)
    return root


def test_golden_dump(tmp_path: Path) -> None:
    db_path = tmp_path / "p.sqlite"
    project_runs(PROJECTION_ROOT, db_path, rebuild=True)
    assert _dump(db_path) == GOLDEN.read_text(encoding="utf-8")


def test_incremental_skips_unchanged(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    first = project_runs(root, db_path, rebuild=False)
    assert (first.projected, first.skipped) == (2, 0)
    second = project_runs(root, db_path, rebuild=False)
    assert (second.projected, second.skipped) == (0, 2)


def test_incremental_reprojects_on_evidence_change(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    project_runs(root, db_path, rebuild=False)
    evidence = root / "version4" / "evidence.jsonl"
    stat = evidence.stat()
    os.utime(evidence, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    again = project_runs(root, db_path, rebuild=False)
    assert again.projected == 1 and again.skipped == 1


def test_rebuild_matches_incremental(tmp_path: Path) -> None:
    incremental = tmp_path / "incremental.sqlite"
    rebuilt = tmp_path / "rebuilt.sqlite"
    project_runs(PROJECTION_ROOT, incremental, rebuild=False)
    project_runs(PROJECTION_ROOT, incremental, rebuild=False)
    project_runs(PROJECTION_ROOT, rebuilt, rebuild=True)
    assert _dump(incremental) == _dump(rebuilt)


def test_projecting_twice_is_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    project_runs(PROJECTION_ROOT, first, rebuild=True)
    project_runs(PROJECTION_ROOT, second, rebuild=True)
    assert _dump(first) == _dump(second)


def test_records_ingest_error(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    broken = root / "broken"
    broken.mkdir(parents=True)
    shutil.copy(FIXTURES / "version4" / "manifest.json", broken / "manifest.json")
    (broken / "evidence.jsonl").write_text("not json\n", encoding="utf-8")
    db_path = tmp_path / "p.sqlite"
    summary = project_runs(root, db_path, rebuild=True)
    assert summary.ingest_errors == 1
    store = open_store(db_path)
    try:
        connection = store._connection  # noqa: SLF001 - test inspects stored rows
        run = connection.execute("SELECT ingest_error FROM runs WHERE run_id = 'broken'").fetchone()
        assert run is not None and run[0]
        for table in ("run_manifest", "cohorts", "proposals", "pairs"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        store.close()


def _journal(root: Path, run_id: str, run_dir: Path) -> None:
    import json

    root.mkdir(parents=True, exist_ok=True)
    with (root / "launches.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "launch",
                    "record": {"run_id": run_id, "run_dir": str(run_dir), "pid": 1, "argv": []},
                }
            )
            + "\n"
        )
        terminal = {"event": "terminal", "run_id": run_id, "outcome": "exited"}
        handle.write(json.dumps(terminal) + "\n")


def test_surfaces_a_launch_that_never_wrote_a_manifest(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = root / "doomed"
    run_dir.mkdir(parents=True)
    (run_dir / "launch.err").write_text(
        "tuner failed: objective file does not exist\n", encoding="utf-8"
    )
    _journal(root, "doomed", run_dir)
    db_path = tmp_path / "p.sqlite"

    summary = project_runs(root, db_path, rebuild=True)
    assert summary.ingest_errors == 1
    store = open_store(db_path)
    try:
        connection = store._connection  # noqa: SLF001 - test inspects stored rows
        row = connection.execute("SELECT ingest_error FROM runs WHERE run_id = 'doomed'").fetchone()
        assert row is not None and "objective file does not exist" in row[0]
    finally:
        store.close()

    # It is stable across a re-projection and never pruned while the journal
    # still lists it.
    again = project_runs(root, db_path, rebuild=False)
    assert again.skipped == 1 and again.pruned == 0


def test_a_manifest_written_later_supersedes_the_startup_failure_row(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = root / "version4"
    shutil.copytree(FIXTURES / "version4", run_dir)
    manifest = run_dir / "manifest.json"
    manifest_body = manifest.read_text(encoding="utf-8")
    manifest.unlink()
    _journal(root, "version4", run_dir)
    db_path = tmp_path / "p.sqlite"

    first = project_runs(root, db_path, rebuild=True)
    assert first.ingest_errors == 1

    manifest.write_text(manifest_body, encoding="utf-8")
    second = project_runs(root, db_path, rebuild=False)
    store = open_store(db_path)
    try:
        connection = store._connection  # noqa: SLF001
        row = connection.execute(
            "SELECT ingest_error FROM runs WHERE run_id = 'version4'"
        ).fetchone()
        assert row is not None and row[0] is None
    finally:
        store.close()
    assert second.pruned == 0


def test_prunes_removed_runs(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    project_runs(root, db_path, rebuild=False)
    shutil.rmtree(root / "version4-active-halving")
    summary = project_runs(root, db_path, rebuild=False)
    assert summary.pruned == 1
    store = open_store(db_path)
    try:
        assert store.projected_run_ids() == ["version4"]
    finally:
        store.close()


@pytest.mark.parametrize("rebuild", [True, False])
def test_schema_version_row_present(tmp_path: Path, rebuild: bool) -> None:
    db_path = tmp_path / "p.sqlite"
    project_runs(PROJECTION_ROOT, db_path, rebuild=rebuild)
    store = open_store(db_path)
    try:
        value = store._connection.execute(  # noqa: SLF001
            "SELECT value FROM projection_meta WHERE key = 'projection_schema_version'"
        ).fetchone()
        assert value == ("1",)
    finally:
        store.close()
