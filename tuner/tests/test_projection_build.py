"""Orchestrator behavior: incremental skip, rebuild equivalence, the
ingest-error path, idempotence, and the golden canonical dump."""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

import tuner_projection.build as build_module
from tuner_projection.build import project_pass, project_runs
from tuner_projection.store import Store, open_store

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


def test_watch_shape_reuses_one_store_over_a_growing_run(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    store = open_store(db_path)
    try:
        first = project_pass(root, store, rebuild=False)
        assert (first.projected, first.skipped) == (2, 0)
        evidence = root / "version4" / "evidence.jsonl"
        with evidence.open("a", encoding="utf-8") as handle:
            handle.write(
                '{"schema_version":5,"sequence":999,"type":"run_interrupted",'
                '"payload":{"stage":"s","pair_id":null}}\n'
            )
        stat = evidence.stat()
        os.utime(evidence, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        second = project_pass(root, store, rebuild=False)
        assert (second.projected, second.skipped) == (1, 1)
        assert sorted(store.projected_run_ids()) == ["version4", "version4-active-halving"]
    finally:
        store.close()


def test_records_a_last_pass_stamp_kept_out_of_the_golden_dump(tmp_path: Path) -> None:
    db_path = tmp_path / "p.sqlite"
    project_runs(PROJECTION_ROOT, db_path, rebuild=True)
    store = open_store(db_path)
    try:
        assert store.last_pass_at() is not None
        # The stamp is wall-clock state; the canonical dump must not carry it,
        # so the golden fixture stays byte-stable across passes.
        assert "last_pass_at" not in store.canonical_dump()
    finally:
        store.close()
    assert _dump(db_path) == GOLDEN.read_text(encoding="utf-8")


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


def test_forget_run_scoped(tmp_path: Path) -> None:
    """`delete_run` removes exactly the target run's rows and leaves siblings
    intact."""
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    project_runs(root, db_path, rebuild=False)
    store = open_store(db_path)
    try:
        assert sorted(store.projected_run_ids()) == ["version4", "version4-active-halving"]
        store.delete_run("version4-active-halving")
        assert store.projected_run_ids() == ["version4"]
        connection = store._connection  # noqa: SLF001
        for table in ("run_manifest", "cohorts", "candidates", "pairs", "ingest_state"):
            gone = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id = 'version4-active-halving'"
            ).fetchone()[0]
            assert gone == 0, table
            kept = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id = 'version4'"
            ).fetchone()[0]
            assert kept > 0, table
    finally:
        store.close()


def test_forget_cli_removes_one_run(tmp_path: Path) -> None:
    from tuner_projection.__main__ import main

    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    project_runs(root, db_path, rebuild=False)

    assert main(["--db", str(db_path), "--forget", "version4-active-halving"]) == 0

    store = open_store(db_path)
    try:
        assert store.projected_run_ids() == ["version4"]
    finally:
        store.close()


def test_deleted_run_is_not_resurrected_as_a_startup_failure(tmp_path: Path) -> None:
    """A tombstoned run whose directory has been removed must not reappear as
    an orphan launch failure on the next pass."""
    import json

    root = tmp_path / "runs"
    run_dir = root / "doomed"
    run_dir.mkdir(parents=True)
    (run_dir / "launch.err").write_text("boom\n", encoding="utf-8")
    _journal(root, "doomed", run_dir)
    db_path = tmp_path / "p.sqlite"
    first = project_runs(root, db_path, rebuild=True)
    assert first.ingest_errors == 1

    # The bench server tombstones the run and removes its directory.
    with (root / "launches.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": "run_deleted", "run_id": "doomed", "deleted_at": "2026-01-01T00:00:00Z"}
            )
            + "\n"
        )
    shutil.rmtree(run_dir)

    second = project_runs(root, db_path, rebuild=False)
    assert second.pruned == 1
    store = open_store(db_path)
    try:
        assert store.projected_run_ids() == []
    finally:
        store.close()


def test_pass_with_no_prunes_does_not_vacuum(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    store = open_store(db_path)
    try:
        first = project_pass(root, store, rebuild=False)
        assert first.pruned == 0 and first.vacuumed is False

        calls = []
        store.vacuum = lambda: calls.append(True)  # type: ignore[method-assign]
        second = project_pass(root, store, rebuild=False)
        assert second.pruned == 0 and second.vacuumed is False
        assert calls == []
    finally:
        store.close()


def test_pass_with_a_prune_still_vacuums(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    store = open_store(db_path)
    try:
        project_pass(root, store, rebuild=False)
        shutil.rmtree(root / "version4-active-halving")

        calls = []
        store.vacuum = lambda: calls.append(True)  # type: ignore[method-assign]
        summary = project_pass(root, store, rebuild=False)
        assert summary.pruned == 1 and summary.vacuumed is True
        assert calls == [True]
    finally:
        store.close()


def test_watch_loop_folds_only_the_new_tail_after_the_first_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task 14d: a live run's steady-state passes must not re-read/re-decode
    the whole evidence log -- only `tail_events` past the last checkpoint."""
    run_dir = tmp_path / "runs" / "version4"
    run_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "version4" / "manifest.json", run_dir / "manifest.json")
    lines = (
        (FIXTURES / "version4" / "evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    evidence = run_dir / "evidence.jsonl"
    evidence.write_text("", encoding="utf-8")

    calls = []
    original_read_events = build_module.read_events

    def counting_read_events(path: Path) -> list:
        calls.append(path)
        return original_read_events(path)

    monkeypatch.setattr(build_module, "read_events", counting_read_events)

    db_path = tmp_path / "p.sqlite"
    store = open_store(db_path)
    try:
        chunk = len(lines) // 4
        for end in (chunk, 2 * chunk, 3 * chunk, len(lines)):
            evidence.write_text("".join(lines[:end]), encoding="utf-8")
            os.utime(evidence)
            summary = project_pass(tmp_path / "runs", store, rebuild=False)
            assert summary.ingest_errors == 0
    finally:
        store.close()

    # Only the very first pass (no checkpoint yet) does a full read; every
    # later pass, despite the log growing each time, resumes from the prior
    # checkpoint via `tail_events` instead.
    assert calls == [evidence]

    rebuilt_db = tmp_path / "rebuilt.sqlite"
    project_runs(tmp_path / "runs", rebuilt_db, rebuild=True)
    assert _dump(db_path) == _dump(rebuilt_db)


def test_checkpoint_version_mismatch_falls_back_to_full_replay(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    project_runs(root, db_path, rebuild=False)

    store = open_store(db_path)
    store._connection.execute(  # noqa: SLF001 - test corrupts stored checkpoint on purpose
        "UPDATE run_checkpoints SET checkpoint_version = 999 WHERE run_id = 'version4'"
    )
    store._connection.commit()  # noqa: SLF001
    store.close()

    evidence = root / "version4" / "evidence.jsonl"
    stat = evidence.stat()
    os.utime(evidence, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    summary = project_runs(root, db_path, rebuild=False)
    assert summary.ingest_errors == 0

    rebuilt_db = tmp_path / "rebuilt.sqlite"
    project_runs(PROJECTION_ROOT, rebuilt_db, rebuild=True)
    assert _dump(db_path) == _dump(rebuilt_db)


def test_rebuild_ignores_an_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_root(tmp_path)
    db_path = tmp_path / "p.sqlite"
    project_runs(root, db_path, rebuild=False)

    store = open_store(db_path)
    # A checkpoint so broken that reading it would raise if `--rebuild` ever
    # looked at it -- proving the rebuild path is the escape hatch it claims
    # to be, not merely "usually skips a valid one".
    store._connection.execute(  # noqa: SLF001
        "UPDATE run_checkpoints SET state = ? WHERE run_id = 'version4'", (b"not a pickle",)
    )
    store._connection.commit()  # noqa: SLF001

    checked = []
    original_checkpoint = Store.checkpoint

    def spying_checkpoint(self: Store, run_id: str, *, manifest_fingerprint: str):
        checked.append(run_id)
        return original_checkpoint(self, run_id, manifest_fingerprint=manifest_fingerprint)

    monkeypatch.setattr(Store, "checkpoint", spying_checkpoint)
    try:
        summary = project_pass(root, store, rebuild=True)
    finally:
        store.close()
    assert summary.ingest_errors == 0
    assert checked == []

    rebuilt_db = tmp_path / "rebuilt.sqlite"
    project_runs(PROJECTION_ROOT, rebuilt_db, rebuild=True)
    assert _dump(db_path) == _dump(rebuilt_db)


@pytest.mark.parametrize("rebuild", [True, False])
def test_schema_version_row_present(tmp_path: Path, rebuild: bool) -> None:
    db_path = tmp_path / "p.sqlite"
    project_runs(PROJECTION_ROOT, db_path, rebuild=rebuild)
    store = open_store(db_path)
    try:
        value = store._connection.execute(  # noqa: SLF001
            "SELECT value FROM projection_meta WHERE key = 'projection_schema_version'"
        ).fetchone()
        assert value == ("2",)
    finally:
        store.close()


def test_open_store_rebuilds_on_schema_version_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "p.sqlite"
    project_runs(PROJECTION_ROOT, db_path, rebuild=True)
    original_ids = sorted(open_store(db_path).projected_run_ids())

    stale = sqlite3.connect(db_path)
    stale.execute("UPDATE projection_meta SET value = '1' WHERE key = 'projection_schema_version'")
    stale.commit()
    stale.close()

    store = open_store(db_path)
    try:
        assert store._connection.execute(  # noqa: SLF001
            "SELECT value FROM projection_meta WHERE key = 'projection_schema_version'"
        ).fetchone() == ("2",)
        # The file was dropped and recreated fresh, so it holds no run rows until
        # the next projection pass re-populates it.
        assert store.projected_run_ids() == []
    finally:
        store.close()

    project_runs(PROJECTION_ROOT, db_path, rebuild=False)
    assert sorted(open_store(db_path).projected_run_ids()) == original_ids
