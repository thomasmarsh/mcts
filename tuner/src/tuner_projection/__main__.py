"""``tuner-project`` console entry point."""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path
from types import FrameType

from .build import project_pass, project_runs
from .store import open_store


def _run_watch(runs_root: Path, db_path: Path, *, interval: float) -> int:
    """Reproject in a loop from one long-lived process.

    The interpreter, import graph, and SQLite connection are paid for once; the
    per-run ``_fingerprint`` skip keeps each pass cheap when nothing changed.
    """
    stopping = False

    def _stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    store = open_store(db_path)
    try:
        while not stopping:
            summary = project_pass(runs_root, store, rebuild=False)
            print(
                f"projected={summary.projected} skipped={summary.skipped} "
                f"ingest_errors={summary.ingest_errors} pruned={summary.pruned}",
                flush=True,
            )
            deadline = time.monotonic() + interval
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    finally:
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tuner-project",
        description="Build a rebuildable read-only SQLite projection of version-4 tuner runs.",
    )
    parser.add_argument("--runs-root", type=Path, help="directory of run subdirs")
    parser.add_argument("--db", required=True, type=Path, help="SQLite projection file to write")
    parser.add_argument(
        "--forget",
        metavar="RUN_ID",
        help="delete exactly one run's rows from the projection and exit "
        "(the run directory itself is removed by the bench server)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the file and re-project every run instead of skipping unchanged runs",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="reproject in a loop from one process until SIGTERM/SIGINT",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=4.0,
        help="seconds between passes in --watch mode (default 4)",
    )
    args = parser.parse_args(argv)
    db_path: Path = args.db
    rebuild: bool = args.rebuild

    if args.forget is not None:
        store = open_store(db_path)
        try:
            store.delete_run(args.forget)
        finally:
            store.close()
        print(f"forgot run {args.forget}")
        return 0

    if args.runs_root is None:
        parser.error("--runs-root is required unless --forget is given")
    runs_root: Path = args.runs_root
    if not runs_root.is_dir():
        parser.error(f"runs root is not a directory: {runs_root}")
    if args.watch:
        if rebuild:
            parser.error("--rebuild cannot be combined with --watch")
        return _run_watch(runs_root, db_path, interval=args.interval)
    summary = project_runs(runs_root, db_path, rebuild=rebuild)
    print(
        f"projected={summary.projected} skipped={summary.skipped} "
        f"ingest_errors={summary.ingest_errors} pruned={summary.pruned}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
