"""``tuner-project`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build import project_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tuner-project",
        description="Build a rebuildable read-only SQLite projection of version-4 tuner runs.",
    )
    parser.add_argument("--runs-root", required=True, type=Path, help="directory of run subdirs")
    parser.add_argument("--db", required=True, type=Path, help="SQLite projection file to write")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the file and re-project every run instead of skipping unchanged runs",
    )
    args = parser.parse_args(argv)
    runs_root: Path = args.runs_root
    db_path: Path = args.db
    rebuild: bool = args.rebuild
    if not runs_root.is_dir():
        parser.error(f"runs root is not a directory: {runs_root}")
    summary = project_runs(runs_root, db_path, rebuild=rebuild)
    print(
        f"projected={summary.projected} skipped={summary.skipped} "
        f"ingest_errors={summary.ingest_errors} pruned={summary.pruned}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
