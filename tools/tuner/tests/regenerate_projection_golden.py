"""Regenerate the checked-in projection golden dump.

Run from the tuner project root::

    uv run python tests/regenerate_projection_golden.py

The dump is produced only from the checked-in version-4 run fixtures via the
public replay and report codecs, so any diff here reflects a real change to what
the projection would materialize.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tuner_projection.build import project_runs
from tuner_projection.store import open_store

PROJECTION_ROOT = Path(__file__).parent / "fixtures" / "projection-root"
GOLDEN = Path(__file__).parent / "fixtures" / "projection" / "version4.dump.sql"


def build_dump() -> str:
    with tempfile.TemporaryDirectory() as raw:
        db_path = Path(raw) / "projection.sqlite"
        project_runs(PROJECTION_ROOT, db_path, rebuild=True)
        store = open_store(db_path)
        try:
            return store.canonical_dump()
        finally:
            store.close()


def main() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(build_dump(), encoding="utf-8")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
