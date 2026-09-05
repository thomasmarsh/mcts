"""SQLite read-model persistence: DDL application, per-run transactional
replacement, change-fingerprint bookkeeping, and a canonical dump.

No scientific logic lives here -- the store only writes the frozen row
dataclasses that ``rows.py`` produces.
"""

from __future__ import annotations

import pickle
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tuner_cli.replay import ReplayCheckpoint

from .rows import (
    ActiveEliminationDecisionRow,
    CandidateRow,
    CohortRow,
    ComputePhaseRow,
    GameRow,
    ObservationRow,
    PairRow,
    ProposalRow,
    RunManifestRow,
    RunReportRow,
    RunRow,
    ShadowDecisionRow,
    ValidationRow,
)
from .schema import CONTENT_TABLES, DDL, PROJECTION_SCHEMA_VERSION

# Bumped whenever `ReplayCheckpoint`'s pickled shape changes incompatibly. A
# mismatch (or any other unpickling problem) is treated as "no checkpoint" --
# the next pass falls back to a full read+replay and writes a fresh one, so
# this is a perf-only fallback, never a correctness one.
CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ChangeFingerprint:
    evidence_size: int
    evidence_mtime_ns: int
    manifest_fingerprint: str


@dataclass(frozen=True, slots=True)
class RunProjection:
    """Every row a single run contributes, keyed the way the tables are keyed."""

    run: RunRow
    fingerprint: ChangeFingerprint
    manifest: RunManifestRow | None
    report: RunReportRow | None
    cohorts: Sequence[CohortRow] = ()
    candidates: Sequence[CandidateRow] = ()
    proposals: Sequence[ProposalRow] = ()
    pairs: Sequence[PairRow] = ()
    games: Sequence[GameRow] = ()
    observations: Sequence[ObservationRow] = ()
    shadow_decisions: Sequence[ShadowDecisionRow] = ()
    active_elimination_decisions: Sequence[ActiveEliminationDecisionRow] = ()
    validation_rows: Sequence[ValidationRow] = ()
    compute_phases: Sequence[ComputePhaseRow] = ()


_CHILD_TABLES: tuple[str, ...] = (
    "run_manifest",
    "run_report",
    "cohorts",
    "candidates",
    "proposals",
    "pairs",
    "games",
    "observations",
    "shadow_decisions",
    "active_elimination_decisions",
    "validation_rows",
    "compute_phases",
)


class Store:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def close(self) -> None:
        self._connection.close()

    def fingerprint(self, run_id: str) -> ChangeFingerprint | None:
        cursor = self._connection.execute(
            "SELECT evidence_size, evidence_mtime_ns, manifest_fingerprint "
            "FROM ingest_state WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return ChangeFingerprint(int(row[0]), int(row[1]), str(row[2]))

    def projected_run_ids(self) -> list[str]:
        cursor = self._connection.execute("SELECT run_id FROM runs ORDER BY run_id")
        return [str(item[0]) for item in cursor.fetchall()]

    def delete_run(self, run_id: str) -> None:
        with self._connection:
            for table in (*_CHILD_TABLES, "runs", "ingest_state", "run_checkpoints"):
                self._connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))

    def checkpoint(
        self, run_id: str, *, manifest_fingerprint: str
    ) -> tuple[int, ReplayCheckpoint] | None:
        """`(last_sequence, checkpoint)` for `run_id`, or `None` if there isn't
        a usable one.

        `None` covers every reason a checkpoint can't be trusted: none exists
        yet, its `checkpoint_version` doesn't match this build, it was taken
        against a different manifest, or the pickled blob fails to load (a
        stale/foreign format). Every case is a full-replay fallback, not an
        error -- see `CHECKPOINT_VERSION`.
        """
        cursor = self._connection.execute(
            "SELECT checkpoint_version, last_sequence, manifest_fingerprint, state "
            "FROM run_checkpoints WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        version, last_sequence, stored_fingerprint, blob = row
        if int(version) != CHECKPOINT_VERSION or str(stored_fingerprint) != manifest_fingerprint:
            return None
        try:
            checkpoint = pickle.loads(blob)
        except Exception:
            return None
        if not isinstance(checkpoint, ReplayCheckpoint):
            return None
        return int(last_sequence), checkpoint

    def save_checkpoint(
        self,
        run_id: str,
        *,
        last_sequence: int,
        manifest_fingerprint: str,
        checkpoint: ReplayCheckpoint,
    ) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO run_checkpoints VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "checkpoint_version = excluded.checkpoint_version, "
                "last_sequence = excluded.last_sequence, "
                "manifest_fingerprint = excluded.manifest_fingerprint, "
                "state = excluded.state",
                (
                    run_id,
                    CHECKPOINT_VERSION,
                    last_sequence,
                    manifest_fingerprint,
                    pickle.dumps(checkpoint),
                ),
            )

    def replace_run(self, projection: RunProjection) -> None:
        run_id = projection.run.run_id
        with self._connection:
            for table in (*_CHILD_TABLES, "runs", "ingest_state"):
                self._connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            _insert(self._connection, "runs", [projection.run])
            manifest = [projection.manifest] if projection.manifest is not None else []
            report = [projection.report] if projection.report is not None else []
            _insert(self._connection, "run_manifest", manifest)
            _insert(self._connection, "run_report", report)
            _insert(self._connection, "cohorts", projection.cohorts)
            _insert(self._connection, "candidates", projection.candidates)
            _insert(self._connection, "proposals", projection.proposals)
            _insert(self._connection, "pairs", projection.pairs)
            _insert(self._connection, "games", projection.games)
            _insert(self._connection, "observations", projection.observations)
            _insert(self._connection, "shadow_decisions", projection.shadow_decisions)
            _insert(
                self._connection,
                "active_elimination_decisions",
                projection.active_elimination_decisions,
            )
            _insert(self._connection, "validation_rows", projection.validation_rows)
            _insert(self._connection, "compute_phases", projection.compute_phases)
            self._connection.execute(
                "INSERT INTO ingest_state VALUES (?, ?, ?, ?)",
                (
                    run_id,
                    projection.fingerprint.evidence_size,
                    projection.fingerprint.evidence_mtime_ns,
                    projection.fingerprint.manifest_fingerprint,
                ),
            )

    def vacuum(self) -> None:
        self._connection.execute("VACUUM")

    def record_pass(self, timestamp: str) -> None:
        """Stamp the wall-clock time this projection pass committed.

        Read by the fleet API so "projection refreshed N s ago" reflects the
        headless follower's last pass, not the last time a browser tab was
        open. Deliberately excluded from :meth:`canonical_dump` -- it is
        machine- and clock-specific, like ``ingest_state``.
        """
        with self._connection:
            self._connection.execute(
                "INSERT INTO projection_meta VALUES ('last_pass_at', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (timestamp,),
            )

    def last_pass_at(self) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM projection_meta WHERE key = 'last_pass_at'"
        ).fetchone()
        return None if row is None else str(row[0])

    def canonical_dump(self) -> str:
        lines: list[str] = []
        for table in CONTENT_TABLES:
            columns = _column_names(self._connection, table)
            order = ", ".join(str(index + 1) for index in range(len(columns)))
            cursor = self._connection.execute(f"SELECT * FROM {table} ORDER BY {order}")
            for row in cursor.fetchall():
                # The follower's last-pass stamp is wall-clock state, not
                # projected content -- keep it out of the golden dump.
                if table == "projection_meta" and row and row[0] == "last_pass_at":
                    continue
                rendered = ", ".join(_render(value) for value in row)
                lines.append(f"{table}({', '.join(columns)}): {rendered}")
        return "\n".join(lines) + "\n"


def _column_names(connection: sqlite3.Connection, table: str) -> list[str]:
    cursor = connection.execute(f"PRAGMA table_info({table})")
    return [str(item[1]) for item in cursor.fetchall()]


def _render(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"unexpected column type: {type(value)!r}")


def _insert(connection: sqlite3.Connection, table: str, rows: Sequence[object]) -> None:
    if not rows:
        return
    columns = _column_names(connection, table)
    placeholders = ", ".join("?" for _ in columns)
    payload = [tuple(_field(row, column) for column in columns) for row in rows]
    connection.executemany(f"INSERT INTO {table} VALUES ({placeholders})", payload)


def _field(row: object, name: str) -> str | int | float | None:
    value = getattr(row, name)
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise TypeError(f"row field {name!r} is not a SQLite scalar: {type(value)!r}")


def open_store(db_path: Path) -> Store:
    if db_path.exists() and _schema_version(db_path) != PROJECTION_SCHEMA_VERSION:
        # The projection is a rebuildable read model, so a file left behind by
        # an older schema is not an error: drop it and re-project from scratch.
        db_path.unlink()
    fresh = not db_path.exists()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    if fresh:
        connection.executescript(DDL)
        connection.execute(
            "INSERT INTO projection_meta VALUES ('projection_schema_version', ?)",
            (str(PROJECTION_SCHEMA_VERSION),),
        )
        connection.commit()
    _check_schema_version(connection)
    return Store(connection)


def _schema_version(db_path: Path) -> int | None:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.execute(
            "SELECT value FROM projection_meta WHERE key = 'projection_schema_version'"
        )
        row = cursor.fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        connection.close()
    return int(row[0]) if row is not None else None


def _check_schema_version(connection: sqlite3.Connection) -> None:
    cursor = connection.execute(
        "SELECT value FROM projection_meta WHERE key = 'projection_schema_version'"
    )
    row = cursor.fetchone()
    if row is None or int(row[0]) != PROJECTION_SCHEMA_VERSION:
        raise ValueError("projection database has an incompatible schema version")
