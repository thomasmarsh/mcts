"""Versioned lifecycle evidence for a tuning session."""

from __future__ import annotations

import json
import hashlib
import math
import os
import fcntl
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NewType, Sequence
from uuid import uuid4, uuid5, NAMESPACE_URL

from .config import json_default

LIFECYCLE_SCHEMA_VERSION: Final = 1

SessionId = NewType("SessionId", str)
AttemptId = NewType("AttemptId", str)
TrialId = NewType("TrialId", str)
EventId = NewType("EventId", str)

EVENT_TYPES: Final = frozenset(
    {
        "session_started",
        "attempt_started",
        "pool_revised",
        "trial_created",
        "trial_started",
        "trial_reported",
        "pair_started",
        "game_finished",
        "pair_finished",
        "pair_failed",
        "trial_completed",
        "trial_pruned",
        "trial_failed",
        "trial_cancelled",
        "attempt_completed",
        "attempt_failed",
        "attempt_stopped",
    }
)
TRIAL_TERMINAL_EVENTS: Final = frozenset(
    {"trial_completed", "trial_pruned", "trial_failed", "trial_cancelled"}
)


def make_attempt_id() -> AttemptId:
    """Return a fresh opaque identifier for one physical coordinator process."""
    return AttemptId(f"attempt-{uuid4().hex}")


def trial_id_for(session_id: SessionId, trial_number: int) -> TrialId:
    """Return a stable opaque identifier for an Optuna trial number."""
    value = uuid5(NAMESPACE_URL, f"mcts-tuner:{session_id}:trial:{trial_number}")
    return TrialId(f"trial-{value.hex}")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_json_value(value: Any) -> Any:
    """Convert lifecycle values to the portable JSON subset Rust accepts."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "infinity" if value > 0 else "-infinity"
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    try:
        return json_default(value)
    except TypeError:
        return value


def strict_json_dumps(value: Any, *, sort_keys: bool = False) -> str:
    """Serialize portable JSON without JavaScript-only non-finite numbers."""
    return json.dumps(
        normalize_json_value(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    )


def pool_snapshot_fingerprint(anchors: Sequence[Any]) -> str:
    """Fingerprint ordered matchmaking identity without provenance metadata."""
    snapshot = [
        {
            "anchor_id": anchor.id,
            "config": anchor.config,
            "mu": anchor.mu,
            "sigma": anchor.sigma,
        }
        for anchor in anchors
    ]
    return hashlib.sha256(
        strict_json_dumps(snapshot, sort_keys=True).encode()
    ).hexdigest()


class LifecycleWriter:
    """Append ordered lifecycle records for one attempt.

    Only the optimization coordinator owns an instance.  Workers return data to
    that coordinator and never open or sequence this file.
    """

    def __init__(self, path: str | Path, session_id: SessionId, attempt_id: AttemptId) -> None:
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.session_id = session_id
        self.attempt_id = attempt_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.lock_path.open("a", encoding="utf-8")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError(f"lifecycle journal is already locked: {self.path}") from None
        try:
            (
                self._sequence,
                self._session_started,
                self._manifest_fingerprint,
                self._terminal_trials,
            ) = self._existing_state()
            needs_separator = self._needs_record_separator()
            self._file = self.path.open("a", encoding="utf-8")
        except BaseException:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            raise
        if needs_separator:
            self._file.write("\n")
            self._file.flush()
            os.fsync(self._file.fileno())

    @property
    def has_session_started(self) -> bool:
        return self._session_started

    @property
    def manifest_fingerprint(self) -> str | None:
        """Return the immutable manifest fingerprint from the session start."""
        return self._manifest_fingerprint

    def _existing_state(self) -> tuple[int, bool, str | None, set[TrialId]]:
        if not self.path.exists():
            return 0, False, None, set()

        sequence = 0
        session_started = False
        manifest_fingerprint: str | None = None
        terminal_trials: set[TrialId] = set()
        with self.path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("session_id") != self.session_id:
                    raise ValueError(
                        f"lifecycle journal at {self.path} belongs to a different session"
                    )
                sequence = max(sequence, int(record.get("session_sequence", 0)))
                payload = record.get("payload")
                if record.get("event_type") == "session_started":
                    fingerprint = payload.get("manifest_fingerprint") if isinstance(payload, dict) else None
                    if session_started:
                        raise ValueError("lifecycle journal contains multiple session_started events")
                    session_started = True
                    manifest_fingerprint = fingerprint if isinstance(fingerprint, str) else None
                if record.get("event_type") in TRIAL_TERMINAL_EVENTS and isinstance(payload, dict):
                    trial_id = payload.get("trial_id")
                    if isinstance(trial_id, str):
                        terminal_trials.add(TrialId(trial_id))
        return sequence, session_started, manifest_fingerprint, terminal_trials

    def _needs_record_separator(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with self.path.open("rb") as source:
            source.seek(-1, os.SEEK_END)
            return source.read(1) != b"\n"

    def emit(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Write and flush one versioned lifecycle event."""
        if event_type == "session_started" and self._session_started:
            raise ValueError("session_started may occur only once")
        if event_type == "pool_revised" and not self._session_started:
            raise ValueError("pool_revised requires session_started")
        record = self._build_record(event_type, payload)
        self._append_record(record)
        if event_type == "session_started":
            self._session_started = True
            fingerprint = payload.get("manifest_fingerprint")
            self._manifest_fingerprint = fingerprint if isinstance(fingerprint, str) else None
        return record

    def _build_record(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate one event and assign its next coordinator-owned sequence."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported lifecycle event type {event_type!r}")
        self._sequence += 1
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "event_id": EventId(f"event-{uuid4().hex}"),
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "session_sequence": self._sequence,
            "timestamp": _timestamp(),
            "event_type": event_type,
            "payload": payload,
        }

    def _append_record(self, record: dict[str, Any]) -> None:
        """Append one serialized record and make it observable to the ingester."""
        self._file.write(strict_json_dumps(record) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def emit_trial_terminal(
        self, event_type: str, trial_id: TrialId, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Write the sole terminal record for a trial."""
        if event_type not in TRIAL_TERMINAL_EVENTS:
            raise ValueError(f"{event_type!r} is not a trial terminal event")
        if trial_id in self._terminal_trials:
            raise ValueError(f"trial {trial_id} already has terminal lifecycle evidence")
        record = self.emit(event_type, {"trial_id": trial_id, **payload})
        self._terminal_trials.add(trial_id)
        return record

    def has_trial_terminal(self, trial_id: TrialId) -> bool:
        return trial_id in self._terminal_trials

    def close(self) -> None:
        self._file.close()
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
        self._lock.close()

    def __enter__(self) -> LifecycleWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
