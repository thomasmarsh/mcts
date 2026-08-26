"""Versioned lifecycle evidence for a tuning session."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NewType
from uuid import NAMESPACE_URL, uuid4, uuid5

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
        "attempt_recovered",
        "pool_revised",
        "pool_anchor_decided",
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
ATTEMPT_TERMINAL_EVENTS: Final = frozenset(
    {"attempt_completed", "attempt_failed", "attempt_stopped"}
)
PAIR_TERMINAL_EVENTS: Final = frozenset({"pair_finished", "pair_failed"})


@dataclass(frozen=True)
class RecoveredTrial:
    """One incomplete trial that must not be scheduled again."""

    trial_id: TrialId
    trial_number: int


@dataclass(frozen=True)
class OrphanedAttempt:
    """The latest attempt whose journal lacks terminal lifecycle evidence."""

    attempt_id: AttemptId
    bench_run_id: str | None
    trials: tuple[RecoveredTrial, ...]
    pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class JournalSnapshot:
    """Pure replay of attempts and their nonterminal work in one journal."""

    orphaned_attempt: OrphanedAttempt | None


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


def replay_journal(path: str | Path, session_id: SessionId) -> JournalSnapshot:
    """Return the sole recoverable prior attempt without changing the journal.

    A final partial line is ignored because a coordinator can die between a
    write and its newline.  Complete records must still agree on ownership and
    the deterministic trial identity; guessing at either would make recovery
    less safe than stopping before new work is scheduled.
    """
    path = Path(path)
    attempts: dict[str, dict[str, Any]] = {}
    trials: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return JournalSnapshot(None)

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("session_id") != session_id:
            raise ValueError("lifecycle journal belongs to a different session")
        event_type = record.get("event_type")
        attempt_id = record.get("attempt_id")
        payload = record.get("payload")
        if not isinstance(attempt_id, str) or not isinstance(payload, dict):
            raise ValueError("lifecycle journal has an invalid event envelope")
        if event_type == "attempt_started":
            if attempt_id in attempts:
                raise ValueError(
                    "lifecycle journal contains duplicate attempt_started evidence"
                )
            bench_run_id = payload.get("bench_run_id", payload.get("run_id"))
            if bench_run_id is not None and not isinstance(bench_run_id, str):
                raise ValueError("lifecycle journal has an invalid bench run identity")
            attempts[attempt_id] = {
                "sequence": record.get("session_sequence"),
                "bench_run_id": bench_run_id,
                "terminal": False,
            }
            continue
        if event_type == "attempt_recovered":
            prior = payload.get("prior_attempt_id")
            if (
                attempt_id not in attempts
                or not isinstance(prior, str)
                or prior not in attempts
                or attempts[prior]["terminal"]
            ):
                raise ValueError(
                    "recovery evidence references an invalid prior attempt"
                )
            recovered_trials = payload.get("trials")
            recovered_pairs = payload.get("pair_ids")
            if not isinstance(recovered_trials, list) or not isinstance(
                recovered_pairs, list
            ):
                raise ValueError("recovery evidence has an invalid scope")
            if any(
                not isinstance(item, dict)
                or not isinstance(item.get("trial_id"), str)
                or not isinstance(item.get("trial_number"), int)
                or isinstance(item.get("trial_number"), bool)
                for item in recovered_trials
            ) or any(not isinstance(pair_id, str) for pair_id in recovered_pairs):
                raise ValueError("recovery evidence does not match prior work")
            expected_trials = {
                (trial_id, trial["number"])
                for trial_id, trial in trials.items()
                if trial["attempt_id"] == prior and not trial["terminal"]
            }
            listed_trials = {
                (item["trial_id"], item["trial_number"]) for item in recovered_trials
            }
            expected_pairs = {
                pair_id
                for pair_id, pair in pairs.items()
                if pair["attempt_id"] == prior and not pair["terminal"]
            }
            if (
                len(listed_trials) != len(recovered_trials)
                or listed_trials != expected_trials
                or len(set(recovered_pairs)) != len(recovered_pairs)
                or set(recovered_pairs) != expected_pairs
            ):
                raise ValueError("recovery evidence does not match prior work")
            attempts[prior]["terminal"] = True
            continue
        if event_type in ATTEMPT_TERMINAL_EVENTS:
            if attempt_id not in attempts:
                raise ValueError("attempt terminal evidence precedes attempt_started")
            attempts[attempt_id]["terminal"] = True
            continue
        if event_type not in {
            "trial_created",
            "trial_started",
            "trial_reported",
            *TRIAL_TERMINAL_EVENTS,
            "pair_started",
            "game_finished",
            *PAIR_TERMINAL_EVENTS,
        }:
            continue
        if attempt_id not in attempts:
            raise ValueError("work evidence precedes attempt_started")
        trial_id = payload.get("trial_id")
        if not isinstance(trial_id, str):
            raise ValueError("work evidence is missing its trial identity")
        if event_type == "trial_created":
            number = payload.get("trial_number")
            if not isinstance(number, int) or isinstance(number, bool):
                raise ValueError("trial_created has an invalid trial number")
            if trial_id != trial_id_for(session_id, number):
                raise ValueError(
                    "trial id does not match its deterministic trial number"
                )
            if trial_id in trials:
                raise ValueError(
                    "lifecycle journal contains duplicate trial_created evidence"
                )
            trials[trial_id] = {
                "attempt_id": attempt_id,
                "number": number,
                "terminal": False,
            }
            continue
        trial = trials.get(trial_id)
        if trial is None or trial["attempt_id"] != attempt_id:
            raise ValueError("trial evidence has conflicting attempt ownership")
        number = payload.get("trial_number")
        if number is not None and number != trial["number"]:
            raise ValueError("trial evidence has a conflicting trial number")
        if event_type in TRIAL_TERMINAL_EVENTS:
            if trial["terminal"]:
                raise ValueError(
                    "lifecycle journal contains duplicate trial terminal evidence"
                )
            trial["terminal"] = True
            continue
        if event_type == "pair_started":
            pair_id = payload.get("pair_id")
            if not isinstance(pair_id, str) or pair_id in pairs:
                raise ValueError(
                    "pair_started has an invalid or duplicate pair identity"
                )
            pairs[pair_id] = {
                "attempt_id": attempt_id,
                "trial_id": trial_id,
                "terminal": False,
            }
            continue
        if event_type in {"game_finished", *PAIR_TERMINAL_EVENTS}:
            pair_id = payload.get("pair_id")
            pair = pairs.get(pair_id) if isinstance(pair_id, str) else None
            if (
                pair is None
                or pair["attempt_id"] != attempt_id
                or pair["trial_id"] != trial_id
            ):
                raise ValueError("pair evidence has conflicting ownership")
            if event_type in PAIR_TERMINAL_EVENTS:
                if pair["terminal"]:
                    raise ValueError(
                        "lifecycle journal contains duplicate pair terminal evidence"
                    )
                pair["terminal"] = True

    open_attempts = [
        (attempt_id, value)
        for attempt_id, value in attempts.items()
        if not value["terminal"]
    ]
    if not open_attempts:
        return JournalSnapshot(None)
    if len(open_attempts) != 1:
        raise ValueError("lifecycle journal has multiple unterminated attempts")
    attempt_id, attempt = open_attempts[0]
    recovered_trials = tuple(
        RecoveredTrial(TrialId(trial_id), trial["number"])
        for trial_id, trial in trials.items()
        if trial["attempt_id"] == attempt_id and not trial["terminal"]
    )
    pair_ids = tuple(
        pair_id
        for pair_id, pair in pairs.items()
        if pair["attempt_id"] == attempt_id and not pair["terminal"]
    )
    return JournalSnapshot(
        OrphanedAttempt(
            AttemptId(attempt_id), attempt["bench_run_id"], recovered_trials, pair_ids
        )
    )


class LifecycleWriter:
    """Append ordered lifecycle records for one attempt.

    Only the optimization coordinator owns an instance.  Workers return data to
    that coordinator and never open or sequence this file.
    """

    def __init__(
        self, path: str | Path, session_id: SessionId, attempt_id: AttemptId
    ) -> None:
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
            raise RuntimeError(
                f"lifecycle journal is already locked: {self.path}"
            ) from None
        try:
            (
                self._sequence,
                self._session_started,
                self._manifest_fingerprint,
                self._terminal_trials,
            ) = self._existing_state()
            self._journal_snapshot = replay_journal(self.path, self.session_id)
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

    @property
    def journal_snapshot(self) -> JournalSnapshot:
        """Return the immutable replay captured while this writer acquired the lock."""
        return self._journal_snapshot

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
                    fingerprint = (
                        payload.get("manifest_fingerprint")
                        if isinstance(payload, dict)
                        else None
                    )
                    if session_started:
                        raise ValueError(
                            "lifecycle journal contains multiple session_started events"
                        )
                    session_started = True
                    manifest_fingerprint = (
                        fingerprint if isinstance(fingerprint, str) else None
                    )
                if record.get("event_type") in TRIAL_TERMINAL_EVENTS and isinstance(
                    payload, dict
                ):
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
            self._manifest_fingerprint = (
                fingerprint if isinstance(fingerprint, str) else None
            )
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
            raise ValueError(
                f"trial {trial_id} already has terminal lifecycle evidence"
            )
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
