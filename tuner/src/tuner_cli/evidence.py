"""Strict version-4 append-only evidence records and atomic file publishing."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .artifacts import SCHEMA_VERSION
from .codec import JsonObject, JsonValue, integer, object_fields, strict_json, string
from .domain import DiagnosticPairResult, GameResult, PairResult, PairTask
from .event_payloads import (
    SCIENTIFIC,
    DiagnosticPairCompletedPayload,
    EventPayload,
    EventType,
    PairCompletedPayload,
    PairIdentity,
    decode_payload,
    narrow_event_type,
)
from .identity import canonical_json, game_id
from .statistics import pair_utility
from .target import parse_pair_output

__all__ = [
    "SCIENTIFIC",
    "EventPayload",
    "EventType",
    "EvidenceEvent",
    "EvidenceWriter",
    "atomic_json",
    "decode_event",
    "decode_pair_payload",
    "game_payload",
    "pair_payload",
    "read_events",
    "scientific_projection",
    "write_manifest",
]


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    sequence: int
    type: EventType
    payload: EventPayload

    def encoded(self) -> JsonObject:
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "type": self.type,
            "payload": self.payload.encode(),
        }


def atomic_json(path: Path, value: JsonValue, *, create_once: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical_json(value) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if create_once:
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_manifest(path: Path, manifest: JsonObject) -> JsonObject:
    if "fingerprint" not in manifest:
        raise ValueError("manifest must be decoded and fingerprinted before publishing")
    atomic_json(path, manifest, create_once=True)
    return dict(manifest)


def _object(value: object, fields: set[str], label: str) -> JsonObject:
    return object_fields(value, fields, label)


def _int(value: object, label: str, *, positive: bool = False) -> int:
    return integer(value, label, positive=positive)


def _string(value: object, label: str) -> str:
    return string(value, label)


def game_payload(game: GameResult) -> JsonObject:
    return {
        "game_id": game.game_id,
        "candidate_side": game.candidate_side,
        "outcome": game.outcome,
        "derived_seed": game.derived_seed,
        "round": game.round,
        "seq": game.seq,
        "trace_game_seq": game.trace_game_seq,
        "plies": game.plies,
        "elapsed_ms": game.elapsed_ms,
        "candidate_metrics": {
            "iterations_total": game.candidate_metrics.iterations_total,
            "iterations_first_half": game.candidate_metrics.iterations_first_half,
            "move_time_ms": game.candidate_metrics.move_time_ms,
        },
        "opponent_metrics": {
            "iterations_total": game.opponent_metrics.iterations_total,
            "iterations_first_half": game.opponent_metrics.iterations_first_half,
            "move_time_ms": game.opponent_metrics.move_time_ms,
        },
        "raw_record": game.raw_record,
    }


def pair_payload(result: PairResult) -> PairCompletedPayload:
    task = result.task
    return PairCompletedPayload(
        PairIdentity(
            task.task_case.phase,
            task.candidate_id,
            task.task_case.task_id,
            task.pair_id,
            task.task_case.opponent_id,
            task.budget,
        ),
        (game_payload(result.games[0]), game_payload(result.games[1])),
        pair_utility(result),
    )


def diagnostic_pair_payload(result: DiagnosticPairResult) -> DiagnosticPairCompletedPayload:
    return DiagnosticPairCompletedPayload(
        result.task, (game_payload(result.games[0]), game_payload(result.games[1]))
    )


def decode_pair_payload(payload: PairCompletedPayload, task: PairTask) -> PairResult:
    identity = payload.identity
    if (
        identity.phase != task.task_case.phase
        or identity.candidate_id != task.candidate_id
        or identity.task_id != task.task_case.task_id
        or identity.pair_id != task.pair_id
        or identity.opponent_id != task.task_case.opponent_id
        or identity.search_effort != task.budget
    ):
        raise ValueError("pair completion does not match expected pair")
    raw_records: list[JsonObject] = []
    game_fields = {
        "game_id",
        "candidate_side",
        "outcome",
        "derived_seed",
        "round",
        "seq",
        "trace_game_seq",
        "plies",
        "elapsed_ms",
        "candidate_metrics",
        "opponent_metrics",
        "raw_record",
    }
    for position, encoded in enumerate(payload.games):
        game = _object(encoded, game_fields, "completed game")
        raw = _string(game["raw_record"], "raw game record")
        parsed = strict_json(raw, "raw game record")
        if not isinstance(parsed, dict) or canonical_json(parsed) != raw:
            raise ValueError("raw game record must be a canonical JSON object")
        raw_records.append(parsed)
        side = ("first", "second")[position]
        if game["candidate_side"] != side or game["game_id"] != game_id(task, side):
            raise ValueError("completed games are not in deterministic seat order")
    outcomes = [string(record["outcome"], "raw game outcome") for record in raw_records]
    summary = {
        "type": "configured_comparison_summary",
        "games": 2,
        "wins": outcomes.count("candidate_win"),
        "losses": outcomes.count("baseline_win"),
        "draws": outcomes.count("draw"),
    }
    decoded = parse_pair_output("\n".join(canonical_json(x) for x in [*raw_records, summary]), task)
    if not isinstance(decoded, PairResult):
        raise ValueError("objective completion decoded as diagnostic pair")
    for expected, actual in zip(payload.games, decoded.games, strict=True):
        if expected != game_payload(actual):
            raise ValueError("typed game fields disagree with raw game record")
    if payload.pair_utility != pair_utility(decoded):
        raise ValueError("pair utility disagrees with raw games")
    return decoded


def decode_event(value: object, expected_sequence: int | None = None) -> EvidenceEvent:
    item = _object(value, {"schema_version", "sequence", "type", "payload"}, "evidence event")
    if item["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported evidence schema version: {item['schema_version']!r}")
    sequence = _int(item["sequence"], "evidence sequence", positive=True)
    if expected_sequence is not None and sequence != expected_sequence:
        raise ValueError("evidence sequence is not contiguous")
    event_type = narrow_event_type(item["type"])
    return EvidenceEvent(sequence, event_type, decode_payload(event_type, item["payload"]))


def read_events(path: Path) -> list[EvidenceEvent]:
    if not path.is_file():
        raise ValueError(f"missing evidence log: {path}")
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("evidence log has an unterminated final line")
    if "\n\n" in text:
        raise ValueError("evidence log contains a blank line")
    return [
        decode_event(strict_json(line, "evidence line"), sequence)
        for sequence, line in enumerate(text.splitlines(), 1)
    ]


class EvidenceWriter:
    def __init__(self, path: Path, *, reopen: bool = False) -> None:
        self.path = path
        if reopen:
            self._sequence = len(read_events(path))
        else:
            self._sequence = 0
            with path.open("x", encoding="utf-8"):
                pass

    @classmethod
    def open(cls, path: Path) -> EvidenceWriter:
        return cls(path, reopen=True)

    @property
    def sequence(self) -> int:
        return self._sequence

    def append(self, payload: EventPayload) -> EvidenceEvent:
        event = EvidenceEvent(self._sequence + 1, payload.event_type, payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event.encoded()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._sequence = event.sequence
        return event


def scientific_projection(events: list[EvidenceEvent]) -> str:
    return canonical_json(
        [
            {"type": event.type, "payload": event.payload.encode()}
            for event in events
            if event.type in SCIENTIFIC
        ]
    )
