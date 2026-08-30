"""Strict version-2 append-only evidence records and atomic file publishing."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .artifacts import SCHEMA_VERSION, strict_json
from .domain import GameResult, PairResult, PairTask
from .identity import canonical_json, game_id
from .statistics import pair_utility
from .target import parse_pair_output

EventType = Literal[
    "proposal_created",
    "proposal_accepted",
    "proposal_rejected",
    "cohort_accepted",
    "pair_started",
    "pair_completed",
    "pair_failed",
    "run_interrupted",
    "run_failed",
    "observation_completed",
    "finalists_selected",
    "run_completed",
]
SCIENTIFIC = {
    "proposal_created",
    "proposal_accepted",
    "proposal_rejected",
    "cohort_accepted",
    "pair_completed",
    "observation_completed",
    "finalists_selected",
    "run_completed",
}


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    sequence: int
    type: EventType
    payload: dict[str, object]

    def encoded(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "type": self.type,
            "payload": self.payload,
        }


def atomic_json(path: Path, value: object, *, create_once: bool = False) -> None:
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


def write_manifest(path: Path, manifest: object) -> dict[str, object]:
    if not isinstance(manifest, dict) or "fingerprint" not in manifest:
        raise ValueError("manifest must be decoded and fingerprinted before publishing")
    atomic_json(path, manifest, create_once=True)
    return dict(manifest)


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} has invalid fields")
    return value


def _int(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        raise ValueError(f"{label} must be{' positive' if positive else ' an'} integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _candidate_payload(value: object, *, disposition: bool = False) -> dict[str, object]:
    fields = {
        "proposal_index",
        "source",
        "proposer_version",
        "candidate_id",
        "fingerprint",
        "canonical_config",
    }
    if disposition:
        fields = {"proposal_index", "candidate_id", "fingerprint", "canonical_config"}
    item = _object(value, fields, "proposal payload")
    _int(item["proposal_index"], "proposal index")
    for key in fields - {"proposal_index"}:
        _string(item[key], key)
    if not disposition and item["source"] not in {"schema_default", "configspace_random"}:
        raise ValueError("unknown proposal source")
    return item


def _pair_identity(value: object, *, seed: bool = False) -> dict[str, object]:
    fields = {"phase", "candidate_id", "task_id", "pair_id", "opponent_id", "budget"}
    if seed:
        fields.add("task_seed")
    item = _object(value, fields, "pair identity")
    if item["phase"] not in {"tuning", "validation"}:
        raise ValueError("invalid pair phase")
    for key in fields - {"budget", "task_seed"}:
        _string(item[key], key)
    _int(item["budget"], "pair budget", positive=True)
    if seed:
        _int(item["task_seed"], "task seed")
    return item


def game_payload(game: GameResult) -> dict[str, object]:
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


def pair_payload(result: PairResult) -> dict[str, object]:
    task = result.task
    return {
        "phase": task.task_case.phase,
        "candidate_id": task.candidate_id,
        "task_id": task.task_case.task_id,
        "pair_id": task.pair_id,
        "opponent_id": task.task_case.opponent_id,
        "budget": task.budget.max_iterations,
        "games": [game_payload(game) for game in result.games],
        "pair_utility": pair_utility(result),
    }


def decode_pair_payload(payload: object, task: PairTask) -> PairResult:
    item = _object(
        payload,
        {
            "phase",
            "candidate_id",
            "task_id",
            "pair_id",
            "opponent_id",
            "budget",
            "games",
            "pair_utility",
        },
        "pair completion",
    )
    actual_identity = {
        "phase": task.task_case.phase,
        "candidate_id": task.candidate_id,
        "task_id": task.task_case.task_id,
        "pair_id": task.pair_id,
        "opponent_id": task.task_case.opponent_id,
        "budget": task.budget.max_iterations,
    }
    if (
        {key: item[key] for key in actual_identity} != actual_identity
        or not isinstance(item["games"], list)
        or len(item["games"]) != 2
    ):
        raise ValueError("pair completion does not match expected pair")
    raw_records: list[object] = []
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
    for position, encoded in enumerate(item["games"]):
        game = _object(encoded, game_fields, "completed game")
        raw = _string(game["raw_record"], "raw game record")
        parsed = strict_json(raw, "raw game record")
        if not isinstance(parsed, dict) or canonical_json(parsed) != raw:
            raise ValueError("raw game record must be a canonical JSON object")
        raw_records.append(parsed)
        side = ("first", "second")[position]
        if game["candidate_side"] != side or game["game_id"] != game_id(task, side):
            raise ValueError("completed games are not in deterministic seat order")
    outcomes = [record["outcome"] for record in raw_records if isinstance(record, dict)]
    summary = {
        "type": "configured_comparison_summary",
        "games": 2,
        "wins": outcomes.count("candidate_win"),
        "losses": outcomes.count("baseline_win"),
        "draws": outcomes.count("draw"),
    }
    decoded = parse_pair_output("\n".join(canonical_json(x) for x in [*raw_records, summary]), task)
    for expected, actual in zip(item["games"], decoded.games, strict=True):
        if expected != game_payload(actual):
            raise ValueError("typed game fields disagree with raw game record")
    utility = item["pair_utility"]
    if (
        not isinstance(utility, (int, float))
        or isinstance(utility, bool)
        or utility != pair_utility(decoded)
    ):
        raise ValueError("pair utility disagrees with raw games")
    return decoded


def _validate_payload(event_type: str, payload: object) -> dict[str, object]:
    if event_type == "proposal_created":
        return _candidate_payload(payload)
    if event_type == "proposal_accepted":
        return _candidate_payload(payload, disposition=True)
    if event_type == "proposal_rejected":
        item = _object(
            payload,
            {
                "proposal_index",
                "candidate_id",
                "fingerprint",
                "canonical_config",
                "reason",
                "errors",
            },
            "proposal rejection",
        )
        _candidate_payload(
            {
                key: item[key]
                for key in {"proposal_index", "candidate_id", "fingerprint", "canonical_config"}
            },
            disposition=True,
        )
        if item["reason"] not in {"duplicate", "semantic_validation"} or not isinstance(
            item["errors"], list
        ):
            raise ValueError("invalid proposal rejection")
        return item
    if event_type == "cohort_accepted":
        item = _object(payload, {"candidate_ids", "validation_response_fingerprint"}, "cohort")
        if (
            not isinstance(item["candidate_ids"], list)
            or not all(isinstance(x, str) for x in item["candidate_ids"])
            or not isinstance(item["validation_response_fingerprint"], str)
        ):
            raise ValueError("invalid cohort")
        return item
    if event_type == "pair_started":
        return _pair_identity(payload, seed=True)
    if event_type == "pair_completed":
        return _object(
            payload,
            {
                "phase",
                "candidate_id",
                "task_id",
                "pair_id",
                "opponent_id",
                "budget",
                "games",
                "pair_utility",
            },
            "pair completion",
        )
    if event_type == "pair_failed":
        item = _object(
            payload,
            {
                "phase",
                "candidate_id",
                "task_id",
                "pair_id",
                "opponent_id",
                "budget",
                "kind",
                "command",
                "returncode",
                "stderr",
                "stdout",
                "partial_output",
            },
            "pair failure",
        )
        _pair_identity(
            {
                key: item[key]
                for key in {"phase", "candidate_id", "task_id", "pair_id", "opponent_id", "budget"}
            }
        )
        if (
            not isinstance(item["command"], list)
            or not all(isinstance(x, str) for x in item["command"])
            or (
                item["returncode"] is not None
                and (
                    not isinstance(item["returncode"], int) or isinstance(item["returncode"], bool)
                )
            )
            or not all(isinstance(item[key], str) for key in {"kind", "stderr", "stdout"})
            or not isinstance(item["partial_output"], list)
        ):
            raise ValueError("invalid pair failure")
        return item
    if event_type == "run_interrupted":
        item = _object(payload, {"stage", "pair_id"}, "interruption")
        if not isinstance(item["stage"], str) or (
            item["pair_id"] is not None and not isinstance(item["pair_id"], str)
        ):
            raise ValueError("invalid interruption")
        return item
    if event_type == "run_failed":
        item = _object(payload, {"kind", "message"}, "run failure")
        if item["kind"] != "configuration" or not isinstance(item["message"], str):
            raise ValueError("invalid run failure")
        return item
    if event_type == "observation_completed":
        item = _object(
            payload,
            {
                "candidate_id",
                "phase",
                "block_id",
                "prefix_length",
                "budget",
                "pair_utilities",
                "estimate",
                "counts",
            },
            "observation",
        )
        if (
            item["phase"] not in {"tuning", "validation"}
            or not isinstance(item["candidate_id"], str)
            or not isinstance(item["block_id"], str)
            or not isinstance(item["pair_utilities"], list)
        ):
            raise ValueError("invalid observation")
        _int(item["prefix_length"], "prefix length", positive=True)
        _int(item["budget"], "observation budget", positive=True)
        return item
    if event_type == "finalists_selected":
        item = _object(
            payload,
            {
                "finalist_ids",
                "tuning_estimates",
                "source_block",
                "budget",
                "selection_rule_version",
            },
            "finalists",
        )
        if (
            not isinstance(item["finalist_ids"], list)
            or not all(isinstance(x, str) for x in item["finalist_ids"])
            or not isinstance(item["tuning_estimates"], dict)
        ):
            raise ValueError("invalid finalist selection")
        return item
    if event_type == "run_completed":
        item = _object(
            payload,
            {
                "manifest_fingerprint",
                "accepted_ids",
                "finalist_ids",
                "evidence_counts",
                "validation_claim",
            },
            "run completion",
        )
        if not all(
            isinstance(item[key], str) for key in {"manifest_fingerprint", "validation_claim"}
        ) or not all(
            isinstance(item[key], list) and all(isinstance(x, str) for x in item[key])
            for key in {"accepted_ids", "finalist_ids"}
        ):
            raise ValueError("invalid run completion")
        return item
    raise ValueError(f"unknown evidence event type {event_type!r}")


def decode_event(value: object, expected_sequence: int | None = None) -> EvidenceEvent:
    item = _object(value, {"schema_version", "sequence", "type", "payload"}, "evidence event")
    if item["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported evidence schema version: {item['schema_version']!r}")
    sequence = _int(item["sequence"], "evidence sequence", positive=True)
    if expected_sequence is not None and sequence != expected_sequence:
        raise ValueError("evidence sequence is not contiguous")
    if not isinstance(item["type"], str):
        raise ValueError("event type must be a string")
    return EvidenceEvent(sequence, item["type"], _validate_payload(item["type"], item["payload"]))  # type: ignore[arg-type]


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

    def append(self, event_type: EventType, payload: object) -> EvidenceEvent:
        event = EvidenceEvent(
            self._sequence + 1, event_type, _validate_payload(event_type, payload)
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event.encoded()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._sequence = event.sequence
        return event


def scientific_projection(events: list[EvidenceEvent]) -> str:
    return canonical_json(
        [
            {"type": event.type, "payload": event.payload}
            for event in events
            if event.type in SCIENTIFIC
        ]
    )
