"""Strict version-4 append-only evidence records and atomic file publishing."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .artifacts import SCHEMA_VERSION
from .codec import JsonObject, JsonValue, integer, object_fields, strict_json, string, strings
from .domain import GameResult, PairResult, PairTask
from .identity import canonical_json, game_id
from .statistics import pair_utility
from .target import parse_pair_output

EventType = Literal[
    "proposal_created",
    "proposal_accepted",
    "proposal_rejected",
    "cohort_completed",
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
    "cohort_completed",
    "pair_completed",
    "observation_completed",
    "finalists_selected",
    "run_completed",
}


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    sequence: int
    type: EventType
    payload: JsonObject

    def encoded(self) -> JsonObject:
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "type": self.type,
            "payload": self.payload,
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


def _candidate_payload(value: object, *, disposition: str = "created") -> JsonObject:
    fields = {
        "proposal_index",
        "cohort_slot",
        "source",
        "source_attempt",
        "candidate_id",
        "fingerprint",
        "canonical_config",
    }
    if disposition == "created":
        fields |= {
            "frontier_id",
            "frontier_observation_ids",
            "proposer_version",
            "origin",
            "acquisition",
            "prediction",
            "uncertainty",
            "parent_candidate_id",
        }
    elif disposition == "accepted":
        fields.add("panel_response_fingerprints")
    item = _object(value, fields, "proposal payload")
    _int(item["proposal_index"], "proposal index")
    _int(item["cohort_slot"], "cohort slot")
    _int(item["source_attempt"], "source attempt", positive=True)
    for key in {"source", "candidate_id", "fingerprint", "canonical_config"}:
        _string(item[key], key)
    if item["source"] not in {"schema_default", "bootstrap_random", "smac_model", "random_reserve"}:
        raise ValueError("unknown proposal source")
    if disposition == "created":
        _string(item["frontier_id"], "frontier id")
        if not isinstance(item["frontier_observation_ids"], list) or not all(
            isinstance(value, str) for value in item["frontier_observation_ids"]
        ):
            raise ValueError("proposal frontier is invalid")
        _string(item["proposer_version"], "proposer version")
        for key in ("origin", "parent_candidate_id"):
            if item[key] is not None and not isinstance(item[key], str):
                raise ValueError(f"proposal {key} is invalid")
        for key in ("acquisition", "prediction", "uncertainty"):
            value = item[key]
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"proposal {key} is invalid")
    return item


def _pair_identity(value: object, *, seed: bool = False) -> JsonObject:
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


def pair_payload(result: PairResult) -> JsonObject:
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
    outcomes = [string(record["outcome"], "raw game outcome") for record in raw_records]
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


def _validate_payload(event_type: EventType, payload: object) -> JsonObject:
    if event_type == "proposal_created":
        return _candidate_payload(payload)
    if event_type == "proposal_accepted":
        item = _candidate_payload(payload, disposition="accepted")
        if not isinstance(item["panel_response_fingerprints"], list) or not all(
            isinstance(value, str) for value in item["panel_response_fingerprints"]
        ):
            raise ValueError("proposal acceptance panel fingerprints are invalid")
        return item
    if event_type == "proposal_rejected":
        item = _object(
            payload,
            {
                "proposal_index",
                "cohort_slot",
                "source",
                "source_attempt",
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
                for key in {
                    "proposal_index",
                    "cohort_slot",
                    "source",
                    "source_attempt",
                    "candidate_id",
                    "fingerprint",
                    "canonical_config",
                }
            },
            disposition="disposition",
        )
        if item["reason"] not in {"duplicate", "semantic_validation"} or not isinstance(
            item["errors"], list
        ):
            raise ValueError("invalid proposal rejection")
        if any(
            not isinstance(error, dict)
            or set(error) != {"opponent_id", "errors"}
            or not isinstance(error["opponent_id"], str)
            or not isinstance(error["errors"], list)
            for error in item["errors"]
        ):
            raise ValueError("proposal rejection errors must identify panel opponents")
        return item
    if event_type == "cohort_completed":
        item = _object(
            payload, {"candidate_ids", "sources", "schedule_version", "final_frontier_id"}, "cohort"
        )
        if (
            not isinstance(item["candidate_ids"], list)
            or not all(isinstance(x, str) for x in item["candidate_ids"])
            or not isinstance(item["sources"], list)
            or not all(isinstance(value, str) for value in item["sources"])
            or not isinstance(item["schedule_version"], str)
            or not isinstance(item["final_frontier_id"], str)
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
                "observation_id",
                "candidate_id",
                "phase",
                "objective_epoch_id",
                "corpus_id",
                "prefix_id",
                "prefix_task_ids",
                "prefix_length",
                "search_effort",
                "pair_utilities",
                "estimate",
                "counts",
            },
            "observation",
        )
        if (
            item["phase"] not in {"tuning", "validation"}
            or not isinstance(item["candidate_id"], str)
            or not isinstance(item["observation_id"], str)
            or not all(
                isinstance(item[key], str)
                for key in {"objective_epoch_id", "corpus_id", "prefix_id"}
            )
            or not isinstance(item["prefix_task_ids"], list)
            or not all(isinstance(value, str) for value in item["prefix_task_ids"])
            or not isinstance(item["pair_utilities"], list)
        ):
            raise ValueError("invalid observation")
        _int(item["prefix_length"], "prefix length", positive=True)
        _int(item["search_effort"], "observation search effort", positive=True)
        return item
    if event_type == "finalists_selected":
        item = _object(
            payload,
            {
                "finalist_ids",
                "tuning_estimates",
                "objective_epoch_id",
                "corpus_id",
                "prefix_id",
                "prefix_task_ids",
                "search_effort",
                "selection_rule_version",
            },
            "finalists",
        )
        if (
            not isinstance(item["finalist_ids"], list)
            or not all(isinstance(x, str) for x in item["finalist_ids"])
            or not isinstance(item["tuning_estimates"], dict)
            or not all(
                isinstance(item[key], str)
                for key in {"objective_epoch_id", "corpus_id", "prefix_id"}
            )
            or not isinstance(item["prefix_task_ids"], list)
            or not all(isinstance(value, str) for value in item["prefix_task_ids"])
            or not isinstance(item["search_effort"], int)
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
                "objective_epoch_id",
                "validation_prefix_id",
                "validation_search_effort",
                "missing_production_axes",
                "cohort_frontier_id",
            },
            "run completion",
        )
        if not all(
            isinstance(item[key], str)
            for key in {
                "manifest_fingerprint",
                "validation_claim",
                "objective_epoch_id",
                "validation_prefix_id",
                "cohort_frontier_id",
            }
        ):
            raise ValueError("invalid run completion")
        strings(item["accepted_ids"], "accepted IDs")
        strings(item["finalist_ids"], "finalist IDs")
        if (
            not isinstance(item["validation_search_effort"], int)
            or not isinstance(item["missing_production_axes"], list)
            or not all(isinstance(value, str) for value in item["missing_production_axes"])
        ):
            raise ValueError("invalid completion fidelity")
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
    event_type = cast(EventType, item["type"])
    return EvidenceEvent(sequence, event_type, _validate_payload(event_type, item["payload"]))


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
