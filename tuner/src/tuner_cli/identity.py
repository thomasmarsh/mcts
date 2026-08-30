"""Canonical JSON and deterministic identities used by tuner artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from .domain import Candidate, IterationBudget, PairTask, TaskBlock, TaskCase

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _normalize(value: object) -> JsonValue:
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value and isinstance(scalar, (type(None), bool, int, float, str)):
            return _normalize(scalar)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow non-finite floats")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON requires string mapping keys")
            normalized[key] = _normalize(child)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_normalize(child) for child in value]
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return compact, sorted, UTF-8 JSON after safe scalar normalization."""
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(kind: str, identity_payload: object) -> str:
    return f"{kind}-{fingerprint(identity_payload)}"


def derive_task_seed(root_seed: int, phase: str, ordinal: int) -> int:
    if phase not in {"tuning", "validation"} or ordinal < 0:
        raise ValueError("invalid task seed inputs")
    payload = {
        "namespace": "mcts-tuner-task-seed-v1",
        "root_seed": root_seed,
        "phase": phase,
        "ordinal": ordinal,
    }
    digest = hashlib.sha256(canonical_json(payload).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 53) - 1)


def candidate_from_config(config: object) -> Candidate:
    """Construct the only candidate identity used by artifacts and execution."""
    canonical = canonical_json(config)
    config_fingerprint = fingerprint(json.loads(canonical))
    return Candidate(f"candidate-{config_fingerprint}", config_fingerprint, canonical)


def candidate_from_canonical_config(canonical: str) -> Candidate:
    """Verify a stored configuration spelling before reconstructing its identity."""
    try:
        parsed = json.loads(canonical)
    except json.JSONDecodeError as error:
        raise ValueError("candidate configuration is not JSON") from error
    if canonical_json(parsed) != canonical:
        raise ValueError("candidate configuration is not canonical JSON")
    return candidate_from_config(parsed)


def task_case(
    phase: str,
    ordinal: int,
    root_seed: int,
    opponent: Candidate,
    game_config_fingerprint: str,
) -> TaskCase:
    if phase not in {"tuning", "validation"}:
        raise ValueError("invalid task phase")
    seed = derive_task_seed(root_seed, phase, ordinal)
    payload = {
        "phase": phase,
        "ordinal": ordinal,
        "seed": seed,
        "opponent_fingerprint": opponent.fingerprint,
        "game_config_fingerprint": game_config_fingerprint,
        "start": "default",
    }
    return TaskCase(
        stable_id("task", payload),
        phase,  # type: ignore[arg-type]
        ordinal,
        seed,
        f"opponent-default-{opponent.fingerprint}",
        opponent.fingerprint,
        game_config_fingerprint,
    )


def task_block(
    phase: str,
    count: int,
    root_seed: int,
    opponent: Candidate,
    game_config_fingerprint: str,
) -> TaskBlock:
    cases = tuple(
        task_case(phase, ordinal, root_seed, opponent, game_config_fingerprint)
        for ordinal in range(count)
    )
    return TaskBlock(
        stable_id("block", {"phase": phase, "task_ids": [case.task_id for case in cases]}),
        phase,  # type: ignore[arg-type]
        cases,
    )


def pair_task(candidate: Candidate, case: TaskCase, budget: IterationBudget) -> PairTask:
    pair_id = stable_id(
        "pair",
        {
            "candidate_fingerprint": candidate.fingerprint,
            "task_id": case.task_id,
            "opponent_fingerprint": case.opponent_fingerprint,
            "max_iterations": budget.max_iterations,
        },
    )
    return PairTask(pair_id, candidate.candidate_id, case, budget)


def game_id(task: PairTask, candidate_side: str) -> str:
    if candidate_side not in {"first", "second"}:
        raise ValueError("invalid candidate side")
    return stable_id("game", {"pair_id": task.pair_id, "candidate_side": candidate_side})
