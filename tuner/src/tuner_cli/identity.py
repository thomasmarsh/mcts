"""Canonical JSON and deterministic identities used by tuner artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

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
