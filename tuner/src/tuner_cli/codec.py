"""Strict JSON decoding primitives shared by artifacts and objectives."""

from __future__ import annotations

import json
import math


def _constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("JSON object has duplicate keys")
    return result


def strict_json(text: str, label: str = "JSON") -> object:
    try:
        value = json.loads(text, parse_constant=_constant, object_pairs_hook=_unique)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict {label}: {error}") from error
    _finite(value, label)
    return value


def _finite(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child, label)
    elif isinstance(value, list):
        for child in value:
            _finite(child, label)


def object_fields(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = set(value) if isinstance(value, dict) else set()
        missing, unknown = sorted(fields - actual), sorted(actual - fields)
        raise ValueError(f"{label} has invalid fields (missing={missing}, unknown={unknown})")
    return value


def string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(f"{label} must be {'a non-empty string' if nonempty else 'a string'}")
    return value


def integer(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        raise ValueError(f"{label} must be {'a positive integer' if positive else 'an integer'}")
    return value
