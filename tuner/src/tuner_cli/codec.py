"""Strict JSON algebra and narrowing primitives at the transport boundary."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import TypeVar
from typing_extensions import TypeGuard

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
_Literal = TypeVar("_Literal", bound=str)


def _constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    result: JsonObject = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("JSON object has duplicate keys")
    return result


def strict_json(text: str, label: str = "JSON") -> JsonValue:
    try:
        value: JsonValue = json.loads(text, parse_constant=_constant, object_pairs_hook=_unique)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict {label}: {error}") from error
    finite(value, label)
    return value


def finite(value: JsonValue, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            finite(child, label)
    elif isinstance(value, list):
        for child in value:
            finite(child, label)


def _list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return not isinstance(value, float) or math.isfinite(value)
    if _list(value):
        return all(is_json_value(item) for item in value)
    return _mapping(value) and all(
        isinstance(key, str) and is_json_value(item) for key, item in value.items()
    )


def is_json_object(value: object) -> TypeGuard[JsonObject]:
    """Recognize a complete JSON object without exposing untyped mappings."""
    return _mapping(value) and all(
        isinstance(key, str) and is_json_value(item) for key, item in value.items()
    )


def objects(value: object, label: str) -> tuple[JsonObject, ...]:
    if not _list(value):
        raise ValueError(f"{label} must be an array of JSON objects")
    result: list[JsonObject] = []
    for item in value:
        result.append(json_object(item, label))
    return tuple(result)


def json_object(value: object, label: str) -> JsonObject:
    if not is_json_object(value):
        raise ValueError(f"{label} must be a JSON object")
    result: JsonObject = {}
    for key, child in value.items():
        result[key] = child
    return result


def object_fields(value: object, fields: set[str], label: str) -> JsonObject:
    item = json_object(value, label)
    if set(item) != fields:
        actual = set(item)
        missing, unknown = sorted(fields - actual), sorted(actual - fields)
        raise ValueError(f"{label} has invalid fields (missing={missing}, unknown={unknown})")
    return item


def string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(f"{label} must be {'a non-empty string' if nonempty else 'a string'}")
    return value


def literal(value: object, values: tuple[_Literal, ...], label: str) -> _Literal:
    item = string(value, label)
    for allowed in values:
        if item == allowed:
            return allowed
    raise ValueError(f"{label} must be one of {values!r}")


def integer(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        raise ValueError(f"{label} must be {'a positive integer' if positive else 'an integer'}")
    return value


def number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def strings(value: object, label: str) -> tuple[str, ...]:
    if not _list(value):
        raise ValueError(f"{label} must be an array of strings")
    result: list[str] = []
    for item in value:
        result.append(string(item, label))
    return tuple(result)


def optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return string(value, label)


def optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    return number(value, label)
