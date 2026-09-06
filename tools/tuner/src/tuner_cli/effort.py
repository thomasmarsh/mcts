"""Strict serialization and limited comparisons for frozen search effort."""

from __future__ import annotations

from .codec import JsonObject, object_fields
from .domain import SearchEffort


def encode_effort(effort: SearchEffort) -> JsonObject:
    return {"kind": effort.kind, "value": effort.value}


def decode_effort(value: object, label: str = "search effort") -> SearchEffort:
    item = object_fields(value, {"kind", "value"}, label)
    kind = item["kind"]
    raw_value = item["value"]
    if kind not in {"iterations", "time_ms"}:
        raise ValueError(f"{label} kind is invalid")
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
        raise ValueError(f"{label} value must be a positive integer")
    if kind == "iterations":
        return SearchEffort("iterations", raw_value)
    return SearchEffort("time_ms", raw_value)


def exceeds_same_kind(observed: SearchEffort, production: SearchEffort) -> bool:
    """Return whether a comparable observed effort exceeds production effort."""
    return observed.kind == production.kind and observed.value > production.value
