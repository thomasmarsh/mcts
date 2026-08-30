"""Typed decoding of Druid's top-level ``describe`` response."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .identity import JsonValue, canonical_json, fingerprint

ParameterKind = Literal["float", "int", "categorical", "bool", "constant"]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    kind: ParameterKind
    bounds: tuple[float, float] | None
    choices: tuple[JsonValue, ...] | None
    default: JsonValue | None
    constant_value: JsonValue | None


@dataclass(frozen=True, slots=True)
class ActivationCondition:
    parent: str
    values: tuple[JsonValue, ...]
    children: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TuningSchema:
    id: str
    baselines: tuple[str, ...]
    eval_rounds: int
    parameters: tuple[ParameterSpec, ...]
    conditions: tuple[ActivationCondition, ...]
    game_config: str


@dataclass(frozen=True, slots=True)
class GameSpec:
    kind: str
    label: str
    description: str
    default_game_config: str
    tuning: TuningSchema
    description_fingerprint: str
    schema_fingerprint: str
    binary_path: Path
    binary_sha256: str
    raw_description: str


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _parameter(raw: object) -> ParameterSpec:
    item = _object(raw, "parameter")
    name = _string(item.get("name"), "parameter name")
    kind = _string(item.get("type"), f"parameter {name} type")
    if kind not in {"float", "int", "categorical", "bool", "constant"}:
        raise ValueError(f"unknown parameter type {kind!r}")
    if kind in {"float", "int"}:
        bounds = item.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"numeric parameter {name} needs two bounds")
        lo, hi = (
            _number(bounds[0], f"{name} lower bound"),
            _number(bounds[1], f"{name} upper bound"),
        )
        if lo > hi or (kind == "int" and (not lo.is_integer() or not hi.is_integer())):
            raise ValueError(f"invalid bounds for {name}")
        default = item.get("default")
        if default is not None:
            number = _number(default, f"{name} default")
            if not lo <= number <= hi or (kind == "int" and not number.is_integer()):
                raise ValueError(f"invalid default for {name}")
        return ParameterSpec(name, kind, (lo, hi), None, default, None)
    if kind == "categorical":
        choices = item.get("choices")
        if not isinstance(choices, list) or not choices or item.get("default") not in choices:
            raise ValueError(f"categorical parameter {name} has invalid choices/default")
        return ParameterSpec(name, kind, None, tuple(choices), item["default"], None)
    if kind == "bool":
        if not isinstance(item.get("default"), bool):
            raise ValueError(f"Boolean parameter {name} needs a Boolean default")
        return ParameterSpec(name, kind, None, (False, True), item["default"], None)
    if "value" not in item:
        raise ValueError(f"constant parameter {name} needs value")
    return ParameterSpec(name, kind, None, None, None, item["value"])


def _condition(raw: object) -> ActivationCondition:
    item = _object(raw, "condition")
    predicate = _object(item.get("if"), "condition if")
    if len(predicate) != 1:
        raise ValueError("condition if must have exactly one parent")
    parent, raw_values = next(iter(predicate.items()))
    values = tuple(raw_values) if isinstance(raw_values, list) else (raw_values,)
    children = item.get("then")
    if not isinstance(parent, str) or not values or not isinstance(children, list) or not children:
        raise ValueError("malformed activation condition")
    if not all(isinstance(child, str) for child in children):
        raise ValueError("condition children must be strings")
    return ActivationCondition(parent, values, tuple(children))


def decode_druid_spec(raw: JsonValue, binary_path: Path, binary_sha256: str) -> GameSpec:
    top = _object(raw, "describe response")
    kind = _string(top.get("kind"), "kind")
    if kind != "druid":
        raise ValueError(f"expected Druid describe response, got {kind!r}")
    tuning_raw = top.get("tuning")
    if tuning_raw is None:
        raise ValueError("Druid does not provide tuning metadata")
    tuning = _object(tuning_raw, "tuning")
    parameters = tuple(_parameter(item) for item in tuning.get("parameters", []))
    if not parameters or len({parameter.name for parameter in parameters}) != len(parameters):
        raise ValueError("tuning parameters must be non-empty and uniquely named")
    conditions = tuple(_condition(item) for item in tuning.get("conditions", []))
    schema_game_config = canonical_json(tuning.get("game_config"))
    schema = TuningSchema(
        _string(tuning.get("id"), "tuning id"),
        tuple(_string(value, "baseline") for value in tuning.get("baselines", [])),
        int(_number(tuning.get("eval_rounds"), "eval_rounds")),
        parameters,
        conditions,
        schema_game_config,
    )
    schema_payload = {
        "id": schema.id,
        "baselines": list(schema.baselines),
        "eval_rounds": schema.eval_rounds,
        "parameters": tuning.get("parameters"),
        "conditions": tuning.get("conditions"),
        "game_config": json.loads(schema_game_config),
    }
    return GameSpec(
        kind,
        _string(top.get("label"), "label"),
        _string(top.get("description"), "description"),
        canonical_json(top.get("default_config")),
        schema,
        fingerprint(top),
        fingerprint(schema_payload),
        binary_path,
        binary_sha256,
        canonical_json(top),
    )
