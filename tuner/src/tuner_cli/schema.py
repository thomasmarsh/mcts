"""Strict typed decoding of a game-host ``describe`` response."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .codec import JsonObject, JsonValue, json_object, object_fields
from .identity import canonical_json, fingerprint

ParameterKind = Literal["float", "int", "categorical", "bool", "constant"]
_SCALAR_TYPES = (type(None), bool, int, float, str)


@dataclass(frozen=True, slots=True)
class AiPresetSpec:
    id: str
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    kind: ParameterKind
    bounds: tuple[float | int, float | int] | None
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
    ai_presets: tuple[AiPresetSpec, ...]
    tuning: TuningSchema
    description_fingerprint: str
    schema_fingerprint: str
    binary_path: Path
    binary_sha256: str
    engine_fingerprint: str
    raw_description: str


def _object(value: object, label: str, fields: set[str]) -> JsonObject:
    return object_fields(value, fields, label)


def _string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if nonempty and not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _scalar(value: object, label: str) -> JsonValue:
    if not isinstance(value, _SCALAR_TYPES):
        raise ValueError(f"{label} must be a JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _same_scalar(left: JsonValue, right: JsonValue) -> bool:
    """Compare schema values without Python's bool/int numeric equivalence."""
    return type(left) is type(right) and left == right


def _unique_scalars(values: list[JsonValue], label: str) -> None:
    if any(
        _same_scalar(left, right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ):
        raise ValueError(f"{label} must be unique")


def _parameter(raw: object) -> ParameterSpec:
    preliminary = json_object(raw, "parameter")
    kind = _string(preliminary.get("type"), "parameter type")
    expected = {
        "float": {"name", "type", "bounds", "default"},
        "int": {"name", "type", "bounds", "default"},
        "categorical": {"name", "type", "choices", "default"},
        "bool": {"name", "type", "default"},
        "constant": {"name", "type", "value"},
    }
    if kind not in expected:
        raise ValueError(f"unknown parameter type {kind!r}")
    if kind == "float":
        parameter_kind: ParameterKind = "float"
    elif kind == "int":
        parameter_kind = "int"
    elif kind == "categorical":
        parameter_kind = "categorical"
    elif kind == "bool":
        parameter_kind = "bool"
    else:
        parameter_kind = "constant"
    item = _object(preliminary, "parameter", expected[parameter_kind])
    name = _string(item["name"], "parameter name", nonempty=True)
    if parameter_kind == "float":
        bounds = item["bounds"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"numeric parameter {name} needs two bounds")
        lo, hi = (
            _number(bounds[0], f"{name} lower bound"),
            _number(bounds[1], f"{name} upper bound"),
        )
        default = _number(item["default"], f"{name} default")
        if lo > hi or not lo <= default <= hi:
            raise ValueError(f"invalid bounds/default for {name}")
        return ParameterSpec(name, parameter_kind, (lo, hi), None, default, None)
    if parameter_kind == "int":
        bounds = item["bounds"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"numeric parameter {name} needs two bounds")
        lo, hi = (
            _integer(bounds[0], f"{name} lower bound"),
            _integer(bounds[1], f"{name} upper bound"),
        )
        default = _integer(item["default"], f"{name} default")
        if lo > hi or not lo <= default <= hi:
            raise ValueError(f"invalid bounds/default for {name}")
        return ParameterSpec(name, parameter_kind, (lo, hi), None, default, None)
    if parameter_kind == "categorical":
        raw_choices = item["choices"]
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError(f"categorical parameter {name} needs non-empty choices")
        choices = [_scalar(value, f"{name} choice") for value in raw_choices]
        _unique_scalars(choices, f"categorical choices for {name}")
        default = _scalar(item["default"], f"{name} default")
        if not any(_same_scalar(default, choice) for choice in choices):
            raise ValueError(f"invalid default for {name}")
        return ParameterSpec(name, parameter_kind, None, tuple(choices), default, None)
    if parameter_kind == "bool":
        if not isinstance(item["default"], bool):
            raise ValueError(f"Boolean parameter {name} needs a Boolean default")
        return ParameterSpec(name, parameter_kind, None, (False, True), item["default"], None)
    return ParameterSpec(
        name, parameter_kind, None, None, None, _scalar(item["value"], f"{name} value")
    )


def _condition(raw: object) -> ActivationCondition:
    item = _object(raw, "condition", {"if", "then"})
    predicate = item["if"]
    if not isinstance(predicate, dict) or len(predicate) != 1:
        raise ValueError("condition if must have exactly one parent")
    parent, raw_values = next(iter(predicate.items()))
    if not parent:
        raise ValueError("condition parent must be a non-empty string")
    if isinstance(raw_values, list):
        if not raw_values:
            raise ValueError("condition values must be non-empty")
        values = [_scalar(value, "condition value") for value in raw_values]
    else:
        values = [_scalar(raw_values, "condition value")]
    _unique_scalars(values, "condition values")
    children = item["then"]
    if not isinstance(children, list) or not children:
        raise ValueError("condition then must be a non-empty array")
    decoded_children = [_string(child, "condition child", nonempty=True) for child in children]
    if len(set(decoded_children)) != len(decoded_children):
        raise ValueError("condition children must be unique")
    return ActivationCondition(parent, tuple(values), tuple(decoded_children))


def _value_in_domain(value: JsonValue, parameter: ParameterSpec) -> bool:
    if parameter.kind == "float":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and parameter.bounds is not None
            and parameter.bounds[0] <= float(value) <= parameter.bounds[1]
        )
    if parameter.kind == "int":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and parameter.bounds is not None
            and parameter.bounds[0] <= value <= parameter.bounds[1]
        )
    if parameter.kind in {"categorical", "bool"}:
        return parameter.choices is not None and any(
            _same_scalar(value, choice) for choice in parameter.choices
        )
    return _same_scalar(value, parameter.constant_value)


def _validate_conditions(
    parameters: tuple[ParameterSpec, ...], conditions: tuple[ActivationCondition, ...]
) -> None:
    by_name = {parameter.name: parameter for parameter in parameters}
    edges: dict[str, set[str]] = {name: set() for name in by_name}
    for condition in conditions:
        if condition.parent not in by_name:
            raise ValueError(f"condition references unknown parent {condition.parent!r}")
        for value in condition.values:
            if not _value_in_domain(value, by_name[condition.parent]):
                raise ValueError(f"condition value is outside {condition.parent!r}'s domain")
        for child in condition.children:
            if child not in by_name:
                raise ValueError(f"condition references unknown child {child!r}")
            if child == condition.parent:
                raise ValueError("condition cannot reference itself")
            edges[condition.parent].add(child)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("activation conditions contain a cycle")
        if name in visited:
            return
        visiting.add(name)
        for child in edges[name]:
            visit(child)
        visiting.remove(name)
        visited.add(name)

    for name in edges:
        visit(name)


def decode_game_spec(raw: JsonValue, binary_path: Path, binary_sha256: str) -> GameSpec:
    top = _object(
        raw,
        "describe response",
        {"kind", "label", "description", "default_config", "ai_presets", "tuning"},
    )
    kind = _string(top["kind"], "kind", nonempty=True)
    label = _string(top["label"], "label")
    description = _string(top["description"], "description")
    presets_raw = top["ai_presets"]
    if not isinstance(presets_raw, list):
        raise ValueError("ai_presets must be an array")
    presets = tuple(
        AiPresetSpec(
            _string(item["id"], "preset id", nonempty=True),
            _string(item["label"], "preset label"),
            _string(item["description"], "preset description"),
        )
        for raw_item in presets_raw
        for item in [_object(raw_item, "AI preset", {"id", "label", "description"})]
    )
    if len({preset.id for preset in presets}) != len(presets):
        raise ValueError("AI preset IDs must be unique")
    tuning_raw = top["tuning"]
    if tuning_raw is None:
        raise ValueError("game does not provide tuning metadata")
    tuning = _object(
        tuning_raw,
        "tuning",
        {"id", "baselines", "eval_rounds", "parameters", "conditions", "game_config"},
    )
    baselines_raw = tuning["baselines"]
    if not isinstance(baselines_raw, list):
        raise ValueError("baselines must be an array")
    baselines = tuple(_string(value, "baseline", nonempty=True) for value in baselines_raw)
    if len(set(baselines)) != len(baselines) or not set(baselines) <= {
        preset.id for preset in presets
    }:
        raise ValueError("baselines must be unique AI preset IDs")
    eval_rounds = _integer(tuning["eval_rounds"], "eval_rounds")
    if eval_rounds <= 0:
        raise ValueError("eval_rounds must be positive")
    parameters_raw = tuning["parameters"]
    conditions_raw = tuning["conditions"]
    if not isinstance(parameters_raw, list) or not isinstance(conditions_raw, list):
        raise ValueError("parameters and conditions must be arrays")
    parameters = tuple(_parameter(item) for item in parameters_raw)
    if not parameters or len({parameter.name for parameter in parameters}) != len(parameters):
        raise ValueError("tuning parameters must be non-empty and uniquely named")
    conditions = tuple(_condition(item) for item in conditions_raw)
    _validate_conditions(parameters, conditions)
    default_game_config = canonical_json(top["default_config"])
    game_config = canonical_json(tuning["game_config"])
    if game_config != default_game_config:
        raise ValueError("tuning game_config must equal default_config")
    schema = TuningSchema(
        _string(tuning["id"], "tuning id", nonempty=True),
        baselines,
        eval_rounds,
        parameters,
        conditions,
        game_config,
    )
    description_fingerprint = fingerprint(top)
    return GameSpec(
        kind,
        label,
        description,
        default_game_config,
        presets,
        schema,
        description_fingerprint,
        fingerprint(tuning),
        binary_path,
        binary_sha256,
        fingerprint(
            {"binary_sha256": binary_sha256, "description_fingerprint": description_fingerprint}
        ),
        canonical_json(top),
    )
