"""Run-scoped tuning-space overrides: constrain (never widen) the declared schema.

Family exclusion is the only run-scoped space control the tuner had; these
overrides add the rest -- fixing a parameter to a constant, narrowing a numeric
range, or restricting a categorical to a subset of its choices -- for a single
run. They are validated against the resolved schema (never widen it, only
constrain), recorded in ``manifest.json``, folded into the objective-epoch
fingerprint, and threaded into :func:`tuner_cli.space.build_space` so every
proposer sees the same constrained space. This mirrors
:mod:`tuner_cli.family_exclusions`: reject statically, never learn a constraint
through games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .codec import JsonObject, JsonValue, elements, json_object
from .schema import ParameterSpec, TuningSchema, same_scalar, value_in_domain

SPACE_OVERRIDE_POLICY_VERSION = "run-scoped-space-overrides-v1"

ParamScalar = bool | int | float | str


@dataclass(frozen=True, slots=True)
class FixOverride:
    """Treat the parameter as ``Constant(name, value)``."""

    value: ParamScalar


@dataclass(frozen=True, slots=True)
class RangeOverride:
    """Replace a float/int parameter's bounds with a sub-range of the schema's."""

    low: int | float
    high: int | float


@dataclass(frozen=True, slots=True)
class ChoicesOverride:
    """Restrict a categorical/bool parameter to a proper subset of its choices."""

    choices: tuple[ParamScalar, ...]


SpaceOverride = FixOverride | RangeOverride | ChoicesOverride
SpaceOverrides = dict[str, SpaceOverride]


def no_space_overrides() -> SpaceOverrides:
    """An empty override map, for dataclass ``field(default_factory=...)``."""
    return {}


def _param_scalar(value: object, label: str) -> ParamScalar:
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
        return value
    raise ValueError(f"{label} must be a JSON scalar (bool, number, or string)")


def _number(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _decode_one(name: str, spec: object) -> SpaceOverride:
    entry = json_object(spec, f"space override for {name}")
    if len(entry) != 1:
        raise ValueError(f"space override for {name!r} must carry exactly one of fix/range/choices")
    (kind, body) = next(iter(entry.items()))
    if kind == "fix":
        return FixOverride(_param_scalar(body, f"fix value for {name!r}"))
    if kind == "range":
        items = elements(body, f"range for {name!r}")
        if len(items) != 2:
            raise ValueError(f"range for {name!r} must be [low, high]")
        return RangeOverride(
            _number(items[0], f"range low for {name!r}"),
            _number(items[1], f"range high for {name!r}"),
        )
    if kind == "choices":
        items = elements(body, f"choices for {name!r}")
        if not items:
            raise ValueError(f"choices for {name!r} must be non-empty")
        choices = tuple(_param_scalar(item, f"choice for {name!r}") for item in items)
        for index, left in enumerate(choices):
            if any(same_scalar(left, right) for right in choices[index + 1 :]):
                raise ValueError(f"choices for {name!r} must be unique")
        return ChoicesOverride(choices)
    raise ValueError(f"unknown space override kind {kind!r} for {name!r}")


def decode_space_overrides(raw: object) -> SpaceOverrides:
    """Strictly decode the wire form ``{name: {fix|range|choices: ...}}``."""
    if raw is None:
        return {}
    obj = json_object(raw, "space overrides")
    result: SpaceOverrides = {}
    for name, spec in obj.items():
        if not name or name != name.strip():
            raise ValueError(
                "space override parameter names must be nonempty and free of surrounding whitespace"
            )
        result[name] = _decode_one(name, spec)
    return result


def encode_space_overrides(overrides: SpaceOverrides) -> JsonObject:
    """Canonical (name-sorted) wire form for the manifest and epoch fingerprint."""
    encoded: JsonObject = {}
    for name in sorted(overrides):
        override = overrides[name]
        if isinstance(override, FixOverride):
            encoded[name] = {"fix": override.value}
        elif isinstance(override, RangeOverride):
            encoded[name] = {"range": [override.low, override.high]}
        else:
            encoded[name] = {"choices": list(override.choices)}
    return encoded


def _condition_still_reachable(
    parent_override: SpaceOverride | None, values: tuple[JsonValue, ...]
) -> bool:
    if parent_override is None:
        return True
    if isinstance(parent_override, FixOverride):
        return any(same_scalar(parent_override.value, value) for value in values)
    if isinstance(parent_override, ChoicesOverride):
        return any(
            same_scalar(choice, value) for choice in parent_override.choices for value in values
        )
    return any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and parent_override.low <= value <= parent_override.high
        for value in values
    )


def validate_space_overrides(schema: TuningSchema, overrides: SpaceOverrides) -> None:
    """Reject an override that widens, mistypes, empties, or orphans a parameter."""
    if not overrides:
        return
    by_name = {parameter.name: parameter for parameter in schema.parameters}
    for name, override in overrides.items():
        parameter = by_name.get(name)
        if parameter is None:
            raise ValueError(f"space override references unknown parameter {name!r}")
        if parameter.kind == "constant":
            raise ValueError(f"space override cannot touch schema-constant parameter {name!r}")
        _validate_one(name, parameter, override)
    for condition in schema.conditions:
        if not _condition_still_reachable(overrides.get(condition.parent), condition.values):
            children = ", ".join(condition.children)
            raise ValueError(
                f"space override on {condition.parent!r} leaves conditional "
                f"parameter(s) {children} unreachable"
            )


def constrained_schema(schema: TuningSchema, overrides: SpaceOverrides) -> TuningSchema:
    """Return ``schema`` with ``overrides`` baked into its parameter set.

    A ``fix`` becomes a schema constant, a ``range`` replaces the numeric bounds
    (default clamped in), and a ``choices`` restricts the choice set (default
    fixed up). Every proposer then draws from this rewritten schema, so the
    constraint reaches ConfigSpace-backed and hand-rolled proposers alike.
    """
    validate_space_overrides(schema, overrides)
    if not overrides:
        return schema
    parameters = tuple(
        _apply_override(parameter, overrides.get(parameter.name)) for parameter in schema.parameters
    )
    return replace(schema, parameters=parameters)


def _apply_override(parameter: ParameterSpec, override: SpaceOverride | None) -> ParameterSpec:
    if override is None:
        return parameter
    if isinstance(override, FixOverride):
        return replace(
            parameter,
            kind="constant",
            bounds=None,
            choices=None,
            default=None,
            constant_value=override.value,
        )
    if isinstance(override, RangeOverride):
        assert isinstance(parameter.default, (int, float))
        clamped = min(max(parameter.default, override.low), override.high)
        return replace(parameter, bounds=(override.low, override.high), default=clamped)
    assert parameter.choices is not None
    kept = tuple(
        choice
        for choice in parameter.choices
        if any(same_scalar(choice, allowed) for allowed in override.choices)
    )
    default = (
        parameter.default
        if any(same_scalar(parameter.default, choice) for choice in kept)
        else kept[0]
    )
    return replace(parameter, choices=kept, default=default)


def _validate_one(name: str, parameter: ParameterSpec, override: SpaceOverride) -> None:
    if isinstance(override, FixOverride):
        if not value_in_domain(override.value, parameter):
            raise ValueError(f"fix value for {name!r} is outside its schema domain")
        return
    if isinstance(override, RangeOverride):
        if parameter.kind not in ("float", "int") or parameter.bounds is None:
            raise ValueError(f"range override for {name!r} needs a numeric parameter")
        low, high = override.low, override.high
        if parameter.kind == "int" and (isinstance(low, float) or isinstance(high, float)):
            raise ValueError(f"range override for integer {name!r} needs integer bounds")
        if not low < high:
            raise ValueError(f"range override for {name!r} must have low < high")
        if low < parameter.bounds[0] or high > parameter.bounds[1]:
            raise ValueError(f"range override for {name!r} escapes its schema bounds")
        return
    if parameter.kind not in ("categorical", "bool") or parameter.choices is None:
        raise ValueError(f"choices override for {name!r} needs a categorical parameter")
    for choice in override.choices:
        if not any(same_scalar(choice, allowed) for allowed in parameter.choices):
            raise ValueError(f"choices override for {name!r} includes {choice!r} not in the schema")
    if len(override.choices) >= len(parameter.choices):
        raise ValueError(f"choices override for {name!r} must be a proper subset of the schema")
