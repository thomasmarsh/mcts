"""Run-scoped tuning-space constraints: narrow (never widen) the declared schema.

Unifies the tuner's two prior run-scoped space controls -- named-family
exclusion and the ``fix``/``range``/``choices`` space overrides -- into one
predicated form. A constraint is a ``set`` of per-parameter narrowings,
optionally guarded by a ``when`` predicate over categorical parameters.
"Exclude algorithm/axis-variant X" is now expressed as a ``choices`` narrowing
of the ``algorithm`` / ``select`` / ``simulate`` categorical.

Constraints are validated against the resolved schema (never widen, only
narrow), recorded in ``manifest.json``, folded into the objective-epoch
fingerprint, and threaded into :func:`tuner_cli.space.build_space` so every
proposer -- ConfigSpace-backed or not -- draws from the same constrained space.
Reject statically, never learn a constraint through games.

Wire form is a list of ``{"when": {...}, "set": {...}}`` entries; ``when`` may be
omitted. The bare map ``{name: {fix|range|choices: ...}}`` is still accepted as
sugar for a single un-predicated constraint, so pre-cutover snapshots and
scripts keep working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .codec import JsonObject, JsonValue, elements, json_object, json_value, strict_json
from .domain import Candidate
from .schema import ParameterSpec, TuningSchema, same_scalar, value_in_domain

CONSTRAINT_POLICY_VERSION = "space-constraints-v1"

ParamScalar = bool | int | float | str


@dataclass(frozen=True, slots=True)
class FixOp:
    """Treat the parameter as ``Constant(name, value)``."""

    value: ParamScalar


@dataclass(frozen=True, slots=True)
class RangeOp:
    """Replace a float/int parameter's bounds with a sub-range of the schema's."""

    low: int | float
    high: int | float


@dataclass(frozen=True, slots=True)
class ChoicesOp:
    """Restrict a categorical/bool parameter to a proper subset of its choices."""

    choices: tuple[ParamScalar, ...]


SetOp = FixOp | RangeOp | ChoicesOp


@dataclass(frozen=True, slots=True)
class Constraint:
    """A ``set`` of per-parameter narrowings, optionally guarded by ``when``.

    ``when`` maps a categorical parameter name to the values for which this
    constraint's ``sets`` apply; an empty ``when`` applies unconditionally.
    Both tuples are sorted by key for a canonical form.
    """

    when: tuple[tuple[str, tuple[ParamScalar, ...]], ...]
    sets: tuple[tuple[str, SetOp], ...]


Constraints = tuple[Constraint, ...]


def no_constraints() -> Constraints:
    """An empty constraint list, for dataclass ``field(default_factory=...)``."""
    return ()


# --- decoding -------------------------------------------------------------------


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


def _decode_set_op(name: str, spec: object) -> SetOp:
    entry = json_object(spec, f"constraint set for {name}")
    if len(entry) != 1:
        raise ValueError(f"constraint set for {name!r} must carry exactly one of fix/range/choices")
    (kind, body) = next(iter(entry.items()))
    if kind == "fix":
        return FixOp(_param_scalar(body, f"fix value for {name!r}"))
    if kind == "range":
        items = elements(body, f"range for {name!r}")
        if len(items) != 2:
            raise ValueError(f"range for {name!r} must be [low, high]")
        return RangeOp(
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
        return ChoicesOp(choices)
    raise ValueError(f"unknown constraint set kind {kind!r} for {name!r}")


def _clean_name(name: str, what: str) -> str:
    if not name or name != name.strip():
        raise ValueError(f"{what} names must be nonempty and free of surrounding whitespace")
    return name


def _decode_sets(raw: object) -> tuple[tuple[str, SetOp], ...]:
    obj = json_object(raw, "constraint set")
    if not obj:
        raise ValueError("constraint set must narrow at least one parameter")
    decoded = {
        _clean_name(name, "constraint set parameter"): _decode_set_op(name, spec)
        for name, spec in obj.items()
    }
    return tuple(sorted(decoded.items()))


def _decode_when(raw: object) -> tuple[tuple[str, tuple[ParamScalar, ...]], ...]:
    obj = json_object(raw, "constraint when")
    if not obj:
        raise ValueError("constraint when must predicate on at least one parameter")
    decoded: dict[str, tuple[ParamScalar, ...]] = {}
    for name, spec in obj.items():
        clean = _clean_name(name, "constraint when parameter")
        items = elements(spec, f"when values for {name!r}")
        if not items:
            raise ValueError(f"when values for {name!r} must be non-empty")
        values = tuple(_param_scalar(item, f"when value for {name!r}") for item in items)
        for index, left in enumerate(values):
            if any(same_scalar(left, right) for right in values[index + 1 :]):
                raise ValueError(f"when values for {name!r} must be unique")
        decoded[clean] = values
    return tuple(sorted(decoded.items()))


def _decode_one(raw: object) -> Constraint:
    entry = json_object(raw, "constraint")
    unknown = set(entry) - {"when", "set"}
    if unknown:
        raise ValueError(f"constraint has unknown field(s): {sorted(unknown)}")
    if "set" not in entry:
        raise ValueError("constraint must carry a 'set'")
    when = _decode_when(entry["when"]) if "when" in entry else ()
    return Constraint(when=when, sets=_decode_sets(entry["set"]))


def decode_constraints(raw: object) -> Constraints:
    """Strictly decode the wire form (list of entries, or bare-map sugar)."""
    if raw is None:
        return ()
    value = json_value(raw, "constraints")
    if isinstance(value, list):
        return tuple(_decode_one(item) for item in value)
    # Bare-map sugar: one un-predicated constraint narrowing every named parameter.
    if isinstance(value, dict):
        return (Constraint(when=(), sets=_decode_sets(value)),)
    raise ValueError("constraints must be a JSON array or object")


def _encode_set_op(op: SetOp) -> JsonValue:
    if isinstance(op, FixOp):
        return {"fix": op.value}
    if isinstance(op, RangeOp):
        return {"range": [op.low, op.high]}
    return {"choices": list(op.choices)}


def encode_constraints(constraints: Constraints) -> list[JsonObject]:
    """Canonical (key-sorted list) wire form for the manifest and epoch fingerprint."""
    encoded: list[JsonObject] = []
    for constraint in constraints:
        entry: JsonObject = {}
        if constraint.when:
            entry["when"] = {name: list(values) for name, values in constraint.when}
        entry["set"] = {name: _encode_set_op(op) for name, op in constraint.sets}
        encoded.append(entry)
    return encoded


# --- validation ----------------------------------------------------------------


def _validate_set_op(name: str, parameter: ParameterSpec, op: SetOp) -> None:
    if isinstance(op, FixOp):
        if not value_in_domain(op.value, parameter):
            raise ValueError(f"fix value for {name!r} is outside its schema domain")
        return
    if isinstance(op, RangeOp):
        if parameter.kind not in ("float", "int") or parameter.bounds is None:
            raise ValueError(f"range constraint for {name!r} needs a numeric parameter")
        low, high = op.low, op.high
        if parameter.kind == "int" and (isinstance(low, float) or isinstance(high, float)):
            raise ValueError(f"range constraint for integer {name!r} needs integer bounds")
        if not low < high:
            raise ValueError(f"range constraint for {name!r} must have low < high")
        if low < parameter.bounds[0] or high > parameter.bounds[1]:
            raise ValueError(f"range constraint for {name!r} escapes its schema bounds")
        return
    if parameter.kind not in ("categorical", "bool") or parameter.choices is None:
        raise ValueError(f"choices constraint for {name!r} needs a categorical parameter")
    for choice in op.choices:
        if not any(same_scalar(choice, allowed) for allowed in parameter.choices):
            raise ValueError(
                f"choices constraint for {name!r} includes {choice!r} not in the schema"
            )
    if len(op.choices) >= len(parameter.choices):
        raise ValueError(f"choices constraint for {name!r} must be a proper subset of the schema")


def _residual_choices(parameter: ParameterSpec, op: SetOp | None) -> tuple[JsonValue, ...]:
    assert parameter.choices is not None
    if op is None:
        return parameter.choices
    if isinstance(op, FixOp):
        return tuple(c for c in parameter.choices if same_scalar(c, op.value))
    if isinstance(op, ChoicesOp):
        return tuple(c for c in parameter.choices if any(same_scalar(c, k) for k in op.choices))
    return parameter.choices


def _unconditional_set(constraints: Constraints, name: str) -> SetOp | None:
    hits = [
        op
        for constraint in constraints
        if not constraint.when
        for key, op in constraint.sets
        if key == name
    ]
    if len(hits) > 1:
        raise ValueError(f"parameter {name!r} is constrained more than once unconditionally")
    return hits[0] if hits else None


def _condition_reachable(op: SetOp | None, values: tuple[JsonValue, ...]) -> bool:
    if op is None:
        return True
    if isinstance(op, FixOp):
        return any(same_scalar(op.value, value) for value in values)
    if isinstance(op, ChoicesOp):
        return any(same_scalar(choice, value) for choice in op.choices for value in values)
    return any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and op.low <= value <= op.high
        for value in values
    )


def validate_constraints(schema: TuningSchema, constraints: Constraints) -> None:
    """Reject a constraint that widens, mistypes, empties, or orphans a parameter."""
    if not constraints:
        return
    by_name = {parameter.name: parameter for parameter in schema.parameters}
    for constraint in constraints:
        for parent, values in constraint.when:
            parameter = by_name.get(parent)
            if parameter is None:
                raise ValueError(f"constraint when references unknown parameter {parent!r}")
            if parameter.kind not in ("categorical", "bool"):
                raise ValueError(f"constraint when parameter {parent!r} must be categorical")
            for value in values:
                if not value_in_domain(value, parameter):
                    raise ValueError(
                        f"constraint when value {value!r} is outside {parent!r}'s domain"
                    )
        for name, op in constraint.sets:
            parameter = by_name.get(name)
            if parameter is None:
                raise ValueError(f"constraint set references unknown parameter {name!r}")
            if parameter.kind == "constant":
                raise ValueError(f"constraint cannot touch schema-constant parameter {name!r}")
            _validate_set_op(name, parameter, op)

    constrained_names = {
        name for constraint in constraints if not constraint.when for name, _ in constraint.sets
    }
    for name in constrained_names:
        _unconditional_set(constraints, name)  # rejects a doubly-constrained parameter

    for parameter in schema.parameters:
        if parameter.kind in ("categorical", "bool") and parameter.choices is not None:
            residual = _residual_choices(parameter, _unconditional_set(constraints, parameter.name))
            if not residual:
                raise ValueError(f"constraints leave parameter {parameter.name!r} with no choices")

    for condition in schema.conditions:
        op = _unconditional_set(constraints, condition.parent)
        if not _condition_reachable(op, condition.values):
            children = ", ".join(condition.children)
            raise ValueError(
                f"constraint on {condition.parent!r} leaves conditional "
                f"parameter(s) {children} unreachable"
            )


# --- candidate gate -----------------------------------------------------------


def _predicate_matches(
    when: tuple[tuple[str, tuple[ParamScalar, ...]], ...], config: dict[str, JsonValue]
) -> bool:
    for parent, values in when:
        if parent not in config:
            return False
        if not any(same_scalar(config[parent], value) for value in values):
            return False
    return True


def _set_op_satisfied(op: SetOp, value: JsonValue) -> bool:
    if isinstance(op, FixOp):
        return same_scalar(value, op.value)
    if isinstance(op, ChoicesOp):
        return any(same_scalar(value, choice) for choice in op.choices)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and op.low <= value <= op.high
    )


def require_candidate_allowed(candidate: Candidate, constraints: Constraints) -> None:
    """Reject a proposed candidate that violates any active constraint."""
    if not constraints:
        return
    raw = strict_json(candidate.canonical_config, "candidate configuration")
    if not isinstance(raw, dict):
        raise ValueError("candidate configuration must be a JSON object")
    config: dict[str, JsonValue] = raw
    for constraint in constraints:
        if not _predicate_matches(constraint.when, config):
            continue
        for name, op in constraint.sets:
            if name not in config:
                continue
            if not _set_op_satisfied(op, config[name]):
                raise ValueError(f"candidate violates constraint on {name!r}: {config[name]!r}")


# --- schema rewrite ----------------------------------------------------------


def _apply_op(parameter: ParameterSpec, op: SetOp | None) -> ParameterSpec:
    if op is None:
        return parameter
    if isinstance(op, FixOp):
        return replace(
            parameter,
            kind="constant",
            bounds=None,
            choices=None,
            default=None,
            constant_value=op.value,
        )
    if isinstance(op, RangeOp):
        assert isinstance(parameter.default, (int, float))
        clamped = min(max(parameter.default, op.low), op.high)
        return replace(parameter, bounds=(op.low, op.high), default=clamped)
    assert parameter.choices is not None
    kept = tuple(
        choice
        for choice in parameter.choices
        if any(same_scalar(choice, allowed) for allowed in op.choices)
    )
    default = (
        parameter.default
        if any(same_scalar(parameter.default, choice) for choice in kept)
        else kept[0]
    )
    return replace(parameter, choices=kept, default=default)


def constrained_schema(schema: TuningSchema, constraints: Constraints) -> TuningSchema:
    """Return ``schema`` with the un-predicated ``constraints`` baked in.

    ``when``-predicated constraints cannot be expressed as a static domain
    narrowing here (the child domain varies with the parent), so they are
    enforced at proposal time by :func:`require_candidate_allowed`.
    """
    validate_constraints(schema, constraints)
    if not constraints:
        return schema
    parameters = tuple(
        _apply_op(parameter, _unconditional_set(constraints, parameter.name))
        for parameter in schema.parameters
    )
    return replace(schema, parameters=parameters)
