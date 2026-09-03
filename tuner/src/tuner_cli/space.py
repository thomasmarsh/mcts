"""ConfigSpace bridge used only for reproducible default/random proposals."""

# ConfigSpace's own stubs annotate the `meta` kwarg on Categorical/Float/Integer/Constant
# and the `add()` methods as bare `dict`/`Hyperparameter[...]`, which pyright's strict mode
# widens to Unknown; that Unknown then propagates through every Hyperparameter-returning
# call in this file. This is a library stub gap, not a real type-safety issue here, so it's
# suppressed only in this bridge module rather than loosening the workspace's strict mode.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false

from __future__ import annotations

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenGreaterThanClause,
    ForbiddenInClause,
    ForbiddenLessThanClause,
    Integer,
    OrConjunction,
)
from ConfigSpace.forbidden import ForbiddenClause
from ConfigSpace.hyperparameters import Hyperparameter

from .constraints import ChoicesOp, Constraints, FixOp, RangeOp, SetOp, dynamic_forbiddens
from .family_exclusions import validate_family_exclusions
from .schema import ParameterSpec, TuningSchema, same_scalar


def build_space(
    schema: TuningSchema,
    seed: int,
    excluded_families: tuple[str, ...] = (),
    constraints: Constraints = (),
) -> ConfigurationSpace:
    """Build the ConfigSpace for ``schema``.

    Run-scoped tuning-space constraints are applied in two places. Un-predicated
    narrowings (and statically-entailed predicated ones) are baked into
    ``schema`` beforehand by :func:`tuner_cli.constraints.constrained_schema`, so
    every proposer -- ConfigSpace-backed or not -- draws from the same narrowed
    parameter set. A ``when``-predicated constraint that genuinely crosses the
    space is passed here as ``constraints`` and emitted as ConfigSpace forbidden
    clauses so ConfigSpace-backed proposals honour it directly.
    """
    validate_family_exclusions(schema, excluded_families)
    space = ConfigurationSpace(seed=seed)
    parameters = []
    for spec in schema.parameters:
        if spec.kind == "constant":
            parameter = Constant(spec.name, spec.constant_value)
        elif spec.kind == "float":
            assert spec.bounds is not None
            default = spec.default if spec.default is not None else spec.bounds[0]
            assert isinstance(default, (int, float))
            parameter = Float(spec.name, spec.bounds, default=float(default))
        elif spec.kind == "int":
            assert spec.bounds is not None
            default = spec.default if spec.default is not None else spec.bounds[0]
            assert isinstance(default, (int, float))
            parameter = Integer(
                spec.name,
                (int(spec.bounds[0]), int(spec.bounds[1])),
                default=int(default),
            )
        else:
            assert spec.choices is not None
            default = spec.default
            if spec.name == "family" and default in excluded_families:
                default = next(choice for choice in spec.choices if choice not in excluded_families)
            parameter = Categorical(spec.name, list(spec.choices), default=default)
        parameters.append(parameter)
    space.add(parameters)
    atoms: dict[str, list[EqualsCondition]] = {}
    for condition in schema.conditions:
        for child in condition.children:
            for value in condition.values:
                atoms.setdefault(child, []).append(
                    EqualsCondition(space[child], space[condition.parent], value)
                )
    for _child, child_atoms in atoms.items():
        space.add(child_atoms[0] if len(child_atoms) == 1 else OrConjunction(*child_atoms))
    for family in excluded_families:
        family_parameter = space["family"]
        # A `fix` override on `family` (rewritten to a schema constant) collapses
        # the parameter; the constant already constrains it, and a forbidden
        # clause on a constant is rejected by ConfigSpace.
        if isinstance(family_parameter, Constant):
            continue
        space.add(ForbiddenEqualsClause(family_parameter, family))
    _add_predicated_forbiddens(space, schema, constraints)
    return space


def _violation_clauses(spec: ParameterSpec, hp: Hyperparameter, op: SetOp) -> list[ForbiddenClause]:
    """Clauses each describing one way to violate ``op`` on ``hp`` (child side only)."""
    if isinstance(op, RangeOp):
        low, high = op.low, op.high
    elif isinstance(op, FixOp) and spec.kind in ("float", "int"):
        assert isinstance(op.value, (int, float))
        low = high = op.value
    else:  # categorical / bool fix or choices restriction
        assert spec.choices is not None
        allowed: tuple[object, ...] = op.choices if isinstance(op, ChoicesOp) else (op.value,)
        return [
            ForbiddenEqualsClause(hp, choice)
            for choice in spec.choices
            if not any(same_scalar(choice, ok) for ok in allowed)
        ]
    assert spec.bounds is not None
    clauses: list[ForbiddenClause] = []
    if low > spec.bounds[0]:
        clauses.append(ForbiddenLessThanClause(hp, low))
    if high < spec.bounds[1]:
        clauses.append(ForbiddenGreaterThanClause(hp, high))
    return clauses


def _add_predicated_forbiddens(
    space: ConfigurationSpace, schema: TuningSchema, constraints: Constraints
) -> None:
    by_name = {spec.name: spec for spec in schema.parameters}
    for guard, child, op in dynamic_forbiddens(schema, constraints):
        child_spec = by_name[child]
        guard_clauses: list[ForbiddenClause] = []
        for parent, values in guard:
            parent_hp = space[parent]
            if isinstance(parent_hp, Constant):
                continue
            guard_clauses.append(
                ForbiddenEqualsClause(parent_hp, values[0])
                if len(values) == 1
                else ForbiddenInClause(parent_hp, list(values))
            )
        for bad in _violation_clauses(child_spec, space[child], op):
            clauses = [*guard_clauses, bad]
            space.add(clauses[0] if len(clauses) == 1 else ForbiddenAndConjunction(*clauses))


ParamValue = bool | int | float | str
ParamValues = dict[str, ParamValue]


def param_value(value: object, label: str = "hyperparameter value") -> ParamValue:
    """Narrow one hyperparameter value, unwrapping the numpy scalars ConfigSpace stores."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool | int | float | str):
        return value
    raise ValueError(f"{label} is not a scalar: {value!r}")


def active_values(configuration: Configuration) -> ParamValues:
    values = dict(configuration)
    return {name: param_value(value) for name, value in values.items() if value is not None}


def default_values(space: ConfigurationSpace) -> ParamValues:
    return active_values(space.get_default_configuration())


def random_values(space: ConfigurationSpace) -> ParamValues:
    return active_values(space.sample_configuration())


def configuration_from_values(space: ConfigurationSpace, values: ParamValues) -> Configuration:
    """Build a ConfigSpace configuration from already canonical active values."""
    return Configuration(space, values=values)


def conditional_values(schema: TuningSchema, values: dict[str, ParamValue]) -> ParamValues:
    """Project proposed values through the schema's stable conditional order.

    Callers provide values for every non-constant parameter; inactive values are
    deliberately omitted so candidate identity stays canonical.
    """
    conditions = {
        child: condition for condition in schema.conditions for child in condition.children
    }
    result: ParamValues = {}
    for parameter in schema.parameters:
        condition = conditions.get(parameter.name)
        if condition is not None and result.get(condition.parent) not in condition.values:
            continue
        if parameter.kind == "constant":
            assert parameter.constant_value is not None
            result[parameter.name] = param_value(parameter.constant_value)
        else:
            result[parameter.name] = values[parameter.name]
    return result


def nonconstant_parameters(schema: TuningSchema) -> tuple[ParameterSpec, ...]:
    return tuple(item for item in schema.parameters if item.kind != "constant")
