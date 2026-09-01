"""ConfigSpace bridge used only for reproducible default/random proposals."""

# ConfigSpace's own stubs annotate the `meta` kwarg on Categorical/Float/Integer/Constant
# and the `add()` methods as bare `dict`/`Hyperparameter[...]`, which pyright's strict mode
# widens to Unknown; that Unknown then propagates through every Hyperparameter-returning
# call in this file. This is a library stub gap, not a real type-safety issue here, so it's
# suppressed only in this bridge module rather than loosening the workspace's strict mode.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    Float,
    ForbiddenEqualsClause,
    Integer,
    OrConjunction,
)

from .family_exclusions import validate_family_exclusions
from .schema import ParameterSpec, TuningSchema


def build_space(
    schema: TuningSchema, seed: int, excluded_families: tuple[str, ...] = ()
) -> ConfigurationSpace:
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
        space.add(ForbiddenEqualsClause(space["family"], family))
    return space


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
