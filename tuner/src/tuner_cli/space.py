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
    Integer,
    OrConjunction,
)

from .schema import TuningSchema


def build_space(schema: TuningSchema, seed: int) -> ConfigurationSpace:
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
            parameter = Categorical(spec.name, list(spec.choices), default=spec.default)
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
    return space


def _to_python(value: object) -> object:
    """ConfigSpace stores hyperparameter values as numpy scalars internally."""
    return value.item() if isinstance(value, np.generic) else value


def active_values(configuration: Configuration) -> dict[str, object]:
    values = dict(configuration)
    return {name: _to_python(value) for name, value in values.items() if value is not None}


def default_values(space: ConfigurationSpace) -> dict[str, object]:
    return active_values(space.get_default_configuration())


def random_values(space: ConfigurationSpace) -> dict[str, object]:
    return active_values(space.sample_configuration())


def configuration_from_values(
    space: ConfigurationSpace, values: dict[str, object]
) -> Configuration:
    """Build a ConfigSpace configuration from already canonical active values."""
    return Configuration(space, values=values)
