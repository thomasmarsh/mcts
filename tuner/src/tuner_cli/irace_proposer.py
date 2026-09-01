"""A small, stateless elite-centred generational proposer."""

from __future__ import annotations

import random

from .domain import Candidate, ProposalRequest, ProposedConfiguration
from .identity import candidate_from_config
from .schema import TuningSchema
from .smac_proposer import active_values_from_candidate
from .space import ParamValue, conditional_values, nonconstant_parameters, param_value

ADAPTER_VERSION = "irace-elite-generational-v1"


class IraceProposer:
    adapter_version = ADAPTER_VERSION

    def __init__(self, schema: TuningSchema, excluded_families: tuple[str, ...]) -> None:
        self._schema, self._excluded = schema, excluded_families
        self._parameters = nonconstant_parameters(schema)

    def ask(self, request: ProposalRequest) -> ProposedConfiguration:
        if not request.ranked_parents:
            raise ValueError("irace needs ranked parents")
        rng = random.Random(request.attempt.seed)
        parent = _choose_parent(request.ranked_parents, rng)
        parent_values = active_values_from_candidate(parent.canonical_config)
        v, m, g = (
            len(self._parameters),
            request.guided_candidates_per_generation,
            request.generation_index,
        )
        if m <= 0:
            raise ValueError("irace needs positive guided candidate count")
        values: dict[str, ParamValue] = {}
        for parameter in self._parameters:
            inherited = parent_values.get(parameter.name)
            if parameter.kind in {"float", "int"}:
                assert parameter.bounds is not None
                low, high = float(parameter.bounds[0]), float(parameter.bounds[1])
                mean = float(inherited) if inherited is not None else (low + high) / 2
                if parameter.kind == "int":
                    mean += 0.5
                std = (high - low) / 2 * (1 / m) ** (g / v)
                drawn = _truncated_normal(rng, mean, std, low, high)
                values[parameter.name] = drawn if parameter.kind == "float" else int(drawn // 1)
            else:
                assert parameter.choices is not None
                choices = tuple(
                    param_value(item, f"{parameter.name} choice")
                    for item in parameter.choices
                    if parameter.name != "family" or item not in self._excluded
                )
                values[parameter.name] = _categorical(rng, choices, inherited, g, v)
        return ProposedConfiguration(
            candidate_from_config(conditional_values(self._schema, values)),
            "irace_elite",
            parent_candidate_id=parent.candidate_id,
        )


def _choose_parent(parents: tuple[Candidate, ...], rng: random.Random) -> Candidate:
    return rng.choices(parents, weights=tuple(range(len(parents), 0, -1)), k=1)[0]


def _truncated_normal(
    rng: random.Random, mean: float, std: float, low: float, high: float
) -> float:
    for _ in range(100):
        value = rng.gauss(mean, std)
        if low <= value <= high:
            return value
    return min(high, max(low, mean))


def _categorical(
    rng: random.Random,
    choices: tuple[ParamValue, ...],
    parent: ParamValue | None,
    generation: int,
    dimensions: int,
) -> ParamValue:
    cap = 0.2 ** (1 / dimensions)
    parent_mass = generation / (generation + 1)
    weights = [1 - parent_mass for _ in choices]
    if parent in choices:
        weights[choices.index(parent)] += parent_mass
    weights = [min(cap, item) for item in weights]
    return rng.choices(choices, weights=weights, k=1)[0]
