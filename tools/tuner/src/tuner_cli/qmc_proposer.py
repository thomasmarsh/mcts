"""Deterministic scrambled-Sobol proposer over the canonical conditional schema."""

from __future__ import annotations

from scipy.stats import qmc

from .domain import ProposalRequest, ProposedConfiguration
from .identity import candidate_from_config
from .schema import TuningSchema
from .space import ParamValue, conditional_values, nonconstant_parameters, param_value

ADAPTER_VERSION = "scipy-sobol-scrambled-v1"


class QmcProposer:
    adapter_version = ADAPTER_VERSION

    def __init__(self, schema: TuningSchema, stream_seed: int) -> None:
        self._schema = schema
        self._parameters = nonconstant_parameters(schema)
        self._engine = qmc.Sobol(d=len(self._parameters), scramble=True, rng=stream_seed)
        self._points: list[list[float]] = []

    def ask(self, request: ProposalRequest) -> ProposedConfiguration:
        index = request.attempt.source_attempt - 1
        while len(self._points) <= index:
            self._points.extend(self._engine.random(1).tolist())
        values: dict[str, ParamValue] = {}
        for parameter, coordinate in zip(self._parameters, self._points[index], strict=True):
            if parameter.kind == "float":
                assert parameter.bounds is not None
                values[parameter.name] = parameter.bounds[0] + coordinate * (
                    parameter.bounds[1] - parameter.bounds[0]
                )
            elif parameter.kind == "int":
                assert parameter.bounds is not None
                low, high = int(parameter.bounds[0]), int(parameter.bounds[1])
                values[parameter.name] = min(high, low + int(coordinate * (high - low + 1)))
            else:
                assert parameter.choices is not None
                choices = parameter.choices
                values[parameter.name] = param_value(
                    choices[min(len(choices) - 1, int(coordinate * len(choices)))],
                    parameter.name,
                )
        return ProposedConfiguration(
            candidate_from_config(conditional_values(self._schema, values)), "qmc_sobol"
        )
