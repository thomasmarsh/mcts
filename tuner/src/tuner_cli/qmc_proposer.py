"""Deterministic scrambled-Sobol proposer over the canonical conditional schema."""

from __future__ import annotations

from scipy.stats import qmc

from .domain import ProposalRequest, ProposedConfiguration
from .identity import candidate_from_config
from .schema import TuningSchema
from .space import ParamValue, conditional_values, nonconstant_parameters

ADAPTER_VERSION = "scipy-sobol-scrambled-v1"


class QmcProposer:
    adapter_version = ADAPTER_VERSION

    def __init__(
        self, schema: TuningSchema, stream_seed: int, excluded_families: tuple[str, ...]
    ) -> None:
        self._schema, self._excluded = schema, excluded_families
        self._parameters = nonconstant_parameters(schema)
        self._engine = qmc.Sobol(d=len(self._parameters), scramble=True, rng=stream_seed)  # type: ignore[call-arg]
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
                choices = tuple(
                    item
                    for item in parameter.choices
                    if parameter.name != "family" or item not in self._excluded
                )
                values[parameter.name] = choices[
                    min(len(choices) - 1, int(coordinate * len(choices)))
                ]  # type: ignore[assignment]
        return ProposedConfiguration(
            candidate_from_config(conditional_values(self._schema, values)), "qmc_sobol"
        )
