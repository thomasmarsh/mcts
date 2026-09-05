"""Persistent public SMAC ask/tell adapter with no evaluation authority."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ConfigSpace import ConfigurationSpace

from .domain import ModelObservation, ProposalRequest, ProposedConfiguration
from .identity import candidate_from_config
from .proposer import ModelProposer
from .space import ParamValues, active_values, configuration_from_values, param_value

if TYPE_CHECKING:
    from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade

ADAPTER_VERSION = "smac-2.4-persistent-facade-v1"


class SmacProposer(ModelProposer):
    """Holds one SMAC facade for the process lifetime.

    Each block-0 frontier observation is ``tell``-d to the standing facade
    exactly once, as it first appears; ``ask`` reads from that facade. Per-``ask``
    cost stops growing with the run -- SMAC is not rebuilt in a fresh temp dir
    and the whole frontier is not replayed through ``tell`` on every call.

    ``request.observations`` is the full block-0 frontier in global proposal
    order, an append-only prefix that only ever grows. The adapter tracks what it
    has already told and, on each ``ask``, tells only the new tail. A
    ``request.observations`` that is not an in-order extension of the told
    sequence (a reorder, a drop, a changed cost) means an upstream assumption
    broke; the adapter fails loudly rather than silently desyncing the model from
    the evidence.

    On ``--resume`` there is no special priming path: the first ``ask`` after a
    resume receives the whole frontier and the same tail-tell logic tells all of
    it once, because the told sequence starts empty. That reaches the facade
    state an uninterrupted run holds at the same point -- same observations, same
    order, same costs, same tells.
    """

    adapter_version = ADAPTER_VERSION

    def __init__(self, space: ConfigurationSpace) -> None:
        self._space = space
        self._facade: AlgorithmConfigurationFacade | None = None
        self._output: tempfile.TemporaryDirectory[str] | None = None
        self._seed: int | None = None
        self._told: list[tuple[str, float]] = []

    def ask(self, request: ProposalRequest) -> ProposedConfiguration:
        facade = self._ensure_facade(request.attempt.seed)
        self._tell_tail(facade, request.observations)
        trial = facade.ask()
        candidate = candidate_from_config(active_values(trial.config))
        return ProposedConfiguration(candidate, trial.config.origin)

    def _ensure_facade(self, seed: int) -> AlgorithmConfigurationFacade:
        """Build the facade lazily, so a ``SmacProposer`` that is never asked --
        the ``random`` policy constructs one only for its space -- stays free."""
        if self._facade is None:
            self._output = tempfile.TemporaryDirectory(prefix="mcts-tuner-smac-")
            self._seed = seed
            self._facade = _facade(self._space, seed, Path(self._output.name))
        return self._facade

    def _tell_tail(
        self,
        facade: AlgorithmConfigurationFacade,
        observations: tuple[ModelObservation, ...],
    ) -> None:
        assert self._seed is not None
        if len(observations) < len(self._told):
            raise ValueError(
                "SMAC frontier shrank: told "
                f"{len(self._told)} observations, received {len(observations)}"
            )
        for index, told in enumerate(self._told):
            current = observations[index]
            if (current.reference.observation_id, current.cost) != told:
                raise ValueError(
                    "SMAC frontier is not an in-order extension of what the "
                    f"facade was already told; diverged at observation {index}"
                )
        for observation in observations[len(self._told) :]:
            _tell(facade, self._space, observation, self._seed)
            self._told.append((observation.reference.observation_id, observation.cost))


def _facade(space: ConfigurationSpace, seed: int, output: Path) -> AlgorithmConfigurationFacade:
    from smac import AlgorithmConfigurationFacade, Scenario
    from smac.initial_design.default_design import DefaultInitialDesign
    from smac.random_design.probability_design import ProbabilityRandomDesign

    scenario = Scenario(
        space,
        output_directory=output,
        deterministic=True,
        n_workers=1,
        n_trials=1_000_000,
        seed=seed,
    )
    return AlgorithmConfigurationFacade(
        scenario,
        target_function=None,
        initial_design=DefaultInitialDesign(scenario, n_configs=0),
        random_design=ProbabilityRandomDesign(0.0, seed=seed),
        intensifier=AlgorithmConfigurationFacade.get_intensifier(scenario, max_config_calls=1),
        config_selector=AlgorithmConfigurationFacade.get_config_selector(scenario, retrain_after=1),
        logging_level=False,
    )


def _tell(
    facade: AlgorithmConfigurationFacade,
    space: ConfigurationSpace,
    observation: ModelObservation,
    seed: int,
) -> None:
    from smac.runhistory.dataclasses import TrialInfo, TrialValue

    configuration = configuration_from_values(
        space, active_values_from_candidate(observation.candidate.canonical_config)
    )
    facade.tell(TrialInfo(configuration, seed=seed), TrialValue(cost=observation.cost), save=False)


def active_values_from_candidate(canonical_config: str) -> ParamValues:
    from .codec import strict_json

    value = strict_json(canonical_config, "candidate configuration")
    if not isinstance(value, dict):
        raise ValueError("candidate configuration is not an object")
    return {key: param_value(item, "candidate configuration value") for key, item in value.items()}
