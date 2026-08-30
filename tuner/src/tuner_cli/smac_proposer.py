"""One-shot public SMAC ask/tell adapter with no evaluation authority."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ConfigSpace import ConfigurationSpace

from .domain import ModelAttempt, ModelObservation, ObservationFrontier, ProposedConfiguration
from .identity import candidate_from_config
from .proposer import ModelProposer
from .space import active_values, configuration_from_values

if TYPE_CHECKING:
    from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade

ADAPTER_VERSION = "smac-2.4-public-ask-v1"


class SmacProposer(ModelProposer):
    """Rebuilds SMAC from authoritative observations for each individual ask."""

    def __init__(self, space: ConfigurationSpace) -> None:
        self._space = space

    def ask(
        self,
        observations: tuple[ModelObservation, ...],
        frontier: ObservationFrontier,
        excluded_fingerprints: frozenset[str],
        attempt: ModelAttempt,
    ) -> ProposedConfiguration:
        del frontier, excluded_fingerprints
        with tempfile.TemporaryDirectory(prefix="mcts-tuner-smac-") as output:
            facade = _facade(self._space, attempt.seed, Path(output))
            for observation in observations:
                _tell(facade, self._space, observation, attempt.seed)
            trial = facade.ask()
            candidate = candidate_from_config(active_values(trial.config))
        return ProposedConfiguration(candidate, trial.config.origin)


def _facade(space: ConfigurationSpace, seed: int, output: Path) -> AlgorithmConfigurationFacade:
    from smac import AlgorithmConfigurationFacade, Scenario
    from smac.initial_design.default_design import DefaultInitialDesign
    from smac.random_design.probability_design import ProbabilityRandomDesign

    scenario = Scenario(
        space,
        output_directory=output,
        deterministic=True,
        n_workers=1,
        n_trials=1,
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


def active_values_from_candidate(canonical_config: str) -> dict[str, object]:
    from .codec import strict_json

    value = strict_json(canonical_config, "candidate configuration")
    if not isinstance(value, dict):
        raise ValueError("candidate configuration is not an object")
    return dict(value)
