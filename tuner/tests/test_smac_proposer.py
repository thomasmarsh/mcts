from __future__ import annotations

import json

from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition

from tuner_cli.domain import (
    ModelAttempt,
    ModelObservation,
    ObservationFrontier,
    ObservationReference,
    SearchEffort,
)
from tuner_cli.identity import candidate_from_config
from tuner_cli.smac_proposer import SmacProposer


def test_public_smac_adapter_warm_starts_and_returns_active_values() -> None:
    space = ConfigurationSpace(seed=3)
    family = Categorical("family", ["a", "b"])
    depth = Categorical("depth", [1, 2])
    space.add([family, depth])
    space.add(EqualsCondition(depth, family, "b"))
    effort = SearchEffort("iterations", 3)
    references = (
        ObservationReference("one", "candidate-one", "epoch", "prefix", ("task",), effort),
        ObservationReference("two", "candidate-two", "epoch", "prefix", ("task",), effort),
    )
    observations = (
        ModelObservation(candidate_from_config({"family": "a"}), references[0], 0.75),
        ModelObservation(candidate_from_config({"family": "b", "depth": 2}), references[1], 0.25),
    )
    frontier = ObservationFrontier("frontier", "epoch", "prefix", ("task",), effort, ("one", "two"))
    proposed = SmacProposer(space).ask(observations, frontier, frozenset(), ModelAttempt(1, 3))
    values = json.loads(proposed.candidate.canonical_config)
    assert set(values) in ({"family"}, {"family", "depth"})
    assert proposed.origin is not None
