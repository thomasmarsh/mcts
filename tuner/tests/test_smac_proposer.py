from __future__ import annotations

import json

from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition

from tuner_cli.domain import (
    ModelAttempt,
    ModelObservation,
    ObservationFrontier,
    ObservationReference,
    ProposalRequest,
    SearchEffort,
)
from tuner_cli.identity import candidate_from_config
from tuner_cli.smac_proposer import SmacProposer


def test_public_smac_adapter_warm_starts_and_returns_active_values() -> None:
    space = ConfigurationSpace(seed=3)
    algorithm = Categorical("algorithm", ["a", "b"])
    depth = Categorical("depth", [1, 2])
    space.add([algorithm, depth])
    space.add(EqualsCondition(depth, algorithm, "b"))
    effort = SearchEffort("iterations", 3)
    references = (
        ObservationReference("one", "candidate-one", "epoch", "prefix", ("task",), effort),
        ObservationReference("two", "candidate-two", "epoch", "prefix", ("task",), effort),
    )
    observations = (
        ModelObservation(candidate_from_config({"algorithm": "a"}), references[0], 0.75),
        ModelObservation(
            candidate_from_config({"algorithm": "b", "depth": 2}), references[1], 0.25
        ),
    )
    frontier = ObservationFrontier("frontier", "epoch", "prefix", ("task",), effort, ("one", "two"))
    request = ProposalRequest(observations, frontier, frozenset(), ModelAttempt(1, 3), 0, (), 1)
    proposed = SmacProposer(space).ask(request)
    values = json.loads(proposed.candidate.canonical_config)
    assert set(values) in ({"algorithm"}, {"algorithm", "depth"})
    assert proposed.origin is not None
