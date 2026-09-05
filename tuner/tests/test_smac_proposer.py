from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition

from tuner_cli import smac_proposer
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

EFFORT = SearchEffort("iterations", 3)


def _space() -> ConfigurationSpace:
    space = ConfigurationSpace(seed=3)
    algorithm = Categorical("algorithm", ["a", "b"])
    depth = Categorical("depth", [1, 2])
    space.add([algorithm, depth])
    space.add(EqualsCondition(depth, algorithm, "b"))
    return space


def _observation(index: int, cost: float) -> ModelObservation:
    reference = ObservationReference(
        f"obs-{index}", f"candidate-{index}", "epoch", "prefix", ("task",), EFFORT
    )
    return ModelObservation(candidate_from_config({"algorithm": "a"}), reference, cost)


def _request(observations: tuple[ModelObservation, ...], attempt: int = 1) -> ProposalRequest:
    ids = tuple(item.reference.observation_id for item in observations)
    frontier = ObservationFrontier("frontier", "epoch", "prefix", ("task",), EFFORT, ids)
    return ProposalRequest(observations, frontier, frozenset(), ModelAttempt(attempt, 3), 0, (), 1)


class _FakeFacade:
    """Records every ``tell`` and hands back a fixed configuration on ``ask``."""

    def __init__(self, space: ConfigurationSpace) -> None:
        self._space = space
        self.told_costs: list[float] = []

    def tell(self, info: object, value: object, save: bool) -> None:  # noqa: FBT001
        self.told_costs.append(value.cost)  # type: ignore[attr-defined]

    def ask(self) -> object:
        return SimpleNamespace(config=self._space.get_default_configuration())


@pytest.fixture
def fake_facades(monkeypatch: pytest.MonkeyPatch) -> list[_FakeFacade]:
    built: list[_FakeFacade] = []

    def _build(space: ConfigurationSpace, seed: int, output: object) -> _FakeFacade:
        facade = _FakeFacade(space)
        built.append(facade)
        return facade

    monkeypatch.setattr(smac_proposer, "_facade", _build)
    return built


def test_public_smac_adapter_warm_starts_and_returns_active_values() -> None:
    space = _space()
    references = (
        ObservationReference("one", "candidate-one", "epoch", "prefix", ("task",), EFFORT),
        ObservationReference("two", "candidate-two", "epoch", "prefix", ("task",), EFFORT),
    )
    observations = (
        ModelObservation(candidate_from_config({"algorithm": "a"}), references[0], 0.75),
        ModelObservation(
            candidate_from_config({"algorithm": "b", "depth": 2}), references[1], 0.25
        ),
    )
    frontier = ObservationFrontier("frontier", "epoch", "prefix", ("task",), EFFORT, ("one", "two"))
    request = ProposalRequest(observations, frontier, frozenset(), ModelAttempt(1, 3), 0, (), 1)
    proposed = SmacProposer(space).ask(request)
    values = json.loads(proposed.candidate.canonical_config)
    assert set(values) in ({"algorithm"}, {"algorithm", "depth"})
    assert proposed.origin is not None


def test_a_proposer_that_is_never_asked_builds_no_facade(fake_facades: list[_FakeFacade]) -> None:
    SmacProposer(_space())
    assert fake_facades == []


def test_each_frontier_observation_is_told_exactly_once_across_a_growing_run(
    fake_facades: list[_FakeFacade],
) -> None:
    proposer = SmacProposer(_space())
    observations = tuple(_observation(index, cost=index / 10) for index in range(6))

    proposer.ask(_request(observations[:2], attempt=1))
    proposer.ask(_request(observations[:2], attempt=2))  # no new observations
    proposer.ask(_request(observations[:5], attempt=3))
    proposer.ask(_request(observations, attempt=4))

    assert len(fake_facades) == 1
    assert fake_facades[0].told_costs == [item.cost for item in observations]


def test_resume_tells_the_same_sequence_as_an_uninterrupted_run(
    fake_facades: list[_FakeFacade],
) -> None:
    observations = tuple(_observation(index, cost=index / 10) for index in range(8))

    uninterrupted = SmacProposer(_space())
    for split in (2, 4, 6, 8):
        uninterrupted.ask(_request(observations[:split], attempt=split))

    resumed = SmacProposer(_space())
    resumed.ask(_request(observations, attempt=1))  # first ask sees the whole frontier

    assert fake_facades[0].told_costs == fake_facades[1].told_costs


def test_a_reordered_frontier_fails_loudly(fake_facades: list[_FakeFacade]) -> None:
    proposer = SmacProposer(_space())
    observations = tuple(_observation(index, cost=index / 10) for index in range(4))
    proposer.ask(_request(observations, attempt=1))

    reordered = (observations[1], observations[0], observations[2], observations[3])
    with pytest.raises(ValueError, match="in-order extension"):
        proposer.ask(_request(reordered, attempt=2))


def test_a_dropped_frontier_observation_fails_loudly(fake_facades: list[_FakeFacade]) -> None:
    proposer = SmacProposer(_space())
    observations = tuple(_observation(index, cost=index / 10) for index in range(4))
    proposer.ask(_request(observations, attempt=1))

    with pytest.raises(ValueError, match="frontier shrank"):
        proposer.ask(_request(observations[:3], attempt=2))


def test_a_changed_cost_fails_loudly(fake_facades: list[_FakeFacade]) -> None:
    proposer = SmacProposer(_space())
    observations = tuple(_observation(index, cost=index / 10) for index in range(4))
    proposer.ask(_request(observations, attempt=1))

    mutated = observations[:2] + (_observation(2, cost=0.99), observations[3])
    with pytest.raises(ValueError, match="in-order extension"):
        proposer.ask(_request(mutated, attempt=2))
