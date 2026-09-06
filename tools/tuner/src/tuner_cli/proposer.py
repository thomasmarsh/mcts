"""Pure mixed-proposal policy and the model-proposer boundary."""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

from .domain import (
    Candidate,
    ModelObservation,
    Observation,
    ObservationContext,
    ObservationFrontier,
    ObservationReference,
    ProposalRequest,
    ProposalSource,
    ProposedConfiguration,
)
from .identity import canonical_json, observation_frontier, observation_reference

POLICY_VERSION = "whole-run-proposer-policy-v1"
COST_POLICY_VERSION = "smac_pair_mean_cost_v1"
SAMPLER_VERSION = "configspace-independent-v1"
ProposerPolicy = Literal["smac_mixed", "random", "qmc", "irace_generational"]
POLICIES: tuple[ProposerPolicy, ...] = ("random", "qmc", "smac_mixed", "irace_generational")


class ModelProposer(Protocol):
    """Produces one configuration from completed comparable tuning observations."""

    adapter_version: str

    def ask(self, request: ProposalRequest) -> ProposedConfiguration: ...


def source_schedule(
    cohort_size: int,
    bootstrap_candidates: int,
    random_reserve_candidates: int,
    policy: ProposerPolicy = "smac_mixed",
) -> tuple[ProposalSource, ...]:
    _validate_counts(cohort_size, bootstrap_candidates, random_reserve_candidates)
    model = cohort_size - bootstrap_candidates - random_reserve_candidates
    post = _weighted_sources(model, random_reserve_candidates, _guided_source(policy))
    bootstrap: tuple[ProposalSource, ...] = ("bootstrap_random",) * (bootstrap_candidates - 1)
    return ("schema_default", *bootstrap, *post)


def challenger_source_schedule(
    cohort_size: int,
    finalists: int,
    random_reserve_candidates: int,
    policy: ProposerPolicy = "smac_mixed",
) -> tuple[ProposalSource, ...]:
    if any(
        isinstance(value, bool) for value in (cohort_size, finalists, random_reserve_candidates)
    ):
        raise ValueError("proposal counts must be integers")
    challengers = cohort_size - finalists
    if finalists < 1 or challengers < 1 or random_reserve_candidates < 1:
        raise ValueError("invalid finalist, reserve, and cohort count relationship")
    reserve = min(random_reserve_candidates, challengers - 1)
    return _weighted_sources(challengers - reserve, reserve, _guided_source(policy))


def _guided_source(policy: ProposerPolicy) -> ProposalSource:
    sources: dict[str, ProposalSource] = {
        "smac_mixed": "smac_model",
        "random": "random_search",
        "qmc": "qmc_search",
        "irace_generational": "irace_model",
    }
    result = sources.get(policy)
    return result if result is not None else _invalid_policy(policy)


def _invalid_policy(policy: object) -> ProposalSource:
    raise ValueError(f"unknown proposer policy {policy!r}")


def _validate_counts(cohort_size: int, bootstrap: int, reserve: int) -> None:
    values = (cohort_size, bootstrap, reserve)
    if any(isinstance(value, bool) for value in values):
        raise ValueError("proposal counts must be integers")
    post = cohort_size - bootstrap
    if bootstrap < 2 or post < 2 or not 1 <= reserve < post:
        raise ValueError("invalid bootstrap, reserve, and cohort count relationship")


def _weighted_sources(
    model: int, reserve: int, guided: ProposalSource
) -> tuple[ProposalSource, ...]:
    result: list[ProposalSource] = []
    sources: tuple[ProposalSource, ProposalSource] = (guided, "random_reserve")
    emitted: dict[ProposalSource, int] = {guided: 0, "random_reserve": 0}
    limits: dict[ProposalSource, int] = {guided: model, "random_reserve": reserve}
    for index in range(model + reserve):
        choices: list[ProposalSource] = [
            source for source in sources if emitted[source] < limits[source]
        ]
        source = max(
            choices,
            key=lambda item: (
                (index + 1) * limits[item] - emitted[item] * (model + reserve),
                item == guided,
            ),
        )
        result.append(source)
        emitted[source] += 1
    return tuple(result)


def derived_seed(root_seed: int, namespace: str, ordinal: int = 0) -> int:
    if (
        namespace not in {"bootstrap", "reserve", "smac", "random_search", "irace", "qmc"}
        or ordinal < 0
    ):
        raise ValueError("invalid proposal seed namespace")
    payload = {
        "version": "proposal-seed-v1",
        "root_seed": root_seed,
        "namespace": namespace,
        "ordinal": ordinal,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(payload).encode()).digest()[:4], "big")


def empty_frontier(context: ObservationContext) -> ObservationFrontier:
    return ObservationFrontier(
        "frontier-empty-v1",
        context.objective_epoch_id,
        context.task_prefix.prefix_id,
        context.task_prefix.task_ids,
        context.search_effort,
        (),
    )


def tuning_frontier(observations: tuple[Observation, ...]) -> ObservationFrontier:
    if not observations:
        raise ValueError("a model frontier needs completed tuning observations")
    if any(item.phase != "tuning" for item in observations):
        raise ValueError("validation observations cannot enter a model frontier")
    return observation_frontier(tuple(observation_reference(item) for item in observations))


def model_observations(
    observations: tuple[Observation, ...],
    candidate_by_id: dict[str, Candidate],
    frontier: ObservationFrontier,
) -> tuple[ModelObservation, ...]:
    result: list[ModelObservation] = []
    for item in observations:
        reference = observation_reference(item)
        _check_tuning_context(reference, frontier)
        candidate = candidate_by_id.get(item.candidate_id)
        if candidate is None:
            raise ValueError("model observation candidate is absent")
        result.append(ModelObservation(candidate, reference, cost_from_observation(item)))
    if tuple(item.reference.observation_id for item in result) != frontier.observation_ids:
        raise ValueError("model observations do not match the visible frontier")
    return tuple(result)


def _check_tuning_context(reference: ObservationReference, frontier: ObservationFrontier) -> None:
    expected = (
        frontier.objective_epoch_id,
        frontier.prefix_id,
        frontier.task_ids,
        frontier.search_effort,
    )
    actual = (
        reference.objective_epoch_id,
        reference.prefix_id,
        reference.task_ids,
        reference.search_effort,
    )
    labels = ("epoch", "prefix", "ordered task IDs", "search effort")
    for label, left, right in zip(labels, actual, expected, strict=True):
        if left != right:
            raise ValueError(f"model observation differs on {label}")


def cost_from_observation(value: Observation) -> float:
    if value.phase != "tuning":
        raise ValueError("only tuning observations have a model cost")
    cost = 1.0 - value.estimate.mean
    if not 0.0 <= cost <= 1.0:
        raise ValueError("tuning observation cost is outside [0, 1]")
    return cost
