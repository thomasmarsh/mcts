from __future__ import annotations

import pytest

from tuner_cli.domain import ObservationContext, SearchEffort, TaskPrefix
from tuner_cli.observations import observation
from tuner_cli.proposer import (
    challenger_source_schedule,
    cost_from_observation,
    derived_seed,
    source_schedule,
    tuning_frontier,
)


def test_fixed_source_schedule_uses_weighted_fair_model_and_reserve_slots() -> None:
    assert source_schedule(4, 2, 1) == (
        "schema_default",
        "bootstrap_random",
        "smac_model",
        "random_reserve",
    )


def test_challenger_schedule_reserves_only_nonfinalist_slots() -> None:
    assert challenger_source_schedule(4, 3, 1) == ("smac_model",)
    assert challenger_source_schedule(8, 3, 1) == (
        "smac_model",
        "smac_model",
        "random_reserve",
        "smac_model",
        "smac_model",
    )
    assert source_schedule(8, 3, 2) == (
        "schema_default",
        "bootstrap_random",
        "bootstrap_random",
        "smac_model",
        "random_reserve",
        "smac_model",
        "random_reserve",
        "smac_model",
    )


def test_seed_namespaces_are_deterministic_and_independent() -> None:
    bootstrap = derived_seed(7, "bootstrap", 1)
    assert bootstrap == derived_seed(7, "bootstrap", 1)
    assert bootstrap != derived_seed(7, "reserve", 1)
    assert bootstrap != derived_seed(7, "bootstrap", 2)


def test_model_cost_and_frontier_require_tuning_context() -> None:
    prefix = TaskPrefix("prefix", "corpus", 2, ("one", "two"))
    tuning = observation(
        "candidate-a", ObservationContext("epoch", "tuning", prefix, SearchEffort(3)), (1.0, 0.5)
    )
    validation = observation(
        "candidate-a",
        ObservationContext("epoch", "validation", prefix, SearchEffort(3)),
        (1.0, 0.5),
    )
    assert cost_from_observation(tuning) == 0.25
    assert tuning_frontier((tuning,)).observation_ids == (tuning.observation_id,)
    with pytest.raises(ValueError, match="validation"):
        tuning_frontier((validation,))
