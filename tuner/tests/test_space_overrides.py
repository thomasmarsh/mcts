from __future__ import annotations

import pytest

from tuner_cli.schema import ActivationCondition, ParameterSpec, TuningSchema
from tuner_cli.space import build_space, default_values, random_values
from tuner_cli.space_overrides import (
    ChoicesOverride,
    FixOverride,
    RangeOverride,
    constrained_schema,
    decode_space_overrides,
    encode_space_overrides,
    validate_space_overrides,
)


def _schema() -> TuningSchema:
    return TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("family", "categorical", None, ("ucb", "grave", "puct"), "ucb", None),
            ParameterSpec("c", "float", (0.5, 3.0), None, 1.4, None),
            ParameterSpec("rollout_depth", "int", (1, 20), None, 10, None),
            ParameterSpec(
                "q_init", "categorical", None, ("Parent", "Zero", "Infinity"), "Parent", None
            ),
            ParameterSpec("grave_ref", "int", (1, 500), None, 50, None),
        ),
        (ActivationCondition("family", ("grave",), ("grave_ref",)),),
        "{}",
    )


def test_decode_round_trips_and_rejects_bad_shapes() -> None:
    raw = {"c": {"range": [1.0, 2.0]}, "q_init": {"choices": ["Zero", "Infinity"]}}
    decoded = decode_space_overrides(raw)
    assert encode_space_overrides(decoded) == {
        "c": {"range": [1.0, 2.0]},
        "q_init": {"choices": ["Zero", "Infinity"]},
    }
    for bad in (
        {"c": {}},
        {"c": {"fix": 1, "range": [1, 2]}},
        {"c": {"nope": 1}},
        {" c": {"fix": 1}},
    ):
        with pytest.raises(ValueError):
            decode_space_overrides(bad)


def test_fix_is_constant_across_every_proposer() -> None:
    overrides = decode_space_overrides({"c": {"fix": 2.25}, "family": {"fix": "grave"}})
    schema = constrained_schema(_schema(), overrides)
    space = build_space(schema, 7)
    for _ in range(25):
        sample = random_values(space)
        assert sample["c"] == 2.25
        assert sample["family"] == "grave"
    assert default_values(space)["c"] == 2.25


def test_range_checks() -> None:
    schema = _schema()
    within = constrained_schema(schema, decode_space_overrides({"c": {"range": [1.2, 1.8]}}))
    space = build_space(within, 3)
    assert all(1.2 <= random_values(space)["c"] <= 1.8 for _ in range(30))
    assert default_values(space)["c"] == 1.4  # schema default clamped in

    with pytest.raises(ValueError):
        validate_space_overrides(schema, decode_space_overrides({"c": {"range": [0.1, 1.0]}}))
    with pytest.raises(ValueError):
        validate_space_overrides(schema, decode_space_overrides({"c": {"range": [2.0, 1.0]}}))
    with pytest.raises(ValueError):
        validate_space_overrides(
            schema, decode_space_overrides({"rollout_depth": {"range": [1.5, 4.5]}})
        )


def test_choices_checks() -> None:
    schema = _schema()
    restricted = constrained_schema(
        schema, decode_space_overrides({"q_init": {"choices": ["Zero", "Infinity"]}})
    )
    space = build_space(restricted, 4)
    assert {random_values(space)["q_init"] for _ in range(40)} <= {"Zero", "Infinity"}

    with pytest.raises(ValueError):
        validate_space_overrides(schema, decode_space_overrides({"q_init": {"choices": ["Nope"]}}))
    with pytest.raises(ValueError):
        validate_space_overrides(
            schema,
            decode_space_overrides({"q_init": {"choices": ["Parent", "Zero", "Infinity"]}}),
        )
    with pytest.raises(ValueError):
        validate_space_overrides(schema, decode_space_overrides({"c": {"choices": ["Zero"]}}))


def test_conditional_reachability() -> None:
    schema = _schema()
    # Fixing family away from `grave` orphans the conditional `grave_ref`.
    with pytest.raises(ValueError):
        validate_space_overrides(schema, decode_space_overrides({"family": {"fix": "ucb"}}))
    with pytest.raises(ValueError):
        validate_space_overrides(
            schema, decode_space_overrides({"family": {"choices": ["ucb", "puct"]}})
        )
    # Keeping `grave` reachable is fine.
    validate_space_overrides(
        schema, decode_space_overrides({"family": {"choices": ["ucb", "grave"]}})
    )


def test_compose_with_exclusions() -> None:
    schema = constrained_schema(
        _schema(), decode_space_overrides({"q_init": {"choices": ["Parent", "Zero"]}})
    )
    space = build_space(schema, 9, ("grave",))
    for _ in range(30):
        sample = random_values(space)
        assert sample["family"] != "grave"
        assert sample["q_init"] in {"Parent", "Zero"}


def test_unknown_or_constant_parameter_is_rejected() -> None:
    schema = TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("family", "categorical", None, ("ucb", "grave"), "ucb", None),
            ParameterSpec("frozen", "constant", None, None, None, "yes"),
        ),
        (),
        "{}",
    )
    with pytest.raises(ValueError):
        validate_space_overrides(schema, {"missing": FixOverride(1)})
    with pytest.raises(ValueError):
        validate_space_overrides(schema, {"frozen": FixOverride("no")})


def test_typed_override_objects_encode() -> None:
    overrides = {
        "c": RangeOverride(1.0, 2.0),
        "q_init": ChoicesOverride(("Zero",)),
        "rollout_depth": FixOverride(5),
    }
    assert encode_space_overrides(overrides) == {
        "c": {"range": [1.0, 2.0]},
        "q_init": {"choices": ["Zero"]},
        "rollout_depth": {"fix": 5},
    }
