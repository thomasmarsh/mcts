from __future__ import annotations

import pytest

from tuner_cli.constraints import (
    ChoicesOp,
    Constraint,
    FixOp,
    RangeOp,
    constrained_schema,
    decode_constraints,
    encode_constraints,
    require_candidate_allowed,
    validate_constraints,
)
from tuner_cli.identity import candidate_from_config
from tuner_cli.schema import ActivationCondition, ParameterSpec, TuningSchema
from tuner_cli.space import build_space, default_values, random_values


def _schema() -> TuningSchema:
    return TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec(
                "algorithm", "categorical", None, ("mcts", "flat_mc", "random"), "mcts", None
            ),
            ParameterSpec(
                "select", "categorical", None, ("ucb1", "ucb1_tuned", "rave"), "ucb1", None
            ),
            ParameterSpec("c", "float", (0.5, 3.0), None, 1.4, None),
            ParameterSpec("rollout_depth", "int", (1, 20), None, 10, None),
            ParameterSpec(
                "q_init", "categorical", None, ("Parent", "Zero", "Infinity"), "Parent", None
            ),
            ParameterSpec("rave_ref", "int", (1, 500), None, 50, None),
        ),
        (ActivationCondition("select", ("rave",), ("rave_ref",)),),
        "{}",
    )


def test_decode_list_and_bare_map_sugar() -> None:
    listed = decode_constraints(
        [
            {"set": {"algorithm": {"choices": ["mcts"]}}},
            {"when": {"select": ["ucb1", "ucb1_tuned"]}, "set": {"c": {"range": [1.2, 1.8]}}},
        ]
    )
    assert len(listed) == 2
    assert listed[1].when == (("select", ("ucb1", "ucb1_tuned")),)

    sugar = decode_constraints(
        {"c": {"range": [1.0, 2.0]}, "q_init": {"choices": ["Zero", "Infinity"]}}
    )
    assert sugar == (
        Constraint(
            when=(),
            sets=(("c", RangeOp(1.0, 2.0)), ("q_init", ChoicesOp(("Zero", "Infinity")))),
        ),
    )
    assert encode_constraints(sugar) == [
        {"set": {"c": {"range": [1.0, 2.0]}, "q_init": {"choices": ["Zero", "Infinity"]}}}
    ]


def test_decode_rejects_bad_shapes() -> None:
    for bad in (
        [{"set": {"c": {}}}],
        [{"set": {"c": {"fix": 1, "range": [1, 2]}}}],
        [{"set": {"c": {"nope": 1}}}],
        [{"nope": 1, "set": {"c": {"fix": 1}}}],
        [{"when": {"select": ["ucb1"]}}],
        [{"set": {}}],
        {" c": {"fix": 1}},
    ):
        with pytest.raises(ValueError):
            decode_constraints(bad)


def test_fix_is_constant_across_every_proposer() -> None:
    constraints = decode_constraints({"c": {"fix": 2.25}, "select": {"fix": "rave"}})
    schema = constrained_schema(_schema(), constraints)
    space = build_space(schema, 7)
    for _ in range(25):
        sample = random_values(space)
        assert sample["c"] == 2.25
        assert sample["select"] == "rave"
    assert default_values(space)["c"] == 2.25


def test_range_checks() -> None:
    schema = _schema()
    within = constrained_schema(schema, decode_constraints({"c": {"range": [1.2, 1.8]}}))
    space = build_space(within, 3)
    assert all(1.2 <= random_values(space)["c"] <= 1.8 for _ in range(30))
    assert default_values(space)["c"] == 1.4

    with pytest.raises(ValueError):
        validate_constraints(schema, decode_constraints({"c": {"range": [0.1, 1.0]}}))
    with pytest.raises(ValueError):
        validate_constraints(schema, decode_constraints({"c": {"range": [2.0, 1.0]}}))
    with pytest.raises(ValueError):
        validate_constraints(schema, decode_constraints({"rollout_depth": {"range": [1.5, 4.5]}}))


def test_choices_checks() -> None:
    schema = _schema()
    restricted = constrained_schema(
        schema, decode_constraints({"q_init": {"choices": ["Zero", "Infinity"]}})
    )
    space = build_space(restricted, 4)
    assert {random_values(space)["q_init"] for _ in range(40)} <= {"Zero", "Infinity"}

    with pytest.raises(ValueError):
        validate_constraints(schema, decode_constraints({"q_init": {"choices": ["Nope"]}}))
    with pytest.raises(ValueError):
        validate_constraints(
            schema, decode_constraints({"q_init": {"choices": ["Parent", "Zero", "Infinity"]}})
        )
    with pytest.raises(ValueError):
        validate_constraints(schema, decode_constraints({"c": {"choices": ["Zero"]}}))


def test_conditional_reachability() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        validate_constraints(schema, decode_constraints({"select": {"fix": "ucb1"}}))
    with pytest.raises(ValueError):
        validate_constraints(
            schema, decode_constraints({"select": {"choices": ["ucb1", "ucb1_tuned"]}})
        )
    validate_constraints(schema, decode_constraints({"select": {"choices": ["ucb1", "rave"]}}))


def test_residual_domain_must_be_non_empty() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        validate_constraints(
            schema, decode_constraints({"algorithm": {"choices": ["mcts", "flat_mc", "random"]}})
        )


def test_double_unconditional_constraint_rejected() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        validate_constraints(
            schema,
            decode_constraints(
                [{"set": {"c": {"range": [1.0, 2.0]}}}, {"set": {"c": {"fix": 1.5}}}]
            ),
        )


def test_unknown_or_constant_parameter_is_rejected() -> None:
    schema = TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("select", "categorical", None, ("ucb1", "rave"), "ucb1", None),
            ParameterSpec("frozen", "constant", None, None, None, "yes"),
        ),
        (),
        "{}",
    )
    with pytest.raises(ValueError):
        validate_constraints(schema, (Constraint(when=(), sets=(("missing", FixOp(1)),)),))
    with pytest.raises(ValueError):
        validate_constraints(schema, (Constraint(when=(), sets=(("frozen", FixOp("no")),)),))


def test_candidate_gate() -> None:
    constraints = decode_constraints(
        [
            {"set": {"select": {"choices": ["ucb1", "rave"]}}},
            {"when": {"select": ["ucb1"]}, "set": {"c": {"range": [1.2, 1.8]}}},
        ]
    )
    require_candidate_allowed(
        candidate_from_config({"algorithm": "mcts", "select": "ucb1", "c": 1.5}), constraints
    )
    with pytest.raises(ValueError):
        require_candidate_allowed(
            candidate_from_config({"algorithm": "mcts", "select": "ucb1_tuned", "c": 1.5}),
            constraints,
        )
    with pytest.raises(ValueError):
        require_candidate_allowed(
            candidate_from_config({"algorithm": "mcts", "select": "ucb1", "c": 2.5}), constraints
        )
    # Predicate not matched -> the `c` range does not apply.
    require_candidate_allowed(
        candidate_from_config({"algorithm": "mcts", "select": "rave", "c": 2.5}), constraints
    )


def test_when_validation() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        validate_constraints(
            schema,
            decode_constraints([{"when": {"missing": ["x"]}, "set": {"c": {"fix": 1.0}}}]),
        )
    with pytest.raises(ValueError):
        validate_constraints(
            schema,
            decode_constraints([{"when": {"c": [1.0]}, "set": {"rollout_depth": {"fix": 5}}}]),
        )
    with pytest.raises(ValueError):
        validate_constraints(
            schema,
            decode_constraints([{"when": {"select": ["nope"]}, "set": {"c": {"fix": 1.0}}}]),
        )
