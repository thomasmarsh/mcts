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
                "algorithm", "categorical", None, ("mcts", "bandit", "random"), "mcts", None
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


def test_narrowing_prunes_unreachable_conditionals() -> None:
    schema = _schema()
    # Excluding `rave` leaves the `rave_ref` condition with no trigger value:
    # the condition and its orphaned child are pruned, not rejected.
    for constraints in (
        decode_constraints({"select": {"fix": "ucb1"}}),
        decode_constraints({"select": {"choices": ["ucb1", "ucb1_tuned"]}}),
    ):
        validate_constraints(schema, constraints)
        pruned = constrained_schema(schema, constraints)
        assert pruned.conditions == ()
        assert "rave_ref" not in {parameter.name for parameter in pruned.parameters}

    # Keeping `rave` keeps the condition and its child intact.
    kept = constrained_schema(schema, decode_constraints({"select": {"choices": ["ucb1", "rave"]}}))
    assert kept.conditions == schema.conditions
    assert "rave_ref" in {parameter.name for parameter in kept.parameters}


def test_residual_domain_must_be_non_empty() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        validate_constraints(
            schema, decode_constraints({"algorithm": {"choices": ["mcts", "bandit", "random"]}})
        )


def test_pruning_propagates_through_a_condition_chain() -> None:
    # `deep` is active only when `mid == "on"`, which is active only when
    # `top == "mcts"`. Excluding `mcts` must prune both conditions and both
    # children, not leave `deep` dangling on a vanished parent.
    schema = TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("top", "categorical", None, ("mcts", "negamax"), "mcts", None),
            ParameterSpec("mid", "categorical", None, ("on", "off"), "on", None),
            ParameterSpec("deep", "int", (1, 10), None, 5, None),
        ),
        (
            ActivationCondition("top", ("mcts",), ("mid",)),
            ActivationCondition("mid", ("on",), ("deep",)),
        ),
        "{}",
    )
    pruned = constrained_schema(schema, decode_constraints({"top": {"choices": ["negamax"]}}))
    assert pruned.conditions == ()
    assert {parameter.name for parameter in pruned.parameters} == {"top"}


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


def test_predicated_range_is_a_forbidden_clause_in_configspace() -> None:
    constraints = decode_constraints(
        [{"when": {"select": ["ucb1", "ucb1_tuned"]}, "set": {"c": {"range": [1.2, 1.8]}}}]
    )
    schema = constrained_schema(_schema(), constraints)
    # The predicate crosses the space, so `c` keeps its full domain in the schema.
    assert next(p for p in schema.parameters if p.name == "c").bounds == (0.5, 3.0)
    space = build_space(schema, 11, constraints=constraints)
    seen_other = False
    for _ in range(300):
        sample = random_values(space)
        if sample["select"] in ("ucb1", "ucb1_tuned"):
            assert 1.2 <= sample["c"] <= 1.8
        else:
            seen_other = True
            if not 1.2 <= sample["c"] <= 1.8:
                break
    assert seen_other, "expected some samples on the unconstrained `select` branch"


def test_predicated_choices_is_a_forbidden_clause() -> None:
    constraints = decode_constraints(
        [{"when": {"select": ["rave"]}, "set": {"q_init": {"choices": ["Zero"]}}}]
    )
    schema = constrained_schema(_schema(), constraints)
    space = build_space(schema, 5, constraints=constraints)
    for _ in range(300):
        sample = random_values(space)
        if sample["select"] == "rave":
            assert sample["q_init"] == "Zero"


def test_predicated_constraint_folds_when_parent_is_entailed() -> None:
    # `select` fixed to `rave` unconditionally makes the predicate always hold.
    constraints = decode_constraints(
        [
            {"set": {"select": {"fix": "rave"}}},
            {"when": {"select": ["rave"]}, "set": {"c": {"range": [2.0, 2.5]}}},
        ]
    )
    schema = constrained_schema(_schema(), constraints)
    c = next(p for p in schema.parameters if p.name == "c")
    assert c.bounds == (2.0, 2.5)
    space = build_space(schema, 3, constraints=constraints)
    assert all(2.0 <= random_values(space)["c"] <= 2.5 for _ in range(30))


def test_predicated_constraint_dropped_when_parent_is_contradicted() -> None:
    constraints = decode_constraints(
        [
            {"set": {"select": {"choices": ["ucb1", "rave"]}}},
            {"when": {"select": ["ucb1_tuned"]}, "set": {"c": {"range": [2.0, 2.5]}}},
        ]
    )
    schema = constrained_schema(_schema(), constraints)
    assert next(p for p in schema.parameters if p.name == "c").bounds == (0.5, 3.0)
    space = build_space(schema, 3, constraints=constraints)
    assert any(random_values(space)["c"] < 2.0 for _ in range(50))


def test_predicated_default_is_retargeted_so_configspace_accepts_the_space() -> None:
    # Schema default select=ucb1, c=1.4 would violate the predicated range.
    constraints = decode_constraints(
        [{"when": {"select": ["ucb1"]}, "set": {"c": {"range": [2.0, 2.5]}}}]
    )
    schema = constrained_schema(_schema(), constraints)
    assert next(p for p in schema.parameters if p.name == "c").default == 2.0
    assert default_values(build_space(schema, 1, constraints=constraints))["c"] == 2.0


def test_predicated_multi_parent_when_is_conjunctive() -> None:
    schema_in = TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("select", "categorical", None, ("ucb1", "rave"), "ucb1", None),
            ParameterSpec("q_init", "categorical", None, ("Parent", "Zero"), "Parent", None),
            ParameterSpec("c", "float", (0.5, 3.0), None, 1.4, None),
        ),
        (),
        "{}",
    )
    constraints = decode_constraints(
        [{"when": {"select": ["ucb1"], "q_init": ["Zero"]}, "set": {"c": {"range": [1.2, 1.4]}}}]
    )
    schema = constrained_schema(schema_in, constraints)
    space = build_space(schema, 7, constraints=constraints)
    for _ in range(400):
        sample = random_values(space)
        if sample["select"] == "ucb1" and sample["q_init"] == "Zero":
            assert 1.2 <= sample["c"] <= 1.4


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
