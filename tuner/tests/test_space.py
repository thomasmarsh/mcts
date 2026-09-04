from __future__ import annotations

from tuner_cli.schema import ActivationCondition, ParameterSpec, TuningSchema
from tuner_cli.space import build_space, default_values, random_values


def test_configspace_preserves_zero_defaults_and_active_values() -> None:
    schema = TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("algorithm", "categorical", None, ("a", "b"), "a", None),
            ParameterSpec("zero", "float", (0.0, 1.0), None, 0.0, None),
            ParameterSpec("enabled", "bool", None, (False, True), False, None),
            ParameterSpec("child", "int", (0.0, 2.0), None, 0, None),
            ParameterSpec("fixed", "constant", None, None, None, "yes"),
        ),
        (ActivationCondition("algorithm", ("b",), ("child",)),),
        "{}",
    )
    values = default_values(build_space(schema, 5))
    assert values == {"algorithm": "a", "zero": 0.0, "enabled": False, "fixed": "yes"}


def test_constrained_default_uses_first_allowed_family_and_active_children() -> None:
    from tuner_cli.constraints import constrained_schema, decode_constraints

    schema = TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("algorithm", "categorical", None, ("a", "b", "c"), "a", None),
            ParameterSpec("depth", "int", (1.0, 3.0), None, 2, None),
        ),
        (ActivationCondition("algorithm", ("b",), ("depth",)),),
        "{}",
    )
    narrowed = constrained_schema(
        schema, decode_constraints({"algorithm": {"choices": ["b", "c"]}})
    )
    assert default_values(build_space(narrowed, 5)) == {"algorithm": "b", "depth": 2}


def test_transitively_conditioned_parameter_builds_a_valid_space() -> None:
    # `c` is active only when `select == rave`, and `select` itself is active
    # only when `algorithm == mcts`: ConfigSpace needs the full chain.
    schema = TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("algorithm", "categorical", None, ("mcts", "random"), "mcts", None),
            ParameterSpec("select", "categorical", None, ("ucb1", "rave"), "ucb1", None),
            ParameterSpec("c", "float", (0.5, 3.0), None, 1.4, None),
        ),
        (
            ActivationCondition("algorithm", ("mcts",), ("select",)),
            ActivationCondition("select", ("rave",), ("c",)),
        ),
        "{}",
    )
    space = build_space(schema, 5)
    for _ in range(200):
        sample = random_values(space)
        if "c" in sample:
            assert sample["select"] == "rave" and sample["algorithm"] == "mcts"
        if sample["algorithm"] == "random":
            assert "select" not in sample and "c" not in sample
