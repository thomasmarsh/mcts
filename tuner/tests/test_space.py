from __future__ import annotations

from tuner_cli.schema import ActivationCondition, ParameterSpec, TuningSchema
from tuner_cli.space import build_space, default_values


def test_configspace_preserves_zero_defaults_and_active_values() -> None:
    schema = TuningSchema(
        "strategy",
        (),
        1,
        (
            ParameterSpec("family", "categorical", None, ("a", "b"), "a", None),
            ParameterSpec("zero", "float", (0.0, 1.0), None, 0.0, None),
            ParameterSpec("enabled", "bool", None, (False, True), False, None),
            ParameterSpec("child", "int", (0.0, 2.0), None, 0, None),
            ParameterSpec("fixed", "constant", None, None, None, "yes"),
        ),
        (ActivationCondition("family", ("b",), ("child",)),),
        "{}",
    )
    values = default_values(build_space(schema, 5))
    assert values == {"family": "a", "zero": 0.0, "enabled": False, "fixed": "yes"}
