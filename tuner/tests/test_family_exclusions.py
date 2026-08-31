from __future__ import annotations

import pytest

from tuner_cli.family_exclusions import (
    normalize_family_exclusions,
    require_candidate_family_allowed,
    validate_family_exclusions,
)
from tuner_cli.identity import candidate_from_config
from tuner_cli.schema import ActivationCondition, ParameterSpec, TuningSchema


def _schema() -> TuningSchema:
    return TuningSchema(
        "strategy",
        (),
        1,
        (ParameterSpec("family", "categorical", None, ("a", "b", "c"), "a", None),),
        (),
        "{}",
    )


def test_normalization_is_sorted_exact_and_strict() -> None:
    assert normalize_family_exclusions(("b", "a", "b")) == ("a", "b")
    with pytest.raises(ValueError):
        normalize_family_exclusions((" a",))
    with pytest.raises(ValueError):
        normalize_family_exclusions(("",))
    with pytest.raises(ValueError):
        normalize_family_exclusions((1,))  # type: ignore[arg-type]


def test_schema_and_candidate_membership_are_strict() -> None:
    schema = _schema()
    validate_family_exclusions(schema, ("a",))
    with pytest.raises(ValueError):
        validate_family_exclusions(schema, ("d",))
    with pytest.raises(ValueError):
        validate_family_exclusions(schema, ("a", "b", "c"))
    require_candidate_family_allowed(candidate_from_config({"family": "b"}), ("a",))
    with pytest.raises(ValueError):
        require_candidate_family_allowed(candidate_from_config({"family": "a"}), ("a",))


def test_empty_exclusions_need_no_family_parameter() -> None:
    schema = TuningSchema("strategy", (), 1, (), (), "{}")
    validate_family_exclusions(schema, ())
    conditional = TuningSchema(
        "strategy",
        (),
        1,
        (ParameterSpec("family", "categorical", None, ("a", "b"), "a", None),),
        (ActivationCondition("flag", (True,), ("family",)),),
        "{}",
    )
    with pytest.raises(ValueError):
        validate_family_exclusions(conditional, ("a",))
