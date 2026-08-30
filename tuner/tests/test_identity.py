from __future__ import annotations

import math

import pytest

from tuner_cli.identity import canonical_json, derive_task_seed, fingerprint, stable_id


def test_canonical_identity_is_stable_and_rejects_unsafe_values() -> None:
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert stable_id("candidate", {"a": 1}).startswith("candidate-")
    with pytest.raises(ValueError):
        canonical_json({1: "not a string key"})
    with pytest.raises(ValueError):
        canonical_json(math.nan)


def test_task_seed_is_stable_and_phase_namespaced() -> None:
    tuning = [derive_task_seed(7, "tuning", index) for index in range(8)]
    validation = [derive_task_seed(7, "validation", index) for index in range(8)]
    assert tuning == [derive_task_seed(7, "tuning", index) for index in range(8)]
    assert not set(tuning) & set(validation)
