"""Search-config parsing -- pure dict/YAML parsing, no binary required."""

from __future__ import annotations

import pytest

from tuner_cli.__main__ import _apply_overrides
from tuner_cli.config import SearchConfig


def test_config_parses_parameter_types_and_conditions():
    cfg = SearchConfig._from_dict(
        {
            "parameters": {
                "family": {
                    "type": "categorical",
                    "choices": ["a", "b"],
                    "default": "a",
                },
                "depth": {"type": "int", "bounds": [1, 10], "default": 3},
                "enabled": {"type": "bool", "default": True},
                "fixed": {"type": "constant", "value": "yes"},
            },
            "conditions": [{"if": {"family": "b"}, "then": ["depth"]}],
        }
    )
    assert [parameter.name for parameter in cfg.parameters] == [
        "family",
        "depth",
        "enabled",
        "fixed",
    ]
    assert cfg.conditions[0].parent == "family"
    assert cfg.conditions[0].values == ["b"]


def test_default_resource_and_rating_policy_reproduces_current_behavior():
    cfg = SearchConfig.defaults()

    assert cfg.optimizer.resource.min_pairs == 5
    assert cfg.optimizer.resource.max_pairs == 15
    assert cfg.optimizer.rating.sigma_stop == 2.0
    assert cfg.optimizer.rating.conservative_k == 3.0
    assert not cfg.optimizer.pruning.enabled


def test_nested_policy_yaml_and_cli_overrides_resolve_to_typed_values():
    cfg = SearchConfig._from_dict(
        {
            "optimizer": {
                "resource": {"min_pairs": 7, "max_pairs": 19},
                "rating": {"sigma_stop": None, "conservative_k": 2.5},
                "pruning": {"startup_trials": 12},
                "sampler": {"startup_trials": 8},
            }
        }
    )

    _apply_overrides(
        cfg,
        {
            "optimizer.resource.max_pairs": "21",
            "optimizer.sampler.startup_trials": "11",
        },
    )
    cfg.validate()

    assert cfg.optimizer.resource.min_pairs == 7
    assert cfg.optimizer.resource.max_pairs == 21
    assert cfg.optimizer.rating.sigma_stop is None
    assert cfg.optimizer.rating.conservative_k == 2.5
    assert cfg.optimizer.pruning.startup_trials == 12
    assert cfg.optimizer.sampler.startup_trials == 11


def test_legacy_eta_sets_reduction_factor_but_conflicting_aliases_fail():
    cfg = SearchConfig._from_dict({"optimizer": {"eta": 4}})
    assert cfg.optimizer.pruning.reduction_factor == 4

    with pytest.raises(ValueError, match="conflicts"):
        SearchConfig._from_dict(
            {
                "optimizer": {
                    "eta": 4,
                    "pruning": {"reduction_factor": 3},
                }
            }
        )


@pytest.mark.parametrize(
    ("optimizer", "message"),
    [
        ({"resource": {"min_pairs": 6, "max_pairs": 5}}, "must not exceed"),
        ({"rating": {"sigma_stop": 0}}, "positive finite"),
        ({"sampler": {"kind": "random"}}, "must be 'tpe'"),
        ({"pruning": {"kind": "median"}}, "must be 'hyperband'"),
        (
            {"pruning": {"reduction_factor": 2.0}},
            "must be an integer at least 2",
        ),
        (
            {"pruning": {"reduction_factor": 1}},
            "must be an integer at least 2",
        ),
        ({"sampler": {"startup_trials": -1}}, "nonnegative integer"),
        ({"resource": {"minimum_pairs": 5}}, "unsupported keys"),
    ],
)
def test_invalid_policy_values_are_rejected(optimizer, message):
    with pytest.raises(ValueError, match=message):
        SearchConfig._from_dict({"optimizer": optimizer})


def test_pruning_accepts_parallel_and_automatic_evaluation_slots():
    cfg = SearchConfig._from_dict({"optimizer": {"pruning": {"enabled": True}}})

    assert cfg.optimizer.n_workers is None
    assert (
        SearchConfig._from_dict(
            {"optimizer": {"n_workers": 2, "pruning": {"enabled": True}}}
        ).optimizer.n_workers
        == 2
    )
