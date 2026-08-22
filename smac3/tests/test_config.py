"""Search-config parsing and binary-provided search-space metadata."""

from __future__ import annotations

from pathlib import Path

from smac3_cli.config import SearchConfig


def test_config_parses_parameter_types_and_conditions():
    cfg = SearchConfig._from_dict(
        {
            "parameters": {
                "family": {"type": "categorical", "choices": ["a", "b"], "default": "a"},
                "depth": {"type": "int", "bounds": [1, 10], "default": 3},
                "enabled": {"type": "bool", "default": True},
                "fixed": {"type": "constant", "value": "yes"},
            },
            "conditions": [{"if": {"family": "b"}, "then": ["depth"]}],
        }
    )
    assert [parameter.name for parameter in cfg.parameters] == ["family", "depth", "enabled", "fixed"]
    assert cfg.conditions[0].parent == "family"
    assert cfg.conditions[0].values == ["b"]


def test_parameters_from_binary_reports_search_space_and_baselines(game_nim_binary: Path):
    parameters, conditions, baselines = SearchConfig.parameters_from_binary(game_nim_binary)
    assert parameters
    assert isinstance(conditions, list)
    assert "strong" in baselines
