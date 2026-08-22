"""Search-config parsing -- pure dict/YAML parsing, no binary required."""

from __future__ import annotations

from tuner_cli.config import SearchConfig


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