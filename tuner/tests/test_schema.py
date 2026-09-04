from __future__ import annotations

import copy

import pytest

from tuner_cli.schema import decode_game_spec


def _description() -> dict[str, object]:
    return {
        "kind": "example",
        "label": "Example",
        "description": "test",
        "default_config": {"size": 5},
        "ai_presets": [{"id": "default", "label": "Default", "description": "test"}],
        "tuning": {
            "id": "strategy",
            "baselines": ["default"],
            "eval_rounds": 1,
            "game_config": {"size": 5},
            "parameters": [
                {"name": "ratio", "type": "float", "bounds": [0.0, 1.0], "default": 0.5},
                {"name": "depth", "type": "int", "bounds": [1, 3], "default": 2},
                {"name": "algorithm", "type": "categorical", "choices": ["a", "b"], "default": "a"},
                {"name": "flag", "type": "bool", "default": False},
                {"name": "fixed", "type": "constant", "value": "fixed"},
            ],
            "conditions": [{"if": {"flag": True}, "then": ["algorithm"]}],
        },
    }


def test_decoder_keeps_game_and_strategy_configs_separate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    raw = _description()
    spec = decode_game_spec(raw, tmp_path / "game-druid", "0" * 64)
    assert spec.default_game_config == '{"size":5}'
    assert spec.tuning.game_config == '{"size":5}'
    assert spec.tuning.conditions[0].values == (True,)
    raw["kind"] = "nim"
    assert decode_game_spec(raw, tmp_path / "game-nim", "0" * 64).kind == "nim"


def test_decoder_rejects_contract_and_condition_errors(tmp_path) -> None:
    raw = _description()
    raw["unknown"] = True
    with pytest.raises(ValueError, match="invalid fields"):
        decode_game_spec(raw, tmp_path / "game", "0" * 64)

    raw = _description()
    tuning = raw["tuning"]
    assert isinstance(tuning, dict)
    tuning["conditions"] = [
        {"if": {"flag": True}, "then": ["algorithm"]},
        {"if": {"algorithm": "a"}, "then": ["flag"]},
    ]
    with pytest.raises(ValueError, match="cycle"):
        decode_game_spec(raw, tmp_path / "game", "0" * 64)


def test_decoder_accepts_and_describes_a_game_config_axis(tmp_path) -> None:
    raw = _description()
    raw["default_config"] = {"size": 13}
    raw["config_schema"] = {
        "parameters": [{"name": "size", "type": "int", "bounds": [3, 19], "default": 13}],
        "conditions": [],
    }
    tuning = raw["tuning"]
    assert isinstance(tuning, dict)
    tuning["game_config"] = {"size": 13}
    tuning["game_config_schema"] = raw["config_schema"]
    spec = decode_game_spec(raw, tmp_path / "game-atarigo", "0" * 64)
    assert not spec.game_config_schema.is_empty
    assert spec.game_config_schema.validate_config({"size": 9}) == []
    assert spec.game_config_schema.validate_config({"size": 99}) != []
    assert spec.game_config_schema.validate_config({"width": 9}) != []


def test_decoder_no_longer_requires_game_config_to_equal_default(tmp_path) -> None:
    raw = _description()
    tuning = raw["tuning"]
    assert isinstance(tuning, dict)
    tuning["game_config"] = {"size": 6}
    spec = decode_game_spec(raw, tmp_path / "game", "0" * 64)
    assert spec.default_game_config == '{"size":5}'
    assert spec.tuning.game_config == '{"size":6}'


def test_decoder_rejects_inconsistent_config_schema_siblings(tmp_path) -> None:
    raw = _description()
    raw["config_schema"] = {
        "parameters": [{"name": "size", "type": "int", "bounds": [3, 19], "default": 13}],
        "conditions": [],
    }
    tuning = raw["tuning"]
    assert isinstance(tuning, dict)
    tuning["game_config_schema"] = {
        "parameters": [{"name": "size", "type": "int", "bounds": [3, 25], "default": 13}],
        "conditions": [],
    }
    with pytest.raises(ValueError, match="disagrees"):
        decode_game_spec(raw, tmp_path / "game", "0" * 64)


def test_engine_identity_binds_binary_content_and_description(tmp_path) -> None:
    raw = _description()
    first = decode_game_spec(raw, tmp_path / "first", "a" * 64)
    moved = decode_game_spec(copy.deepcopy(raw), tmp_path / "second", "a" * 64)
    changed_binary = decode_game_spec(raw, tmp_path / "first", "b" * 64)
    changed_description = copy.deepcopy(raw)
    changed_description["label"] = "Changed"
    changed_metadata = decode_game_spec(changed_description, tmp_path / "first", "a" * 64)
    assert first.engine_fingerprint == moved.engine_fingerprint
    assert first.engine_fingerprint != changed_binary.engine_fingerprint
    assert first.engine_fingerprint != changed_metadata.engine_fingerprint
