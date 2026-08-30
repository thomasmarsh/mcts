from __future__ import annotations

import pytest

from tuner_cli.schema import decode_druid_spec


def test_decoder_keeps_game_and_strategy_configs_separate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    raw = {
        "kind": "druid",
        "label": "Druid",
        "description": "test",
        "default_config": {"size": 5},
        "ai_presets": [],
        "tuning": {
            "id": "strategy",
            "baselines": [],
            "eval_rounds": 1,
            "game_config": {"size": 5},
            "parameters": [
                {"name": "family", "type": "categorical", "choices": ["x", "y"], "default": "x"},
                {"name": "flag", "type": "bool", "default": False},
            ],
            "conditions": [{"if": {"flag": True}, "then": ["family"]}],
        },
    }
    spec = decode_druid_spec(raw, tmp_path / "game-druid", "0" * 64)
    assert spec.default_game_config == '{"size":5}'
    assert spec.tuning.game_config == '{"size":5}'
    assert spec.tuning.conditions[0].values == (True,)
    raw["kind"] = "nim"
    with pytest.raises(ValueError, match="Druid"):
        decode_druid_spec(raw, tmp_path / "game-druid", "0" * 64)
