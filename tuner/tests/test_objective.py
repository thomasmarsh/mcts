from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuner_cli.identity import candidate_from_config
from tuner_cli.objective import resolve_objective


def _write(path: Path, opponents: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "objective_id": "example-v1",
                "game_kind": "example",
                "opponents": opponents,
                "start_distribution": {"kind": "default_only"},
            }
        )
    )
    return path


def test_resolved_panel_is_content_identified_not_path_identified(tmp_path: Path) -> None:
    opponents = [
        {
            "id": "default",
            "label": "Default",
            "role": "default",
            "weight": 1,
            "config": {"source": "schema_default"},
        },
        {
            "id": "historical",
            "label": "Historical",
            "role": "historical_reference",
            "weight": 2,
            "config": {"source": "inline", "value": {"family": "b"}},
        },
    ]
    default = candidate_from_config({"family": "a"})
    first = resolve_objective(_write(tmp_path / "one.json", opponents), "example", default)
    second = resolve_objective(_write(tmp_path / "two.json", opponents), "example", default)
    assert first.fingerprint == second.fingerprint
    assert first.panel.fingerprint == second.panel.fingerprint
    assert first.panel.total_weight == 3


@pytest.mark.parametrize(
    "opponents",
    [
        [
            {
                "id": "default",
                "label": "Default",
                "role": "default",
                "weight": 1,
                "config": {"source": "schema_default"},
            }
        ],
        [
            {
                "id": "default",
                "label": "Default",
                "role": "default",
                "weight": 2,
                "config": {"source": "schema_default"},
            },
            {
                "id": "historical",
                "label": "Historical",
                "role": "historical_reference",
                "weight": 2,
                "config": {"source": "inline", "value": {"family": "b"}},
            },
        ],
    ],
)
def test_objective_rejects_insufficient_or_nonreduced_panels(tmp_path: Path, opponents) -> None:
    with pytest.raises(ValueError):
        resolve_objective(
            _write(tmp_path / "bad.json", opponents),
            "example",
            candidate_from_config({"family": "a"}),
        )
