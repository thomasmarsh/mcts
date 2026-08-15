"""`IncumbentTracker` (`smac3_cli.callback`) must emit a `{"type":
"incumbent", ...}` JSONL line on stdout whenever SMAC3's tracked incumbent
changes, so the ingest loop has something to upsert into the `incumbents`
table. Runs a tiny real optimize (same fixture/pattern as `test_resume.py`)
rather than mocking SMAC3's `Callback`/`SMBO` internals, since faking
`smbo.intensifier.get_incumbent()` correctly would mean re-deriving the
exact thing this test is meant to catch drift in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smac3_cli.__main__ import build_optimizer
from smac3_cli.config import OptimizerConfig, SearchConfig, TargetConfig


def _cfg(binary: Path) -> SearchConfig:
    return SearchConfig(
        optimizer=OptimizerConfig(n_trials=3, n_workers=1, deterministic=True, seed=7),
        target=TargetConfig(binary=binary, rounds=2, baselines=["strong"]),
    )


@pytest.fixture
def run_id(tmp_path, monkeypatch) -> str:
    monkeypatch.chdir(tmp_path)
    return "incumbent-test-run"


def _incumbent_lines(raw_stdout: str) -> list[dict]:
    """Parse JSONL trial/incumbent records out of stdout, same tolerance for
    interleaved non-JSON lines (logging, warnings) as the Rust ingest loop's
    own `serde_json::from_str(&line).ok()` skip-on-parse-error handling."""
    records = []
    for line in raw_stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return [r for r in records if r.get("type") == "incumbent"]


def test_incumbent_tracker_emits_jsonl_on_change(game_nim_binary: Path, run_id: str, capsys):
    optimizer = build_optimizer(_cfg(game_nim_binary), run_id=run_id)
    optimizer.optimize()

    incumbents = _incumbent_lines(capsys.readouterr().out)
    assert len(incumbents) >= 1, "expected at least one incumbent record"

    for record in incumbents:
        assert isinstance(record["config"], dict)
        assert isinstance(record["cost"], (int, float))

    # The final line must match SMAC3's own tracked incumbent -- config
    # values as well as cost, not just the presence of the fields.
    final = incumbents[-1]
    tracked_incumbent = optimizer.intensifier.get_incumbent()
    assert final["config"] == dict(tracked_incumbent)
    assert final["cost"] == pytest.approx(optimizer.runhistory.get_cost(tracked_incumbent))
