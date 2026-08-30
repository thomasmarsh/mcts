"""Explicit low-budget Druid acceptance check; not collected by pytest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tuner_cli.run import RunOptions, run_foreground


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    binary = root / "target/release/game-druid"
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise SystemExit(f"build the release Druid binary first: {binary}")
    with tempfile.TemporaryDirectory(prefix="mcts-tuner-druid-") as temporary:
        run_dir = Path(temporary) / "run"
        run_foreground(
            RunOptions(
                run_dir,
                seed=7,
                cohort_size=3,
                finalists=2,
                tuning_pairs=1,
                validation_pairs=1,
                tuning_max_iterations=16,
                validation_max_iterations=32,
                production_max_iterations=10_000,
            )
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        report = json.loads((run_dir / "report.json").read_text())
        events = [
            json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()
        ]
        completed = [event["payload"] for event in events if event["type"] == "pair_completed"]
        assert manifest["kind"] == "druid"
        assert (
            len(
                next(event["payload"] for event in events if event["type"] == "cohort_accepted")[
                    "candidate_ids"
                ]
            )
            == 3
        )
        assert len([pair for pair in completed if pair["phase"] == "tuning"]) == 3
        assert len([pair for pair in completed if pair["phase"] == "validation"]) == 2
        assert all(len(pair["game_ids"]) == 2 for pair in completed)
        assert not {case["task_id"] for case in manifest["tuning_tasks"]["cases"]} & {
            case["task_id"] for case in manifest["validation_tasks"]["cases"]
        }
        assert report["status"] == "complete"
        assert report["validation_claim"] == "mechanics_smoke"
        assert all(entry["pairs"] == 1 for entry in report["validation_order"])


if __name__ == "__main__":
    main()
