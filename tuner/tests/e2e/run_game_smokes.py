"""Explicit low-budget generic game acceptance checks; not collected by pytest."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _check_game(binary: Path) -> None:
    description = subprocess.run(
        [str(binary), "describe"], check=False, capture_output=True, text=True
    )
    assert description.returncode == 0, description.stderr
    expected_kind = json.loads(description.stdout)["kind"]
    with tempfile.TemporaryDirectory(prefix="mcts-tuner-game-") as temporary:
        run_dir = Path(temporary) / "run"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tuner_cli",
                "--game-binary",
                str(binary),
                "--run-dir",
                str(run_dir),
                "--seed",
                "7",
                "--cohort-size",
                "3",
                "--finalists",
                "2",
                "--tuning-pairs",
                "1",
                "--validation-pairs",
                "1",
                "--tuning-max-iterations",
                "16",
                "--validation-max-iterations",
                "32",
                "--production-max-iterations",
                "10000",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        manifest = json.loads((run_dir / "manifest.json").read_text())
        report = json.loads((run_dir / "report.json").read_text())
        events = [
            json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()
        ]
        completed = [event["payload"] for event in events if event["type"] == "pair_completed"]
        assert manifest["kind"] == expected_kind
        assert manifest["schema_version"] == 2
        assert manifest["binary"]["sha256"]
        assert manifest["engine_fingerprint"]
        assert manifest["tuning_schema_fingerprint"]
        assert manifest["game_config_fingerprint"]
        assert report["frozen"]["kind"] == manifest["kind"]
        assert report["frozen"]["engine_fingerprint"] == manifest["engine_fingerprint"]
        assert (
            report["frozen"]["tuning_schema_fingerprint"] == manifest["tuning_schema_fingerprint"]
        )
        assert report["frozen"]["game_config_fingerprint"] == manifest["game_config_fingerprint"]
        accepted = next(event["payload"] for event in events if event["type"] == "cohort_accepted")
        assert len(accepted["candidate_ids"]) == 3
        assert len([pair for pair in completed if pair["phase"] == "tuning"]) == 3
        assert len([pair for pair in completed if pair["phase"] == "validation"]) == 2
        assert all(len(pair["games"]) == 2 for pair in completed)
        assert not {case["seed"] for case in manifest["tuning_tasks"]["cases"]} & {
            case["seed"] for case in manifest["validation_tasks"]["cases"]
        }
        assert report["status"] == "complete"
        assert report["validation_claim"] == "mechanics_smoke"
        assert all(entry["pairs"] == 1 for entry in report["validation_order"])
        original_report = (run_dir / "report.json").read_bytes()
        (run_dir / "report.json").unlink()
        rebuilt = subprocess.run(
            [
                sys.executable,
                "-m",
                "tuner_cli",
                "--game-binary",
                str(binary),
                "--run-dir",
                str(run_dir),
                "--resume",
                "--seed",
                "7",
                "--cohort-size",
                "3",
                "--finalists",
                "2",
                "--tuning-pairs",
                "1",
                "--validation-pairs",
                "1",
                "--tuning-max-iterations",
                "16",
                "--validation-max-iterations",
                "32",
                "--production-max-iterations",
                "10000",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rebuilt.returncode == 0, rebuilt.stderr
        assert (run_dir / "report.json").read_bytes() == original_report


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    for name in ("game-druid", "game-ttt"):
        binary = root / "target/release" / name
        if not binary.is_file() or not binary.stat().st_mode & 0o111:
            raise SystemExit(f"build the release game binary first: {binary}")
        _check_game(binary)


if __name__ == "__main__":
    main()
