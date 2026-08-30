"""Explicit low-budget generic game acceptance checks; not collected by pytest."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _check_game(binary: Path, objective: Path) -> None:
    description = subprocess.run(
        [str(binary), "describe"], check=False, capture_output=True, text=True
    )
    assert description.returncode == 0, description.stderr
    expected_kind = json.loads(description.stdout)["kind"]
    panel = json.loads(objective.read_text())["opponents"]
    total_weight = sum(item["weight"] for item in panel)
    with tempfile.TemporaryDirectory(prefix="mcts-tuner-game-") as temporary:
        run_dir = Path(temporary) / "run"
        command = [
            sys.executable,
            "-m",
            "tuner_cli",
            "--game-binary",
            str(binary),
            "--objective-file",
            str(objective),
            "--run-dir",
            str(run_dir),
            "--seed",
            "7",
            "--task-seed",
            "11",
            "--cohort-size",
            "4",
            "--finalists",
            "2",
            "--bootstrap-candidates",
            "2",
            "--random-reserve-candidates",
            "1",
            "--tuning-pairs",
            str(total_weight * 2),
            "--validation-pairs",
            str(total_weight),
            "--production-validation-pairs",
            str(total_weight * 2),
            "--tuning-max-iterations",
            "16",
            "--validation-max-iterations",
            "32",
            "--production-max-iterations",
            "64",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        manifest = json.loads((run_dir / "manifest.json").read_text())
        report = json.loads((run_dir / "report.json").read_text())
        events = [
            json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()
        ]
        completed_pairs = [
            event["payload"] for event in events if event["type"] == "pair_completed"
        ]
        assert manifest["schema_version"] == 4
        assert manifest["kind"] == expected_kind
        assert manifest["objective"]["fingerprint"] and manifest["opponent_panel"]["fingerprint"]
        assert manifest["epoch"]["fingerprint"]
        assert len(manifest["tuning_blocks"]) == 2
        assert manifest["proposer"]["source_schedule"] == [
            "schema_default",
            "bootstrap_random",
            "smac_model",
            "random_reserve",
        ]
        assert manifest["proposer"]["challenger_source_schedule"] == [
            "smac_model",
            "random_reserve",
        ]
        accepted_sources = [
            event["payload"]["source"] for event in events if event["type"] == "proposal_accepted"
        ]
        assert accepted_sources == [
            *manifest["proposer"]["source_schedule"],
            *manifest["proposer"]["challenger_source_schedule"],
        ]
        cohorts = [event["payload"] for event in events if event["type"] == "cohort_completed"]
        assert [cohort["cohort_index"] for cohort in cohorts] == [0, 1]
        retained = [
            event["payload"]["allocation"]
            for event in events
            if event["type"] == "allocation_decided"
            and event["payload"]["allocation"]["kind"] == "retain_elites"
        ]
        assert len(retained) == 1
        assert retained[0]["candidate_ids"] == cohorts[1]["retained_candidate_ids"]
        assert cohorts[1]["candidate_ids"][:2] == retained[0]["candidate_ids"]
        tuning_pairs_by_candidate: dict[str, set[str]] = {}
        for pair in completed_pairs:
            if pair["phase"] == "tuning":
                tuning_pairs_by_candidate.setdefault(pair["candidate_id"], set()).add(
                    pair["pair_id"]
                )
        assert all(
            len(tuning_pairs_by_candidate[candidate_id]) == total_weight * 2
            for candidate_id in retained[0]["candidate_ids"]
        )
        assert (
            len({pair["opponent_id"] for pair in completed_pairs if pair["phase"] == "tuning"}) > 1
        )
        assert (
            len({pair["opponent_id"] for pair in completed_pairs if pair["phase"] == "validation"})
            > 1
        )
        assert report["status"] == "complete"
        tuning_prefixes = {
            event["payload"]["prefix_id"]
            for event in events
            if event["type"] == "observation_completed" and event["payload"]["phase"] == "tuning"
        }
        assert len(tuning_prefixes) == 2
        for prefix_id in tuning_prefixes:
            observed = {
                event["payload"]["candidate_id"]
                for event in events
                if event["type"] == "observation_completed"
                and event["payload"]["phase"] == "tuning"
                and event["payload"]["prefix_id"] == prefix_id
            }
            assert set(cohorts[1]["candidate_ids"]) <= observed
        assert report["validation_claim"] == {
            "claim": "mechanics_smoke",
            "missing_production_axes": ["task_count", "search_effort"],
        }
        assert all(
            len(entry["opponent_matchups"]) == len(panel) for entry in report["validation_order"]
        )
        original_report = (run_dir / "report.json").read_bytes()
        (run_dir / "report.json").unlink()
        rebuilt = subprocess.run(
            [*command, "--resume"], check=False, capture_output=True, text=True
        )
        assert rebuilt.returncode == 0, rebuilt.stderr
        assert (run_dir / "report.json").read_bytes() == original_report


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    _check_game(
        root / "target/release/game-druid", root / "tuner/objectives/druid-reference-v1.json"
    )
    _check_game(
        root / "target/release/game-ttt", root / "tuner/tests/e2e/objectives/ttt-smoke-v1.json"
    )


if __name__ == "__main__":
    main()
