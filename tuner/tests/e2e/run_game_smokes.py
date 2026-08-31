"""Explicit low-budget generic game acceptance checks; not collected by pytest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _check_game(
    binary: Path,
    objective: Path,
    evaluator_workers: int = 1,
    excluded_family: str | None = None,
) -> None:
    description = subprocess.run(
        [str(binary), "describe"], check=False, capture_output=True, text=True
    )
    assert description.returncode == 0, description.stderr
    expected_kind = json.loads(description.stdout)["kind"]
    panel = json.loads(objective.read_text())["opponents"]
    total_weight = sum(item["weight"] for item in panel)
    tuning_pairs = total_weight * 2
    # The initial cohort costs cohort_size * tuning_pairs and each challenger
    # cohort costs (cohort_size - finalists) * tuning_pairs, so a budget of
    # 16 * total_weight admits exactly three cohorts; a fourth would not fit.
    # The validation budget gives each of the two finalists exactly one complete
    # weighted panel cycle.
    tuning_pair_budget = 16 * total_weight
    validation_pair_budget = 2 * total_weight
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
            str(tuning_pairs),
            "--tuning-pair-budget",
            str(tuning_pair_budget),
            "--validation-pair-budget",
            str(validation_pair_budget),
            "--production-validation-pairs",
            str(total_weight * 2),
            "--tuning-max-iterations",
            "16",
            "--validation-max-iterations",
            "32",
            "--production-max-iterations",
            "64",
            "--evaluator-workers",
            str(evaluator_workers),
        ]
        if excluded_family is not None:
            command.extend(["--exclude-family", excluded_family])
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
        assert manifest["proposer"]["excluded_families"] == (
            [] if excluded_family is None else [excluded_family]
        )
        assert (
            report["proposal_search"]["configured"]["excluded_families"]
            == manifest["proposer"]["excluded_families"]
        )
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
        proposal_configs = [
            json.loads(event["payload"]["canonical_config"])
            for event in events
            if event["type"] == "proposal_created"
        ]
        assert all(config.get("family") != excluded_family for config in proposal_configs)
        assert accepted_sources == [
            *manifest["proposer"]["source_schedule"],
            *manifest["proposer"]["challenger_source_schedule"],
            *manifest["proposer"]["challenger_source_schedule"],
        ]
        cohorts = [event["payload"] for event in events if event["type"] == "cohort_completed"]
        assert [cohort["cohort_index"] for cohort in cohorts] == [0, 1, 2]
        allocations = [
            event["payload"] for event in events if event["type"] == "allocation_decided"
        ]
        assert all(item["policy_version"] == "budgeted-multi-cohort-v1" for item in allocations)
        retained = [
            item["allocation"]
            for item in allocations
            if item["allocation"]["kind"] == "retain_elites"
        ]
        assert [item["cohort_index"] for item in retained] == [1, 2]
        for allocation, cohort in zip(retained, cohorts[1:], strict=True):
            assert cohort["retained_candidate_ids"] == allocation["candidate_ids"]
            assert cohort["candidate_ids"][:2] == allocation["candidate_ids"]
        tuning_pairs_by_candidate: dict[str, set[str]] = {}
        for pair in completed_pairs:
            if pair["phase"] == "tuning":
                tuning_pairs_by_candidate.setdefault(pair["candidate_id"], set()).add(
                    pair["pair_id"]
                )
        # Retained elites reuse their exact pair evidence: they are never
        # re-executed in later cohorts.
        assert all(
            len(tuning_pairs_by_candidate[candidate_id]) == tuning_pairs
            for candidate_id in retained[-1]["candidate_ids"]
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
        shadow_races = [
            event["payload"] for event in events if event["type"] == "shadow_race_decided"
        ]
        assert len(shadow_races) == len(cohorts)
        assert {race["cohort_index"] for race in shadow_races} == set(range(len(cohorts)))
        assert all(len(race["decisions"]) == 4 for race in shadow_races)
        shadow = report["shadow_elimination"]
        assert shadow["policy"]["enforced"] is False
        assert shadow["scope"]["held_out_validation_used"] is False
        assert shadow["scope"]["recorded_looks"] == sum(
            len(path["looks"]) for cohort in shadow["cohorts"] for path in cohort["candidate_paths"]
        )
        assert shadow["summary"]["counterfactual_eliminations"] == sum(
            path["first_elimination_prefix_id"] is not None
            for cohort in shadow["cohorts"]
            for path in cohort["candidate_paths"]
        )
        assert all(
            0.0 <= look["promotion_probability"] <= 1.0
            for cohort in shadow["cohorts"]
            for path in cohort["candidate_paths"]
            for look in path["looks"]
        )
        for prefix_id in tuning_prefixes:
            observed = {
                event["payload"]["candidate_id"]
                for event in events
                if event["type"] == "observation_completed"
                and event["payload"]["phase"] == "tuning"
                and event["payload"]["prefix_id"] == prefix_id
            }
            assert set(cohorts[-1]["candidate_ids"]) <= observed
        compute = report["compute"]
        assert compute["policy_version"] == "safe-boundary-pair-attempts-v1"
        assert compute["budget"] == {
            "tuning_pair_attempts": tuning_pair_budget,
            "validation_pair_attempts": validation_pair_budget,
        }
        assert compute["tuning"]["pair_attempts"] == tuning_pair_budget
        assert compute["tuning"]["completed_pairs"] == tuning_pair_budget
        assert compute["tuning"]["unspent_pair_attempts"] == 0
        assert compute["tuning"]["overrun_pair_attempts"] == 0
        assert compute["validation"]["pair_attempts"] == validation_pair_budget
        assert compute["validation"]["completed_pairs"] == validation_pair_budget
        assert report["proposal_search"]["configured"]["cohorts"] == 3
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
        root / "target/release/game-druid",
        root / "tuner/objectives/druid-reference-v1.json",
        2 if (os.cpu_count() or 1) >= 2 else 1,
        "meta_mcts",
    )
    _check_game(
        root / "target/release/game-ttt", root / "tuner/tests/e2e/objectives/ttt-smoke-v1.json"
    )


if __name__ == "__main__":
    main()
