"""Pure manifest/evidence read model for completed foreground runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .evidence import atomic_json, read_events
from .statistics import marginal_interval, paired_difference, tie_relation


def build_report(run_dir: Path) -> dict[str, object]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = list(read_events(run_dir / "evidence.jsonl"))
    if not events or events[-1]["type"] != "run_completed":
        raise ValueError("report requires completed evidence")
    selected = next(
        (event["payload"] for event in events if event["type"] == "finalists_selected"), None
    )
    if not isinstance(selected, dict):
        raise ValueError("missing finalist selection")
    finalists = selected["finalist_ids"]
    if not isinstance(finalists, list):
        raise ValueError("invalid finalist selection")
    completed = [event["payload"] for event in events if event["type"] == "pair_completed"]
    game_events = {
        event["payload"]["game_id"]: event["payload"]
        for event in events
        if event["type"] == "game_finished" and isinstance(event["payload"], dict)
    }
    validation = [
        item for item in completed if isinstance(item, dict) and item.get("phase") == "validation"
    ]
    expected_tasks = [case["task_id"] for case in manifest["validation_tasks"]["cases"]]
    budget = manifest["budgets"]["validation"]
    per_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair in validation:
        candidate_id = pair.get("candidate_id")
        if isinstance(candidate_id, str):
            per_candidate[candidate_id].append(pair)
    utilities: dict[str, tuple[float, ...]] = {}
    entries = []
    configs = {
        event["payload"]["candidate_id"]: event["payload"]
        for event in events
        if event["type"] == "proposal_created" and isinstance(event["payload"], dict)
    }
    for candidate_id in finalists:
        pairs = sorted(
            per_candidate[candidate_id], key=lambda item: expected_tasks.index(item["task_id"])
        )
        if [item.get("task_id") for item in pairs] != expected_tasks or any(
            item.get("budget") != budget for item in pairs
        ):
            raise ValueError("finalist lacks the complete common validation block")
        values = tuple(float(item["pair_utility"]) for item in pairs)
        utilities[candidate_id] = values
        interval = marginal_interval(values)
        game_ids = [game_id for item in pairs for game_id in item["game_ids"]]
        games = [game_events[game_id] for game_id in game_ids]
        wins = sum(game["outcome"] == "candidate_win" for game in games)
        draws = sum(game["outcome"] == "draw" for game in games)
        candidate = configs[candidate_id]
        entries.append(
            {
                "candidate_id": candidate_id,
                "candidate_fingerprint": candidate["fingerprint"],
                "config": candidate["canonical_config"],
                "estimate": interval.mean,
                "interval": {"lower": interval.lower, "upper": interval.upper},
                "pairs": len(pairs),
                "games": len(games),
                "unique_task_count": len({pair["task_id"] for pair in pairs}),
                "unique_opponent_count": 1,
                "unique_start_count": 1,
                "wins": wins,
                "draws": draws,
                "losses": len(games) - wins - draws,
                "candidate_iterations_total": sum(
                    game["candidate_metrics"]["iterations_total"] for game in games
                ),
                "opponent_iterations_total": sum(
                    game["opponent_metrics"]["iterations_total"] for game in games
                ),
                "candidate_move_time_ms": sum(
                    game["candidate_metrics"]["move_time_ms"] for game in games
                ),
                "opponent_move_time_ms": sum(
                    game["opponent_metrics"]["move_time_ms"] for game in games
                ),
                "elapsed_ms": sum(game["elapsed_ms"] for game in games),
                "tied_with": [],
            }
        )
    comparisons = []
    ties: dict[str, list[str]] = defaultdict(list)
    for left in finalists:
        for right in finalists:
            if left == right:
                continue
            difference = paired_difference(utilities[left], utilities[right])
            relation = tie_relation(difference)
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "mean_difference": difference.mean,
                    "interval": {"lower": difference.lower, "upper": difference.upper},
                    "relation": relation,
                }
            )
            if relation == "tie":
                ties[left].append(right)
    for entry in entries:
        entry["tied_with"] = sorted(ties[entry["candidate_id"]])
    entries.sort(key=lambda entry: (-entry["estimate"], entry["candidate_fingerprint"]))
    claim = (
        "production"
        if manifest["budgets"]["validation"] == manifest["budgets"]["production"]
        else "mechanics_smoke"
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "manifest_fingerprint": manifest["fingerprint"],
        "validation_claim": claim,
        "frozen": {
            "kind": manifest["kind"],
            "engine_fingerprint": manifest["engine_fingerprint"],
            "tuning_schema_fingerprint": manifest["tuning_schema_fingerprint"],
            "game_config_fingerprint": manifest["game_config_fingerprint"],
            "opponent_fingerprint": manifest["opponent"]["fingerprint"],
            "validation_task_block": manifest["validation_tasks"]["block_id"],
            "budgets": manifest["budgets"],
            "utility_formula_version": manifest["utility_formula_version"],
        },
        "selection": selected,
        "validation_order": entries,
        "pairwise_comparisons": comparisons,
        "interval_method": "hoeffding_pair_bound_v1",
        "tie_rule": "paired_hoeffding_v1; tie means not distinguished by this declared paired rule",
        "limitations": [
            "one default opponent",
            "one default starting state",
            "fixed task counts",
            "conservative small-sample intervals",
            "no resume",
        ],
    }


def write_report(run_dir: Path) -> dict[str, object]:
    report = build_report(run_dir)
    atomic_json(run_dir / "report.json", report)
    return report
