"""Completed report projection built only from strict typed replay state."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .artifacts import read_manifest
from .evidence import atomic_json, read_events
from .replay import replay
from .statistics import marginal_interval, paired_difference, tie_relation


def build_report(run_dir: Path) -> dict[str, object]:
    manifest = read_manifest(run_dir / "manifest.json")
    state = replay(manifest, read_events(run_dir / "evidence.jsonl"))
    if state.terminal_status != "complete" or state.finalists is None or state.cohort is None:
        raise ValueError("report requires completed evidence")
    validation = defaultdict(list)
    for pair in state.completed_pairs:
        if pair.task.task_case.phase == "validation":
            validation[pair.task.candidate_id].append(pair)
    utilities: dict[str, tuple[float, ...]] = {}
    entries: list[dict[str, object]] = []
    for candidate in state.finalists:
        pairs = sorted(
            validation[candidate.candidate_id], key=lambda pair: pair.task.task_case.ordinal
        )
        if [pair.task.task_case.task_id for pair in pairs] != [
            case.task_id for case in manifest.validation.cases
        ]:
            raise ValueError("finalist lacks complete common validation evidence")
        values = tuple(
            sum(
                {"candidate_win": 1.0, "draw": 0.5, "baseline_win": 0.0}[game.outcome]
                for game in pair.games
            )
            / 2
            for pair in pairs
        )
        utilities[candidate.candidate_id] = values
        estimate = marginal_interval(values)
        games = [game for pair in pairs for game in pair.games]
        wins = sum(game.outcome == "candidate_win" for game in games)
        draws = sum(game.outcome == "draw" for game in games)
        entries.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_fingerprint": candidate.fingerprint,
                "config": candidate.canonical_config,
                "estimate": estimate.mean,
                "interval": {"lower": estimate.lower, "upper": estimate.upper},
                "pairs": len(pairs),
                "games": len(games),
                "unique_task_count": len({pair.task.task_case.task_id for pair in pairs}),
                "unique_opponent_count": 1,
                "unique_start_count": 1,
                "wins": wins,
                "draws": draws,
                "losses": len(games) - wins - draws,
                "candidate_iterations_total": sum(
                    game.candidate_metrics.iterations_total for game in games
                ),
                "opponent_iterations_total": sum(
                    game.opponent_metrics.iterations_total for game in games
                ),
                "candidate_move_time_ms": sum(
                    game.candidate_metrics.move_time_ms for game in games
                ),
                "opponent_move_time_ms": sum(game.opponent_metrics.move_time_ms for game in games),
                "elapsed_ms": sum(game.elapsed_ms for game in games),
                "tied_with": [],
            }
        )
    comparisons: list[dict[str, object]] = []
    ties: dict[str, list[str]] = defaultdict(list)
    for left in state.finalists:
        for right in state.finalists:
            if left == right:
                continue
            difference = paired_difference(
                utilities[left.candidate_id], utilities[right.candidate_id]
            )
            relation = tie_relation(difference)
            comparisons.append(
                {
                    "left": left.candidate_id,
                    "right": right.candidate_id,
                    "mean_difference": difference.mean,
                    "interval": {"lower": difference.lower, "upper": difference.upper},
                    "relation": relation,
                }
            )
            if relation == "tie":
                ties[left.candidate_id].append(right.candidate_id)
    for entry in entries:
        entry["tied_with"] = sorted(ties[entry["candidate_id"]])
    entries.sort(key=lambda entry: (-entry["estimate"], entry["candidate_fingerprint"]))  # type: ignore[operator]
    claim = (
        "production"
        if manifest.budgets["validation"] == manifest.budgets["production"]
        else "mechanics_smoke"
    )
    return {
        "schema_version": 2,
        "status": "complete",
        "manifest_fingerprint": manifest.fingerprint,
        "validation_claim": claim,
        "frozen": {
            "kind": manifest.spec.kind,
            "engine_fingerprint": manifest.spec.engine_fingerprint,
            "tuning_schema_fingerprint": manifest.spec.schema_fingerprint,
            "game_config_fingerprint": manifest.raw["game_config_fingerprint"],
            "opponent_fingerprint": manifest.opponent.fingerprint,
            "validation_task_block": manifest.validation.block_id,
            "budgets": manifest.budgets,
            "utility_formula_version": manifest.raw["utility_formula_version"],
        },
        "selection": {"finalist_ids": [candidate.candidate_id for candidate in state.finalists]},
        "validation_order": entries,
        "pairwise_comparisons": comparisons,
        "interval_method": "hoeffding_pair_bound_v1",
        "tie_rule": "paired_hoeffding_v1; tie means not distinguished by this declared paired rule",
        "limitations": [
            "one default opponent",
            "one default starting state",
            "fixed task counts",
            "conservative small-sample intervals",
            "explicit resume",
        ],
    }


def write_report(run_dir: Path) -> dict[str, object]:
    report = build_report(run_dir)
    atomic_json(run_dir / "report.json", report)
    return report
