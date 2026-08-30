"""Completed report projection built only from strict typed replay state."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .artifacts import production_claim, read_manifest
from .evidence import atomic_json, read_events
from .observations import paired_difference
from .proposer import tuning_frontier
from .replay import replay
from .statistics import marginal_interval, pair_utility, tie_relation


def _context(value) -> dict[str, object]:
    return {
        "objective_epoch_id": value.objective_epoch_id,
        "phase": value.phase,
        "corpus_id": value.task_prefix.corpus_id,
        "prefix_id": value.task_prefix.prefix_id,
        "task_ids": list(value.task_prefix.task_ids),
        "search_effort": value.search_effort.max_iterations,
    }


def _counts(pairs) -> dict[str, int]:
    games = [game for pair in pairs for game in pair.games]
    wins, draws = (
        sum(game.outcome == "candidate_win" for game in games),
        sum(game.outcome == "draw" for game in games),
    )
    return {
        "pairs": len(pairs),
        "games": len(games),
        "tasks": len({pair.task.task_case.task_id for pair in pairs}),
        "opponents": len({pair.task.task_case.opponent_id for pair in pairs}),
        "starts": len({pair.task.task_case.start for pair in pairs}),
        "wins": wins,
        "draws": draws,
        "losses": len(games) - wins - draws,
        "candidate_iterations_total": sum(
            game.candidate_metrics.iterations_total for game in games
        ),
        "opponent_iterations_total": sum(game.opponent_metrics.iterations_total for game in games),
        "candidate_move_time_ms": sum(game.candidate_metrics.move_time_ms for game in games),
        "opponent_move_time_ms": sum(game.opponent_metrics.move_time_ms for game in games),
        "elapsed_ms": sum(game.elapsed_ms for game in games),
    }


def _matchups(manifest, pairs) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair.task.task_case.opponent_id].append(pair)
    rows = []
    for opponent in manifest.panel.opponents:
        evidence = sorted(
            grouped[opponent.opponent_id], key=lambda pair: pair.task.task_case.ordinal
        )
        values = tuple(pair_utility(pair) for pair in evidence)
        counts = _counts(evidence)
        estimate = marginal_interval(values) if values else None
        rows.append(
            {
                "opponent_id": opponent.opponent_id,
                "opponent_label": opponent.label,
                "opponent_fingerprint": opponent.configuration_fingerprint,
                "declared_weight": opponent.weight,
                "estimate": estimate.mean if estimate else None,
                "interval": None
                if estimate is None
                else {"lower": estimate.lower, "upper": estimate.upper},
                **counts,
            }
        )
    return rows


def _candidate_entry(manifest, candidate, observation, pairs) -> dict[str, object]:
    ordered = sorted(pairs, key=lambda pair: pair.task.task_case.ordinal)
    if (
        tuple(pair.task.task_case.task_id for pair in ordered)
        != observation.context.task_prefix.task_ids
    ):
        raise ValueError("finalist lacks complete selected validation evidence")
    counts = _counts(ordered)
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_fingerprint": candidate.fingerprint,
        "config": candidate.canonical_config,
        "context": _context(observation.context),
        "weighted_marginal": {
            "estimate": observation.estimate.mean,
            "interval": {"lower": observation.estimate.lower, "upper": observation.estimate.upper},
        },
        **counts,
        "opponent_matchups": _matchups(manifest, ordered),
        "tied_with": [],
    }


def _comparisons(
    observations,
) -> tuple[list[dict[str, object]], dict[str, list[str]], list[dict[str, str]]]:
    rows, tied, unresolved = [], defaultdict(list), []
    for index, left in enumerate(observations):
        for right in observations[index + 1 :]:
            difference = paired_difference(left, right)
            relation = tie_relation(difference)
            row = {
                "left_candidate_id": left.candidate_id,
                "right_candidate_id": right.candidate_id,
                "context": _context(left.context),
                "mean_difference": difference.mean,
                "interval": {"lower": difference.lower, "upper": difference.upper},
                "relation": relation,
            }
            rows.append(row)
            if relation == "tie":
                tied[left.candidate_id].append(right.candidate_id)
                tied[right.candidate_id].append(left.candidate_id)
                unresolved.append(
                    {
                        "left_candidate_id": left.candidate_id,
                        "right_candidate_id": right.candidate_id,
                    }
                )
    return rows, tied, unresolved


def build_report(run_dir: Path) -> dict[str, object]:
    manifest = read_manifest(run_dir / "manifest.json")
    state = replay(manifest, read_events(run_dir / "evidence.jsonl"))
    if state.terminal_status != "complete" or state.finalists is None or state.cohort is None:
        raise ValueError("report requires completed evidence")
    validation = defaultdict(list)
    for pair in state.completed_pairs:
        if pair.task.task_case.phase == "validation":
            validation[pair.task.candidate_id].append(pair)
    observations = [
        next(
            (
                item
                for item in state.observations
                if item.phase == "validation" and item.candidate_id == candidate.candidate_id
            ),
            None,
        )
        for candidate in state.finalists
    ]
    if any(item is None for item in observations):
        raise ValueError("finalist lacks held-out validation observation")
    values = [item for item in observations if item is not None]
    entries = [
        _candidate_entry(manifest, candidate, observation, validation[candidate.candidate_id])
        for candidate, observation in zip(state.finalists, values, strict=True)
    ]
    comparisons, tied, unresolved = _comparisons(values)
    for entry in entries:
        entry["tied_with"] = sorted(tied[entry["candidate_id"]])
    entries.sort(
        key=lambda entry: (-entry["weighted_marginal"]["estimate"], entry["candidate_fingerprint"])
    )
    claim, missing = production_claim(
        manifest.validation_prefix,
        manifest.production_validation_corpus,
        manifest.efforts["validation"],
        manifest.efforts["production"],
    )
    proposal_search = _proposal_search(manifest, state)
    return {
        "schema_version": 4,
        "status": "complete",
        "manifest_fingerprint": manifest.fingerprint,
        "validation_claim": {"claim": claim, "missing_production_axes": list(missing)},
        "frozen": {
            "objective_id": manifest.objective_id,
            "objective_fingerprint": manifest.objective_fingerprint,
            "epoch_id": manifest.epoch.epoch_id,
            "engine_fingerprint": manifest.spec.engine_fingerprint,
            "tuning_schema_fingerprint": manifest.spec.schema_fingerprint,
            "game_config_fingerprint": manifest.raw["game_config_fingerprint"],
            "panel_fingerprint": manifest.panel.fingerprint,
            "panel": [
                {
                    "opponent_id": item.opponent_id,
                    "label": item.label,
                    "role": item.role,
                    "weight": item.weight,
                    "configuration_fingerprint": item.configuration_fingerprint,
                }
                for item in manifest.panel.opponents
            ],
            "start_distribution": "default_only",
            "corpora": {
                "tuning": {
                    "fingerprint": manifest.tuning_corpus.fingerprint,
                    "count": len(manifest.tuning_corpus.cases),
                    "selected_prefix": manifest.tuning_prefix.prefix_id,
                },
                "production_validation": {
                    "fingerprint": manifest.production_validation_corpus.fingerprint,
                    "count": len(manifest.production_validation_corpus.cases),
                    "selected_prefix": manifest.validation_prefix.prefix_id,
                },
            },
            "fidelity": {
                "observed_task_count": manifest.validation_prefix.length,
                "production_task_count": len(manifest.production_validation_corpus.cases),
                "observed_search_effort": manifest.efforts["validation"].max_iterations,
                "production_search_effort": manifest.efforts["production"].max_iterations,
            },
            "utility_formula_version": manifest.raw["utility_formula_version"],
            "interval_method": manifest.raw["interval_method"],
            "tie_rule_version": manifest.raw["tie_rule_version"],
        },
        "selection": {"finalist_ids": [candidate.candidate_id for candidate in state.finalists]},
        "proposal_search": proposal_search,
        "validation_order": entries,
        "paired_finalist_comparisons": comparisons,
        "unresolved_ties": unresolved,
        "limitations": [
            "default-only starting state",
            "conservative small-sample intervals",
            "explicit resume",
        ],
    }


def _proposal_search(manifest, state) -> dict[str, object]:
    dispositions = dict(state.dispositions)
    accepted = []
    rejected = {source: 0 for source in manifest.source_schedule}
    for proposal in state.proposals:
        disposition = dispositions.get(proposal.proposal_index)
        if disposition == "rejected":
            rejected[proposal.source] = rejected.get(proposal.source, 0) + 1
        if disposition == "accepted":
            accepted.append(
                {
                    "cohort_slot": proposal.cohort_slot,
                    "candidate_id": proposal.candidate.candidate_id,
                    "source": proposal.source,
                    "proposal_index": proposal.proposal_index,
                    "frontier_id": proposal.frontier.frontier_id,
                    "visible_observation_count": len(proposal.frontier.observation_ids),
                    "origin": proposal.provenance.origin,
                    "acquisition": proposal.provenance.acquisition,
                    "prediction": proposal.provenance.prediction,
                    "uncertainty": proposal.provenance.uncertainty,
                    "parent_candidate_id": proposal.provenance.parent_candidate_id,
                }
            )
    tuning = tuple(item for item in state.observations if item.phase == "tuning")
    frontier = tuning_frontier(tuning)
    return {
        "policy_version": manifest.proposer.encoded()["policy_version"],
        "model_version": manifest.proposer.encoded()["smac_adapter_version"],
        "cost_policy_version": manifest.proposer.encoded()["cost_policy_version"],
        "configured": {
            "bootstrap": manifest.bootstrap_candidates,
            "model": manifest.proposer.model_candidates,
            "random_reserve": manifest.random_reserve_candidates,
        },
        "accepted": accepted,
        "rejections_by_source": rejected,
        "final_frontier_id": frontier.frontier_id,
        "final_observation_count": len(frontier.observation_ids),
    }


def write_report(run_dir: Path) -> dict[str, object]:
    report = build_report(run_dir)
    atomic_json(run_dir / "report.json", report)
    return report
