"""Per-cohort scientific sections shared by the completed report and the
projection.

Each ``*_section`` here shapes one block of the scientific report from values
that already exist in a *partial* ``ReplayState`` -- a completed cohort's race,
its tuning observations, the completed pairs, the diagnostic pairs, and the
shadow audit. None of it requires ``terminal_status == "complete"``, so the
projector can emit these rows per completed cohort while a run is still live.

``build_report`` calls the same functions, so the completed report's bytes are
unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

from .artifacts import Manifest
from .codec import JsonObject, JsonValue
from .diagnostic_graph import DiagnosticGraph
from .domain import (
    Candidate,
    CohortRecord,
    Observation,
    ObservationContext,
    PairResult,
)
from .effort import encode_effort
from .opponent_interactions import OpponentResponseAnalysis
from .shadow_audit import CandidatePathAudit, ShadowAudit


def context_json(value: ObservationContext) -> JsonObject:
    return {
        "objective_epoch_id": value.objective_epoch_id,
        "phase": value.phase,
        "corpus_id": value.task_prefix.corpus_id,
        "prefix_id": value.task_prefix.prefix_id,
        "task_ids": list(value.task_prefix.task_ids),
        "search_effort": encode_effort(value.search_effort),
    }


def json_array(values: Iterable[JsonValue]) -> list[JsonValue]:
    return list(values)


def evidence_counts(pairs: list[PairResult]) -> JsonObject:
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


def opponent_response_section(
    manifest: Manifest,
    cohort_index: int,
    observation: Observation,
    analysis: OpponentResponseAnalysis,
) -> JsonObject:
    responses = {(item.candidate_id, item.opponent_id): item for item in analysis.responses}
    candidates: list[JsonObject] = []
    for candidate_id in dict.fromkeys(item.candidate_id for item in analysis.responses):
        rows: list[JsonObject] = []
        for opponent in manifest.panel.opponents:
            response = responses[candidate_id, opponent.opponent_id]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "opponent_id": opponent.opponent_id,
                    "mean": response.estimate.mean,
                    "interval": {
                        "lower": response.estimate.lower,
                        "upper": response.estimate.upper,
                    },
                    "pair_count": response.pair_count,
                    **evidence_counts(list(response.pairs)),
                }
            )
        candidates.append({"candidate_id": candidate_id, "opponent_responses": json_array(rows)})
    return {
        "scope": {
            "phase": "tuning",
            "cohort_index": cohort_index,
            "prefix_id": observation.context.task_prefix.prefix_id,
            "opponent_ids": [item.opponent_id for item in manifest.panel.opponents],
            "interval_method": "hoeffding_pair_bound_v1",
            "interaction_rule": "opposite-paired-hoeffding-relations-v1",
        },
        "candidates": json_array(candidates),
        "pairwise_interactions": json_array(
            {
                "left_candidate_id": item.left_candidate_id,
                "right_candidate_id": item.right_candidate_id,
                "contrasts": json_array(
                    {
                        "opponent_id": contrast.opponent_id,
                        "mean_difference": contrast.paired_difference.mean,
                        "interval": {
                            "lower": contrast.paired_difference.lower,
                            "upper": contrast.paired_difference.upper,
                        },
                        "relation": contrast.relation,
                    }
                    for contrast in item.contrasts
                ),
                "ranking_reversals": json_array(
                    {
                        "left_opponent_id": reversal.left_opponent_id,
                        "right_opponent_id": reversal.right_opponent_id,
                    }
                    for reversal in item.ranking_reversals
                ),
            }
            for item in analysis.interactions
        ),
    }


def _shadow_look(value: object) -> JsonObject:
    from .shadow_audit import ShadowLookAudit

    if not isinstance(value, ShadowLookAudit):
        raise TypeError("shadow audit look expected")
    common: JsonObject = {
        "prefix_id": value.prefix_id,
        "candidate_id": value.candidate_id,
        "boundary_candidate_id": value.boundary_candidate_id,
        "disposition": value.disposition,
        "early_mean_difference": value.early_mean_difference,
        "maximum_mean_difference": value.maximum_mean_difference,
        "final_reaches_recorded_boundary": value.final_reaches_recorded_boundary,
        "strata": [
            {
                "stratum_id": item.stratum_id,
                "early_mean_difference": item.early_mean_difference,
                "maximum_mean_difference": item.maximum_mean_difference,
                "reversal": item.reversal,
                **(
                    {
                        "favorable_resamples": item.favorable_resamples,
                        "favorable_probability": item.favorable_probability,
                    }
                    if item.favorable_resamples is not None
                    else {}
                ),
            }
            for item in value.strata
        ],
    }
    if value.policy_kind == "paired_bootstrap":
        assert value.favorable_resamples is not None and value.total_resamples is not None
        return {
            **common,
            "favorable_resamples": value.favorable_resamples,
            "total_resamples": value.total_resamples,
            "promotion_probability": value.favorable_resamples / value.total_resamples,
        }
    return {
        **common,
        "rank": value.rank,
        "prior_survivor_count": value.prior_survivor_count,
        "target_survivor_count": value.target_survivor_count,
        "newly_eliminated": value.newly_eliminated,
    }


def _shadow_path(value: CandidatePathAudit) -> JsonObject:
    compute = value.avoided_compute
    return {
        "cohort_index": value.cohort_index,
        "candidate_id": value.candidate_id,
        "protected": value.protected,
        "final_top_set": value.final_top_set,
        "first_elimination_prefix_id": value.first_elimination_prefix_id,
        "avoided_work": {
            "pair_attempts": compute.pair_attempts,
            "completed_pairs": compute.completed_pairs,
            "failed_attempts": compute.failed_attempts,
            "censored_attempts": compute.censored_attempts,
            "unique_pairs": value.avoided_unique_pairs,
            "physical_games": compute.physical_games,
            "search_iterations": compute.search_iterations,
            "wall_time_ms": compute.wall_time_ms,
        },
        "looks": [_shadow_look(item) for item in value.looks],
    }


def shadow_elimination_section(manifest: Manifest, audit: ShadowAudit) -> JsonObject:
    policy = manifest.shadow_policy
    compute = audit.recorded_compute_after_first_elimination
    active_looks = sum(
        len(path.looks)
        if path.first_elimination_prefix_id is None and not path.protected
        else (
            path.looks.index(next(item for item in path.looks if item.disposition == "eliminate"))
            + 1
            if not path.protected
            else 0
        )
        for path in audit.paths
    )
    false_rate = (
        None
        if audit.eligible_top_set_paths == 0
        else audit.top_set_false_eliminations / audit.eligible_top_set_paths
    )
    precision = (
        None
        if audit.counterfactual_eliminations == 0
        else audit.true_trash_eliminations / audit.counterfactual_eliminations
    )
    return {
        "policy": (
            {
                "kind": "paired_bootstrap",
                "method_version": policy.method_version,
                "practical_effect_margin": policy.practical_effect_margin,
                "elimination_probability_threshold": policy.elimination_probability_threshold,
                "resamples": policy.resamples,
                "minimum_eligible_prefix_pairs": policy.minimum_eligible_prefix_pairs,
                "enforced": False,
            }
            if policy.kind == "paired_bootstrap"
            else {
                "kind": "successive_halving",
                "method_version": policy.method_version,
                "reduction_factor": policy.reduction_factor,
                "survivor_floor": policy.survivor_floor,
                "ranking_rule": policy.ranking_rule,
                "practical_effect_margin": policy.practical_effect_margin,
                "minimum_eligible_prefix_pairs": policy.minimum_eligible_prefix_pairs,
                "enforced": False,
            }
        ),
        "scope": {
            "truth": "same-cohort-maximum-tuning-prefix-v1",
            "held_out_validation_used": False,
            "completed_cohorts": len({path.cohort_index for path in audit.paths}),
            "recorded_looks": sum(len(path.looks) for path in audit.paths),
            "active_path_looks": active_looks,
            "superseded_roster_looks": audit.superseded_roster_looks,
        },
        "summary": {
            "counterfactual_eliminations": audit.counterfactual_eliminations,
            "eligible_top_set_paths": audit.eligible_top_set_paths,
            "top_set_false_eliminations": audit.top_set_false_eliminations,
            "top_set_false_elimination_rate": false_rate,
            "true_trash_eliminations": audit.true_trash_eliminations,
            "trash_precision": precision,
            "brier_score": audit.brier_score,
        },
        "recorded_compute_after_first_elimination": {
            "pair_attempts": compute.pair_attempts,
            "completed_pairs": compute.completed_pairs,
            "failed_attempts": compute.failed_attempts,
            "censored_attempts": compute.censored_attempts,
            "unique_pairs": sum(path.avoided_unique_pairs for path in audit.paths),
            "physical_games": compute.physical_games,
            "search_iterations": compute.search_iterations,
            "wall_time_ms": compute.wall_time_ms,
        },
        "calibration_bins": [
            {
                "lower": item.lower,
                "upper": item.upper,
                "count": item.count,
                "mean_prediction": item.mean_prediction,
                "observed_success_rate": item.observed_success_rate,
            }
            for item in audit.calibration_bins
        ],
        "strata": [
            {
                "stratum_id": item.stratum_id,
                "looks": item.looks,
                "reversals": item.reversals,
                "elimination_reversals": item.elimination_reversals,
            }
            for item in audit.strata
        ],
        "cohorts": [
            {
                "cohort_index": index,
                "candidate_paths": [
                    _shadow_path(path) for path in audit.paths if path.cohort_index == index
                ],
            }
            for index in sorted({path.cohort_index for path in audit.paths})
        ],
    }


def diagnostic_section(
    manifest: Manifest,
    cohort: CohortRecord,
    order: tuple[Candidate, ...],
    graph: DiagnosticGraph,
    objective: tuple[Candidate, ...],
    finalists: tuple[Candidate, ...],
    reserve: str | None,
    displaced: str | None,
) -> JsonObject:
    rank = {item.candidate_id: index for index, item in enumerate(order)}
    return {
        "scope": {
            "context": "direct_candidate_diagnostic",
            "cohort_index": cohort.cohort_index,
            "candidate_ids": [item.candidate_id for item in order],
            "pair_attempt_budget": manifest.compute_budget.diagnostic_pair_attempts,
            "search_effort": encode_effort(manifest.efforts["tuning"]),
            "edge_policy_version": manifest.diagnostic_policy.encoded()["edge_policy_version"],
            "graph_rule_version": manifest.diagnostic_policy.encoded()["graph_rule_version"],
            "objective_evidence_used_for_priority": True,
            "objective_evidence_used_for_edge_estimates": False,
        },
        "allocations": {
            "count": len(state_pairs := graph.edges)
            and sum(len(edge.pair_results) for edge in state_pairs)
            or 0,
            "by_reason": {},
        },
        "nodes": [
            {
                "candidate_id": item.candidate_id,
                "candidate_fingerprint": item.fingerprint,
                "objective_rank": rank[item.candidate_id],
            }
            for item in order
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "left_candidate_id": edge.left_candidate_id,
                "right_candidate_id": edge.right_candidate_id,
                "pair_count": len(edge.pair_results),
                "game_count": 2 * len(edge.pair_results),
                "estimate": edge.estimate.mean if edge.estimate else None,
                "interval": None
                if edge.estimate is None
                else {"lower": edge.estimate.lower, "upper": edge.estimate.upper},
                "material_direction": edge.material_direction,
            }
            for edge in graph.edges
        ],
        "material_cycle_components": [
            {
                "candidate_ids": list(item.candidate_ids),
                "witness_cycle_candidate_ids": list(item.witness_cycle_candidate_ids),
            }
            for item in graph.material_cycle_components
        ],
        "shortlist_effect": {
            "shortlist_rule_version": manifest.diagnostic_policy.encoded()[
                "shortlist_rule_version"
            ],
            "maximum_reserve_slots": 1,
            "objective_candidate_ids": [item.candidate_id for item in objective],
            "reserve_candidate_id": reserve,
            "displaced_candidate_id": displaced,
            "finalist_ids": [item.candidate_id for item in finalists],
        },
    }
