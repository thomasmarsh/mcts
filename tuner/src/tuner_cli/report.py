"""Completed report projection built only from strict typed replay state."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .active_audit import build_active_audit
from .artifacts import Manifest, production_claim, read_manifest
from .codec import JsonObject, JsonValue
from .cohort import latest_completed_cohort
from .diagnostic_graph import DiagnosticGraph, build_diagnostic_graph
from .domain import (
    Candidate,
    CohortRecord,
    Observation,
    ObservationContext,
    PairResult,
    PhaseCompute,
    ReplayState,
)
from .effort import encode_effort
from .evidence import atomic_json, read_events
from .observations import comparable_prefix_observations, paired_difference
from .opponent_interactions import OpponentResponseAnalysis, build_opponent_response_analysis
from .proposer import tuning_frontier
from .replay import replay
from .selection import select_top_candidates, select_validation_shortlist
from .shadow_audit import CandidatePathAudit, ShadowAudit, build_shadow_audit
from .statistics import marginal_interval, pair_utility, tie_relation


def _context(value: ObservationContext) -> JsonObject:
    return {
        "objective_epoch_id": value.objective_epoch_id,
        "phase": value.phase,
        "corpus_id": value.task_prefix.corpus_id,
        "prefix_id": value.task_prefix.prefix_id,
        "task_ids": list(value.task_prefix.task_ids),
        "search_effort": encode_effort(value.search_effort),
    }


def _array(values: Iterable[JsonValue]) -> list[JsonValue]:
    return list(values)


def _counts(pairs: list[PairResult]) -> JsonObject:
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


def _matchups(manifest: Manifest, pairs: list[PairResult]) -> list[JsonObject]:
    grouped: defaultdict[str, list[PairResult]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.task.task_case.opponent_id].append(pair)
    rows: list[JsonObject] = []
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


def _candidate_entry(
    manifest: Manifest,
    candidate: Candidate,
    observation: Observation,
    pairs: list[PairResult],
    tied_with: list[str],
) -> JsonObject:
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
        "opponent_matchups": _array(_matchups(manifest, ordered)),
        "tied_with": _array(tied_with),
    }


def _comparisons(
    observations: list[Observation],
) -> tuple[list[JsonObject], dict[str, list[str]], list[JsonObject]]:
    rows: list[JsonObject] = []
    tied: defaultdict[str, list[str]] = defaultdict(list)
    unresolved: list[JsonObject] = []
    for index, left in enumerate(observations):
        for right in observations[index + 1 :]:
            difference = paired_difference(left, right)
            relation = tie_relation(difference)
            row: JsonObject = {
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


def _frozen(manifest: Manifest) -> JsonObject:
    return {
        "objective_id": manifest.objective_id,
        "objective_fingerprint": manifest.objective_fingerprint,
        "epoch_id": manifest.epoch.epoch_id,
        "engine_fingerprint": manifest.spec.engine_fingerprint,
        "tuning_schema_fingerprint": manifest.spec.schema_fingerprint,
        "game_config_fingerprint": manifest.game_config_fingerprint,
        "panel_fingerprint": manifest.panel.fingerprint,
        "panel": _array(
            {
                "opponent_id": item.opponent_id,
                "label": item.label,
                "role": item.role,
                "weight": item.weight,
                "configuration_fingerprint": item.configuration_fingerprint,
            }
            for item in manifest.panel.opponents
        ),
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
            "observed_search_effort": encode_effort(manifest.efforts["validation"]),
            "production_search_effort": encode_effort(manifest.efforts["production"]),
        },
        "utility_formula_version": "pair_mean_v1",
        "interval_method": "hoeffding_pair_bound_v1",
        "tie_rule_version": "paired_hoeffding_v1",
    }


def _finalist_validation_observations(
    state: ReplayState, finalists: tuple[Candidate, ...]
) -> list[Observation]:
    resolved = [
        next(
            (
                item
                for item in state.observations
                if item.phase == "validation" and item.candidate_id == candidate.candidate_id
            ),
            None,
        )
        for candidate in finalists
    ]
    if any(item is None for item in resolved):
        raise ValueError("finalist lacks held-out validation observation")
    return [item for item in resolved if item is not None]


def _validation_pairs_by_candidate(state: ReplayState) -> dict[str, list[PairResult]]:
    grouped: defaultdict[str, list[PairResult]] = defaultdict(list)
    for pair in state.completed_pairs:
        if pair.task.task_case.phase == "validation":
            grouped[pair.task.candidate_id].append(pair)
    return grouped


def build_report(run_dir: Path) -> JsonObject:
    manifest = read_manifest(run_dir / "manifest.json")
    events = read_events(run_dir / "evidence.jsonl")
    state = replay(manifest, events)
    if (
        state.terminal_status != "complete"
        or state.finalists is None
        or latest_completed_cohort(state) is None
    ):
        raise ValueError("report requires completed evidence")
    finalists = state.finalists
    observations = _finalist_validation_observations(state, finalists)
    validation = _validation_pairs_by_candidate(state)
    cohort = latest_completed_cohort(state)
    if cohort is None:
        raise ValueError("report requires a completed cohort")
    tuning = comparable_prefix_observations(
        state.observations, cohort.candidates, manifest.tuning_prefix
    )
    opponent_analysis = build_opponent_response_analysis(
        manifest.panel,
        cohort,
        tuning,
        tuple(state.completed_pairs),
    )
    objective_order = select_top_candidates(cohort.candidates, tuning, len(cohort.candidates))
    rank = {item.candidate_id: index for index, item in enumerate(objective_order)}
    diagnostic_graph = build_diagnostic_graph(cohort.candidates, state.diagnostic_pairs, rank)
    objective_shortlist = select_top_candidates(cohort.candidates, tuning, manifest.finalists)
    _shortlist, reserve, displaced = select_validation_shortlist(
        cohort.candidates, tuning, manifest.finalists, diagnostic_graph
    )
    comparisons, tied, unresolved = _comparisons(observations)
    ranked = sorted(
        zip(finalists, observations, strict=True),
        key=lambda item: (-item[1].estimate.mean, item[0].fingerprint),
    )
    entries = [
        _candidate_entry(
            manifest,
            candidate,
            observation,
            validation[candidate.candidate_id],
            sorted(tied[candidate.candidate_id]),
        )
        for candidate, observation in ranked
    ]
    claim, missing = production_claim(
        manifest.validation_prefix,
        manifest.production_validation_corpus,
        manifest.efforts["validation"],
        manifest.efforts["production"],
    )
    shadow = (
        build_shadow_audit(manifest, state, events) if manifest.active_elimination is None else None
    )
    return {
        "schema_version": 4,
        "status": "complete",
        "manifest_fingerprint": manifest.fingerprint,
        "validation_claim": {"claim": claim, "missing_production_axes": list(missing)},
        "frozen": _frozen(manifest),
        "selection": {"finalist_ids": [candidate.candidate_id for candidate in finalists]},
        "proposal_search": _proposal_search(manifest, state),
        "candidate_lifecycle": _candidate_lifecycle(manifest, state),
        "validation_order": _array(entries),
        "opponent_response_analysis": _opponent_response_analysis(
            manifest, cohort.cohort_index, tuning[0], opponent_analysis
        ),
        "diagnostic_matchup_graph": _diagnostic_matchup_graph(
            manifest,
            cohort,
            objective_order,
            diagnostic_graph,
            objective_shortlist,
            finalists,
            reserve,
            displaced,
        ),
        "paired_finalist_comparisons": _array(comparisons),
        "unresolved_ties": _array(unresolved),
        "compute": _compute_section(manifest, state),
        "shadow_elimination": _shadow_elimination(manifest, shadow) if shadow is not None else None,
        "active_elimination": (
            None if manifest.active_elimination is None else build_active_audit(manifest, state)
        ),
        "limitations": [
            "default-only starting state",
            "conservative small-sample intervals",
            (
                "opponent-response intervals are per-contrast only, with no across-matrix "
                "multiplicity correction; ranking reversals are descriptive evidence, not a "
                "family-wise error guarantee"
            ),
            "explicit resume",
            (
                "shadow-elimination outcomes are same-run maximum-tuning "
                "counterfactuals, not held-out production validation"
            ),
            (
                "the bootstrap score is empirically assessed and not an "
                "anytime-valid false-elimination guarantee"
            ),
            "active shadow pruning remains disabled"
            if manifest.active_elimination is None
            else (
                "pruned outcomes remain unobserved; randomized audit continuations "
                "estimate boundary reversals"
            ),
        ],
    }


def _opponent_response_analysis(
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
                    **_counts(list(response.pairs)),
                }
            )
        candidates.append({"candidate_id": candidate_id, "opponent_responses": _array(rows)})
    return {
        "scope": {
            "phase": "tuning",
            "cohort_index": cohort_index,
            "prefix_id": observation.context.task_prefix.prefix_id,
            "opponent_ids": [item.opponent_id for item in manifest.panel.opponents],
            "interval_method": "hoeffding_pair_bound_v1",
            "interaction_rule": "opposite-paired-hoeffding-relations-v1",
        },
        "candidates": _array(candidates),
        "pairwise_interactions": _array(
            {
                "left_candidate_id": item.left_candidate_id,
                "right_candidate_id": item.right_candidate_id,
                "contrasts": _array(
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
                "ranking_reversals": _array(
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


def _shadow_elimination(manifest: Manifest, audit: ShadowAudit) -> JsonObject:
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


def _proposal_search(manifest: Manifest, state: ReplayState) -> JsonObject:
    cohort = latest_completed_cohort(state)
    if cohort is None:
        raise ValueError("report requires a completed cohort")
    dispositions = dict(state.dispositions)
    accepted: list[JsonObject] = []
    rejected = {source: 0 for source in manifest.source_schedule}
    for proposal in state.proposals:
        disposition = dispositions.get(proposal.proposal_index)
        if disposition == "rejected":
            rejected[proposal.source] = rejected.get(proposal.source, 0) + 1
        if disposition == "accepted":
            accepted.append(
                {
                    "cohort_slot": proposal.cohort_slot,
                    "cohort_index": proposal.cohort_index,
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
    completed_cohorts = len(state.completed_cohorts)
    # Derive configured slots from the actual completed cohort count and the
    # repeated frozen schedules.
    guided_source = manifest.proposer.encoded()["guided_source"]
    assert isinstance(guided_source, str)
    model_slots = manifest.source_schedule.count(guided_source) + (
        completed_cohorts - 1
    ) * manifest.challenger_source_schedule.count(guided_source)
    reserve_slots = manifest.source_schedule.count("random_reserve") + (
        completed_cohorts - 1
    ) * manifest.challenger_source_schedule.count("random_reserve")
    tuning = comparable_prefix_observations(
        state.observations, cohort.candidates, manifest.tuning_prefix
    )
    frontier = tuning_frontier(tuning)
    rejected_json: JsonObject = {source: count for source, count in rejected.items()}
    return {
        "policy_version": manifest.proposer.encoded()["policy_version"],
        "model_version": manifest.proposer.encoded()["guided_adapter_version"],
        "cost_policy_version": manifest.proposer.encoded()["cost_policy_version"],
        "configured": {
            "bootstrap": manifest.bootstrap_candidates,
            "model": model_slots,
            "random_reserve": reserve_slots,
            "cohorts": completed_cohorts,
            "retained_elites": manifest.finalists,
            "family_exclusion_policy_version": manifest.proposer.encoded()[
                "family_exclusion_policy_version"
            ],
            "excluded_families": list(manifest.excluded_families),
        },
        "accepted": _array(accepted),
        "rejections_by_source": rejected_json,
        "actual_source_attempts": {
            source: sum(item.source == source for item in state.proposals)
            for source in (
                "schema_default",
                "bootstrap_random",
                "smac_model",
                "random_reserve",
                "random_search",
                "qmc_search",
                "irace_model",
            )
        },
        "final_frontier_id": frontier.frontier_id,
        "final_observation_count": len(frontier.observation_ids),
    }


def _candidate_lifecycle(manifest: Manifest, state: ReplayState) -> JsonObject:
    dispositions = dict(state.dispositions)
    refill = dict(state.refill_attempts)
    attempts: list[JsonObject] = []
    replacement_proposals: list[Candidate] = []
    for proposal in state.proposals:
        if (failed_id := refill.get(proposal.proposal_index)) is None:
            continue
        attempts.append(
            {
                "failed_candidate_id": failed_id,
                "proposal_index": proposal.proposal_index,
                "cohort_index": proposal.cohort_index,
                "cohort_slot": proposal.cohort_slot,
                "source": proposal.source,
                "source_attempt": proposal.provenance.source_attempt,
                "disposition": dispositions.get(proposal.proposal_index),
            }
        )
        replacement_proposals.append(proposal.candidate)
    return {
        "policy": manifest.candidate_failure_policy.encoded(),
        "failed_candidates": _array(
            {
                "cohort_index": item.cohort_index,
                "candidate_id": item.candidate_id,
                "triggering_pair_id": item.triggering_pair_id,
                "started_attempts": item.started_attempts,
                "failed_attempts": item.failed_attempts,
                "censored_attempts": item.censored_attempts,
                "completed_tuning_pair_ids": list(item.completed_tuning_pair_ids),
            }
            for item in state.candidate_failures
        ),
        "replacement_attempts": _array(attempts),
        "accepted_replacements": _array(
            {
                "failed_candidate_id": item["failed_candidate_id"],
                "candidate_id": proposal.candidate_id,
            }
            for item, proposal in zip(attempts, replacement_proposals, strict=True)
            if item["disposition"] == "accepted"
        ),
    }


def _compute_section(manifest: Manifest, state: ReplayState) -> JsonObject:
    """Projection of compute budget and evidence-derived ledger values."""
    ledger = state.compute
    budget = manifest.compute_budget
    return {
        "policy_version": "safe-boundary-pair-attempts-v1",
        "budget": {
            "tuning_pair_attempts": budget.tuning_pair_attempts,
            "validation_pair_attempts": budget.validation_pair_attempts,
            "diagnostic_pair_attempts": budget.diagnostic_pair_attempts,
        },
        "tuning": {
            "pair_attempts": ledger.tuning.pair_attempts,
            "completed_pairs": ledger.tuning.completed_pairs,
            "failed_attempts": ledger.tuning.failed_attempts,
            "censored_attempts": ledger.tuning.censored_attempts,
            "physical_games": ledger.tuning.physical_games,
            "search_iterations": ledger.tuning.search_iterations,
            "wall_time_ms": ledger.tuning.wall_time_ms,
            "unspent_pair_attempts": max(
                0, budget.tuning_pair_attempts - ledger.tuning.pair_attempts
            ),
            "overrun_pair_attempts": max(
                0, ledger.tuning.pair_attempts - budget.tuning_pair_attempts
            ),
        },
        "validation": {
            "pair_attempts": ledger.validation.pair_attempts,
            "completed_pairs": ledger.validation.completed_pairs,
            "failed_attempts": ledger.validation.failed_attempts,
            "censored_attempts": ledger.validation.censored_attempts,
            "physical_games": ledger.validation.physical_games,
            "search_iterations": ledger.validation.search_iterations,
            "wall_time_ms": ledger.validation.wall_time_ms,
            "unspent_pair_attempts": max(
                0, budget.validation_pair_attempts - ledger.validation.pair_attempts
            ),
            "overrun_pair_attempts": max(
                0, ledger.validation.pair_attempts - budget.validation_pair_attempts
            ),
        },
        "diagnostic": _compute_bucket(ledger.diagnostic, budget.diagnostic_pair_attempts),
    }


def _compute_bucket(value: PhaseCompute, budget: int) -> JsonObject:
    return {
        "pair_attempts": value.pair_attempts,
        "completed_pairs": value.completed_pairs,
        "failed_attempts": value.failed_attempts,
        "censored_attempts": value.censored_attempts,
        "physical_games": value.physical_games,
        "search_iterations": value.search_iterations,
        "wall_time_ms": value.wall_time_ms,
        "unspent_pair_attempts": max(0, budget - value.pair_attempts),
        "overrun_pair_attempts": max(0, value.pair_attempts - budget),
    }


def _diagnostic_matchup_graph(
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


def write_report(run_dir: Path) -> JsonObject:
    report = build_report(run_dir)
    atomic_json(run_dir / "report.json", report)
    return report
