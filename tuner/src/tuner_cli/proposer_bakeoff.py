"""Sequential matched proposer-policy experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import read_manifest
from .bakeoff_artifacts import (
    BakeoffSpec,
    SharedRun,
    encode_experiment,
    experiment_fingerprint,
)
from .bakeoff_metrics import ChildFact, aggregate
from .codec import JsonObject
from .domain import Observation
from .evidence import read_events
from .proposer import POLICIES, ProposerPolicy
from .replay import replay
from .run import RunOptions, run_foreground
from .target import Target


@dataclass(frozen=True, slots=True)
class BakeoffCell:
    cell_id: str
    budget: int
    seed: int
    policy: ProposerPolicy
    run_dir: Path


def run_bakeoff(
    spec: BakeoffSpec, experiment_dir: Path, resume: bool = False, target: Target | None = None
) -> Path:
    cells = _cells(spec, experiment_dir)
    manifest_path = experiment_dir / "experiment.json"
    encoded = encode_experiment(spec, [_cell_summary(cell) for cell in cells])
    if manifest_path.exists():
        if manifest_path.read_text() != encoded:
            raise ValueError("existing experiment does not match strict specification")
    elif resume:
        raise ValueError("cannot resume without experiment.json")
    else:
        experiment_dir.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(encoded)
    for cell in cells:
        run_foreground(_options(spec, cell), target=target)
    fingerprint_value = experiment_fingerprint(manifest_path.read_text())
    facts = [_child_fact(cell) for cell in cells]
    results = aggregate(facts, fingerprint_value, spec.decision)
    (experiment_dir / "results.json").write_text(results)
    return experiment_dir / "results.json"


def _cell_summary(cell: BakeoffCell) -> JsonObject:
    return {
        "cell_id": cell.cell_id,
        "budget": cell.budget,
        "seed": cell.seed,
        "policy": cell.policy,
    }


def _cells(spec: BakeoffSpec, directory: Path) -> list[BakeoffCell]:
    result: list[BakeoffCell] = []
    for budget in spec.tuning_pair_budgets:
        for seed in spec.proposal_seeds:
            for policy in POLICIES:
                run_dir = directory / "runs" / policy / f"budget-{budget}" / f"seed-{seed}"
                result.append(
                    BakeoffCell(f"{budget}:{seed}:{policy}", budget, seed, policy, run_dir)
                )
    return result


def _options(spec: BakeoffSpec, cell: BakeoffCell) -> RunOptions:
    shared: SharedRun = spec.shared_run
    return RunOptions(
        spec.game_binary,
        cell.run_dir,
        objective_file=spec.objective_file,
        seed=cell.seed,
        task_seed=spec.task_seed,
        cohort_size=shared.cohort_size,
        finalists=shared.finalists,
        bootstrap_candidates=shared.bootstrap_candidates,
        random_reserve_candidates=shared.random_reserve_candidates,
        tuning_pairs=shared.tuning_pairs,
        tuning_pair_budget=cell.budget,
        validation_pair_budget=shared.validation_pair_budget,
        diagnostic_pair_budget=0,
        production_validation_pairs=shared.production_validation_pairs,
        tuning_effort=shared.tuning_effort,
        validation_effort=shared.validation_effort,
        production_effort=shared.production_effort,
        pair_timeout_seconds=shared.pair_timeout_seconds,
        evaluator_workers=shared.evaluator_workers,
        constraints=shared.constraints,
        active_elimination_audit_probability=None,
        proposer_policy=cell.policy,
        resume=cell.run_dir.exists(),
    )


def _child_fact(cell: BakeoffCell) -> ChildFact:
    manifest = read_manifest(cell.run_dir / "manifest.json")
    state = replay(manifest, read_events(cell.run_dir / "evidence.jsonl"))
    if state.terminal_status != "complete" or not state.finalists:
        raise ValueError("bakeoff child is incomplete")
    finalist_ids = {candidate.candidate_id for candidate in state.finalists}
    finalist_fingerprints = {
        candidate.candidate_id: candidate.fingerprint for candidate in state.finalists
    }
    observations = [
        item
        for item in state.observations
        if item.phase == "validation" and item.candidate_id in finalist_ids
    ]
    if not observations:
        raise ValueError("bakeoff child has no finalist held-out observations")
    means_by_candidate: dict[str, float] = {}
    for item in observations:
        current = means_by_candidate.get(item.candidate_id)
        if current is None or item.estimate.mean > current:
            means_by_candidate[item.candidate_id] = item.estimate.mean
    best = max(observations, key=lambda item: _score_key(item, finalist_fingerprints))
    return ChildFact(
        cell_id=cell.cell_id,
        budget=cell.budget,
        seed=cell.seed,
        policy=cell.policy,
        manifest_fingerprint=manifest.fingerprint,
        best_candidate_fingerprint=finalist_fingerprints[best.candidate_id],
        finalist_fingerprints=tuple(sorted(finalist_fingerprints.values())),
        held_out_means=tuple(
            sorted((finalist_fingerprints[cid], mean) for cid, mean in means_by_candidate.items())
        ),
        held_out_best_score=best.estimate.mean,
        tuning_pair_attempts=state.compute.tuning.pair_attempts,
        tuning_physical_games=state.compute.tuning.physical_games,
        tuning_search_iterations=state.compute.tuning.search_iterations,
        tuning_wall_time_ms=state.compute.tuning.wall_time_ms,
    )


def _score_key(observation: Observation, fingerprints: dict[str, str]) -> tuple[float, str]:
    return observation.estimate.mean, fingerprints[observation.candidate_id]
