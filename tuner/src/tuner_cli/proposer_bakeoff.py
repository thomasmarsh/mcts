"""Sequential matched proposer-policy experiment orchestration."""

from __future__ import annotations

from pathlib import Path

from .artifacts import read_manifest
from .bakeoff_artifacts import BakeoffSpec, encode_experiment
from .bakeoff_metrics import aggregate
from .evidence import read_events
from .identity import fingerprint
from .replay import replay
from .run import RunOptions, run_foreground
from .target import Target


def run_bakeoff(
    spec: BakeoffSpec, experiment_dir: Path, resume: bool = False, target: Target | None = None
) -> Path:
    cells = _cells(spec, experiment_dir)
    manifest_path = experiment_dir / "experiment.json"
    encoded = encode_experiment(
        spec, [{k: v for k, v in cell.items() if k != "run_dir"} for cell in cells]
    )
    if manifest_path.exists():
        if manifest_path.read_text() != encoded:
            raise ValueError("existing experiment does not match strict specification")
    elif resume:
        raise ValueError("cannot resume without experiment.json")
    else:
        experiment_dir.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(encoded)
    for cell in cells:
        run_dir = cell["run_dir"]
        assert isinstance(run_dir, Path)
        options = _options(spec, cell, resume=run_dir.exists())
        run_foreground(options, target=target)
    facts = [_child_fact(cell) for cell in cells]
    raw = strict_experiment(manifest_path.read_text())
    results = aggregate(facts, raw["fingerprint"], spec.decision)
    (experiment_dir / "results.json").write_text(results)
    return experiment_dir / "results.json"


def _cells(spec: BakeoffSpec, directory: Path) -> list[dict[str, object]]:
    result = []
    for budget in spec.tuning_pair_budgets:
        for seed in spec.proposal_seeds:
            for policy in ("random", "qmc", "smac_mixed", "irace_generational"):
                run_dir = directory / "runs" / policy / f"budget-{budget}" / f"seed-{seed}"
                result.append(
                    {
                        "cell_id": f"{budget}:{seed}:{policy}",
                        "budget": budget,
                        "seed": seed,
                        "policy": policy,
                        "run_dir": run_dir,
                    }
                )
    return result


def _options(spec: BakeoffSpec, cell: dict[str, object], resume: bool) -> RunOptions:
    shared = spec.shared_run

    def effort(name: str):
        raw = shared[f"{name}_effort"]
        if not isinstance(raw, dict) or set(raw) != {"kind", "value"}:
            raise ValueError(f"invalid {name} effort")
        from .domain import SearchEffort

        return SearchEffort(raw["kind"], raw["value"])  # type: ignore[arg-type]

    return RunOptions(
        spec.game_binary,
        cell["run_dir"],
        objective_file=spec.objective_file,
        seed=cell["seed"],
        task_seed=spec.task_seed,
        cohort_size=shared["cohort_size"],
        finalists=shared["finalists"],
        bootstrap_candidates=shared["bootstrap_candidates"],
        random_reserve_candidates=shared["random_reserve_candidates"],
        tuning_pairs=shared["tuning_pairs"],
        tuning_pair_budget=cell["budget"],
        validation_pair_budget=shared["validation_pair_budget"],
        diagnostic_pair_budget=0,
        production_validation_pairs=shared["production_validation_pairs"],
        tuning_effort=effort("tuning"),
        validation_effort=effort("validation"),
        production_effort=effort("production"),
        pair_timeout_seconds=shared["pair_timeout_seconds"],
        evaluator_workers=shared["evaluator_workers"],
        excluded_families=tuple(shared.get("excluded_families", [])),
        active_elimination_audit_probability=None,
        proposer_policy=cell["policy"],
        resume=resume,
    )  # type: ignore[arg-type]


def _child_fact(cell: dict[str, object]) -> dict[str, object]:
    manifest = read_manifest(Path(cell["run_dir"]) / "manifest.json")
    state = replay(manifest, read_events(Path(cell["run_dir"]) / "evidence.jsonl"))
    if state.terminal_status != "complete" or not state.finalists:
        raise ValueError("bakeoff child is incomplete")
    observations = [
        item
        for item in state.observations
        if item.phase == "validation"
        and item.candidate_id in {candidate.candidate_id for candidate in state.finalists}
    ]
    best = max(
        observations,
        key=lambda item: (
            item.estimate.mean,
            next(
                candidate.fingerprint
                for candidate in state.finalists
                if candidate.candidate_id == item.candidate_id
            ),
        ),
    )
    candidate = next(
        candidate for candidate in state.finalists if candidate.candidate_id == best.candidate_id
    )
    return {
        "cell_id": cell["cell_id"],
        "budget": cell["budget"],
        "seed": cell["seed"],
        "policy": cell["policy"],
        "manifest_fingerprint": manifest.fingerprint,
        "candidate_fingerprint": candidate.fingerprint,
        "held_out_best_score": best.estimate.mean,
        "tuning_pair_attempts": state.compute.tuning.pair_attempts,
        "tuning_physical_games": state.compute.tuning.physical_games,
        "tuning_search_iterations": state.compute.tuning.search_iterations,
        "tuning_wall_time_ms": state.compute.tuning.wall_time_ms,
    }


def strict_experiment(value: str) -> dict[str, object]:
    from .codec import strict_json

    raw = strict_json(value, "experiment")
    if not isinstance(raw, dict) or "fingerprint" not in raw:
        raise ValueError("invalid experiment manifest")
    copy = dict(raw)
    actual = copy.pop("fingerprint")
    if actual != fingerprint(copy):
        raise ValueError("experiment fingerprint mismatch")
    return raw
