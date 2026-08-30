"""Short foreground proposal, selection, and held-out validation flow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from .domain import (
    Candidate,
    IterationBudget,
    Observation,
    PairResult,
    PairTask,
    Proposal,
    TaskBlock,
    TaskCase,
)
from .evidence import EvidenceWriter, write_manifest
from .identity import canonical_json, derive_task_seed, fingerprint, sha256_file, stable_id
from .report import write_report
from .schema import GameSpec, decode_game_spec
from .space import build_space, default_values, random_values
from .statistics import marginal_interval, pair_utility
from .target import GameBinaryTarget, PairExecutionError, Target


@dataclass(frozen=True, slots=True)
class RunOptions:
    game_binary: Path
    run_dir: Path
    seed: int = 42
    cohort_size: int = 8
    finalists: int = 3
    tuning_pairs: int = 4
    validation_pairs: int = 8
    tuning_max_iterations: int = 1_000
    validation_max_iterations: int = 10_000
    production_max_iterations: int = 10_000
    pair_timeout_seconds: int = 600


def _validate_options(options: RunOptions) -> tuple[Path, Path]:
    values = asdict(options)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for key, value in values.items()
        if key not in {"game_binary", "run_dir"}
    ):
        raise ValueError("all numeric arguments must be positive integers")
    if options.cohort_size < 2 or options.finalists > options.cohort_size:
        raise ValueError("cohort size must be at least 2 and finalists cannot exceed it")
    run_dir = options.run_dir.expanduser().resolve()
    if run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}")
    binary = options.game_binary.expanduser().resolve()
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise ValueError(f"game binary is missing, not a regular executable file: {binary}")
    return binary, run_dir


def _candidate(config: dict[str, object]) -> Candidate:
    canonical = canonical_json(config)
    config_fingerprint = fingerprint(config)
    return Candidate(f"candidate-{config_fingerprint}", config_fingerprint, canonical)


def _tasks(
    phase: Literal["tuning", "validation"],
    count: int,
    root_seed: int,
    opponent: Candidate,
    game_config_fingerprint: str,
) -> TaskBlock:
    cases = []
    for ordinal in range(count):
        seed = derive_task_seed(root_seed, phase, ordinal)
        payload = {
            "phase": phase,
            "ordinal": ordinal,
            "seed": seed,
            "opponent_fingerprint": opponent.fingerprint,
            "game_config_fingerprint": game_config_fingerprint,
            "start": "default",
        }
        cases.append(
            TaskCase(
                stable_id("task", payload),
                phase,
                ordinal,
                seed,
                f"opponent-default-{opponent.fingerprint}",
                opponent.fingerprint,
                game_config_fingerprint,
            )
        )
    task_ids = [case.task_id for case in cases]
    return TaskBlock(
        stable_id("block", {"phase": phase, "task_ids": task_ids}), phase, tuple(cases)
    )


def _task_dict(task: TaskCase) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "phase": task.phase,
        "ordinal": task.ordinal,
        "seed": task.seed,
        "opponent_id": task.opponent_id,
        "opponent_fingerprint": task.opponent_fingerprint,
        "game_config_fingerprint": task.game_config_fingerprint,
        "start": task.start,
    }


def _block_dict(block: TaskBlock) -> dict[str, object]:
    return {
        "block_id": block.block_id,
        "phase": block.phase,
        "cases": [_task_dict(task) for task in block.cases],
    }


def _spec_for(target: Target, binary: Path) -> GameSpec:
    return decode_game_spec(target.describe(), binary, sha256_file(binary))


def _game_payload(pair: PairResult, game: object) -> dict[str, object]:
    # GameResult's scalar-only records make this event a direct report input.
    from .domain import GameResult

    assert isinstance(game, GameResult)
    return {
        "pair_id": pair.task.pair_id,
        "game_id": game.game_id,
        "candidate_id": pair.task.candidate_id,
        "task_id": pair.task.task_case.task_id,
        "phase": pair.task.task_case.phase,
        "candidate_side": game.candidate_side,
        "outcome": game.outcome,
        "derived_seed": game.derived_seed,
        "round": game.round,
        "seq": game.seq,
        "trace_game_seq": game.trace_game_seq,
        "plies": game.plies,
        "elapsed_ms": game.elapsed_ms,
        "candidate_metrics": asdict(game.candidate_metrics),
        "opponent_metrics": asdict(game.opponent_metrics),
        "raw_record": game.raw_record,
    }


def _pair_payload(result: PairResult) -> dict[str, object]:
    return {
        "phase": result.task.task_case.phase,
        "candidate_id": result.task.candidate_id,
        "task_id": result.task.task_case.task_id,
        "pair_id": result.task.pair_id,
        "opponent_id": result.task.task_case.opponent_id,
        "budget": result.task.budget.max_iterations,
        "game_ids": [game.game_id for game in result.games],
        "outcomes": [game.outcome for game in result.games],
        "pair_utility": pair_utility(result),
    }


def _observation(
    candidate: Candidate,
    phase: Literal["tuning", "validation"],
    block: TaskBlock,
    budget: IterationBudget,
    results: list[PairResult],
) -> Observation:
    by_task = {result.task.task_case.task_id: result for result in results}
    if set(by_task) != {case.task_id for case in block.cases}:
        raise ValueError("observation has missing or duplicate task results")
    ordered = tuple(pair_utility(by_task[case.task_id]) for case in block.cases)
    interval = marginal_interval(ordered)
    return Observation(
        candidate.candidate_id, phase, block.block_id, len(ordered), budget, ordered, interval
    )


def _emit_observation(writer: EvidenceWriter, observation: Observation) -> None:
    writer.append(
        "observation_completed",
        {
            "candidate_id": observation.candidate_id,
            "phase": observation.phase,
            "block_id": observation.block_id,
            "prefix_length": observation.prefix_length,
            "budget": observation.budget.max_iterations,
            "pair_utilities": list(observation.pair_utilities),
            "estimate": asdict(observation.estimate),
            "counts": {"pairs": observation.prefix_length, "games": observation.prefix_length * 2},
        },
    )


def _failure_payload(stage: str, error: BaseException) -> dict[str, object]:
    payload: dict[str, object] = {"stage": stage, "kind": "runtime", "message": str(error)}
    if isinstance(error, PairExecutionError):
        payload.update(
            {
                "kind": error.kind,
                "command": error.command,
                "returncode": error.returncode,
                "stderr": error.stderr,
                "stdout": error.stdout,
            }
        )
        partial = []
        for line in error.stdout.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("type") == "configured_match_result":
                partial.append(canonical_json(record))
        if partial:
            payload["partial_output"] = partial
    return payload


def run_foreground(
    options: RunOptions, target: Target | None = None, run_dir: Path | None = None
) -> Path:
    """Execute one non-resumable generic game-tuning run, returning its report path."""
    if run_dir is not None:
        options = replace(options, run_dir=run_dir)
    binary, options_run_dir = _validate_options(options)
    if target is None:
        target = GameBinaryTarget(binary)
    spec = _spec_for(target, binary)
    try:
        space = build_space(spec.tuning, options.seed)
        schema_default = default_values(space)
    except Exception as error:
        raise ValueError(
            f"invalid tuning metadata: ConfigSpace construction/default failed: {error}"
        ) from error
    opponent = _candidate(schema_default)
    tuning = _tasks(
        "tuning",
        options.tuning_pairs,
        options.seed,
        opponent,
        fingerprint(json.loads(spec.default_game_config)),
    )
    validation = _tasks(
        "validation",
        options.validation_pairs,
        options.seed,
        opponent,
        fingerprint(json.loads(spec.default_game_config)),
    )
    if len({case.seed for case in tuning.cases + validation.cases}) != len(tuning.cases) + len(
        validation.cases
    ):
        raise ValueError("task seed derivation collided")
    options_run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "run_id": options_run_dir.name,
        "command_policy_version": "generic-foreground-v1",
        "binary": {"path": str(spec.binary_path), "sha256": spec.binary_sha256},
        "engine_fingerprint": spec.engine_fingerprint,
        "description": spec.raw_description,
        "description_fingerprint": spec.description_fingerprint,
        "kind": spec.kind,
        "label": spec.label,
        "game_description": spec.description,
        "ai_presets": [asdict(preset) for preset in spec.ai_presets],
        "tuning": {
            "id": spec.tuning.id,
            "baselines": list(spec.tuning.baselines),
            "eval_rounds": spec.tuning.eval_rounds,
            "game_config": spec.tuning.game_config,
            "parameters": [asdict(parameter) for parameter in spec.tuning.parameters],
            "conditions": [asdict(condition) for condition in spec.tuning.conditions],
        },
        "tuning_schema_fingerprint": spec.schema_fingerprint,
        "game_config": spec.default_game_config,
        "game_config_fingerprint": fingerprint(json.loads(spec.default_game_config)),
        "parameters": [asdict(parameter) for parameter in spec.tuning.parameters],
        "conditions": [asdict(condition) for condition in spec.tuning.conditions],
        "proposer": {
            "kind": "configspace_random",
            "version": "configspace-random-v1",
            "seed": options.seed,
            "cohort_size": options.cohort_size,
            "finalists": options.finalists,
        },
        "opponent": {
            "id": f"opponent-default-{opponent.fingerprint}",
            "canonical_config": opponent.canonical_config,
            "fingerprint": opponent.fingerprint,
        },
        "tuning_tasks": _block_dict(tuning),
        "validation_tasks": _block_dict(validation),
        "budgets": {
            "tuning": options.tuning_max_iterations,
            "validation": options.validation_max_iterations,
            "production": options.production_max_iterations,
        },
        "utility_formula_version": "pair_mean_v1",
        "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
        "interval_method": "hoeffding_pair_bound_v1",
        "confidence_level": 0.95,
        "tie_rule_version": "paired_hoeffding_v1",
        "limitations": [
            "one opponent",
            "default starting state",
            "sequential execution",
            "fixed iterations",
            "no resume",
        ],
    }
    published = write_manifest(options_run_dir / "manifest.json", manifest)
    writer = EvidenceWriter(options_run_dir / "evidence.jsonl")
    stage = "proposal"
    try:
        accepted: list[Candidate] = []
        seen: set[str] = set()
        index = 0
        default = _candidate(schema_default)
        proposal = Proposal(index, "schema_default", "configspace-random-v1", default)
        writer.append(
            "proposal_created",
            {
                "proposal_index": index,
                "source": proposal.source,
                "proposer_version": proposal.proposer_version,
                "candidate_id": default.candidate_id,
                "fingerprint": default.fingerprint,
                "canonical_config": default.canonical_config,
            },
        )
        seen.add(default.fingerprint)
        validation_result = target.validate([default], opponent, spec.default_game_config)
        if not validation_result.valid:
            raise RuntimeError("schema default failed semantic validation")
        accepted.append(default)
        draws = 0
        cap = max(100, options.cohort_size * 100)
        while len(accepted) < options.cohort_size and draws < cap:
            batch: list[tuple[int, Candidate]] = []
            while len(batch) < min(8, options.cohort_size - len(accepted)) and draws < cap:
                draws += 1
                index += 1
                candidate = _candidate(random_values(space))
                writer.append(
                    "proposal_created",
                    {
                        "proposal_index": index,
                        "source": "configspace_random",
                        "proposer_version": "configspace-random-v1",
                        "candidate_id": candidate.candidate_id,
                        "fingerprint": candidate.fingerprint,
                        "canonical_config": candidate.canonical_config,
                    },
                )
                if candidate.fingerprint in seen:
                    writer.append(
                        "proposal_rejected",
                        {
                            "proposal_index": index,
                            "candidate_id": candidate.candidate_id,
                            "fingerprint": candidate.fingerprint,
                            "canonical_config": candidate.canonical_config,
                            "reason": "duplicate",
                            "errors": [],
                        },
                    )
                    continue
                seen.add(candidate.fingerprint)
                batch.append((index, candidate))
            if not batch:
                continue
            result = target.validate(
                [candidate for _, candidate in batch], opponent, spec.default_game_config
            )
            unscoped = [error for error in result.errors if error.candidate_index is None]
            if unscoped or (not result.valid and not result.errors):
                raise RuntimeError("validation error lacks candidate_index")
            for candidate_index, (proposal_index, candidate) in enumerate(batch):
                errors = [
                    error for error in result.errors if error.candidate_index == candidate_index
                ]
                if errors:
                    writer.append(
                        "proposal_rejected",
                        {
                            "proposal_index": proposal_index,
                            "candidate_id": candidate.candidate_id,
                            "fingerprint": candidate.fingerprint,
                            "canonical_config": candidate.canonical_config,
                            "reason": "semantic_validation",
                            "errors": [asdict(error) for error in errors],
                        },
                    )
                else:
                    accepted.append(candidate)
        if len(accepted) < options.cohort_size:
            raise RuntimeError(
                f"proposal draw cap reached: accepted {len(accepted)}/{options.cohort_size}"
            )
        final_validation = target.validate(accepted, opponent, spec.default_game_config)
        if not final_validation.valid:
            raise RuntimeError("final cohort validation failed")
        writer.append(
            "cohort_accepted",
            {
                "candidate_ids": [candidate.candidate_id for candidate in accepted],
                "validation_response_fingerprint": fingerprint(asdict(final_validation)),
            },
        )
        stage = "tuning"
        tuning_results: dict[str, list[PairResult]] = {
            candidate.candidate_id: [] for candidate in accepted
        }
        tuning_budget = IterationBudget(options.tuning_max_iterations)
        for case in tuning.cases:
            for candidate in accepted:
                task = PairTask(
                    stable_id(
                        "pair",
                        {
                            "candidate_fingerprint": candidate.fingerprint,
                            "task_id": case.task_id,
                            "opponent_fingerprint": opponent.fingerprint,
                            "max_iterations": tuning_budget.max_iterations,
                        },
                    ),
                    candidate.candidate_id,
                    case,
                    tuning_budget,
                )
                writer.append(
                    "pair_started",
                    {
                        "phase": case.phase,
                        "candidate_id": candidate.candidate_id,
                        "task_id": case.task_id,
                        "pair_id": task.pair_id,
                        "opponent_id": case.opponent_id,
                        "budget": task.budget.max_iterations,
                        "task_seed": case.seed,
                    },
                )
                result = target.evaluate(
                    task,
                    candidate,
                    opponent,
                    spec.default_game_config,
                    options.pair_timeout_seconds,
                )
                for game in result.games:
                    writer.append("game_finished", _game_payload(result, game))
                writer.append("pair_completed", _pair_payload(result))
                tuning_results[candidate.candidate_id].append(result)
        tuning_observations = [
            _observation(
                candidate, "tuning", tuning, tuning_budget, tuning_results[candidate.candidate_id]
            )
            for candidate in accepted
        ]
        for observation in tuning_observations:
            _emit_observation(writer, observation)
        ranking = sorted(
            tuning_observations,
            key=lambda item: (
                -item.estimate.mean,
                next(
                    candidate.fingerprint
                    for candidate in accepted
                    if candidate.candidate_id == item.candidate_id
                ),
            ),
        )
        finalist_ids = [item.candidate_id for item in ranking[: options.finalists]]
        finalists = [
            next(candidate for candidate in accepted if candidate.candidate_id == candidate_id)
            for candidate_id in finalist_ids
        ]
        writer.append(
            "finalists_selected",
            {
                "finalist_ids": finalist_ids,
                "tuning_estimates": {
                    item.candidate_id: item.estimate.mean for item in tuning_observations
                },
                "source_block": tuning.block_id,
                "budget": tuning_budget.max_iterations,
                "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
            },
        )
        stage = "validation"
        validation_results: dict[str, list[PairResult]] = {
            candidate.candidate_id: [] for candidate in finalists
        }
        validation_budget = IterationBudget(options.validation_max_iterations)
        for case in validation.cases:
            for candidate in finalists:
                task = PairTask(
                    stable_id(
                        "pair",
                        {
                            "candidate_fingerprint": candidate.fingerprint,
                            "task_id": case.task_id,
                            "opponent_fingerprint": opponent.fingerprint,
                            "max_iterations": validation_budget.max_iterations,
                        },
                    ),
                    candidate.candidate_id,
                    case,
                    validation_budget,
                )
                writer.append(
                    "pair_started",
                    {
                        "phase": case.phase,
                        "candidate_id": candidate.candidate_id,
                        "task_id": case.task_id,
                        "pair_id": task.pair_id,
                        "opponent_id": case.opponent_id,
                        "budget": task.budget.max_iterations,
                        "task_seed": case.seed,
                    },
                )
                result = target.evaluate(
                    task,
                    candidate,
                    opponent,
                    spec.default_game_config,
                    options.pair_timeout_seconds,
                )
                for game in result.games:
                    writer.append("game_finished", _game_payload(result, game))
                writer.append("pair_completed", _pair_payload(result))
                validation_results[candidate.candidate_id].append(result)
        for candidate in finalists:
            _emit_observation(
                writer,
                _observation(
                    candidate,
                    "validation",
                    validation,
                    validation_budget,
                    validation_results[candidate.candidate_id],
                ),
            )
        claim = (
            "production"
            if options.validation_max_iterations == options.production_max_iterations
            else "mechanics_smoke"
        )
        writer.append(
            "run_completed",
            {
                "manifest_fingerprint": published["fingerprint"],
                "accepted_ids": [candidate.candidate_id for candidate in accepted],
                "finalist_ids": finalist_ids,
                "evidence_counts": {"events": writer._sequence + 1},
                "validation_claim": claim,
            },
        )
        report = write_report(options_run_dir)
        summary = ", ".join(item["candidate_id"] for item in report["validation_order"])
        print(f"run directory: {options_run_dir}")
        print(f"validation claim: {claim}")
        print(f"accepted/finalists: {len(accepted)}/{len(finalists)}")
        print(f"ranked: {summary}")
        return options_run_dir / "report.json"
    except BaseException as error:
        writer.append("failure", _failure_payload(stage, error))
        raise
