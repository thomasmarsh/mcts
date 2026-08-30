"""Foreground creation, replay, and sequential continuation for tuner runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .artifacts import (
    Manifest,
    build_manifest,
    configspace_version,
    manifest_json,
    production_claim,
    read_manifest,
)
from .domain import Candidate, Opponent, PairTask, Proposal, ValidationResult
from .evidence import EvidenceWriter, pair_payload, read_events, write_manifest
from .identity import candidate_from_config, canonical_json, sha256_file
from .objective import ResolvedObjective, resolve_objective
from .replay import _observation, _selection, expected_pairs, fold_events, observation_payload
from .report import write_report
from .schema import GameSpec, decode_game_spec
from .space import build_space, default_values, random_values
from .target import GameBinaryTarget, PairExecutionError, Target
from .tasks import validate_cycle_endpoint


@dataclass(frozen=True, slots=True)
class RunOptions:
    game_binary: Path
    run_dir: Path
    objective_file: Path | None = None
    seed: int = 42
    task_seed: int = 43
    cohort_size: int = 8
    finalists: int = 3
    tuning_pairs: int = 4
    validation_pairs: int = 8
    production_validation_pairs: int = 8
    tuning_max_iterations: int = 1_000
    validation_max_iterations: int = 10_000
    production_max_iterations: int = 10_000
    pair_timeout_seconds: int = 600
    resume: bool = False


class RunInterrupted(KeyboardInterrupt):
    """Marks an interruption whose operational event was already persisted."""


def _validate_options(options: RunOptions) -> tuple[Path, Path, Path]:
    numeric = (
        options.seed,
        options.task_seed,
        options.cohort_size,
        options.finalists,
        options.tuning_pairs,
        options.validation_pairs,
        options.production_validation_pairs,
        options.tuning_max_iterations,
        options.validation_max_iterations,
        options.production_max_iterations,
        options.pair_timeout_seconds,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in numeric
    ):
        raise ValueError("all numeric arguments must be positive integers")
    if options.cohort_size < 2 or options.finalists > options.cohort_size:
        raise ValueError("cohort size must be at least 2 and finalists cannot exceed it")
    if options.objective_file is None:
        raise ValueError("--objective-file is required")
    objective = options.objective_file.expanduser().resolve()
    if not objective.is_file():
        raise ValueError(f"objective file does not exist: {objective}")
    run_dir, binary = (
        options.run_dir.expanduser().resolve(),
        options.game_binary.expanduser().resolve(),
    )
    if options.resume:
        if not run_dir.is_dir():
            raise ValueError(f"resume run directory does not exist: {run_dir}")
    elif run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}; use --resume to continue it")
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise ValueError(f"game binary is missing, not a regular executable file: {binary}")
    return binary, run_dir, objective


def _spec_for(target: Target, binary: Path) -> GameSpec:
    return decode_game_spec(target.describe(), binary, sha256_file(binary))


def _proposal_payload(proposal: Proposal) -> dict[str, object]:
    return {
        "proposal_index": proposal.proposal_index,
        "source": proposal.source,
        "proposer_version": proposal.proposer_version,
        "candidate_id": proposal.candidate.candidate_id,
        "fingerprint": proposal.candidate.fingerprint,
        "canonical_config": proposal.candidate.canonical_config,
    }


def _disposition_payload(proposal: Proposal) -> dict[str, object]:
    return {
        "proposal_index": proposal.proposal_index,
        "candidate_id": proposal.candidate.candidate_id,
        "fingerprint": proposal.candidate.fingerprint,
        "canonical_config": proposal.candidate.canonical_config,
    }


def _refresh(manifest: Manifest, run_dir: Path):
    return fold_events(manifest, read_events(run_dir / "evidence.jsonl"))


def _schema_default(spec: GameSpec, seed: int) -> tuple[object, Candidate]:
    space = build_space(spec.tuning, seed)
    return space, candidate_from_config(default_values(space))


def _validate_options_against_panel(options: RunOptions, objective: ResolvedObjective) -> None:
    for count, label in (
        (options.tuning_pairs, "tuning pairs"),
        (options.validation_pairs, "validation pairs"),
        (options.production_validation_pairs, "production validation pairs"),
    ):
        validate_cycle_endpoint(objective.panel, count, label)
    if options.validation_pairs > options.production_validation_pairs:
        raise ValueError("validation pairs cannot exceed production validation pairs")
    if (
        options.tuning_max_iterations > options.production_max_iterations
        or options.validation_max_iterations > options.production_max_iterations
    ):
        raise ValueError("tuning or validation effort cannot exceed production effort")


def _panel_results(
    target: Target, candidates: tuple[Candidate, ...], manifest: Manifest
) -> tuple[ValidationResult, ...]:
    return tuple(
        target.validate(candidates, opponent, manifest.spec.default_game_config)
        for opponent in manifest.panel.opponents
    )


def _preflight_panel(
    target: Target, spec: GameSpec, default: Candidate, objective: ResolvedObjective
) -> None:
    results = tuple(
        target.validate((default,), opponent, spec.default_game_config)
        for opponent in objective.panel.opponents
    )
    if not all(result.valid for result in results):
        bad = [
            opponent.opponent_id
            for opponent, result in zip(objective.panel.opponents, results, strict=True)
            if not result.valid
        ]
        raise ValueError(f"schema default failed panel preflight: {', '.join(bad)}")


def _rejection_errors(
    manifest: Manifest, results: tuple[ValidationResult, ...]
) -> list[dict[str, object]]:
    return [
        {"opponent_id": opponent.opponent_id, "errors": [asdict(error) for error in result.errors]}
        for opponent, result in zip(manifest.panel.opponents, results, strict=True)
        if not result.valid
    ]


def _validate_proposal(
    target: Target, manifest: Manifest, candidate: Candidate
) -> tuple[ValidationResult, ...]:
    return _panel_results(target, (candidate,), manifest)


def _configuration_failure(writer: EvidenceWriter, message: str) -> None:
    writer.append("run_failed", {"kind": "configuration", "message": message})
    raise RuntimeError(message)


def _restore_space(manifest: Manifest):
    if manifest.proposer["configspace_version"] != configspace_version():
        raise ValueError("ConfigSpace version differs from frozen manifest")
    space, default = _schema_default(manifest.spec, manifest.seed)
    if default.canonical_config != manifest.opponent.canonical_config:
        raise ValueError("ConfigSpace default differs from manifest objective")
    return space


def _replay_draws(manifest: Manifest, state, space) -> None:
    for proposal in state.proposals:
        if proposal.proposal_index == 0:
            if proposal.candidate.canonical_config != manifest.opponent.canonical_config:
                raise ValueError("default proposal differs from manifest")
        elif candidate_from_config(random_values(space)) != proposal.candidate:
            raise ValueError("recorded random proposal differs from frozen sampler")


def _finish_cohort(
    manifest: Manifest, state, writer: EvidenceWriter, target: Target, space
) -> tuple[Candidate, ...]:
    _replay_draws(manifest, state, space)
    proposals, dispositions = list(state.proposals), dict(state.dispositions)
    seen = {proposal.candidate.fingerprint for proposal in proposals}
    for proposal in proposals:
        if proposal.proposal_index not in dispositions:
            results = _validate_proposal(target, manifest, proposal.candidate)
            if proposal.proposal_index == 0 and not all(result.valid for result in results):
                _configuration_failure(writer, "schema default failed semantic validation")
            if all(result.valid for result in results):
                writer.append("proposal_accepted", _disposition_payload(proposal))
            else:
                writer.append(
                    "proposal_rejected",
                    {
                        **_disposition_payload(proposal),
                        "reason": "semantic_validation",
                        "errors": _rejection_errors(manifest, results),
                    },
                )
            dispositions = dict(_refresh(manifest, writer.path.parent).dispositions)
    draws, cap = len(proposals) - 1, max(100, manifest.cohort_size * 100)
    while (
        len([value for value in dispositions.values() if value == "accepted"])
        < manifest.cohort_size
        and draws < cap
    ):
        draws += 1
        proposal = Proposal(
            len(proposals),
            "configspace_random",
            "configspace-random-v1",
            candidate_from_config(random_values(space)),
        )
        writer.append("proposal_created", _proposal_payload(proposal))
        proposals.append(proposal)
        if proposal.candidate.fingerprint in seen:
            writer.append(
                "proposal_rejected",
                {**_disposition_payload(proposal), "reason": "duplicate", "errors": []},
            )
        else:
            seen.add(proposal.candidate.fingerprint)
            results = _validate_proposal(target, manifest, proposal.candidate)
            if all(result.valid for result in results):
                writer.append("proposal_accepted", _disposition_payload(proposal))
            else:
                writer.append(
                    "proposal_rejected",
                    {
                        **_disposition_payload(proposal),
                        "reason": "semantic_validation",
                        "errors": _rejection_errors(manifest, results),
                    },
                )
        dispositions = dict(_refresh(manifest, writer.path.parent).dispositions)
    accepted = tuple(
        proposal.candidate
        for proposal in proposals
        if dispositions.get(proposal.proposal_index) == "accepted"
    )
    if len(accepted) != manifest.cohort_size:
        _configuration_failure(
            writer, f"proposal draw cap reached: accepted {len(accepted)}/{manifest.cohort_size}"
        )
    results = _panel_results(target, accepted, manifest)
    if not all(result.valid for result in results):
        _configuration_failure(writer, "final cohort validation failed")
    fingerprints = [
        hashlib.sha256(canonical_json(asdict(result)).encode()).hexdigest() for result in results
    ]
    writer.append(
        "cohort_accepted",
        {
            "candidate_ids": [candidate.candidate_id for candidate in accepted],
            "validation_response_fingerprints": fingerprints,
        },
    )
    return accepted


def _failure_payload(task: PairTask, error: PairExecutionError) -> dict[str, object]:
    partial = []
    for line in error.stdout.splitlines():
        try:
            record = json.loads(line)
            if isinstance(record, dict) and record.get("type") == "configured_match_result":
                partial.append(canonical_json(record))
        except json.JSONDecodeError:
            pass
    return {
        "phase": task.task_case.phase,
        "candidate_id": task.candidate_id,
        "task_id": task.task_case.task_id,
        "pair_id": task.pair_id,
        "opponent_id": task.task_case.opponent_id,
        "budget": task.budget.max_iterations,
        "kind": error.kind,
        "command": error.command,
        "returncode": error.returncode,
        "stderr": error.stderr,
        "stdout": error.stdout,
        "partial_output": partial,
    }


def _opponent(manifest: Manifest, task: PairTask) -> Opponent:
    return next(
        item
        for item in manifest.panel.opponents
        if item.opponent_id == task.task_case.opponent_id
        and item.configuration_fingerprint == task.task_case.opponent_fingerprint
    )


def _emit_observations(
    manifest: Manifest, writer: EvidenceWriter, state, candidates: tuple[Candidate, ...], phase: str
) -> None:
    for candidate in candidates:
        if not any(
            item.phase == phase and item.candidate_id == candidate.candidate_id
            for item in state.observations
        ):
            pairs = [
                pair
                for pair in state.completed_pairs
                if pair.task.candidate_id == candidate.candidate_id
                and pair.task.task_case.phase == phase
            ]
            writer.append(
                "observation_completed",
                observation_payload(
                    _observation(candidate, phase, manifest, pairs),
                    len({pair.task.task_case.opponent_id for pair in pairs}),
                ),
            )
            state = _refresh(manifest, writer.path.parent)


def _continue(manifest: Manifest, writer: EvidenceWriter, target: Target, timeout: int) -> Path:
    state = _refresh(manifest, writer.path.parent)
    if state.terminal_status == "configuration_failed":
        raise ValueError("terminal configuration failure cannot resume")
    if state.terminal_status == "complete":
        return writer.path.parent / "report.json"
    cohort = state.cohort
    if cohort is None:
        cohort = _finish_cohort(manifest, state, writer, target, _restore_space(manifest))
        state = _refresh(manifest, writer.path.parent)
    while True:
        state = _refresh(manifest, writer.path.parent)
        if state.terminal_status == "complete":
            break
        plan = expected_pairs(manifest, cohort, state.finalists)
        done = {item.task.pair_id for item in state.completed_pairs}
        pending = next((item for item in plan if item.pair_id not in done), None)
        if pending is not None:
            candidate = next(
                item
                for item in (state.finalists or cohort)
                if item.candidate_id == pending.candidate_id
            )
            writer.append(
                "pair_started",
                {
                    "phase": pending.task_case.phase,
                    "candidate_id": pending.candidate_id,
                    "task_id": pending.task_case.task_id,
                    "pair_id": pending.pair_id,
                    "opponent_id": pending.task_case.opponent_id,
                    "budget": pending.budget.max_iterations,
                    "task_seed": pending.task_case.seed,
                },
            )
            try:
                result = target.evaluate(
                    pending,
                    candidate,
                    _opponent(manifest, pending),
                    manifest.spec.default_game_config,
                    timeout,
                )
            except PairExecutionError as error:
                writer.append("pair_failed", _failure_payload(pending, error))
                raise
            except KeyboardInterrupt as error:
                writer.append(
                    "run_interrupted", {"stage": "pair_execution", "pair_id": pending.pair_id}
                )
                raise RunInterrupted() from error
            writer.append("pair_completed", pair_payload(result))
            continue
        if state.finalists is None:
            _emit_observations(manifest, writer, state, cohort, "tuning")
            state = _refresh(manifest, writer.path.parent)
            tuning = [item for item in state.observations if item.phase == "tuning"]
            finalists = _selection(cohort, tuning, manifest)
            context = tuning[0].context
            writer.append(
                "finalists_selected",
                {
                    "finalist_ids": [item.candidate_id for item in finalists],
                    "tuning_estimates": {item.candidate_id: item.estimate.mean for item in tuning},
                    "objective_epoch_id": context.objective_epoch_id,
                    "corpus_id": context.task_prefix.corpus_id,
                    "prefix_id": context.task_prefix.prefix_id,
                    "prefix_task_ids": list(context.task_prefix.task_ids),
                    "search_effort": context.search_effort.max_iterations,
                    "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
                },
            )
            continue
        _emit_observations(manifest, writer, state, state.finalists, "validation")
        state = _refresh(manifest, writer.path.parent)
        claim, missing = production_claim(
            manifest.validation_prefix,
            manifest.production_validation_corpus,
            manifest.efforts["validation"],
            manifest.efforts["production"],
        )
        scientific_count = (
            sum(
                event.type
                in {
                    "proposal_created",
                    "proposal_accepted",
                    "proposal_rejected",
                    "cohort_accepted",
                    "pair_completed",
                    "observation_completed",
                    "finalists_selected",
                }
                for event in read_events(writer.path)
            )
            + 1
        )
        writer.append(
            "run_completed",
            {
                "manifest_fingerprint": manifest.fingerprint,
                "accepted_ids": [item.candidate_id for item in cohort],
                "finalist_ids": [item.candidate_id for item in state.finalists],
                "evidence_counts": {"events": scientific_count},
                "validation_claim": claim,
                "objective_epoch_id": manifest.epoch.epoch_id,
                "validation_prefix_id": manifest.validation_prefix.prefix_id,
                "validation_search_effort": manifest.efforts["validation"].max_iterations,
                "missing_production_axes": list(missing),
            },
        )
    return writer.path.parent / "report.json"


def _assert_compatibility(
    options: RunOptions, manifest: Manifest, spec: GameSpec, objective: ResolvedObjective
) -> None:
    rebuilt = build_manifest(
        str(manifest.raw["run_id"]),
        spec,
        objective,
        options.seed,
        options.task_seed,
        options.cohort_size,
        options.finalists,
        options.tuning_pairs,
        options.validation_pairs,
        options.production_validation_pairs,
        options.tuning_max_iterations,
        options.validation_max_iterations,
        options.production_max_iterations,
    )
    comparisons = (
        ("objective", objective.fingerprint, manifest.objective_fingerprint),
        ("panel", objective.panel.fingerprint, manifest.panel.fingerprint),
        ("tuning corpus", rebuilt.tuning_corpus.fingerprint, manifest.tuning_corpus.fingerprint),
        (
            "validation corpus",
            rebuilt.production_validation_corpus.fingerprint,
            manifest.production_validation_corpus.fingerprint,
        ),
        ("tuning prefix", rebuilt.tuning_prefix.prefix_id, manifest.tuning_prefix.prefix_id),
        (
            "validation prefix",
            rebuilt.validation_prefix.prefix_id,
            manifest.validation_prefix.prefix_id,
        ),
        ("epoch", rebuilt.epoch.fingerprint, manifest.epoch.fingerprint),
    )
    for label, current, frozen in comparisons:
        if current != frozen:
            raise ValueError(f"resume scientific input differs from manifest: {label}")
    if (
        spec.binary_sha256 != manifest.spec.binary_sha256
        or spec.raw_description != manifest.spec.raw_description
    ):
        raise ValueError("selected game binary or describe response differs from frozen manifest")


def run_foreground(
    options: RunOptions, target: Target | None = None, run_dir: Path | None = None
) -> Path:
    """Create or explicitly resume a strict foreground tuning run."""
    if run_dir is not None:
        options = replace(options, run_dir=run_dir)
    binary, directory, objective_path = _validate_options(options)
    target = GameBinaryTarget(binary) if target is None else target
    writer: EvidenceWriter | None = None
    try:
        spec = _spec_for(target, binary)
        space, default = _schema_default(spec, options.seed)
        objective = resolve_objective(objective_path, spec.kind, default)
        _validate_options_against_panel(options, objective)
        if options.resume:
            manifest = read_manifest(directory / "manifest.json")
            _assert_compatibility(options, manifest, spec, objective)
            state = _refresh(manifest, directory)
            if state.terminal_status == "configuration_failed":
                raise ValueError("terminal configuration failure cannot resume")
            if state.terminal_status == "complete":
                write_report(directory)
                return directory / "report.json"
            writer = EvidenceWriter.open(directory / "evidence.jsonl")
        else:
            _preflight_panel(target, spec, default, objective)
            manifest = build_manifest(
                directory.name,
                spec,
                objective,
                options.seed,
                options.task_seed,
                options.cohort_size,
                options.finalists,
                options.tuning_pairs,
                options.validation_pairs,
                options.production_validation_pairs,
                options.tuning_max_iterations,
                options.validation_max_iterations,
                options.production_max_iterations,
            )
            directory.mkdir(parents=True)
            write_manifest(directory / "manifest.json", manifest_json(manifest))
            writer = EvidenceWriter(directory / "evidence.jsonl")
            writer.append(
                "proposal_created",
                _proposal_payload(Proposal(0, "schema_default", "configspace-random-v1", default)),
            )
        _continue(manifest, writer, target, options.pair_timeout_seconds)
        write_report(directory)
        return directory / "report.json"
    except RunInterrupted:
        raise
    except KeyboardInterrupt:
        if writer is not None:
            writer.append("run_interrupted", {"stage": "proposal", "pair_id": None})
        raise
