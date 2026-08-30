"""Foreground create, replay, and continuation flow for generic tuner runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .artifacts import Manifest, build_manifest, configspace_version, manifest_json, read_manifest
from .domain import Candidate, PairTask, Proposal
from .evidence import EvidenceWriter, pair_payload, read_events, write_manifest
from .identity import candidate_from_config, canonical_json, sha256_file
from .replay import _observation, _selection, expected_pairs, fold_events, observation_payload
from .report import write_report
from .schema import GameSpec, decode_game_spec
from .space import build_space, default_values, random_values
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
    resume: bool = False


class RunInterrupted(KeyboardInterrupt):
    """Marks an interruption whose operational event was already persisted."""


def _validate_options(options: RunOptions) -> tuple[Path, Path]:
    values = asdict(options)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for key, value in values.items()
        if key not in {"game_binary", "run_dir", "resume"}
    ):
        raise ValueError("all numeric arguments must be positive integers")
    if options.cohort_size < 2 or options.finalists > options.cohort_size:
        raise ValueError("cohort size must be at least 2 and finalists cannot exceed it")
    run_dir = options.run_dir.expanduser().resolve()
    binary = options.game_binary.expanduser().resolve()
    if options.resume:
        if not run_dir.is_dir():
            raise ValueError(f"resume run directory does not exist: {run_dir}")
    elif run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}; use --resume to continue it")
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise ValueError(f"game binary is missing, not a regular executable file: {binary}")
    return binary, run_dir


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


def _assert_compatibility(options: RunOptions, manifest: Manifest, spec: GameSpec) -> None:
    scientific = {
        "seed": manifest.seed,
        "cohort_size": manifest.cohort_size,
        "finalists": manifest.finalists,
        "tuning_pairs": len(manifest.tuning.cases),
        "validation_pairs": len(manifest.validation.cases),
        "tuning_max_iterations": manifest.budgets["tuning"],
        "validation_max_iterations": manifest.budgets["validation"],
        "production_max_iterations": manifest.budgets["production"],
    }
    for name, frozen in scientific.items():
        if getattr(options, name) != frozen:
            raise ValueError(f"resume scientific option differs from manifest: {name}")
    if manifest.raw["proposer"]["configspace_version"] != configspace_version():  # type: ignore[index]
        raise ValueError("ConfigSpace version differs from frozen manifest")
    if (
        spec.binary_sha256 != manifest.spec.binary_sha256
        or spec.raw_description != manifest.spec.raw_description
        or spec.engine_fingerprint != manifest.spec.engine_fingerprint
        or spec.schema_fingerprint != manifest.spec.schema_fingerprint
        or spec.default_game_config != manifest.spec.default_game_config
    ):
        raise ValueError("selected game binary or describe response differs from frozen manifest")
    space = build_space(spec.tuning, manifest.seed)
    if candidate_from_config(default_values(space)) != manifest.opponent:
        raise ValueError("ConfigSpace schema default differs from frozen manifest")


def _restore_space(manifest: Manifest):
    if manifest.raw["proposer"]["configspace_version"] != configspace_version():  # type: ignore[index]
        raise ValueError("ConfigSpace version differs from frozen manifest")
    space = build_space(manifest.spec.tuning, manifest.seed)
    default = candidate_from_config(default_values(space))
    if default != manifest.opponent:
        raise ValueError("ConfigSpace default differs from manifest")
    return space


def _replay_draws(manifest: Manifest, state, space) -> None:  # type: ignore[no-untyped-def]
    for proposal in state.proposals:
        if proposal.proposal_index == 0:
            if proposal.candidate != manifest.opponent:
                raise ValueError("default proposal differs from manifest")
        else:
            regenerated = candidate_from_config(random_values(space))
            if regenerated != proposal.candidate:
                raise ValueError("recorded random proposal differs from frozen sampler")


def _configuration_failure(writer: EvidenceWriter, message: str) -> None:
    writer.append("run_failed", {"kind": "configuration", "message": message})
    raise RuntimeError(message)


def _finish_cohort(
    manifest: Manifest, state, writer: EvidenceWriter, target: Target, space
) -> tuple[Candidate, ...]:  # type: ignore[no-untyped-def]
    _replay_draws(manifest, state, space)
    proposals = list(state.proposals)
    dispositions = dict(state.dispositions)
    seen = {proposal.candidate.fingerprint for proposal in proposals}
    for proposal in proposals:
        if proposal.proposal_index not in dispositions:
            result = target.validate(
                [proposal.candidate], manifest.opponent, manifest.spec.default_game_config
            )
            if proposal.proposal_index == 0 and not result.valid:
                _configuration_failure(writer, "schema default failed semantic validation")
            if result.valid:
                writer.append("proposal_accepted", _disposition_payload(proposal))
            else:
                writer.append(
                    "proposal_rejected",
                    {
                        **_disposition_payload(proposal),
                        "reason": "semantic_validation",
                        "errors": [asdict(error) for error in result.errors],
                    },
                )
            state = _refresh(manifest, writer.path.parent)
            dispositions = dict(state.dispositions)
    cap = max(100, manifest.cohort_size * 100)
    draws = len(proposals) - 1
    while (
        len([item for item in dispositions.values() if item == "accepted"]) < manifest.cohort_size
        and draws < cap
    ):
        draws += 1
        index = len(proposals)
        candidate = candidate_from_config(random_values(space))
        proposal = Proposal(index, "configspace_random", "configspace-random-v1", candidate)
        writer.append("proposal_created", _proposal_payload(proposal))
        proposals.append(proposal)
        if candidate.fingerprint in seen:
            writer.append(
                "proposal_rejected",
                {**_disposition_payload(proposal), "reason": "duplicate", "errors": []},
            )
            state = _refresh(manifest, writer.path.parent)
            dispositions = dict(state.dispositions)
            continue
        seen.add(candidate.fingerprint)
        result = target.validate([candidate], manifest.opponent, manifest.spec.default_game_config)
        if result.valid:
            writer.append("proposal_accepted", _disposition_payload(proposal))
        else:
            writer.append(
                "proposal_rejected",
                {
                    **_disposition_payload(proposal),
                    "reason": "semantic_validation",
                    "errors": [asdict(error) for error in result.errors],
                },
            )
        state = _refresh(manifest, writer.path.parent)
        dispositions = dict(state.dispositions)
    accepted = tuple(
        proposal.candidate
        for proposal in proposals
        if dispositions.get(proposal.proposal_index) == "accepted"
    )
    if len(accepted) != manifest.cohort_size:
        _configuration_failure(
            writer, f"proposal draw cap reached: accepted {len(accepted)}/{manifest.cohort_size}"
        )
    final = target.validate(accepted, manifest.opponent, manifest.spec.default_game_config)
    if not final.valid:
        _configuration_failure(writer, "final cohort validation failed")
    writer.append(
        "cohort_accepted",
        {
            "candidate_ids": [candidate.candidate_id for candidate in accepted],
            "validation_response_fingerprint": __import__("hashlib")
            .sha256(canonical_json(asdict(final)).encode())
            .hexdigest(),
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
            continue
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


def _continue(manifest: Manifest, writer: EvidenceWriter, target: Target, timeout: int) -> Path:
    state = _refresh(manifest, writer.path.parent)
    if state.terminal_status == "configuration_failed":
        raise ValueError("terminal configuration failure cannot resume")
    if state.terminal_status == "complete":
        return writer.path.parent / "report.json"
    if state.cohort is None:
        space = _restore_space(manifest)
        cohort = _finish_cohort(manifest, state, writer, target, space)
        state = _refresh(manifest, writer.path.parent)
    else:
        cohort = state.cohort
    assert cohort is not None
    while True:
        state = _refresh(manifest, writer.path.parent)
        if state.terminal_status == "complete":
            break
        if state.finalists is None:
            plan = list(expected_pairs(manifest, cohort))
        else:
            plan = list(expected_pairs(manifest, cohort, state.finalists))
        done = {item.task.pair_id for item in state.completed_pairs}
        pending = next((item for item in plan if item.pair_id not in done), None)
        if pending is not None:
            candidate = next(item for item in cohort if item.candidate_id == pending.candidate_id)
            if state.finalists is not None:
                candidate = next(
                    item for item in state.finalists if item.candidate_id == pending.candidate_id
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
                    manifest.opponent,
                    manifest.spec.default_game_config,
                    timeout,
                )
            except PairExecutionError as error:
                writer.append("pair_failed", _failure_payload(pending, error))
                raise
            except KeyboardInterrupt as error:
                writer.append(
                    "run_interrupted",
                    {"stage": "pair_execution", "pair_id": pending.pair_id},
                )
                raise RunInterrupted() from error
            writer.append("pair_completed", pair_payload(result))
            continue
        if state.finalists is None:
            tuning_observations = []
            for candidate in cohort:
                if not any(
                    item.phase == "tuning" and item.candidate_id == candidate.candidate_id
                    for item in state.observations
                ):
                    pairs = [
                        pair
                        for pair in state.completed_pairs
                        if pair.task.candidate_id == candidate.candidate_id
                        and pair.task.task_case.phase == "tuning"
                    ]
                    writer.append(
                        "observation_completed",
                        observation_payload(_observation(candidate, "tuning", manifest, pairs)),
                    )
                    state = _refresh(manifest, writer.path.parent)
            tuning_observations = [item for item in state.observations if item.phase == "tuning"]
            finalists = _selection(cohort, tuning_observations, manifest)
            writer.append(
                "finalists_selected",
                {
                    "finalist_ids": [item.candidate_id for item in finalists],
                    "tuning_estimates": {
                        item.candidate_id: item.estimate.mean for item in tuning_observations
                    },
                    "source_block": manifest.tuning.block_id,
                    "budget": manifest.budgets["tuning"],
                    "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
                },
            )
            continue
        for candidate in state.finalists:
            if not any(
                item.phase == "validation" and item.candidate_id == candidate.candidate_id
                for item in state.observations
            ):
                pairs = [
                    pair
                    for pair in state.completed_pairs
                    if pair.task.candidate_id == candidate.candidate_id
                    and pair.task.task_case.phase == "validation"
                ]
                writer.append(
                    "observation_completed",
                    observation_payload(_observation(candidate, "validation", manifest, pairs)),
                )
                state = _refresh(manifest, writer.path.parent)
        state = _refresh(manifest, writer.path.parent)
        claim = (
            "production"
            if manifest.budgets["validation"] == manifest.budgets["production"]
            else "mechanics_smoke"
        )
        writer.append(
            "run_completed",
            {
                "manifest_fingerprint": manifest.fingerprint,
                "accepted_ids": [item.candidate_id for item in cohort],
                "finalist_ids": [item.candidate_id for item in state.finalists],
                "evidence_counts": {
                    "events": sum(
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
                },
                "validation_claim": claim,
            },
        )
    return writer.path.parent / "report.json"


def run_foreground(
    options: RunOptions, target: Target | None = None, run_dir: Path | None = None
) -> Path:
    """Create or explicitly resume a strict foreground tuning run."""
    if run_dir is not None:
        options = replace(options, run_dir=run_dir)
    binary, directory = _validate_options(options)
    if target is None:
        target = GameBinaryTarget(binary)
    writer: EvidenceWriter | None = None
    try:
        spec = _spec_for(target, binary)
        if options.resume:
            manifest = read_manifest(directory / "manifest.json")
            state = _refresh(manifest, directory)
            _assert_compatibility(options, manifest, spec)
            if state.terminal_status == "configuration_failed":
                raise ValueError("terminal configuration failure cannot resume")
            writer = EvidenceWriter.open(directory / "evidence.jsonl")
            if state.terminal_status == "complete":
                write_report(directory)
                return directory / "report.json"
            print(f"continuing frozen run: {directory}")
        else:
            try:
                space = build_space(spec.tuning, options.seed)
                default_values(space)
            except Exception as error:
                raise ValueError(
                    f"invalid tuning metadata: ConfigSpace construction/default failed: {error}"
                ) from error
            manifest = build_manifest(
                directory.name,
                spec,
                options.seed,
                options.cohort_size,
                options.finalists,
                options.tuning_pairs,
                options.validation_pairs,
                options.tuning_max_iterations,
                options.validation_max_iterations,
                options.production_max_iterations,
            )
            directory.mkdir(parents=True)
            write_manifest(directory / "manifest.json", manifest_json(manifest))
            writer = EvidenceWriter(directory / "evidence.jsonl")
            default = Proposal(0, "schema_default", "configspace-random-v1", manifest.opponent)
            writer.append("proposal_created", _proposal_payload(default))
        assert writer is not None
        _continue(manifest, writer, target, options.pair_timeout_seconds)
        write_report(directory)
        state = _refresh(manifest, directory)
        print(f"run directory: {directory}")
        claim = (
            "production"
            if manifest.budgets["validation"] == manifest.budgets["production"]
            else "mechanics_smoke"
        )
        print(f"validation claim: {claim}")
        print(f"accepted/finalists: {len(state.cohort or ())}/{len(state.finalists or ())}")
        return directory / "report.json"
    except RunInterrupted:
        raise
    except KeyboardInterrupt:
        if writer is not None:
            writer.append("run_interrupted", {"stage": "proposal", "pair_id": None})
        raise
