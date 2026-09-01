"""Setup, resume compatibility, and report publication for foreground tuning."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .artifacts import Manifest, build_manifest, manifest_json, read_manifest
from .continuation import continue_run
from .domain import Candidate, SearchEffort
from .effort import exceeds_same_kind
from .evidence import EvidenceWriter, read_events, write_manifest
from .executor import BoundedPairExecutor, PairExecutor, SequentialPairExecutor
from .family_exclusions import normalize_family_exclusions, validate_family_exclusions
from .identity import candidate_from_config, sha256_file
from .objective import ResolvedObjective, resolve_objective
from .proposer import ModelProposer, ProposerPolicy
from .replay import fold_events
from .report import write_report
from .schema import GameSpec, decode_game_spec
from .smac_proposer import SmacProposer
from .space import build_space, default_values
from .target import GameBinaryTarget, Target
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
    bootstrap_candidates: int = 3
    random_reserve_candidates: int = 2
    tuning_pairs: int = 4
    tuning_pair_budget: int = 32
    validation_pair_budget: int = 24
    diagnostic_pair_budget: int = 0
    production_validation_pairs: int = 8
    tuning_effort: SearchEffort = SearchEffort("iterations", 1_000)
    validation_effort: SearchEffort = SearchEffort("iterations", 10_000)
    production_effort: SearchEffort = SearchEffort("iterations", 10_000)
    pair_timeout_seconds: int = 600
    evaluator_workers: int = 1
    shadow_practical_margin: float = 0.0
    shadow_elimination_threshold: float = 0.05
    shadow_policy: Literal["paired_bootstrap", "successive_halving"] = "paired_bootstrap"
    shadow_halving_spare_margin: float = 0.0
    active_elimination_audit_probability: float | None = None
    excluded_families: tuple[str, ...] = ()
    proposer_policy: ProposerPolicy = "smac_mixed"
    resume: bool = False


def run_foreground(
    options: RunOptions,
    target: Target | None = None,
    run_dir: Path | None = None,
    model_proposer: ModelProposer | None = None,
) -> Path:
    """Create or explicitly resume one strict budgeted tuning run."""
    if run_dir is not None:
        options = replace(options, run_dir=run_dir)
    options = replace(
        options, excluded_families=normalize_family_exclusions(options.excluded_families)
    )
    binary, directory, objective_path = validate_options(options)
    executor = pair_executor(options.evaluator_workers)
    active_target = GameBinaryTarget(binary) if target is None else target
    spec = game_spec(active_target, binary)
    validate_family_exclusions(spec.tuning, options.excluded_families)
    objective_default = schema_default(spec, options.seed)
    proposal_default = schema_default(spec, options.seed, options.excluded_families)
    objective = resolve_objective(objective_path, spec.kind, objective_default)
    validate_objective_options(options, objective)
    manifest, writer = open_run(options, directory, spec, objective, active_target)
    if fold_events(manifest, read_events(writer.path)).terminal_status == "complete":
        write_report(directory)
        return directory / "report.json"
    model = model_proposer or proposer_for(options.proposer_policy, spec, manifest)
    continue_run(
        manifest,
        writer,
        active_target,
        proposal_default,
        spec,
        model,
        options.pair_timeout_seconds,
        executor,
    )
    write_report(directory)
    return directory / "report.json"


def proposer_for(policy: ProposerPolicy, spec: GameSpec, manifest: Manifest) -> ModelProposer:
    if policy == "smac_mixed":
        return SmacProposer(
            build_space(spec.tuning, options_seed(manifest), manifest.excluded_families)
        )
    if policy == "qmc":
        from .proposer import derived_seed
        from .qmc_proposer import QmcProposer

        return QmcProposer(
            spec.tuning, derived_seed(manifest.seed, "qmc"), manifest.excluded_families
        )
    if policy == "irace_generational":
        from .irace_proposer import IraceProposer

        return IraceProposer(spec.tuning, manifest.excluded_families)
    if policy == "random":
        # Random policy never asks an adapter.
        return SmacProposer(build_space(spec.tuning, manifest.seed, manifest.excluded_families))
    raise ValueError(f"unknown proposer policy {policy!r}")


def options_seed(manifest: Manifest) -> int:
    return manifest.seed


def validate_options(options: RunOptions) -> tuple[Path, Path, Path]:
    raw_efforts: tuple[object, object, object] = (
        object.__getattribute__(options, "tuning_effort"),
        object.__getattribute__(options, "validation_effort"),
        object.__getattribute__(options, "production_effort"),
    )
    if not all(isinstance(effort, SearchEffort) for effort in raw_efforts):
        raise ValueError("all phase efforts must be resolved SearchEffort values")
    numeric = (
        options.seed,
        options.task_seed,
        options.cohort_size,
        options.finalists,
        options.bootstrap_candidates,
        options.random_reserve_candidates,
        options.tuning_pairs,
        options.tuning_pair_budget,
        options.validation_pair_budget,
        options.production_validation_pairs,
        options.pair_timeout_seconds,
    )
    if any(isinstance(item, bool) or item <= 0 for item in numeric):
        raise ValueError("all numeric arguments must be positive integers")
    if isinstance(options.diagnostic_pair_budget, bool) or options.diagnostic_pair_budget < 0:
        raise ValueError("diagnostic pair budget must be a non-Boolean integer at least zero")
    _validate_shadow_margins(options)
    if options.active_elimination_audit_probability is not None:
        if (
            options.shadow_policy == "successive_halving"
            and options.shadow_halving_spare_margin <= 0.0
        ):
            raise ValueError(
                "active elimination with successive halving requires "
                "--shadow-halving-spare-margin > 0 (the gate-approved spare-near-tie policy)"
            )
        value = options.active_elimination_audit_probability
        if (
            isinstance(value, bool)
            or not isinstance(value, float)
            or not math.isfinite(value)
            or not 0.0 < value < 1.0
        ):
            raise ValueError(
                "active elimination audit probability must be a finite number in (0.0, 1.0)"
            )
    validate_evaluator_workers(options.evaluator_workers, os.cpu_count())
    post_bootstrap = options.cohort_size - options.bootstrap_candidates
    if options.bootstrap_candidates < 2 or post_bootstrap < 2:
        raise ValueError("bootstrap and post-bootstrap stages each need at least two candidates")
    if not 1 <= options.random_reserve_candidates < post_bootstrap:
        raise ValueError(
            "random reserve must be positive and smaller than the post-bootstrap stage"
        )
    if options.finalists >= options.cohort_size:
        raise ValueError("finalists must be smaller than cohort size")
    if options.objective_file is None:
        raise ValueError("--objective-file is required")
    objective = options.objective_file.expanduser().resolve()
    if not objective.is_file():
        raise ValueError(f"objective file does not exist: {objective}")
    binary, directory = (
        options.game_binary.expanduser().resolve(),
        options.run_dir.expanduser().resolve(),
    )
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise ValueError(f"game binary is missing, not a regular executable file: {binary}")
    if options.resume and not directory.is_dir():
        raise ValueError(f"resume run directory does not exist: {directory}")
    if not options.resume and directory.exists():
        raise ValueError(f"run directory already exists: {directory}; use --resume to continue it")
    return binary, directory, objective


def _validate_shadow_margins(options: RunOptions) -> None:
    for value, label, inclusive in (
        (options.shadow_practical_margin, "shadow practical margin", True),
        (options.shadow_elimination_threshold, "shadow elimination threshold", False),
        (options.shadow_halving_spare_margin, "shadow halving spare margin", True),
    ):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
        if not (0.0 <= value <= 1.0 if inclusive else 0.0 < value < 0.5):
            raise ValueError(f"{label} must be in {'[0.0, 1.0]' if inclusive else '(0.0, 0.5)'}")
    if options.shadow_halving_spare_margin > 0.0 and options.shadow_policy != "successive_halving":
        raise ValueError("shadow halving spare margin requires the successive_halving policy")


def validate_evaluator_workers(workers: int, cpu_count: int | None) -> int:
    """Validate the fixed one-search-thread-per-evaluator CPU product."""
    available = 1 if cpu_count is None else cpu_count
    if isinstance(workers, bool) or workers <= 0:
        raise ValueError("evaluator workers must be a positive integer")
    if workers > available:
        raise ValueError(
            f"evaluator workers ({workers}) exceed available logical CPUs ({available})"
        )
    return available


def pair_executor(workers: int) -> PairExecutor:
    return SequentialPairExecutor() if workers == 1 else BoundedPairExecutor(workers)


def game_spec(target: Target, binary: Path) -> GameSpec:
    return decode_game_spec(target.describe(), binary, sha256_file(binary))


def schema_default(spec: GameSpec, seed: int, excluded_families: tuple[str, ...] = ()) -> Candidate:
    return candidate_from_config(default_values(build_space(spec.tuning, seed, excluded_families)))


def validate_objective_options(options: RunOptions, objective: ResolvedObjective) -> None:
    for count, label in (
        (options.tuning_pairs, "tuning pairs"),
        (options.production_validation_pairs, "production validation pairs"),
    ):
        validate_cycle_endpoint(objective.panel, count, label)
    if options.validation_pair_budget % options.finalists:
        raise ValueError("validation pair budget must divide finalists")
    validation_pairs = options.validation_pair_budget // options.finalists
    validate_cycle_endpoint(objective.panel, validation_pairs, "validation pairs")
    if validation_pairs > options.production_validation_pairs:
        raise ValueError("validation pairs cannot exceed production validation pairs")
    if options.tuning_pair_budget < options.cohort_size * options.tuning_pairs:
        raise ValueError("tuning pair budget cannot fund initial cohort")
    if any(
        exceeds_same_kind(observed, options.production_effort)
        for observed in (options.tuning_effort, options.validation_effort)
    ):
        raise ValueError("observed search effort cannot exceed production effort")


def open_run(
    options: RunOptions,
    directory: Path,
    spec: GameSpec,
    objective: ResolvedObjective,
    target: Target,
) -> tuple[Manifest, EvidenceWriter]:
    if options.resume:
        manifest = read_manifest(directory / "manifest.json")
        assert_compatible(options, manifest, spec, objective)
        return manifest, EvidenceWriter.open(directory / "evidence.jsonl")
    preflight_default(target, spec, objective, options.seed, options.excluded_families)
    manifest = manifest_for(options, directory, spec, objective)
    directory.mkdir(parents=True)
    write_manifest(directory / "manifest.json", manifest_json(manifest))
    return manifest, EvidenceWriter(directory / "evidence.jsonl")


def preflight_default(
    target: Target,
    spec: GameSpec,
    objective: ResolvedObjective,
    seed: int,
    excluded_families: tuple[str, ...],
) -> None:
    default = schema_default(spec, seed, excluded_families)
    failures = [
        opponent.opponent_id
        for opponent in objective.panel.opponents
        if not target.validate((default,), opponent, spec.default_game_config).valid
    ]
    if failures:
        raise ValueError(f"schema default failed panel preflight: {', '.join(failures)}")


def manifest_for(
    options: RunOptions, directory: Path, spec: GameSpec, objective: ResolvedObjective
) -> Manifest:
    return build_manifest(
        directory.name,
        spec,
        objective,
        options.seed,
        options.task_seed,
        options.cohort_size,
        options.finalists,
        options.bootstrap_candidates,
        options.random_reserve_candidates,
        options.tuning_pairs,
        options.tuning_pair_budget,
        options.validation_pair_budget,
        options.production_validation_pairs,
        options.tuning_effort,
        options.validation_effort,
        options.production_effort,
        options.shadow_practical_margin,
        options.shadow_elimination_threshold,
        options.shadow_policy,
        options.shadow_halving_spare_margin,
        options.excluded_families,
        options.active_elimination_audit_probability,
        options.diagnostic_pair_budget,
        options.proposer_policy,
    )


def assert_compatible(
    options: RunOptions, manifest: Manifest, spec: GameSpec, objective: ResolvedObjective
) -> None:
    if (
        manifest_for(options, Path(manifest.run_id), spec, objective).fingerprint
        != manifest.fingerprint
    ):
        raise ValueError("resume scientific input differs from manifest")
    if (
        spec.binary_sha256 != manifest.spec.binary_sha256
        or spec.raw_description != manifest.spec.raw_description
    ):
        raise ValueError("selected game binary or describe response differs from frozen manifest")
