"""Setup, resume compatibility, and report publication for foreground tuning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .artifacts import Manifest, build_manifest, manifest_json, read_manifest
from .continuation import continue_run
from .domain import Candidate
from .evidence import EvidenceWriter, read_events, write_manifest
from .identity import candidate_from_config, sha256_file
from .objective import ResolvedObjective, resolve_objective
from .proposer import ModelProposer
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
    production_validation_pairs: int = 8
    tuning_max_iterations: int = 1_000
    validation_max_iterations: int = 10_000
    production_max_iterations: int = 10_000
    pair_timeout_seconds: int = 600
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
    binary, directory, objective_path = validate_options(options)
    active_target = GameBinaryTarget(binary) if target is None else target
    spec = game_spec(active_target, binary)
    default = schema_default(spec, options.seed)
    objective = resolve_objective(objective_path, spec.kind, default)
    validate_objective_options(options, objective)
    manifest, writer = open_run(options, directory, spec, objective, active_target)
    if fold_events(manifest, read_events(writer.path)).terminal_status == "complete":
        write_report(directory)
        return directory / "report.json"
    model = model_proposer or SmacProposer(build_space(spec.tuning, options.seed))
    continue_run(
        manifest, writer, active_target, default, spec, model, options.pair_timeout_seconds
    )
    write_report(directory)
    return directory / "report.json"


def validate_options(options: RunOptions) -> tuple[Path, Path, Path]:
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
        options.tuning_max_iterations,
        options.validation_max_iterations,
        options.production_max_iterations,
        options.pair_timeout_seconds,
    )
    if any(isinstance(item, bool) or item <= 0 for item in numeric):
        raise ValueError("all numeric arguments must be positive integers")
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


def game_spec(target: Target, binary: Path) -> GameSpec:
    return decode_game_spec(target.describe(), binary, sha256_file(binary))


def schema_default(spec: GameSpec, seed: int) -> Candidate:
    return candidate_from_config(default_values(build_space(spec.tuning, seed)))


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
    if (
        max(options.tuning_max_iterations, options.validation_max_iterations)
        > options.production_max_iterations
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
    preflight_default(target, spec, objective, options.seed)
    manifest = manifest_for(options, directory, spec, objective)
    directory.mkdir(parents=True)
    write_manifest(directory / "manifest.json", manifest_json(manifest))
    return manifest, EvidenceWriter(directory / "evidence.jsonl")


def preflight_default(
    target: Target, spec: GameSpec, objective: ResolvedObjective, seed: int
) -> None:
    default = schema_default(spec, seed)
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
        options.tuning_max_iterations,
        options.validation_max_iterations,
        options.production_max_iterations,
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
