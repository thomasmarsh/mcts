"""Command-line interface for the foreground game-binary tuner."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .domain import SearchEffort
from .family_exclusions import normalize_family_exclusions
from .run import RunOptions, run_foreground


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tuner", description="Foreground game strategy tuner.")
    parser.add_argument("--game-binary", type=Path, required=True, metavar="PATH")
    parser.add_argument("--objective-file", type=Path, required=True, metavar="PATH")
    parser.add_argument("--run-dir", type=Path, required=True, metavar="PATH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-seed", type=int, required=True)
    parser.add_argument("--cohort-size", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--bootstrap-candidates", type=int, default=3)
    parser.add_argument("--random-reserve-candidates", type=int, default=2)
    parser.add_argument("--tuning-pairs", type=int, default=4)
    parser.add_argument("--tuning-pair-budget", type=int, required=True, metavar="PAIRS")
    parser.add_argument("--validation-pair-budget", type=int, required=True, metavar="PAIRS")
    parser.add_argument("--production-validation-pairs", type=int, required=True)
    for phase in ("tuning", "validation", "production"):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{phase}-max-iterations", type=int)
        group.add_argument(f"--{phase}-max-time-ms", type=int)
    parser.add_argument("--pair-timeout-seconds", type=int, default=600)
    parser.add_argument("--evaluator-workers", type=int, default=1, metavar="N")
    parser.add_argument("--shadow-practical-margin", type=float, default=0.0)
    parser.add_argument("--shadow-elimination-threshold", type=float, default=0.05)
    parser.add_argument("--exclude-family", action="append", default=[], metavar="FAMILY")
    parser.add_argument("--resume", action="store_true", help="continue a frozen version-4 run")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _options(args: argparse.Namespace) -> RunOptions:
    def effort(phase: str, default: int) -> SearchEffort:
        iterations = getattr(args, f"{phase}_max_iterations")
        time_ms = getattr(args, f"{phase}_max_time_ms")
        return (
            SearchEffort("time_ms", time_ms)
            if time_ms is not None
            else SearchEffort("iterations", iterations if iterations is not None else default)
        )

    return RunOptions(
        game_binary=args.game_binary,
        run_dir=args.run_dir,
        objective_file=args.objective_file,
        seed=args.seed,
        task_seed=args.task_seed,
        cohort_size=args.cohort_size,
        finalists=args.finalists,
        bootstrap_candidates=args.bootstrap_candidates,
        random_reserve_candidates=args.random_reserve_candidates,
        tuning_pairs=args.tuning_pairs,
        tuning_pair_budget=args.tuning_pair_budget,
        validation_pair_budget=args.validation_pair_budget,
        production_validation_pairs=args.production_validation_pairs,
        tuning_effort=effort("tuning", 1_000),
        validation_effort=effort("validation", 10_000),
        production_effort=effort("production", 10_000),
        pair_timeout_seconds=args.pair_timeout_seconds,
        evaluator_workers=args.evaluator_workers,
        shadow_practical_margin=args.shadow_practical_margin,
        shadow_elimination_threshold=args.shadow_elimination_threshold,
        excluded_families=normalize_family_exclusions(args.exclude_family),
        resume=args.resume,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    try:
        run_foreground(_options(args))
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        logging.getLogger("tuner_cli").error("tuner failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
