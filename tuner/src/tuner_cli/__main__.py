"""Command-line interface for the foreground game-binary tuner."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .run import RunOptions, run_foreground


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tuner", description="Foreground game strategy tuner.")
    parser.add_argument("--game-binary", type=Path, required=True, metavar="PATH")
    parser.add_argument("--run-dir", type=Path, required=True, metavar="PATH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cohort-size", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--tuning-pairs", type=int, default=4)
    parser.add_argument("--validation-pairs", type=int, default=8)
    parser.add_argument("--tuning-max-iterations", type=int, default=1_000)
    parser.add_argument("--validation-max-iterations", type=int, default=10_000)
    parser.add_argument("--production-max-iterations", type=int, default=10_000)
    parser.add_argument("--pair-timeout-seconds", type=int, default=600)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        game_binary=args.game_binary,
        run_dir=args.run_dir,
        seed=args.seed,
        cohort_size=args.cohort_size,
        finalists=args.finalists,
        tuning_pairs=args.tuning_pairs,
        validation_pairs=args.validation_pairs,
        tuning_max_iterations=args.tuning_max_iterations,
        validation_max_iterations=args.validation_max_iterations,
        production_max_iterations=args.production_max_iterations,
        pair_timeout_seconds=args.pair_timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    try:
        run_foreground(_options(args))
    except (OSError, RuntimeError, ValueError) as error:
        logging.getLogger("tuner_cli").error("tuner failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
