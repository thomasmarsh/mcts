"""Command-line interface for the foreground game-binary tuner."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .codec import strict_json
from .constraints import Constraints, decode_constraints
from .domain import SearchEffort
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
    parser.add_argument(
        "--proposer-policy",
        choices=("smac_mixed", "random", "qmc", "irace_generational"),
        default="smac_mixed",
    )
    parser.add_argument("--tuning-pairs", type=int, default=4)
    parser.add_argument("--tuning-pair-budget", type=int, required=True, metavar="PAIRS")
    parser.add_argument("--validation-pair-budget", type=int, required=True, metavar="PAIRS")
    parser.add_argument("--diagnostic-pair-budget", type=int, default=0, metavar="PAIRS")
    parser.add_argument("--production-validation-pairs", type=int, required=True)
    for phase in ("tuning", "validation", "production"):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{phase}-max-iterations", type=int)
        group.add_argument(f"--{phase}-max-time-ms", type=int)
    parser.add_argument("--pair-timeout-seconds", type=int, default=600)
    parser.add_argument("--evaluator-workers", type=int, default=1, metavar="N")
    parser.add_argument("--shadow-practical-margin", type=float, default=0.0)
    parser.add_argument("--shadow-elimination-threshold", type=float, default=0.05)
    parser.add_argument(
        "--shadow-policy",
        choices=("paired_bootstrap", "successive_halving"),
        default="paired_bootstrap",
    )
    parser.add_argument("--shadow-halving-spare-margin", type=float, default=0.0)
    parser.add_argument("--active-elimination-audit-probability", type=float)
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        metavar="JSON",
        help=(
            "unified run-scoped tuning-space constraint as JSON -- an array of "
            '{"when"?: {...}, "set": {...}} entries or the bare '
            "{name: {fix|range|choices}} map; repeatable."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="continue a frozen run")
    extend = parser.add_argument_group(
        "budget extension", "raise a frozen run's pair budgets; valid only with --resume"
    )
    extend.add_argument("--extend-tuning-pairs", type=int, default=0, metavar="N")
    extend.add_argument("--extend-validation-pairs", type=int, default=0, metavar="N")
    extend.add_argument("--extend-diagnostic-pairs", type=int, default=0, metavar="N")
    extend.add_argument("--extend-reason", default="", metavar="TEXT")
    extend.add_argument("--extend-requested-at", default="", metavar="ISO8601")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _constraints(args: argparse.Namespace) -> Constraints:
    """The unified run-scoped ``constraints`` from every ``--constraint`` flag.

    Each ``--constraint`` carries the full wire form -- an array of
    ``{"when"?: {...}, "set": {...}}`` entries or the bare
    ``{name: {fix|range|choices}}`` map as sugar for one un-predicated entry.
    """
    constraints: Constraints = ()
    for text in args.constraint:
        value = strict_json(text, "--constraint value")
        if isinstance(value, dict) and ("set" in value or "when" in value):
            value = [value]
        constraints = (*constraints, *decode_constraints(value))
    return constraints


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
        diagnostic_pair_budget=args.diagnostic_pair_budget,
        production_validation_pairs=args.production_validation_pairs,
        tuning_effort=effort("tuning", 1_000),
        validation_effort=effort("validation", 10_000),
        production_effort=effort("production", 10_000),
        pair_timeout_seconds=args.pair_timeout_seconds,
        evaluator_workers=args.evaluator_workers,
        shadow_practical_margin=args.shadow_practical_margin,
        shadow_elimination_threshold=args.shadow_elimination_threshold,
        shadow_policy=args.shadow_policy,
        shadow_halving_spare_margin=args.shadow_halving_spare_margin,
        active_elimination_audit_probability=args.active_elimination_audit_probability,
        constraints=_constraints(args),
        proposer_policy=args.proposer_policy,
        resume=args.resume,
        extend_tuning_pairs=args.extend_tuning_pairs,
        extend_validation_pairs=args.extend_validation_pairs,
        extend_diagnostic_pairs=args.extend_diagnostic_pairs,
        extend_reason=args.extend_reason,
        extend_requested_at=args.extend_requested_at,
    )


def _validate_objective_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tuner validate-objective")
    parser.add_argument("--game-binary", type=Path, required=True, metavar="PATH")
    parser.add_argument("--objective-file", type=Path, required=True, metavar="PATH")
    args = parser.parse_args(argv)
    from .validate import validate_objective_file

    print(json.dumps(validate_objective_file(args.game_binary, args.objective_file)))
    return 0


def _preflight_main(argv: list[str]) -> int:
    """`tuner preflight <same args as a run>` — report, as one JSON line,
    every launch problem knowable before the run dir is created or a game is
    played. Reuses the `run` parser verbatim so the two can't drift."""
    from .preflight import preflight_launch

    args = build_parser().parse_args(argv)
    print(json.dumps(preflight_launch(_options(args))))
    return 0


def _plan_main(argv: list[str]) -> int:
    """`tuner plan <same args as a run>` — emit, as one JSON line, the fully
    resolved shape of the run these options would start (opponent panel,
    tuning space, efforts, budgets, game_config, epoch) plus the preflight
    `ok`/`errors`. Creates no run dir and plays no game."""
    from .plan import plan_launch

    args = build_parser().parse_args(argv)
    print(json.dumps(plan_launch(_options(args))))
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # A subcommand-free argv is the foreground `run` (kept as the default so
    # `mcts_bench::tuner_launch` needs no change).
    if raw and raw[0] == "validate-objective":
        return _validate_objective_main(raw[1:])
    if raw and raw[0] == "preflight":
        return _preflight_main(raw[1:])
    if raw and raw[0] == "plan":
        return _plan_main(raw[1:])
    args = build_parser().parse_args(raw)
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
