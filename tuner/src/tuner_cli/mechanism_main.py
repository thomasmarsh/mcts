"""CLI for the shadow-race mechanism sweep: `calibrate` and `sweep`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .artifacts import read_manifest
from .mechanism_calibration import DEFAULT_BINS, load_calibration, write_calibration
from .mechanism_sim import as_halving
from .mechanism_sweep import (
    DEFAULT_BOUNDARY_GAPS,
    DEFAULT_SPREAD_SCALES,
    DEFAULT_TRIALS,
    format_summary,
    run_sweep,
)


def _calibrate(args: argparse.Namespace) -> int:
    calibration = write_calibration(
        [Path(item) for item in args.run], Path(args.out), bins=args.bins
    )
    logging.info(
        "wrote %s from %d runs; %d bins populated, deviation_correlation=%.3f",
        args.out,
        len(args.run),
        len(calibration.pair_utility_bins),
        calibration.deviation_correlation,
    )
    return 0


def _sweep(args: argparse.Namespace) -> int:
    calibration = load_calibration(Path(args.calibration))
    manifest = as_halving(read_manifest(Path(args.manifest)))
    sweep = run_sweep(
        calibration,
        manifest,
        boundary_gaps=tuple(args.boundary_gaps) if args.boundary_gaps else DEFAULT_BOUNDARY_GAPS,
        spread_scales=(tuple(args.spread_scales) if args.spread_scales else DEFAULT_SPREAD_SCALES),
        trials=args.trials,
        seed=args.seed,
        paired_resamples=args.paired_resamples,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(sweep.to_json(), indent=2, sort_keys=True) + "\n")
    print(format_summary(sweep))
    return 0 if sweep.gate.passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tuner-mechanism")
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="extract a calibration from recorded runs")
    calibrate.add_argument("--run", action="append", required=True, metavar="RUN_DIR")
    calibrate.add_argument("--out", required=True, metavar="PATH")
    calibrate.add_argument("--bins", type=int, default=DEFAULT_BINS)
    calibrate.set_defaults(func=_calibrate)

    sweep = sub.add_parser("sweep", help="run the mechanism grid sweep and gate")
    sweep.add_argument("--calibration", required=True, metavar="PATH")
    sweep.add_argument("--manifest", required=True, metavar="PATH")
    sweep.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--paired-resamples", type=int, default=512, dest="paired_resamples")
    sweep.add_argument("--boundary-gaps", type=float, nargs="*", dest="boundary_gaps")
    sweep.add_argument("--spread-scales", type=float, nargs="*", dest="spread_scales")
    sweep.add_argument("--out", metavar="PATH")
    sweep.set_defaults(func=_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
        assert isinstance(result, int)
        return result
    except (OSError, RuntimeError, ValueError) as error:
        logging.error("tuner-mechanism failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
