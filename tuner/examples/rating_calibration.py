#!/usr/bin/env python3
"""Print production rating updates for scripted, seat-swapped evaluation pairs.

For example:
    uv run --project tuner python examples/rating_calibration.py \
      --pair first:win,second:loss --min-pairs 3 --max-pairs 5 --k 2.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tuner_cli.config import SearchConfig
from tuner_cli.evaluation import OpponentSnapshot, Rating, TrialEvaluationState
from tuner_cli.rating_calibration import (
    calibrate,
    parse_scripted_pair,
    parse_sigma_stop,
    render_calibration,
    resolve_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="FIRST,SECOND",
        help="Explicit seats: first:win,second:loss (repeat for each pair)",
    )
    parser.add_argument("--config", type=Path, help="Tuner YAML with the resolved policy")
    parser.add_argument("--min-pairs", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--sigma-stop",
        type=parse_sigma_stop,
        metavar="NUMBER|none",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--k", "--conservative-k", dest="conservative_k", type=float)
    parser.add_argument("--candidate-mu", type=float)
    parser.add_argument("--candidate-sigma", type=float)
    parser.add_argument("--opponent-mu", type=float, default=25.0)
    parser.add_argument("--opponent-sigma", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        pairs = [parse_scripted_pair(value) for value in args.pair]
        config = SearchConfig.load(args.config) if args.config else SearchConfig.defaults()
        resource, rating = resolve_policy(
            config,
            min_pairs=args.min_pairs,
            max_pairs=args.max_pairs,
            sigma_stop=getattr(args, "sigma_stop", ...),
            conservative_k=args.conservative_k,
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    default_rating = TrialEvaluationState(resource, rating).rating
    initial_rating = Rating(
        default_rating.mu if args.candidate_mu is None else args.candidate_mu,
        default_rating.sigma if args.candidate_sigma is None else args.candidate_sigma,
    )
    opponent = OpponentSnapshot(
        "scripted-opponent",
        {},
        args.opponent_mu,
        args.opponent_sigma,
    )
    print(render_calibration(calibrate(pairs, resource, rating, opponent, initial_rating)))


if __name__ == "__main__":
    main()
