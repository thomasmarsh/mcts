"""CLI adapter for the strict foreground elimination bake-off."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .elimination_bakeoff import read_elimination_spec, run_elimination_bakeoff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tuner-elimination-bakeoff")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_elimination_bakeoff(read_elimination_spec(args.spec), args.experiment_dir, args.resume)
    except (OSError, RuntimeError, ValueError) as error:
        logging.error("elimination bakeoff failed: %s", error)
        return 1
    return 0
