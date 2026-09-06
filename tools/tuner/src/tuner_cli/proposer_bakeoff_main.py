"""CLI adapter for the strict foreground proposer bake-off."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .bakeoff_artifacts import read_spec
from .proposer_bakeoff import run_bakeoff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tuner-proposer-bakeoff")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_bakeoff(read_spec(args.spec), args.experiment_dir, args.resume)
    except (OSError, RuntimeError, ValueError) as error:
        logging.error("proposer bakeoff failed: %s", error)
        return 1
    return 0
