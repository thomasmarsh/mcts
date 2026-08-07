#!/usr/bin/env python3
"""
hyper-cli — command-line entry point.

Run::

    uv run --project scripts/ hyper-cli [--config path] [--override key=value] ...

Or with the installed entry-point::

    hyper-cli --help
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from smac import Scenario
from smac.facade import HyperparameterOptimizationFacade

from .callback import IncumbentTracker
from .config import SearchConfig
from .space import build_space
from .target import make_target

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

logger = logging.getLogger("hyper_cli")


def _parse_overrides(raw: list[str]) -> dict[str, str]:
    """Parse ``key=value`` override strings into a flat dict."""
    overrides: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k] = v
    return overrides


def _apply_overrides(cfg: SearchConfig, overrides: dict[str, str]) -> None:
    """Mutate *cfg* in-place from dotted overrides like ``optimizer.n_trials=500``."""
    import ast
    for key, raw_val in overrides.items():
        parts = key.split(".")
        obj: Any = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        # Try to parse as Python literal first (int, float, bool, None)
        try:
            val = ast.literal_eval(raw_val)
        except (ValueError, SyntaxError):
            val = raw_val  # fallback to string
        setattr(obj, parts[-1], val)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hyper-cli",
        description="SMAC3 hyperparameter optimisation for MCTS (Rust binary).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: packaged config/default.yaml)",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value (e.g. optimizer.n_trials=500)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- load config -------------------------------------------------------
    cfg = SearchConfig.load(args.config) if args.config else SearchConfig.defaults()
    logger.info("Config: %s", cfg._source or "defaults")

    overrides = _parse_overrides(args.override)
    _apply_overrides(cfg, overrides)
    if overrides:
        logger.info("Applied overrides: %s", overrides)

    # -- build configuration space -----------------------------------------
    cs = build_space(cfg)
    logger.info("ConfigSpace: %d parameters, %d conditions", len(cs), len(cs.conditions))

    # -- target function ---------------------------------------------------
    train = make_target(cfg)

    # -- SMAC scenario -----------------------------------------------------
    n_workers = cfg.optimizer.n_workers
    if n_workers is None:
        n_workers = max(1, os.cpu_count() // 2)

    scenario = Scenario(
        cs,
        deterministic=cfg.optimizer.deterministic,
        n_trials=cfg.optimizer.n_trials,
        n_workers=n_workers,
        seed=cfg.optimizer.seed,
    )

    # -- run optimisation --------------------------------------------------
    smac = HyperparameterOptimizationFacade(
        scenario,
        train,
        callbacks=[IncumbentTracker()],
        logging_level=logging.INFO if not args.verbose else logging.DEBUG,
        overwrite=True,
    )

    incumbent = smac.optimize()

    # -- report ------------------------------------------------------------
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"\n{'=' * 60}")
    print(f"Best config:  {dict(incumbent)}")
    best_cost = smac.validate(incumbent)
    print(f"Best cost:    {best_cost:.6f}")
    print(f"Default cost: {default_cost:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()