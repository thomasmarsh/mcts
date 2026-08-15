#!/usr/bin/env python3
"""
smac3 — command-line entry point.

Run::

    uv run --project smac3/ smac3 [--config path] [--override key=value] ...

Or with the installed entry-point::

    smac3 --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from smac import Scenario
from smac.facade import HyperparameterOptimizationFacade

from .callback import IncumbentTracker, TrialTracker
from .config import SearchConfig
from .resume import load_prior_runhistory
from .space import build_space
from .target import make_target

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

logger = logging.getLogger("smac3_cli")


def _parse_overrides(raw: list[str]) -> dict[str, str]:
    """Parse ``key=value`` override strings into a flat dict."""
    overrides: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k] = v
    return overrides


def _parse_baseline_configs(raw: list[str]) -> dict[str, dict]:
    """Parse ``id=json`` strings from repeated ``--baseline-config`` flags.

    Not routed through ``_parse_overrides``/``_apply_overrides`` -- that
    mechanism mutates a single scalar dotted field per flag, and this is a
    dict keyed by ids only known at launch time (an automated-ladder rung's
    own discovered baseline ids), not a fixed field name.
    """
    parsed: dict[str, dict] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--baseline-config must be id=json, got {item!r}")
        instance_id, raw_json = item.split("=", 1)
        parsed[instance_id] = json.loads(raw_json)
    return parsed


def _apply_overrides(cfg: SearchConfig, overrides: dict[str, str]) -> None:
    """Mutate *cfg* in-place from dotted overrides like ``optimizer.n_trials=500``."""
    import ast
    for key, raw_val in overrides.items():
        parts = key.split(".")
        obj: Any = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        # A `Path`-typed field (e.g. `target.binary`) must stay a `Path` --
        # `resolve_binary()` calls `.is_absolute()` on it, which a plain
        # `str` doesn't have.
        if isinstance(getattr(obj, parts[-1]), Path):
            val = Path(raw_val)
        else:
            # Try to parse as Python literal first (int, float, bool, None)
            try:
                val = ast.literal_eval(raw_val)
            except (ValueError, SyntaxError):
                val = raw_val  # fallback to string
        setattr(obj, parts[-1], val)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smac3",
        description="SMAC3 hyperparameter optimisation for MCTS (game binary).",
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
        "--baseline-config",
        action="append",
        default=[],
        metavar="ID=JSON",
        help=(
            "Add an extra baseline instance backed by a raw discovered "
            "config rather than a named preset (e.g. an automated-ladder "
            "rung's own incumbent). Repeatable. The id becomes a member of "
            "Scenario(instances=...); train() forwards it to the game "
            "binary as `tune eval --baseline-config <json>` instead of "
            "`--baseline <id>`."
        ),
    )
    p.add_argument(
        "--game-config",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Game-setup config (e.g. Druid's board size) pinning every "
            "trial in this run to a non-default game config instead of "
            "the game binary's own default. Forwarded verbatim as `tune "
            "eval --game-config <json>` on every trial."
        ),
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    p.add_argument(
        "--git-sha",
        type=str,
        default=None,
        help="Git SHA for attribution (auto-detected if omitted)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Pin Scenario.name to this id (default: SMAC3 auto-hashes a name "
            "from the scenario contents). Makes this run's output directory "
            "(smac3_output/<run-id>/<seed>/) discoverable for a later --resume."
        ),
    )
    p.add_argument(
        "--trace-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Append per-ply move-trace JSONL lines to this file (opened in "
            "append mode by each trial's game-binary subprocess, so every "
            "trial in the run accumulates into the same file). Forwarded "
            "verbatim as `tune eval --trace-path <path>` on every trial. "
            "Omit to disable move tracing."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help=(
            "Seed this run's runhistory from a prior run's saved state "
            "(that prior run's own --run-id), so already-evaluated configs "
            "aren't re-evaluated. See resume.py for why this is done "
            "manually rather than via SMAC3's own continue path."
        ),
    )
    return p


def build_optimizer(
    cfg: SearchConfig,
    *,
    run_id: str | None = None,
    resume: str | None = None,
    git_sha: str | None = None,
    verbose: bool = False,
    trace_path: str | None = None,
) -> HyperparameterOptimizationFacade:
    """Build a ready-to-`.optimize()` SMAC3 facade from *cfg*.

    Factored out of `main()` so tests can drive the resume path (build,
    inspect/optimize, build again with `resume=`) without going through
    `argparse`/a subprocess -- see `tests/test_resume.py`.
    """
    # -- search space, sourced from the binary itself -----------------------
    # The game binary is the single source of truth for the search space
    # (`tune describe`), not hand-maintained YAML -- see
    # `SearchConfig.parameters_from_binary`'s docstring.
    binary = cfg.resolve_binary()
    parameters, conditions, advertised_baselines = SearchConfig.parameters_from_binary(binary)
    cfg.parameters = parameters
    cfg.conditions = conditions
    # `tune describe` advertises available named presets; it does not choose
    # an opponent for the run. Requiring the caller to choose avoids silently
    # tuning against an unintended baseline when launch wiring is incomplete.
    if not cfg.target.baselines:
        raise ValueError(
            "target.baselines must be explicitly provided "
            f"(advertised named presets: {advertised_baselines})"
        )
    logger.info(
        "Search space from %s: %d parameters, %d conditions, baselines=%s",
        binary,
        len(cfg.parameters),
        len(cfg.conditions),
        cfg.target.baselines,
    )

    # -- build configuration space -----------------------------------------
    cs = build_space(cfg)
    logger.info("ConfigSpace: %d parameters, %d conditions", len(cs), len(cs.conditions))

    # -- target function ---------------------------------------------------
    train = make_target(cfg, trace_path=trace_path)

    # -- SMAC scenario -----------------------------------------------------
    n_workers = cfg.optimizer.n_workers
    if n_workers is None:
        n_workers = max(1, os.cpu_count() // 2)

    # `instances` lets SMAC evaluate each trial config against multiple
    # baseline opponents and aggregate cost across them -- without it, a
    # config that reaches 100% win rate against the one fixed baseline
    # floors `cost` at 0.0 and every top candidate ties, with no way to
    # rank them further. Most games report a single-entry `baselines` list
    # (an unchanged, single-instance scenario); druid today is the one game
    # with a genuine second, harder preset ("master") in that list.
    # `baseline_configs` adds further instances backed by a raw discovered
    # config rather than a named preset -- `target.py`'s `train()` is what
    # actually distinguishes the two kinds of instance id when dispatching
    # to the game binary.
    instances = [*cfg.target.baselines, *cfg.target.baseline_configs]
    scenario = Scenario(
        cs,
        name=run_id,
        deterministic=cfg.optimizer.deterministic,
        n_trials=cfg.optimizer.n_trials,
        n_workers=n_workers,
        seed=cfg.optimizer.seed,
        instances=instances,
        termination_cost_threshold=cfg.optimizer.termination_cost_threshold,
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        train,
        callbacks=[IncumbentTracker(), TrialTracker(git_sha=git_sha)],
        logging_level=logging.INFO if not verbose else logging.DEBUG,
        # `False` so that an accidental relaunch into the same (name-pinned)
        # output directory with an *identical* scenario auto-continues
        # rather than silently erasing prior runhistory. A relaunch with a
        # genuinely different scenario (e.g. a bumped `n_trials`) is not
        # routed through this gate at all -- see `--resume` below.
        overwrite=False,
    )

    if resume:
        prior = load_prior_runhistory(resume, cs)
        logger.info(
            "Resuming from run %s: merging %d prior trial(s) into runhistory",
            resume,
            len(prior),
        )
        smac.runhistory.update(prior)

    return smac


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

    baseline_configs = _parse_baseline_configs(args.baseline_config)
    cfg.target.baseline_configs.update(baseline_configs)
    if baseline_configs:
        logger.info("Extra baseline instances: %s", list(baseline_configs))

    if args.game_config:
        cfg.target.game_config = json.loads(args.game_config)
        logger.info("Game config: %s", cfg.target.game_config)

    smac = build_optimizer(
        cfg,
        run_id=args.run_id,
        resume=args.resume,
        git_sha=args.git_sha,
        verbose=args.verbose,
        trace_path=args.trace_path,
    )
    cs = smac.scenario.configspace

    # -- run optimisation --------------------------------------------------
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
