#!/usr/bin/env python3
"""Command-line entry point for MCTS hyperparameter optimization."""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from optuna.trial import TrialState

from .config import SearchConfig
from .coordinator import run_optimization

logger = logging.getLogger("tuner_cli")


def _parse_overrides(raw: list[str]) -> dict[str, str]:
    """Parse ``key=value`` override strings into a flat dict."""
    overrides: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def _parse_baseline_configs(raw: list[str]) -> dict[str, dict]:
    """Parse repeated ``--baseline-config id=json`` flags."""
    parsed: dict[str, dict] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--baseline-config must be id=json, got {item!r}")
        anchor_id, raw_json = item.split("=", 1)
        parsed[anchor_id] = json.loads(raw_json)
    return parsed


def _apply_overrides(cfg: SearchConfig, overrides: dict[str, str]) -> None:
    """Mutate *cfg* in-place from dotted CLI overrides."""
    for key, raw_value in overrides.items():
        obj: Any = cfg
        parts = key.split(".")
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            attr = getattr(obj, parts[-1])
        except AttributeError:
            logger.warning(
                "Ignoring unknown override '%s=%s' — no such field on %s",
                key,
                raw_value,
                type(obj).__name__,
            )
            continue
        if isinstance(attr, Path):
            value = Path(raw_value)
        else:
            try:
                value = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                value = raw_value
        setattr(obj, parts[-1], value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tuner", description="Optuna hyperparameter optimisation for MCTS."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--override", action="append", default=[], help="Override key=value"
    )
    parser.add_argument(
        "--baseline-config", action="append", default=[], metavar="ID=JSON"
    )
    parser.add_argument("--game-config", type=str, default=None, metavar="JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Deprecated alias for --optimizer-id.",
    )
    parser.add_argument(
        "--optimizer-id",
        type=str,
        default=None,
        help="Persistent optimizer id; reusing it resumes its study and opponent pool.",
    )
    parser.add_argument(
        "--bench-run-id",
        type=str,
        default=None,
        help="Physical benchmark run id used to join this attempt's traces.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Logical tuning session id; defaults to --run-id.",
    )
    parser.add_argument(
        "--attempt-id",
        type=str,
        default=None,
        help="Physical process attempt id; defaults to a fresh opaque id.",
    )
    parser.add_argument(
        "--lifecycle-path",
        type=Path,
        default=None,
        help="Append-only lifecycle JSONL artifact path.",
    )
    parser.add_argument(
        "--game-kind",
        type=str,
        default=None,
        help="Stable game kind recorded in the session manifest.",
    )
    parser.add_argument("--trace-path", type=str, default=None, metavar="PATH")
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure the command's established human-readable log format."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_cli_config(args: argparse.Namespace) -> SearchConfig:
    """Apply command-line configuration inputs before optimization starts."""
    cfg = SearchConfig.load(args.config) if args.config else SearchConfig.defaults()
    _apply_overrides(cfg, _parse_overrides(args.override))
    cfg.target.baseline_configs.update(_parse_baseline_configs(args.baseline_config))
    if args.game_config:
        cfg.target.game_config = json.loads(args.game_config)
    cfg.validate()
    return cfg


def _print_run_summary(run_id: str, study, pool) -> None:
    """Print the established completion summary for interactive CLI users."""
    if not study.get_trials(states=(TrialState.COMPLETE,)):
        print("\nNo completed trials.")
        return
    default = next(anchor for anchor in pool.anchors if anchor.id == "default")
    print(f"\n{'=' * 60}")
    print(f"Run id:       {run_id}")
    print(f"Best config:  {study.best_trial.user_attrs['config']}")
    print(f"Best score:   {study.best_value:.6f}")
    print(f"Default:      mu={default.mu:.6f} sigma={default.sigma:.6f}")
    print(f"{'=' * 60}")


def main() -> None:
    """Parse CLI inputs, run the coordinator, and render its completion summary."""
    args = build_parser().parse_args()
    _configure_logging(args.verbose)
    cfg = _load_cli_config(args)
    optimizer_id = args.optimizer_id or args.run_id or f"run-{uuid4().hex[:12]}"
    study, pool = run_optimization(
        cfg,
        optimizer_id=optimizer_id,
        bench_run_id=args.bench_run_id,
        git_sha=args.git_sha,
        trace_path=args.trace_path,
        session_id=args.session_id,
        attempt_id=args.attempt_id,
        lifecycle_path=args.lifecycle_path,
        game_kind=args.game_kind,
    )
    _print_run_summary(optimizer_id, study, pool)


if __name__ == "__main__":
    main()
