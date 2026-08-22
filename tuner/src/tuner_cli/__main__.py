#!/usr/bin/env python3
"""Optuna command-line entry point for MCTS hyperparameter optimisation."""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import optuna

from .callback import _resolve_git_sha, emit_incumbent_record, emit_trial_record
from .config import SearchConfig
from .matchmaking import play_trial
from .pool import Anchor, OpponentPool
from .space_optuna import suggest_config
from .target import preflight_check

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
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if isinstance(getattr(obj, parts[-1]), Path):
            value = Path(raw_value)
        else:
            try:
                value = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                value = raw_value
        setattr(obj, parts[-1], value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tuner", description="Optuna hyperparameter optimisation for MCTS.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[], help="Override key=value")
    parser.add_argument("--baseline-config", action="append", default=[], metavar="ID=JSON",
                        help="Seed an additional frozen raw-config pool anchor. Repeatable.")
    parser.add_argument("--game-config", type=str, default=None, metavar="JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None,
                        help="Persistent run id; reusing it resumes its Optuna study and opponent pool.")
    parser.add_argument("--trace-path", type=str, default=None, metavar="PATH")
    return parser


def run_optimization(
    cfg: SearchConfig,
    *,
    run_id: str,
    git_sha: str | None = None,
    trace_path: str | None = None,
) -> tuple[optuna.Study, OpponentPool]:
    """Run the unfinished portion of a persistent Optuna study sequentially."""
    binary = cfg.resolve_binary()
    parameters, conditions, _advertised_baselines = SearchConfig.parameters_from_binary(binary)
    cfg.parameters = parameters
    cfg.conditions = conditions

    output_dir = Path("optuna_output") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "pool.json"
    pool = OpponentPool.load(pool_path) if pool_path.exists() else OpponentPool.bootstrap(cfg)
    for anchor_id, config in cfg.target.baseline_configs.items():
        if not any(anchor.id == anchor_id for anchor in pool.anchors):
            pool.anchors.append(Anchor(anchor_id, dict(config), mu=25.0, sigma=0.5))
    pool.save(pool_path)

    storage = f"sqlite:///{(output_dir / 'study.db').resolve()}"
    study = optuna.create_study(
        direction="maximize",
        study_name=run_id,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.optimizer.seed),
    )

    preflight_check(cfg, pool.closest(25.0).config, pool.closest(0.0).config)
    resolved_sha = git_sha or _resolve_git_sha()
    remaining = max(0, cfg.optimizer.n_trials - len(study.trials))
    for _ in range(remaining):
        trial = study.ask()
        config = suggest_config(trial, cfg)
        trial.set_user_attr("config", config)
        seed = cfg.optimizer.seed + trial.number
        mu, sigma, games = play_trial(cfg, binary, config, pool, seed_base=seed, trace_path=trace_path)
        study.tell(trial, mu - 3 * sigma)
        emit_trial_record(trial.number, config, seed, mu, sigma, games, resolved_sha)
        if pool.maybe_insert(config, mu, sigma) is not None:
            pool.save(pool_path)
        if study.best_trial.number == trial.number:
            emit_incumbent_record(config, mu, sigma)

    return study, pool


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = SearchConfig.load(args.config) if args.config else SearchConfig.defaults()
    _apply_overrides(cfg, _parse_overrides(args.override))
    cfg.target.baseline_configs.update(_parse_baseline_configs(args.baseline_config))
    if args.game_config:
        cfg.target.game_config = json.loads(args.game_config)
    run_id = args.run_id or f"run-{uuid4().hex[:12]}"
    study, pool = run_optimization(cfg, run_id=run_id, git_sha=args.git_sha, trace_path=args.trace_path)
    default = next(anchor for anchor in pool.anchors if anchor.id == "default")
    print(f"\n{'=' * 60}")
    print(f"Run id:       {run_id}")
    print(f"Best config:  {study.best_trial.user_attrs['config']}")
    print(f"Best score:   {study.best_value:.6f}")
    print(f"Default:      mu={default.mu:.6f} sigma={default.sigma:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
