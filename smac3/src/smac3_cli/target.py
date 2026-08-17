"""Target function — runs the game binary's ``tune eval`` subcommand.

SMAC calls ``train(config, instance=..., seed=...)`` and expects a ``float``
cost back (lower is better).  This module maps the SMAC ``Configuration`` to
the ``--config`` JSON object the binary's ``tune eval`` subcommand expects,
and ``instance`` (when the scenario has one) to a ``--baseline`` flag.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from ConfigSpace import Configuration

from .config import SearchConfig, json_dumps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Floor baselines
# ---------------------------------------------------------------------------

# Baseline-only families `mcts-tune`'s own `make_candidate` (mcts-tune/src/
# lib.rs) builds directly from a raw params object -- not named presets any
# game's `preset_cfg`/`PRESET_CONFIGS` knows about. A game's `tune_eval` only
# recognizes a `--baseline <id>` as one of *its own* named presets (Druid:
# easy/medium/strong/master); passing "flat_mc"/"random" that way fails with
# "unknown baseline" on every trial, which this harness's own error handling
# below scores as `cost = 1.0` -- an apparent 100%-loss streak that's
# actually every trial silently erroring, not a real result. Routing them as
# `--baseline-config` instead (same as a `cfg.target.baseline_configs` entry)
# reaches `mcts_tune::build_search`, which does know these two families.
FLOOR_BASELINES: dict[str, dict] = {
    "flat_mc": {"family": "flat_mc", "q_init": "Infinity"},
    "random": {"family": "random", "q_init": "Infinity"},
}

# ---------------------------------------------------------------------------
# Target function factory
# ---------------------------------------------------------------------------


def make_target(cfg: SearchConfig, *, trace_path: str | None = None):
    """Return a callable suitable as SMAC's ``target_function``.

    The returned closure captures the binary path and evaluation settings so
    that the optimizer only sees ``(config, seed)``. ``trace_path``, when
    given, is forwarded verbatim as ``tune eval --trace-path <path>`` on
    every trial -- the game binary's own `MoveTracer` opens it in append
    mode, so all trials in the run accumulate into the same file.
    """
    binary: Path = cfg.resolve_binary()
    if not binary.is_file():
        raise FileNotFoundError(
            f"Game binary not found at {binary}. Build it with: "
            f"cargo build --release -p game-traffic-lights"
        )

    def train(config: Configuration, instance: str | None = None, seed: int = 0) -> float:
        """Evaluate one hyperparameter configuration.

        Parameters
        ----------
        config:
            The configuration sampled by SMAC.  Only *active* parameters
            (those whose parent conditions are satisfied) are included.
        instance:
            The baseline instance id to evaluate against (from
            ``Scenario(instances=...)``). An id backed by a raw discovered
            config (member of ``cfg.target.baseline_configs``) or one of the
            built-in ``FLOOR_BASELINES`` is forwarded as ``--baseline-config``
            with its params JSON; anything else is assumed to be a named
            preset (member of ``cfg.target.baselines``) and forwarded as
            ``--baseline``. ``None`` when the scenario wasn't given an
            instance list.
        seed:
            Random seed forwarded by SMAC (from the scenario).

        Returns
        -------
        A ``float`` cost (lower = better), parsed from the ``{"cost": ...}``
        JSON line the binary's ``tune eval`` subcommand prints on stdout.
        """
        cmd = [
            str(binary),
            "tune",
            "eval",
            "--config",
            json_dumps(dict(config)),
            "--rounds",
            str(cfg.target.rounds),
            "--seed",
            str(seed),
        ]
        if instance is not None:
            if instance in cfg.target.baseline_configs:
                cmd += ["--baseline-config", json.dumps(cfg.target.baseline_configs[instance])]
            elif instance in FLOOR_BASELINES:
                cmd += ["--baseline-config", json.dumps(FLOOR_BASELINES[instance])]
            else:
                cmd += ["--baseline", instance]

        if cfg.target.game_config is not None:
            cmd += ["--game-config", json.dumps(cfg.target.game_config)]

        if cfg.target.max_iterations is not None:
            cmd += ["--max-iterations", str(cfg.target.max_iterations)]

        if trace_path is not None:
            cmd += ["--trace-path", trace_path]

        logger.debug("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes per trial
            )
        except subprocess.TimeoutExpired:
            logger.warning("Trial timed out after 600 s (seed=%s)", seed)
            return 1.0  # worst possible cost

        if result.returncode != 0:
            logger.error(
                "Binary exited with code %s:\nstdout:\n%s\nstderr:\n%s",
                result.returncode,
                result.stdout,
                result.stderr,
            )
            return 1.0

        # Parse the trailing JSON line: {"cost": ..., "wins": ..., ...}
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "cost" in payload:
                return float(payload["cost"])

        logger.error(
            "No JSON 'cost' line found in output:\n%s\nstderr:\n%s",
            result.stdout,
            result.stderr,
        )
        return 1.0

    return train
