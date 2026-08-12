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

from .config import SearchConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target function factory
# ---------------------------------------------------------------------------


def make_target(cfg: SearchConfig):
    """Return a callable suitable as SMAC's ``target_function``.

    The returned closure captures the binary path and evaluation settings so
    that the optimizer only sees ``(config, seed)``.
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
            ``Scenario(instances=...)``). A named preset (member of
            ``cfg.target.baselines``) is forwarded as ``--baseline``; an id
            backed by a raw discovered config instead (member of
            ``cfg.target.baseline_configs``) is forwarded as
            ``--baseline-config`` with its raw config JSON instead. ``None``
            when the scenario wasn't given an instance list.
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
            json.dumps(dict(config)),
            "--rounds",
            str(cfg.target.rounds),
            "--seed",
            str(seed),
        ]
        if instance is not None:
            if instance in cfg.target.baseline_configs:
                cmd += ["--baseline-config", json.dumps(cfg.target.baseline_configs[instance])]
            else:
                cmd += ["--baseline", instance]

        if cfg.target.game_config is not None:
            cmd += ["--game-config", json.dumps(cfg.target.game_config)]

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
