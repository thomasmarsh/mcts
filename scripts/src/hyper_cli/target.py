"""Target function — runs the Rust MCTS hyperparameter binary as a subprocess.

SMAC calls ``train(config, seed=...)`` and expects a ``float`` cost back
(lower is better).  This module maps the SMAC ``Configuration`` to the
kebab-case CLI flags the Rust binary expects.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

from ConfigSpace import Configuration

from .config import SearchConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Name translation
# ---------------------------------------------------------------------------

_SNAKE_TO_KEBAB = re.compile(r"_")


def _to_flag(key: str) -> str:
    """Convert ``q_init`` → ``--q-init`` etc."""
    return "--" + _SNAKE_TO_KEBAB.sub("-", key)


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
            f"Rust binary not found at {binary}. "
            f"Build it with: cargo build --bin hyper --release"
        )

    def train(config: Configuration, seed: int = 0) -> float:
        """Evaluate one hyperparameter configuration.

        Parameters
        ----------
        config:
            The configuration sampled by SMAC.  Only *active* parameters
            (those whose parent conditions are satisfied) are included.
        seed:
            Random seed forwarded by SMAC (from the scenario).

        Returns
        -------
        A ``float`` cost (lower = better).  The Rust binary prints
        ``cost=<float>`` on stdout.
        """
        # Build CLI args from the config's active params
        cli_args: list[str] = []
        for k, v in dict(config).items():
            cli_args += [_to_flag(k), str(v)]

        cmd = [str(binary), *cli_args, "--seed", str(seed)]

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

        # Parse the "cost=..." line
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("cost="):
                try:
                    return float(line.split("=", 1)[1])
                except ValueError:
                    logger.error("Failed to parse cost from line: %s", line)
                    return 1.0

        logger.error(
            "No 'cost=...' line found in output:\n%s\nstderr:\n%s",
            result.stdout,
            result.stderr,
        )
        return 1.0

    return train