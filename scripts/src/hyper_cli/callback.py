"""Custom SMAC callbacks for hyperparameter optimisation."""

from __future__ import annotations

import json
import logging
import subprocess

from smac import Callback
from smac.main.smbo import SMBO
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.utils.configspace import get_config_hash

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Git SHA resolution (mirrors build_info.rs on the Rust side)
# ---------------------------------------------------------------------------


def _resolve_git_sha() -> str:
    """Shell out to ``git rev-parse HEAD`` and return the full SHA.

    Returns ``"unknown"`` if git is unavailable or the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# IncumbentTracker (existing)
# ---------------------------------------------------------------------------


class IncumbentTracker(Callback):
    """Print each new incumbent configuration and its cost as it's found."""

    def __init__(self) -> None:
        self._last_hash: str | None = None

    def on_next_configurations_end(self, config_selector, config) -> None:
        """Log every configuration the config selector proposes."""
        logger.info("Proposed config: %s", dict(config))

    def on_tell_end(
        self,
        smbo: SMBO,
        info: TrialInfo,
        value: TrialValue,
    ) -> bool | None:
        """Check whether the incumbent changed and print it if so."""
        incumbent = smbo.intensifier.get_incumbent()
        if incumbent is None:
            return None

        h = get_config_hash(incumbent)
        if h != self._last_hash:
            cost = smbo.runhistory.get_cost(incumbent)
            logger.info(
                "New incumbent  hash=%s  cost=%s  config=%s",
                h,
                cost,
                dict(incumbent),
            )
            self._last_hash = h
        return None


# ---------------------------------------------------------------------------
# TrialTracker (JSONL line per completed trial)
# ---------------------------------------------------------------------------


class TrialTracker(Callback):
    """Emit one structured JSONL line per completed trial to stdout.

    Each line is a ``{"type": "trial", "trial_id": ..., "config": ...,
    "seed": ..., "cost": ...}`` record matching the Rust
    ``LogRecord::Trial`` variant so the ingest loop can read it directly.

    Accepts an optional ``git_sha`` for attribution.  If omitted, it
    resolves the current HEAD via ``git rev-parse`` on first call.
    """

    def __init__(self, git_sha: str | None = None) -> None:
        self._counter = 0
        self._git_sha = git_sha

    def on_tell_end(
        self,
        smbo: SMBO,
        info: TrialInfo,
        value: TrialValue,
    ) -> bool | None:
        """Emit a JSONL trial record for every completed trial."""
        # Defer git SHA resolution to first trial so it works even if
        # the callback is constructed before cwd is correct.
        if self._git_sha is None:
            self._git_sha = _resolve_git_sha()

        self._counter += 1

        trial_record = {
            "type": "trial",
            "trial_id": self._counter,
            "config": dict(info.config),
            "seed": info.seed,
            "cost": value.cost,
        }
        print(json.dumps(trial_record), flush=True)
        return None