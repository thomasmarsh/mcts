"""Custom SMAC callbacks for hyperparameter optimisation."""

from __future__ import annotations

import logging
import subprocess

from smac import Callback
from smac.main.smbo import SMBO
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.utils.configspace import get_config_hash

from .config import json_dumps

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
    """Emit a JSONL record each time SMAC3's tracked incumbent changes.

    Sourced from ``smbo.intensifier.get_incumbent()`` rather than
    reconstructed as ``MIN(cost)`` over completed trials -- when a run's
    `Scenario` uses multiple baseline instances, per-trial costs aren't
    directly comparable across instances, and only the intensifier itself
    aggregates them into a single ranking. Each line is a
    ``{"type": "incumbent", "config": ..., "cost": ...}`` record matching the
    Rust ``LogRecord::Incumbent`` variant, which the ingest loop upserts into
    the `incumbents` table (latest wins, one row per run) rather than
    appending -- only the current incumbent is durably useful, not its
    history. ``config`` is already in the exact shape `tune eval
    --baseline-config` expects, so a later run can reuse it directly.
    """

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
        """Emit an incumbent JSONL record when the tracked incumbent changes."""
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
            print(
                json_dumps({"type": "incumbent", "config": dict(incumbent), "cost": cost}),
                flush=True,
            )
            self._last_hash = h
        return None


# ---------------------------------------------------------------------------
# TrialTracker (JSONL line per completed trial)
# ---------------------------------------------------------------------------


class TrialTracker(Callback):
    """Emit one structured JSONL line per completed trial to stdout.

    Each line is a ``{"type": "trial", "trial_id": ..., "config": ...,
    "seed": ..., "cost": ..., "extra": {"instance": ..., "status": ...}}``
    record matching the Rust ``LogRecord::Trial`` variant so the ingest loop
    can read it directly. ``extra.instance`` is only present when the run's
    `Scenario` was given a baseline-instances list -- it's the id of which
    baseline this particular trial's cost was measured against, needed to
    make sense of per-trial cost once multiple instances are in play (see
    `target.py`'s `instance` parameter). ``extra.status`` is only present
    when `target.py`'s ``train()`` reported the trial didn't actually
    produce a real result (``"timeout"``/``"crashed"``) -- its ``cost`` is
    still the worst-case float SMAC needs, but callers doing post-hoc
    analysis on the cost distribution (e.g. checking for sigmoid
    saturation) should filter these out rather than treat them as a real
    100%-loss outcome.

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
        extra: dict = {}
        if info.instance is not None:
            extra["instance"] = info.instance
        status = value.additional_info.get("status")
        if status is not None:
            extra["status"] = status
        if extra:
            trial_record["extra"] = extra
        print(json_dumps(trial_record), flush=True)
        return None
