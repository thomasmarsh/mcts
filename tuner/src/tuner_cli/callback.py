"""Structured JSONL records for the Optuna optimization loop."""

from __future__ import annotations

import subprocess

from .config import json_dumps


def _resolve_git_sha() -> str:
    """Return the current git SHA, or ``"unknown"`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _cost(mu: float, sigma: float) -> float:
    """Encode a higher-is-better OpenSkill estimate for the legacy cost wire field."""
    return -(mu - 3 * sigma)


def emit_trial_record(
    trial_id: int,
    config: dict,
    seed: int,
    mu: float,
    sigma: float,
    games: list[dict],
    git_sha: str | None = None,
) -> None:
    """Print one completed trial in the established ``mcts-bench`` JSONL shape."""
    print(
        json_dumps(
            {
                "type": "trial",
                "trial_id": trial_id,
                "config": config,
                "seed": seed,
                "cost": _cost(mu, sigma),
                "extra": {
                    "mu": mu,
                    "sigma": sigma,
                    "opponents": games,
                    "git_sha": git_sha if git_sha is not None else _resolve_git_sha(),
                },
            }
        ),
        flush=True,
    )


def emit_incumbent_record(config: dict, mu: float, sigma: float) -> None:
    """Print the current best trial in the established incumbent JSONL shape."""
    print(
        json_dumps(
            {
                "type": "incumbent",
                "config": config,
                "cost": _cost(mu, sigma),
                "extra": {"mu": mu, "sigma": sigma},
            }
        ),
        flush=True,
    )
