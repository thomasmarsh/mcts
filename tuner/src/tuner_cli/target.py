"""Target function -- runs the game binary's ``tune eval`` subcommand.

``play_game`` runs one candidate-vs-opponent match (``cfg.target.rounds``
game pairs) and returns the raw win/loss/draw counts. The opponent is
always forwarded as a raw config via ``--baseline-config`` -- matchmaking
(see ``matchmaking.py``) only ever plays against exact configs (an
``OpponentPool`` anchor or another trial's candidate), never a game's own
named preset, so there's no named-``--baseline`` dispatch path here anymore.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .config import SearchConfig, json_dumps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Floor baselines
# ---------------------------------------------------------------------------

# Baseline-only families `mcts-tune`'s own `make_candidate` (mcts-tune/src/
# lib.rs) builds directly from a raw params object -- not named presets any
# game's `preset_cfg`/`PRESET_CONFIGS` knows about. Used to seed the
# opponent pool's "random" anchor.
FLOOR_BASELINES: dict[str, dict] = {
    "flat_mc": {"family": "flat_mc", "q_init": "Infinity"},
    "random": {"family": "random", "q_init": "Infinity"},
}

_HEARTBEAT_INTERVAL_S = 30
_TRIAL_TIMEOUT_S = 600


def _run_with_heartbeat(cmd: list[str], *, timeout: float, seed: int) -> subprocess.CompletedProcess:
    """Like ``subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)``,
    but logs a "still running" line every ``_HEARTBEAT_INTERVAL_S`` seconds
    instead of blocking silently until the process exits or the full timeout
    fires. A trial that's merely slow (a big ``--max-iterations``/
    ``--max-time-ms`` budget, a slow game) was otherwise indistinguishable
    from a hung one until the 600s timeout killed it -- this only adds
    liveness output to ``stdout.log``, it doesn't change what a timeout or
    crash reports downstream (see ``play_game``'s own status tagging).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = 0.0
    while True:
        wait = min(_HEARTBEAT_INTERVAL_S, timeout - elapsed)
        try:
            stdout, stderr = proc.communicate(timeout=wait)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            elapsed += wait
            if elapsed >= timeout:
                proc.kill()
                proc.communicate()
                raise
            logger.info("Trial still running after %.0fs (seed=%s)", elapsed, seed)


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------


def _build_cmd(
    cfg: SearchConfig,
    binary: Path,
    candidate_config: dict,
    opponent_config: dict,
    *,
    rounds: int,
    seed: int,
    trace_path: str | None,
) -> list[str]:
    """Build the ``tune eval`` argv shared by trial evaluation and the preflight check."""
    cmd = [
        str(binary),
        "tune",
        "eval",
        "--config",
        json_dumps(candidate_config),
        "--rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--baseline-config",
        json_dumps(opponent_config),
    ]

    if cfg.target.game_config is not None:
        cmd += ["--game-config", json.dumps(cfg.target.game_config)]

    if cfg.target.max_iterations is not None:
        cmd += ["--max-iterations", str(cfg.target.max_iterations)]
    elif cfg.target.max_time_ms is not None:
        cmd += ["--max-time-ms", str(cfg.target.max_time_ms)]

    if trace_path is not None:
        cmd += ["--trace-path", trace_path]

    return cmd


def play_game(
    cfg: SearchConfig,
    binary: Path,
    candidate_config: dict,
    opponent_config: dict,
    *,
    seed: int,
    trace_path: str | None = None,
) -> tuple[int, int, int, str | None]:
    """Play one candidate-vs-opponent match (``cfg.target.rounds`` round-robin pairs).

    Returns ``(wins, losses, draws, status)``. ``status`` is ``None`` for a
    match that actually produced a result; ``"timeout"``/``"crashed"``
    otherwise (a trial's worth of subprocess failure modes -- the binary
    hung past ``_TRIAL_TIMEOUT_S``, exited non-zero, or printed output with
    no parseable ``{"wins": ..., "losses": ..., "draws": ...}`` line), in
    which case ``wins``/``losses``/``draws`` are all ``0`` since no real
    game was played.
    """
    if cfg.target.max_iterations is not None and cfg.target.max_time_ms is not None:
        raise ValueError(
            "target.max_iterations and target.max_time_ms are mutually exclusive "
            "(matches game-host::run_tune_eval's own --max-iterations/--max-time-ms "
            "constraint) -- unset one before launching"
        )

    cmd = _build_cmd(
        cfg,
        binary,
        candidate_config,
        opponent_config,
        rounds=cfg.target.rounds,
        seed=seed,
        trace_path=trace_path,
    )

    logger.debug("Running: %s", " ".join(cmd))

    try:
        result = _run_with_heartbeat(cmd, timeout=_TRIAL_TIMEOUT_S, seed=seed)
    except subprocess.TimeoutExpired:
        logger.warning("Trial timed out after %ds (seed=%s)", _TRIAL_TIMEOUT_S, seed)
        return 0, 0, 0, "timeout"

    if result.returncode != 0:
        logger.error(
            "Binary exited with code %s:\nstdout:\n%s\nstderr:\n%s",
            result.returncode,
            result.stdout,
            result.stderr,
        )
        return 0, 0, 0, "crashed"

    # Parse the trailing JSON line: {"cost": ..., "wins": ..., "losses": ..., "draws": ...}
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "wins" in payload and "losses" in payload and "draws" in payload:
            return int(payload["wins"]), int(payload["losses"]), int(payload["draws"]), None

    logger.error(
        "No JSON 'wins'/'losses'/'draws' line found in output:\n%s\nstderr:\n%s",
        result.stdout,
        result.stderr,
    )
    return 0, 0, 0, "crashed"


def preflight_check(cfg: SearchConfig, default_config: dict, random_config: dict) -> None:
    """Run one real match before the search loop starts.

    A misconfiguration that makes *every* trial's ``tune eval`` invocation
    fail the same way (an unsupported ``--game-config``, a missing binary
    flag, ...) is otherwise invisible until the whole run finishes: a
    genuine ``play_game`` timeout/crash looks the same in the trial log as
    every other trial's own status tagging. Running the default config
    against the random-floor baseline once, here, with the exit code and
    stderr surfaced directly, turns that into an immediate, readable failure
    instead of a full ``optimizer.n_trials``-trial budget spent on a config
    that could never have succeeded.
    """
    _wins, _losses, _draws, status = play_game(
        cfg,
        cfg.resolve_binary(),
        default_config,
        random_config,
        seed=0,
        trace_path=None,
    )
    if status is not None:
        raise RuntimeError(
            f"Preflight match {status}; aborting before spending the full "
            f"{cfg.optimizer.n_trials}-trial budget."
        )
