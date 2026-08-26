"""One-pair configured-comparison subprocess transport."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .config import SearchConfig, json_dumps
from .evaluation import (
    GameResult,
    PairResult,
    PairTask,
    StrategyMetrics,
    configured_game_seed,
    game_id_for,
)

logger = logging.getLogger(__name__)

FLOOR_BASELINES: dict[str, dict] = {
    "flat_mc": {"family": "flat_mc", "q_init": "Infinity"},
    "random": {"family": "random", "q_init": "Infinity"},
}

_HEARTBEAT_INTERVAL_S = 30
_TRIAL_TIMEOUT_S = 600
_DEFAULT_MAX_ITERATIONS = 10_000


class PairExecutionError(RuntimeError):
    """The evaluator could not return one complete, valid comparison pair."""


def _run_with_heartbeat(
    cmd: list[str], *, timeout: float, seed: int
) -> subprocess.CompletedProcess:
    """Run a subprocess while periodically logging liveness until its timeout."""
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
            logger.info("Pair still running after %.0fs (seed=%s)", elapsed, seed)


def _build_pair_cmd(cfg: SearchConfig, binary: Path, task: PairTask) -> list[str]:
    """Build the strict one-pair configured-comparison command."""
    cmd = [
        str(binary),
        "compare",
        "eval",
        "--candidate-config",
        json_dumps(task.candidate_config),
        "--baseline-config",
        json_dumps(task.opponent.config),
        "--rounds",
        "1",
        "--seed",
        str(task.seed),
    ]
    if cfg.target.game_config is not None:
        cmd += ["--game-config", json_dumps(cfg.target.game_config)]
    if cfg.target.max_time_ms is not None:
        cmd += ["--max-time-ms", str(cfg.target.max_time_ms)]
    else:
        cmd += [
            "--max-iterations",
            str(cfg.target.max_iterations or _DEFAULT_MAX_ITERATIONS),
        ]
    return cmd


def evaluate_pair(cfg: SearchConfig, binary: Path, task: PairTask) -> PairResult:
    """Execute and strictly decode one configured, seat-swapped pair."""
    if cfg.target.max_iterations is not None and cfg.target.max_time_ms is not None:
        raise ValueError("target.max_iterations and target.max_time_ms are mutually exclusive")
    cmd = _build_pair_cmd(cfg, binary, task)
    logger.debug("Running: %s", " ".join(cmd))
    try:
        completed = _run_with_heartbeat(cmd, timeout=_TRIAL_TIMEOUT_S, seed=task.seed)
    except subprocess.TimeoutExpired as error:
        raise PairExecutionError(f"pair timed out after {_TRIAL_TIMEOUT_S}s") from error
    if completed.returncode != 0:
        raise PairExecutionError(
            f"comparison exited with code {completed.returncode}: {completed.stderr}"
        )
    try:
        return parse_pair_output(completed.stdout, task)
    except ValueError as error:
        raise PairExecutionError(f"malformed comparison output: {error}") from error


def parse_pair_output(stdout: str, task: PairTask) -> PairResult:
    """Decode exactly two ordered game records and their matching summary."""
    records = _json_records(stdout)
    if len(records) != 3 or [record.get("type") for record in records] != [
        "configured_match_result",
        "configured_match_result",
        "configured_comparison_summary",
    ]:
        raise ValueError("expected exactly two game records followed by one summary")
    expected_seed = configured_game_seed(task.seed)
    decoded = tuple(_decode_game(record, task, expected_seed) for record in records[:2])
    if [game.candidate_side for game in decoded] != ["first", "second"]:
        raise ValueError("games must be ordered candidate first, then candidate second")
    _validate_summary(records[2], decoded)
    return PairResult(task, (decoded[0], decoded[1]))


def _json_records(stdout: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("output contains non-JSON text") from error
        if not isinstance(record, dict):
            raise ValueError("output record is not an object")
        records.append(record)
    return records


def _decode_game(record: dict[str, Any], task: PairTask, expected_seed: int) -> GameResult:
    side = _string(record, "candidate_side")
    outcome = _string(record, "outcome")
    if side not in ("first", "second") or outcome not in (
        "candidate_win",
        "baseline_win",
        "draw",
    ):
        raise ValueError("game has invalid candidate side or outcome")
    if _integer(record, "round") != 1 or _integer(record, "seed") != expected_seed:
        raise ValueError("game has an unexpected round or seed")
    expected_seq = 1 if side == "first" else 2
    if _integer(record, "seq") != expected_seq:
        raise ValueError("game sequence does not match candidate side")
    trace = record.get("trace_game_seq")
    if trace is not None and (not isinstance(trace, int) or isinstance(trace, bool) or trace < 0):
        raise ValueError("trace_game_seq must be an integer or null")
    if task.trace_game_sequence_start is not None:
        expected_trace = task.trace_game_sequence_start + expected_seq - 1
        if trace != expected_trace:
            raise ValueError("trace_game_seq does not match candidate side")
    return GameResult(
        game_id_for(task.pair_id, side),
        side,
        outcome,
        expected_seed,
        1,
        trace,
        _integer(record, "plies"),
        _integer(record, "elapsed_ms"),
        _decode_metrics(record, "candidate"),
        _decode_metrics(record, "baseline"),
    )


def _decode_metrics(record: dict[str, Any], key: str) -> StrategyMetrics:
    raw = record.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"{key} metrics must be an object")
    return StrategyMetrics(
        _integer(raw, "iterations_total"),
        _integer(raw, "iterations_first_half"),
        _integer(raw, "move_time_ms"),
    )


def _validate_summary(summary: dict[str, Any], games: tuple[GameResult, ...]) -> None:
    if summary.get("type") != "configured_comparison_summary":
        raise ValueError("final record is not a comparison summary")
    outcomes = [game.outcome for game in games]
    expected = {
        "games": 2,
        "wins": outcomes.count("candidate_win"),
        "losses": outcomes.count("baseline_win"),
        "draws": outcomes.count("draw"),
    }
    if any(_integer(summary, key) != value for key, value in expected.items()):
        raise ValueError("summary does not match physical game outcomes")


def _string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def preflight_check(cfg: SearchConfig, default_config: dict, random_config: dict) -> None:
    """Run one complete configured pair before starting an optimization attempt."""
    from .evaluation import OpponentSnapshot, PairId, Rating
    from .lifecycle import SessionId, TrialId

    task = PairTask(
        SessionId("preflight"),
        TrialId("preflight"),
        PairId("pair-preflight"),
        0,
        0,
        default_config,
        OpponentSnapshot("random", random_config, 0.0, 0.5),
        "preflight",
        Rating(25.0, 8.333),
        None,
    )
    evaluate_pair(cfg, cfg.resolve_binary(), task)
