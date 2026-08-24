#!/usr/bin/env python3
"""End-to-end tests that run a real game-nim binary.

These tests build and run a real ``game-nim`` subprocess (``cargo build -p game-nim``)
to exercise the full Optuna ask/tell loop and the resume-from-persistent-state path.
They are intentionally separated from the fast pytest suite (``uv run pytest tests/``)
because compiling and running a Rust binary makes them too slow for the every-edit
cycle -- run them manually with::

    uv run python tests/e2e/run_e2e.py

Before running, ensure the tuner package is installed in the venv: ``uv sync``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path


def _repo_root() -> Path:
    """Walk up from this file to the workspace root (has Cargo.lock)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.lock").is_file():
            return parent
    raise RuntimeError("could not locate workspace root (no Cargo.lock found)")


def _build_game_nim() -> Path:
    """Build ``game-nim`` in debug mode and return the binary path."""
    root = _repo_root()
    subprocess.run(
        ["cargo", "build", "-p", "game-nim", "--bin", "game-nim"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    binary = root / "target" / "debug" / "game-nim"
    assert binary.is_file(), f"expected cargo build to produce {binary}"
    return binary


def _ensure_on_path(root: Path) -> None:
    """Prepend ``target/debug`` to ``PATH`` so the binary's own ``cargo``-built
    dependencies (``mcts-tune``, ``game-nim``) can be found by any subprocess
    that resolves them via ``PATH``."""
    debug_dir = str(root / "target" / "debug")
    env_path = os.environ.get("PATH", "")
    if debug_dir not in env_path:
        os.environ["PATH"] = f"{debug_dir}:{env_path}"


def _assert_rust_projection(lifecycle_path: Path) -> None:
    """Replay the real Python artifact through the Rust persistence contract."""
    env = os.environ.copy()
    env["MCTS_TUNING_LIFECYCLE_PATH"] = str(lifecycle_path)
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "mcts-bench",
            "--test",
            "tuning_lifecycle",
            "real_tuner_artifact_projects_complete_pairs_and_trace_links",
            "--",
            "--ignored",
            "--exact",
        ],
        cwd=_repo_root(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ask_tell_loop_emits_rating_jsonl(binary: Path, tmp_dir: Path) -> None:
    """One real tuning trial preserves pair/game and legacy evidence."""
    from tuner_cli.__main__ import run_optimization
    from tuner_cli.config import OptimizerConfig, SearchConfig, TargetConfig

    cfg = SearchConfig(
        optimizer=OptimizerConfig(n_trials=1, deterministic=True, seed=7),
        target=TargetConfig(binary=binary, rounds=1, max_iterations=50),
    )
    artifact_root = tmp_dir / "bench-runs" / "records" / "tuning-artifacts"

    out = StringIO()
    with redirect_stdout(out), redirect_stderr(sys.stderr):
        os.chdir(str(tmp_dir))
        study, _pool = run_optimization(
            cfg,
            run_id="records",
            git_sha="test-sha",
            bench_run_id="records",
            attempt_id="attempt-records",
            artifact_root=artifact_root,
        )

    lifecycle_path = tmp_dir / "optuna_output" / "records" / "lifecycle.jsonl"
    manifest_path = tmp_dir / "optuna_output" / "records" / "session-manifest.json"
    assert lifecycle_path.is_file(), "lifecycle evidence missing"
    assert manifest_path.is_file(), "session manifest missing"
    lifecycle = [json.loads(line) for line in lifecycle_path.read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    assert lifecycle[0]["event_type"] == "session_started"
    assert lifecycle[0]["session_id"] != lifecycle[0]["attempt_id"]
    assert lifecycle[0]["payload"]["manifest_fingerprint"] == manifest["fingerprint"]
    assert any(record["event_type"] == "trial_completed" for record in lifecycle)

    pair_starts = [r for r in lifecycle if r["event_type"] == "pair_started"]
    pair_finishes = [r for r in lifecycle if r["event_type"] == "pair_finished"]
    games_by_pair: dict[str, list[dict]] = defaultdict(list)
    for record in lifecycle:
        if record["event_type"] == "game_finished":
            games_by_pair[record["payload"]["pair_id"]].append(record["payload"])
    assert 5 <= len(pair_starts) <= 15, "pair count must follow the 5/15 stopping bounds"
    assert len(pair_finishes) == len(pair_starts)
    assert [r["payload"]["pair_index"] for r in pair_starts] == list(range(len(pair_starts)))
    assert len({r["payload"]["pair_id"] for r in pair_starts}) == len(pair_starts)
    for started in pair_starts:
        payload = started["payload"]
        games = games_by_pair[payload["pair_id"]]
        assert len(games) == 2
        assert [game["candidate_side"] for game in games] == ["first", "second"]
        assert len({game["game_id"] for game in games}) == 2
        assert all(isinstance(game["game_id"], str) and game["game_id"] for game in games)
        assert all(isinstance(game["trace_game_seq"], int) for game in games)
        assert len({game["trace_game_seq"] for game in games}) == 2
        assert all(game["seed"] == payload["seed"] and game["round"] == 1 for game in games)
    assert list((artifact_root / "tasks").glob("*/trace.jsonl")), "trace evidence missing"
    _assert_rust_projection(lifecycle_path)

    records = []
    for line in out.getvalue().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    trials = [r for r in records if r.get("type") == "trial"]
    incumbents = [r for r in records if r.get("type") == "incumbent"]
    assert len(trials) == 1, f"expected 1 trial, got {len(trials)}"
    assert incumbents, "expected at least one incumbent record"
    for record in trials:
        assert record["cost"] == pytest_approx(
            -(record["extra"]["mu"] - 3 * record["extra"]["sigma"])
        ), f"cost mismatch in trial {record['trial_id']}"
        assert record["extra"]["git_sha"] == "test-sha", "git_sha not propagated"
        opponents = record["extra"]["opponents"]
        assert isinstance(opponents, list), "opponents missing"
        expected = Counter()
        legacy_outcome = {
            "candidate_win": "win",
            "baseline_win": "loss",
            "draw": "draw",
        }
        for started in pair_starts:
            opponent = started["payload"]["opponent"]["anchor_id"]
            for game in games_by_pair[started["payload"]["pair_id"]]:
                expected[(opponent, legacy_outcome[game["outcome"]])] += 1
        actual = Counter((entry["opponent"], entry["outcome"]) for entry in opponents)
        assert actual == expected, "legacy opponents must derive from physical games"
    last_incumbent = incumbents[-1]
    assert last_incumbent["config"] == study.best_trial.user_attrs["config"], "incumbent config mismatch"
    assert last_incumbent["cost"] == pytest_approx(-study.best_value), "incumbent cost mismatch"

    print("  [PASS] test_ask_tell_loop_emits_rating_jsonl")


def test_parameters_from_binary_reports_search_space_and_baselines(
    binary: Path, _tmp_dir: Path
) -> None:
    """The game binary's ``tune describe`` subcommand exposes its search space."""
    from tuner_cli.config import SearchConfig

    parameters, conditions, baselines = SearchConfig.parameters_from_binary(binary)
    assert parameters, "expected at least one parameter"
    assert isinstance(conditions, list)
    assert "strong" in baselines, "expected 'strong' baseline preset"

    print("  [PASS] test_parameters_from_binary_reports_search_space_and_baselines")


def test_reusing_run_id_completes_only_new_trials_and_reloads_pool(
    binary: Path, tmp_dir: Path
) -> None:
    """Reusing a run id reloads both the Optuna study and its frozen opponent pool."""
    from tuner_cli.__main__ import run_optimization
    from tuner_cli.config import OptimizerConfig, SearchConfig, TargetConfig

    def cfg(n_trials: int) -> SearchConfig:
        return SearchConfig(
            optimizer=OptimizerConfig(n_trials=n_trials, deterministic=True, seed=7),
            target=TargetConfig(binary=binary, rounds=1, max_iterations=50),
        )

    out = StringIO()
    with redirect_stdout(out), redirect_stderr(sys.stderr):
        os.chdir(str(tmp_dir))
        first, first_pool = run_optimization(
            cfg(1), run_id="resume", bench_run_id="resume-1",
            attempt_id="attempt-resume-1",
            artifact_root=tmp_dir / "bench-runs" / "resume-1" / "tuning-artifacts",
        )
    first_ids = [anchor.id for anchor in first_pool.anchors]
    assert len(first.trials) == 1, f"expected 1 trial, got {len(first.trials)}"

    out = StringIO()
    with redirect_stdout(out), redirect_stderr(sys.stderr):
        second, second_pool = run_optimization(
            cfg(2), run_id="resume", bench_run_id="resume-2",
            attempt_id="attempt-resume-2",
            artifact_root=tmp_dir / "bench-runs" / "resume-2" / "tuning-artifacts",
        )
    assert len(second.trials) == 2, f"expected 2 trials after resume, got {len(second.trials)}"
    assert [anchor.id for anchor in second_pool.anchors][: len(first_ids)] == first_ids, (
        "pool anchors from first run not preserved"
    )
    assert (tmp_dir / "optuna_output" / "resume" / "study.db").is_file(), "study.db missing"
    assert (tmp_dir / "optuna_output" / "resume" / "pool.json").is_file(), "pool.json missing"

    print("  [PASS] test_reusing_run_id_completes_only_new_trials_and_reloads_pool")


# ---------------------------------------------------------------------------
# Approx matcher (replaces pytest.approx without the dependency)
# ---------------------------------------------------------------------------


def pytest_approx(expected, rel: float = 1e-6, abs_: float = 1e-12):
    class ApproxFloat:
        def __eq__(self, other):
            if isinstance(expected, (int, float)) and isinstance(other, (int, float)):
                delta = max(abs(expected) * rel, abs_)
                return abs(expected - other) <= delta
            return NotImplemented

        def __repr__(self):
            return f"~{expected}"

        def __ne__(self, other):
            return not self.__eq__(other)

    return ApproxFloat()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    root = _repo_root()
    _ensure_on_path(root)
    binary = _build_game_nim()

    with tempfile.TemporaryDirectory(prefix="tuner-e2e-") as tmp:
        tmp_dir = Path(tmp).resolve()

        # Add the tuner package itself to sys.path so imports work from tmp_dir.
        src_dir = root / "tuner" / "src"
        if str(src_dir.resolve()) not in sys.path:
            sys.path.insert(0, str(src_dir.resolve()))

        failures = 0
        for name, test_fn in [
            ("test_parameters_from_binary_reports_search_space_and_baselines",
             test_parameters_from_binary_reports_search_space_and_baselines),
            ("test_ask_tell_loop_emits_rating_jsonl", test_ask_tell_loop_emits_rating_jsonl),
            ("test_reusing_run_id_completes_only_new_trials_and_reloads_pool",
             test_reusing_run_id_completes_only_new_trials_and_reloads_pool),
        ]:
            try:
                test_fn(binary, tmp_dir)
            except Exception:
                import traceback
                print(f"  [FAIL] {name}")
                traceback.print_exc()
                failures += 1

        if failures:
            print(f"\n{'-' * 60}\n{failures} test(s) FAILED\n{'-' * 60}")
            return 1
        else:
            print(f"\n{'-' * 60}\nAll e2e tests passed.\n{'-' * 60}")
            return 0


if __name__ == "__main__":
    sys.exit(main())
