"""Task-owned evaluation bundle execution tests with a fake game process."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import subprocess

import pytest

from tuner_cli.config import OptimizerConfig, SearchConfig, TargetConfig
from tuner_cli.evaluation import (
    OpponentSnapshot,
    PairTask,
    Rating,
    configured_game_seed,
)
from tuner_cli.lifecycle import SessionId, TrialId
from tuner_cli.task_artifacts import (
    DescriptorCommit,
    TaskDescriptorAllocator,
    read_completion,
)
from tuner_cli.task_execution import TaskDescriptorError, execute_task_bundle


def _task() -> PairTask:
    return PairTask(
        SessionId("session-a"),
        TrialId("trial-a"),
        "pair-a",
        0,
        42,
        {"family": "ucb1"},
        OpponentSnapshot("random", {"family": "random"}, 0.0, 0.5),
        "pool-fingerprint",
        Rating(25.0, 8.3),
    )


def _descriptor(
    tmp_path: Path,
) -> tuple[TaskDescriptorAllocator, DescriptorCommit, Path]:
    allocator = TaskDescriptorAllocator.start(
        tmp_path / "attempt",
        session_id="session-a",
        optimizer_id="optimizer-a",
        attempt_id="attempt-a",
        bench_run_id=None,
        manifest_fingerprint="manifest-a",
    )
    cfg = SearchConfig(
        optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("fake-game"))
    )
    commit = allocator.commit_task(
        _task(), cfg=cfg, binary=Path("fake-game"), pool_snapshot=[]
    )
    return allocator, commit, allocator.layout.descriptor(commit.identity)


def _output(sequence: int = 1) -> bytes:
    seed = configured_game_seed(42)
    metrics = {"iterations_total": 12, "iterations_first_half": 5, "move_time_ms": 7}
    records = [
        {
            "type": "configured_match_result",
            "seq": 1,
            "round": 1,
            "seed": seed,
            "candidate_side": "first",
            "outcome": "candidate_win",
            "trace_game_seq": sequence,
            "plies": 8,
            "elapsed_ms": 9,
            "candidate": metrics,
            "baseline": metrics,
        },
        {
            "type": "configured_match_result",
            "seq": 2,
            "round": 1,
            "seed": seed,
            "candidate_side": "second",
            "outcome": "baseline_win",
            "trace_game_seq": sequence + 1,
            "plies": 10,
            "elapsed_ms": 11,
            "candidate": metrics,
            "baseline": metrics,
        },
        {
            "type": "configured_comparison_summary",
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
        },
    ]
    return b"\n".join(json.dumps(record).encode() for record in records) + b"\n"


class _Process:
    def __init__(
        self, stdout: bytes, stderr: bytes = b"", *, timeout_once: bool = False
    ):
        self.stdout, self.stderr = stdout, stderr
        self.returncode = 0
        self.timeout_once = timeout_once
        self.killed = False

    def communicate(self, timeout=None):
        if self.timeout_once and not self.killed:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("fake-game", timeout)
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True


def _popen(process: _Process, launched: list[list[str]], *, trace: bool = True):
    def run(command, **_kwargs):
        launched.append(command)
        if trace:
            trace_path = Path(command[command.index("--trace-path") + 1])
            trace_path.write_text('{"game_seq":1,"ply":0}\n')
        return process

    return run


def test_success_bundle_has_raw_logs_trace_sequences_heartbeats_and_terminal_marker_last(
    tmp_path: Path, monkeypatch
):
    allocator, commit, path = _descriptor(tmp_path)
    launched: list[list[str]] = []
    process = _Process(_output(), b"raw stderr", timeout_once=True)
    writes: list[str] = []
    import tuner_cli.task_execution as execution

    real_completion = execution.write_completion
    monkeypatch.setattr(
        execution,
        "write_completion",
        lambda directory, completion: (
            writes.append("complete.json"),
            real_completion(directory, completion),
        )[1],
    )
    real_immutable = execution.write_immutable
    monkeypatch.setattr(
        execution,
        "write_immutable",
        lambda destination, contents: (
            writes.append(Path(destination).name),
            real_immutable(destination, contents),
        )[1],
    )
    moments = iter(
        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=30 * i) for i in range(3)
    )

    reference = execute_task_bundle(
        path,
        commit.digest,
        popen=_popen(process, launched),
        clock=lambda: next(moments),
    )

    task = allocator.layout.root / "tasks" / commit.identity.task_id
    assert reference.outcome == "completed"
    assert (task / "stdout.log").read_bytes() == _output()
    assert (task / "stderr.log").read_bytes() == b"raw stderr"
    assert (task / "trace.jsonl").read_text() == '{"game_seq":1,"ply":0}\n'
    assert json.loads((task / "heartbeat.json").read_text())["update_sequence"] == 1
    assert writes[-1] == "complete.json"
    assert "--trace-game-sequence-start" in launched[0]
    assert read_completion(task, commit.identity, commit.digest).outcome == "completed"


@pytest.mark.parametrize(
    ("process", "expected_kind"),
    [
        (_Process(b"not-json", b"parse stderr"), "malformed_output"),
        (_Process(b"", b"exit stderr"), "process_exit"),
    ],
)
def test_normal_failures_preserve_logs_trace_and_commit_typed_failure(
    tmp_path: Path, process: _Process, expected_kind: str
):
    allocator, commit, path = _descriptor(tmp_path)
    if expected_kind == "process_exit":
        process.returncode = 7
    reference = execute_task_bundle(path, commit.digest, popen=_popen(process, []))
    task = allocator.layout.root / "tasks" / commit.identity.task_id
    assert reference.outcome == "failed"
    assert (task / "stdout.log").read_bytes() == process.stdout
    assert (task / "stderr.log").read_bytes() == process.stderr
    assert (task / "trace.jsonl").exists()
    assert json.loads((task / "failure.json").read_text())["kind"] == expected_kind
    assert read_completion(task, commit.identity, commit.digest).outcome == "failed"


def test_timeout_bundle_kills_process_and_keeps_partial_logs(tmp_path: Path):
    allocator, commit, path = _descriptor(tmp_path)

    class _NeverReturns(_Process):
        def communicate(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("fake-game", timeout)
            return self.stdout, self.stderr

    process = _NeverReturns(b"partial out", b"partial err")
    reference = execute_task_bundle(
        path,
        commit.digest,
        popen=_popen(process, []),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    task = allocator.layout.root / "tasks" / commit.identity.task_id
    assert process.killed and reference.outcome == "failed"
    assert json.loads((task / "failure.json").read_text())["kind"] == "timeout"
    assert (task / "stdout.log").read_bytes() == b"partial out"


def test_descriptor_digest_and_schema_rejected_before_launch_or_task_writes(
    tmp_path: Path,
):
    allocator, commit, path = _descriptor(tmp_path)
    launches: list[list[str]] = []
    with pytest.raises(TaskDescriptorError, match="digest"):
        execute_task_bundle(path, "0" * 64, popen=_popen(_Process(_output()), launches))
    payload = json.loads(path.read_text())
    payload["task_directory"] = "tasks/outside"
    path.write_text(json.dumps(payload))
    with pytest.raises(TaskDescriptorError):
        execute_task_bundle(
            path, commit.digest, popen=_popen(_Process(_output()), launches)
        )
    assert launches == []
    assert not (allocator.layout.root / "tasks" / commit.identity.task_id).exists()


def test_process_death_before_result_or_marker_leaves_task_incomplete(
    tmp_path: Path, monkeypatch
):
    allocator, commit, path = _descriptor(tmp_path)
    import tuner_cli.task_execution as execution

    real_immutable = execution.write_immutable

    def die_before_result(destination, contents):
        if Path(destination).name == "result.json":
            raise SystemExit("worker died")
        return real_immutable(destination, contents)

    monkeypatch.setattr(execution, "write_immutable", die_before_result)
    with pytest.raises(SystemExit):
        execute_task_bundle(path, commit.digest, popen=_popen(_Process(_output()), []))
    task = allocator.layout.root / "tasks" / commit.identity.task_id
    assert not (task / "result.json").exists() and not (task / "complete.json").exists()

    allocator, commit, path = _descriptor(tmp_path / "second")
    monkeypatch.setattr(execution, "write_immutable", real_immutable)
    monkeypatch.setattr(
        execution,
        "write_completion",
        lambda *_args: (_ for _ in ()).throw(SystemExit("worker died")),
    )
    with pytest.raises(SystemExit):
        execute_task_bundle(path, commit.digest, popen=_popen(_Process(_output()), []))
    task = allocator.layout.root / "tasks" / commit.identity.task_id
    assert (task / "result.json").exists() and not (task / "complete.json").exists()


def test_task_trace_path_cannot_escape_its_descriptor_root(tmp_path: Path):
    allocator, commit, path = _descriptor(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = json.loads(path.read_text())
    payload["task_directory"] = "../outside"
    path.write_text(json.dumps(payload))
    with pytest.raises(TaskDescriptorError):
        execute_task_bundle(path, commit.digest, popen=_popen(_Process(_output()), []))
    assert list(outside.iterdir()) == []


def test_each_task_subprocess_receives_only_its_own_trace_path(tmp_path: Path):
    allocator, first, first_path = _descriptor(tmp_path)
    second = allocator.commit_task(
        _task(),
        cfg=SearchConfig(
            optimizer=OptimizerConfig(), target=TargetConfig(binary=Path("fake-game"))
        ),
        binary=Path("fake-game"),
        pool_snapshot=[],
    )
    second_path = allocator.layout.descriptor(second.identity)
    first_launches: list[list[str]] = []
    second_launches: list[list[str]] = []
    execute_task_bundle(
        first_path, first.digest, popen=_popen(_Process(_output(1)), first_launches)
    )
    execute_task_bundle(
        second_path,
        second.digest,
        popen=_popen(_Process(_output(3)), second_launches),
    )
    first_trace = first_launches[0][first_launches[0].index("--trace-path") + 1]
    second_trace = second_launches[0][second_launches[0].index("--trace-path") + 1]
    assert first_trace.endswith(f"tasks/{first.identity.task_id}/trace.jsonl")
    assert second_trace.endswith(f"tasks/{second.identity.task_id}/trace.jsonl")
    assert first_trace != second_trace
