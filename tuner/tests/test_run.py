from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from tuner_cli.artifacts import read_manifest
from tuner_cli.cohort import current_active_candidates
from tuner_cli.constraints import decode_constraints, encode_constraints
from tuner_cli.domain import (
    GameResult,
    PairedBootstrapEvidence,
    PairResult,
    PairTask,
    ProposalRequest,
    ProposedConfiguration,
    SearchEffort,
    ShadowCandidateDecision,
    ShadowRaceDecision,
    StrategyMetrics,
    SuccessiveHalvingEvidence,
    ValidationResult,
)
from tuner_cli.event_payloads import PairCompletedPayload, PairFailedPayload, PairStartedPayload
from tuner_cli.evidence import read_events, scientific_projection
from tuner_cli.identity import candidate_from_config, canonical_json, fingerprint, game_id
from tuner_cli.observations import comparable_prefix_observations
from tuner_cli.replay import replay
from tuner_cli.report import write_report
from tuner_cli.run import RunOptions, run_foreground
from tuner_cli.target import PairExecutionError, _splitmix_seed


def _fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "game-fake"
    binary.touch()
    binary.chmod(0o755)
    return binary


def _objective(tmp_path: Path) -> Path:
    path = tmp_path / "objective.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "objective_id": "fake-reference-v1",
                "game_kind": "druid",
                "opponents": [
                    {
                        "id": "schema-default",
                        "label": "Default",
                        "role": "default",
                        "weight": 1,
                        "config": {"source": "schema_default"},
                    },
                    {
                        "id": "historical",
                        "label": "Historical",
                        "role": "historical_reference",
                        "weight": 1,
                        "config": {"source": "inline", "value": {"algorithm": "b"}},
                    },
                ],
                "start_distribution": {"kind": "default_only"},
            }
        )
    )
    return path


class FakeTarget:
    def __init__(self) -> None:
        self.calls: list[PairTask] = []

    def describe(self) -> dict[str, object]:
        return {
            "kind": "druid",
            "label": "Druid",
            "description": "fake",
            "default_config": {"size": 5},
            "ai_presets": [],
            "tuning": {
                "id": "strategy",
                "baselines": [],
                "eval_rounds": 1,
                "game_config": {"size": 5},
                "parameters": [
                    {
                        "name": "algorithm",
                        "type": "categorical",
                        "choices": ["a", "b", "c", "d", "e", "f", "g", "h"],
                        "default": "a",
                    },
                ],
                "conditions": [],
            },
        }

    def validate(self, candidates, opponent, game_config):  # type: ignore[no-untyped-def]
        return ValidationResult(True, ())

    def cancel(self) -> None:
        return None

    def _outcome(self, task: PairTask, candidate_config: dict[str, str]) -> str:
        del task
        return "candidate_win" if candidate_config["algorithm"] == "b" else "draw"

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        self.calls.append(task)
        outcome = self._outcome(task, json.loads(candidate.canonical_config))
        games = []
        for seq, side in ((1, "first"), (2, "second")):
            raw = {
                "type": "configured_match_result",
                "seq": seq,
                "round": 1,
                "seed": _splitmix_seed(task.task_case.seed),
                "candidate_side": side,
                "outcome": outcome,
                "trace_game_seq": None,
                "plies": 1,
                "elapsed_ms": 1,
                "candidate": {"iterations_total": 1, "iterations_first_half": 1, "move_time_ms": 1},
                "baseline": {"iterations_total": 1, "iterations_first_half": 1, "move_time_ms": 1},
            }
            games.append(
                GameResult(
                    game_id(task, side),
                    side,
                    outcome,
                    _splitmix_seed(task.task_case.seed),
                    1,
                    seq,
                    None,
                    1,
                    1,
                    StrategyMetrics(1, 1, 1),
                    StrategyMetrics(1, 1, 1),
                    canonical_json(raw),
                )
            )
        return PairResult(task, tuple(games))


class FakeModel:
    def ask(self, request: ProposalRequest) -> ProposedConfiguration:
        return ProposedConfiguration(
            candidate_from_config({"algorithm": f"model-{request.attempt.source_attempt}"}), None
        )


class ActiveProfileModel:
    def ask(self, request: ProposalRequest) -> ProposedConfiguration:
        source_attempt = request.attempt.source_attempt
        algorithm = "c" if source_attempt == 1 else ("e", "f", "g", "h")[source_attempt - 2]
        return ProposedConfiguration(candidate_from_config({"algorithm": algorithm}), None)


class ActiveProfileTarget(FakeTarget):
    def __init__(self, recovery: bool = False) -> None:
        super().__init__()
        self.recovery = recovery

    def _outcome(self, task: PairTask, candidate_config: dict[str, str]) -> str:
        if task.task_case.phase != "tuning":
            return "draw"
        algorithm = candidate_config["algorithm"]
        if not self.recovery:
            return "candidate_win" if algorithm == "c" else "draw"
        if task.task_case.ordinal < 3:
            return "candidate_win" if algorithm == "c" else "draw"
        if task.task_case.ordinal >= 12:
            return "draw" if algorithm == "c" else "candidate_win"
        return "draw"


class InterruptingTarget(FakeTarget):
    def __init__(self, interrupt_on_call: int) -> None:
        super().__init__()
        self.interrupt_on_call = interrupt_on_call

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        if len(self.calls) + 1 == self.interrupt_on_call:
            self.calls.append(task)
            raise KeyboardInterrupt
        return super().evaluate(task, candidate, opponent, game_config, timeout_seconds)


class FailingTarget(FakeTarget):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        if len(self.calls) + 1 == self.fail_on_call:
            self.calls.append(task)
            raise PairExecutionError("pair_output", "injected failure", ["game"], returncode=1)
        return super().evaluate(task, candidate, opponent, game_config, timeout_seconds)


class ConcurrentFailuresThenInterruptTarget(FakeTarget):
    """Fail the first bounded batch together, then leave its retries censored."""

    def __init__(self) -> None:
        super().__init__()
        self._first_batch = threading.Barrier(2)
        self._lock = threading.Lock()
        self._call_count = 0

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        del candidate, opponent, game_config, timeout_seconds
        with self._lock:
            self._call_count += 1
            call_count = self._call_count
            self.calls.append(task)
        if call_count <= 2:
            self._first_batch.wait(timeout=5)
            raise PairExecutionError("pair_output", "concurrent injected failure", ["game"])
        raise KeyboardInterrupt


def test_foreground_fake_run_has_common_blocks_and_rebuildable_report(tmp_path: Path) -> None:
    target = FakeTarget()
    run_dir = tmp_path / "run"
    run_foreground(
        RunOptions(
            _fake_binary(tmp_path),
            run_dir,
            objective_file=_objective(tmp_path),
            task_seed=9,
            cohort_size=4,
            finalists=1,
            bootstrap_candidates=2,
            random_reserve_candidates=1,
            tuning_pairs=2,
            tuning_pair_budget=16,
            validation_pair_budget=2,
            production_validation_pairs=2,
            tuning_effort=SearchEffort("iterations", 3),
            validation_effort=SearchEffort("iterations", 5),
            production_effort=SearchEffort("iterations", 9),
        ),
        target,
        model_proposer=FakeModel(),
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = [json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["type"] == "run_completed"
    assert not {item["seed"] for item in manifest["corpora"]["tuning"]["cases"]} & {
        item["seed"] for item in manifest["corpora"]["production_validation"]["cases"]
    }
    tuning_starts = [
        event["payload"]
        for event in events
        if event["type"] == "pair_started" and event["payload"]["phase"] == "tuning"
    ]
    assert [item["search_effort"] for item in tuning_starts] == [
        {"kind": "iterations", "value": 3}
    ] * 14
    report = (run_dir / "report.json").read_bytes()
    opponent_analysis = json.loads(report)["opponent_response_analysis"]
    assert opponent_analysis["scope"] == {
        "phase": "tuning",
        "cohort_index": 1,
        "prefix_id": manifest["prefixes"]["tuning"]["prefix_id"],
        "opponent_ids": ["schema-default", "historical"],
        "interval_method": "hoeffding_pair_bound_v1",
        "interaction_rule": "opposite-paired-hoeffding-relations-v1",
    }
    state = replay(
        read_manifest(run_dir / "manifest.json"), read_events(run_dir / "evidence.jsonl")
    )
    assert state.completed_cohorts
    assert [item["candidate_id"] for item in opponent_analysis["candidates"]] == [
        item.candidate_id for item in state.completed_cohorts[-1].candidates
    ]
    assert all(
        [item["opponent_id"] for item in candidate["opponent_responses"]]
        == opponent_analysis["scope"]["opponent_ids"]
        for candidate in opponent_analysis["candidates"]
    )
    write_report(run_dir)
    assert (run_dir / "report.json").read_bytes() == report


def test_validation_claim_depends_only_on_iteration_budgets(tmp_path: Path) -> None:
    run_dir = tmp_path / "production"
    run_foreground(
        RunOptions(
            _fake_binary(tmp_path),
            run_dir,
            objective_file=_objective(tmp_path),
            task_seed=9,
            cohort_size=4,
            finalists=1,
            bootstrap_candidates=2,
            random_reserve_candidates=1,
            tuning_pairs=2,
            tuning_pair_budget=16,
            validation_pair_budget=2,
            production_validation_pairs=2,
            tuning_effort=SearchEffort("iterations", 3),
            validation_effort=SearchEffort("iterations", 5),
            production_effort=SearchEffort("iterations", 5),
        ),
        FakeTarget(),
        model_proposer=FakeModel(),
    )
    assert (
        json.loads((run_dir / "report.json").read_text())["validation_claim"]["claim"]
        == "production"
    )


def test_parallel_workers_preserve_scientific_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tuner_cli.run.os.cpu_count", lambda: 2)
    options = _budgeted_options(tmp_path, 14, run_name="sequential")
    run_foreground(options, FakeTarget(), model_proposer=FakeModel())
    parallel = replace(options, run_dir=tmp_path / "parallel", evaluator_workers=2)
    run_foreground(parallel, FakeTarget(), model_proposer=FakeModel())
    sequential_events = read_events(options.run_dir / "evidence.jsonl")
    parallel_events = read_events(parallel.run_dir / "evidence.jsonl")
    sequential_projection = scientific_projection(sequential_events)
    parallel_projection = scientific_projection(parallel_events)
    sequential_fingerprint = json.loads((options.run_dir / "manifest.json").read_text())[
        "fingerprint"
    ]
    parallel_fingerprint = json.loads((parallel.run_dir / "manifest.json").read_text())[
        "fingerprint"
    ]
    assert sequential_projection.replace(sequential_fingerprint, "<fingerprint>") == (
        parallel_projection.replace(parallel_fingerprint, "<fingerprint>")
    )
    first_completed = next(
        index for index, event in enumerate(parallel_events) if event.type == "pair_completed"
    )
    assert sum(event.type == "pair_started" for event in parallel_events[:first_completed]) >= 2


def test_concurrent_pair_failures_fold_resume_and_rebuild_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tuner_cli.run.os.cpu_count", lambda: 2)
    options = replace(
        _budgeted_options(tmp_path, 14), cohort_size=5, bootstrap_candidates=3, evaluator_workers=2
    )
    interrupted = ConcurrentFailuresThenInterruptTarget()
    with pytest.raises(KeyboardInterrupt):
        run_foreground(options, interrupted, model_proposer=FakeModel())

    manifest = read_manifest(options.run_dir / "manifest.json")
    events = read_events(options.run_dir / "evidence.jsonl")
    first_failures = [event for event in events if isinstance(event.payload, PairFailedPayload)]
    assert len(first_failures) == 2
    first_failure_index = events.index(first_failures[0])
    start_events = [
        event
        for event in events[:first_failure_index]
        if isinstance(event.payload, PairStartedPayload)
    ]
    starts = [event.payload for event in start_events]
    assert len(starts) == 2
    assert len({start.identity.pair_id for start in starts}) == 2
    assert [failure.payload.identity.pair_id for failure in first_failures] == [
        start.identity.pair_id for start in starts
    ]
    # This directly exercises the retained two-failure prefix before any
    # healthy target work can mask a replay failure.
    replay(manifest, events)
    with pytest.raises(ValueError, match="started attempt"):
        replay(manifest, [event for event in events if event is not start_events[1]])
    with pytest.raises(ValueError, match="started attempt"):
        replay(manifest, [*events, first_failures[0], first_failures[0]])
    with pytest.raises(ValueError, match="attempt limit"):
        replay(manifest, [*events, start_events[0]])
    wrong_phase = replace(
        first_failures[0].payload,
        identity=replace(first_failures[0].payload.identity, phase="validation"),
    )
    tampered_events = [*events]
    tampered_events[first_failure_index] = replace(first_failures[0], payload=wrong_phase)
    with pytest.raises(ValueError, match="does not match"):
        replay(manifest, tampered_events)

    healthy = FakeTarget()
    run_foreground(replace(options, resume=True), healthy, model_proposer=FakeModel())
    completed_events = read_events(options.run_dir / "evidence.jsonl")
    report = json.loads((options.run_dir / "report.json").read_text())
    lifecycle = report["candidate_lifecycle"]
    failed_candidates = lifecycle["failed_candidates"]
    assert len(failed_candidates) == 2
    assert all(item["started_attempts"] == 2 for item in failed_candidates)
    assert all(item["failed_attempts"] == 1 for item in failed_candidates)
    assert all(item["censored_attempts"] == 1 for item in failed_candidates)
    assert {item["failed_candidate_id"] for item in lifecycle["accepted_replacements"]} == {
        item["candidate_id"] for item in failed_candidates
    }
    terminal_indices = {
        event.payload.candidate_id: index
        for index, event in enumerate(completed_events)
        if event.type == "candidate_failed"
    }
    assert terminal_indices
    for candidate_id, index in terminal_indices.items():
        assert all(
            event.payload.identity.candidate_id != candidate_id
            for event in completed_events[index + 1 :]
            if isinstance(event.payload, PairStartedPayload)
        )
    assert report["compute"]["tuning"]["failed_attempts"] == 2
    assert report["compute"]["tuning"]["censored_attempts"] == 2

    report_bytes = (options.run_dir / "report.json").read_bytes()
    (options.run_dir / "report.json").unlink()
    recording = FakeTarget()
    run_foreground(replace(options, resume=True), recording, model_proposer=FakeModel())
    assert not recording.calls
    assert (options.run_dir / "report.json").read_bytes() == report_bytes


def test_interrupted_pair_resumes_to_the_same_scientific_artifact(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    options = RunOptions(
        binary,
        tmp_path / "control" / "run",
        objective_file=_objective(tmp_path),
        task_seed=9,
        cohort_size=4,
        finalists=1,
        bootstrap_candidates=2,
        random_reserve_candidates=1,
        tuning_pairs=2,
        tuning_pair_budget=16,
        validation_pair_budget=2,
        production_validation_pairs=2,
        tuning_effort=SearchEffort("iterations", 3),
        validation_effort=SearchEffort("iterations", 5),
        production_effort=SearchEffort("iterations", 9),
    )
    run_foreground(options, FakeTarget(), model_proposer=FakeModel())
    interrupted = InterruptingTarget(interrupt_on_call=2)
    resumed_dir = tmp_path / "resumed" / "run"
    try:
        run_foreground(
            replace(options, run_dir=resumed_dir), interrupted, model_proposer=FakeModel()
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the injected interruption should escape the foreground run")
    before = list(interrupted.calls)
    run_foreground(
        replace(options, run_dir=resumed_dir, resume=True), interrupted, model_proposer=FakeModel()
    )
    control_events = read_events(options.run_dir / "evidence.jsonl")
    resumed_events = read_events(resumed_dir / "evidence.jsonl")
    assert scientific_projection(control_events) == scientific_projection(resumed_events)
    # The scientific projection, selection, and validation rows remain equal;
    # only the resumed compute ledger records the extra censored attempt.
    control_report = json.loads((options.run_dir / "report.json").read_text())
    resumed_report = json.loads((resumed_dir / "report.json").read_text())
    del control_report["compute"]
    del resumed_report["compute"]
    assert control_report == resumed_report
    resumed_compute = json.loads((resumed_dir / "report.json").read_text())["compute"]
    control_compute = json.loads((options.run_dir / "report.json").read_text())["compute"]
    assert control_compute["tuning"]["pair_attempts"] == 14
    assert control_compute["tuning"]["censored_attempts"] == 0
    assert resumed_compute["tuning"]["pair_attempts"] == 15
    assert resumed_compute["tuning"]["completed_pairs"] == 14
    assert resumed_compute["tuning"]["censored_attempts"] == 1
    assert resumed_compute["tuning"]["unspent_pair_attempts"] == 1
    assert resumed_compute["tuning"]["overrun_pair_attempts"] == 0
    completed = [
        event.payload.identity.pair_id
        for event in resumed_events
        if isinstance(event.payload, PairCompletedPayload)
    ]
    assert len(completed) == len(set(completed))
    assert interrupted.calls[len(before)].pair_id == before[-1].pair_id
    report = (resumed_dir / "report.json").read_bytes()
    (resumed_dir / "report.json").unlink()
    completed_target = FakeTarget()
    run_foreground(
        replace(options, run_dir=resumed_dir, resume=True),
        completed_target,
        model_proposer=FakeModel(),
    )
    assert not completed_target.calls
    assert (resumed_dir / "report.json").read_bytes() == report


def test_fresh_run_tolerates_the_launcher_created_wrapper_directory(tmp_path: Path) -> None:
    # The detached launcher must create the run directory before it can
    # redirect the child's stdout/stderr into it, so a fresh run legitimately
    # starts with launch.out / launch.err already present.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "launch.out").write_text("", encoding="utf-8")
    (run_dir / "launch.err").write_text("", encoding="utf-8")
    options = RunOptions(
        _fake_binary(tmp_path),
        run_dir,
        objective_file=_objective(tmp_path),
        task_seed=9,
        cohort_size=4,
        finalists=1,
        bootstrap_candidates=2,
        random_reserve_candidates=1,
        tuning_pairs=2,
        tuning_pair_budget=16,
        validation_pair_budget=2,
        production_validation_pairs=2,
        tuning_effort=SearchEffort("iterations", 3),
        validation_effort=SearchEffort("iterations", 5),
        production_effort=SearchEffort("iterations", 9),
    )
    run_foreground(options, FakeTarget(), model_proposer=FakeModel())
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "report.json").is_file()


def test_fresh_run_still_refuses_a_populated_run_directory(tmp_path: Path) -> None:
    from tuner_cli.run import validate_options

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "launch.err").write_text("", encoding="utf-8")
    (run_dir / "evidence.jsonl").write_text("", encoding="utf-8")
    options = RunOptions(
        _fake_binary(tmp_path),
        run_dir,
        objective_file=_objective(tmp_path),
        tuning_effort=SearchEffort("iterations", 3),
        validation_effort=SearchEffort("iterations", 5),
        production_effort=SearchEffort("iterations", 9),
    )
    with pytest.raises(ValueError, match="already exists"):
        validate_options(options)


def _budgeted_options(
    tmp_path: Path,
    tuning_pair_budget: int,
    *,
    finalists: int = 1,
    validation_pair_budget: int = 2,
    run_name: str = "run",
) -> RunOptions:
    return RunOptions(
        _fake_binary(tmp_path),
        tmp_path / run_name,
        objective_file=_objective(tmp_path),
        task_seed=9,
        cohort_size=4,
        finalists=finalists,
        bootstrap_candidates=2,
        random_reserve_candidates=1,
        tuning_pairs=2,
        tuning_pair_budget=tuning_pair_budget,
        validation_pair_budget=validation_pair_budget,
        production_validation_pairs=2,
        tuning_effort=SearchEffort("iterations", 3),
        validation_effort=SearchEffort("iterations", 5),
        production_effort=SearchEffort("iterations", 9),
    )


def _run_and_load(options: RunOptions, target: FakeTarget | None = None) -> tuple[Path, list, dict]:
    run_dir = options.run_dir
    run_foreground(options, target or FakeTarget(), model_proposer=FakeModel())
    events = [json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()]
    report = json.loads((run_dir / "report.json").read_text())
    return run_dir, events, report


def _completed_cohorts(events: list) -> list:
    return [event["payload"] for event in events if event["type"] == "cohort_completed"]


def _active_options(
    tmp_path: Path,
    run_name: str,
    tuning_pairs: int,
    tuning_pair_budget: int,
    audit: float,
    *,
    shadow_policy: str = "paired_bootstrap",
    shadow_halving_spare_margin: float = 0.0,
) -> RunOptions:
    return RunOptions(
        _fake_binary(tmp_path),
        tmp_path / run_name,
        objective_file=_objective(tmp_path),
        seed=42,
        task_seed=43,
        cohort_size=4,
        finalists=1,
        bootstrap_candidates=2,
        random_reserve_candidates=1,
        tuning_pairs=tuning_pairs,
        tuning_pair_budget=tuning_pair_budget,
        validation_pair_budget=2,
        production_validation_pairs=2,
        tuning_effort=SearchEffort("iterations", 3),
        validation_effort=SearchEffort("iterations", 5),
        production_effort=SearchEffort("iterations", 9),
        active_elimination_audit_probability=audit,
        shadow_policy=shadow_policy,  # type: ignore[arg-type]
        shadow_halving_spare_margin=shadow_halving_spare_margin,
    )


def _mock_eliminating_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    def decide(manifest, state, cohort_index, prefix):  # type: ignore[no-untyped-def]
        candidates = current_active_candidates(state)
        observations = comparable_prefix_observations(state.observations, candidates, prefix)
        boundary = next(
            (item for item in candidates if json.loads(item.canonical_config)["algorithm"] == "c"),
            candidates[0],
        )
        return ShadowRaceDecision(
            cohort_index,
            prefix.prefix_id,
            tuple(item.observation_id for item in observations),
            boundary.candidate_id,
            tuple(
                ShadowCandidateDecision(
                    item.candidate_id,
                    "continue" if item == boundary else "eliminate",
                    PairedBootstrapEvidence(4096 if item == boundary else 0, 4096),
                )
                for item in candidates
            ),
            "paired_bootstrap",
            manifest.shadow_policy.method_version,
        )

    monkeypatch.setattr("tuner_cli.continuation.decide_shadow_race", decide)
    monkeypatch.setattr("tuner_cli.replay.decide_shadow_race", decide)


def test_active_prune_completes_survivor_cohort_and_saves_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_eliminating_shadow(monkeypatch)
    options = _active_options(tmp_path, "active-prune", 14, 56, 0.01)
    run_foreground(options, ActiveProfileTarget(), model_proposer=ActiveProfileModel())
    events = [
        json.loads(line) for line in (options.run_dir / "evidence.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((options.run_dir / "manifest.json").read_text())
    report = json.loads((options.run_dir / "report.json").read_text())
    allocation = next(
        event["payload"]["allocation"]
        for event in events
        if event["type"] == "allocation_decided"
        and event["payload"]["allocation"]["kind"] == "apply_elimination"
    )
    pruned = {
        action["candidate_id"] for action in allocation["actions"] if action["action"] == "prune"
    }
    assert pruned
    cohort = _completed_cohorts(events)[0]
    survivors = set(cohort["candidate_ids"])
    assert len(survivors) < 4
    assert len(survivors) >= 1
    assert survivors.isdisjoint(pruned)
    assert all(
        event["payload"]["candidate_id"] not in pruned
        for event in events
        if event["type"] == "pair_started"
        and event["payload"]["phase"] == "tuning"
        and event["payload"]["task_id"]
        not in {item["task_id"] for item in manifest["corpora"]["tuning"]["cases"][:12]}
    )
    finalists = next(
        event["payload"]["finalist_ids"]
        for event in events
        if event["type"] == "finalists_selected"
    )
    assert set(finalists) <= survivors
    assert events[-1]["type"] == "run_completed"
    assert report["compute"]["tuning"]["pair_attempts"] < 4 * 14
    assert report["active_elimination"]["summary"]["pruned"] == len(pruned)
    report_bytes = (options.run_dir / "report.json").read_bytes()
    (options.run_dir / "report.json").unlink()
    recording = ActiveProfileTarget()
    run_foreground(replace(options, resume=True), recording, model_proposer=ActiveProfileModel())
    assert not recording.calls
    assert (options.run_dir / "report.json").read_bytes() == report_bytes


def test_audited_reversal_suspends_later_foreground_cohorts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_eliminating_shadow(monkeypatch)
    options = _active_options(tmp_path, "active-recovery", 16, 112, 0.999999)
    run_foreground(options, ActiveProfileTarget(recovery=True), model_proposer=ActiveProfileModel())
    events = [
        json.loads(line) for line in (options.run_dir / "evidence.jsonl").read_text().splitlines()
    ]
    report = json.loads((options.run_dir / "report.json").read_text())
    actions = [
        action
        for event in events
        if event["type"] == "allocation_decided"
        and event["payload"]["allocation"]["kind"] == "apply_elimination"
        for action in event["payload"]["allocation"]["actions"]
    ]
    audited = {action["candidate_id"] for action in actions if action["action"] == "audit_continue"}
    assert audited
    first_cohort = _completed_cohorts(events)[0]
    assert audited <= set(first_cohort["candidate_ids"])
    suspension = next(
        event["payload"]["allocation"]
        for event in events
        if event["type"] == "allocation_decided"
        and event["payload"]["allocation"]["kind"] == "suspend_active_elimination"
    )
    assert suspension["after_cohort_index"] == 0
    assert set(suspension["triggering_candidate_ids"]) <= audited
    assert len(_completed_cohorts(events)) == 2
    assert all(
        event["payload"]["allocation"]["cohort_index"] == 0
        for event in events
        if event["type"] == "allocation_decided"
        and event["payload"]["allocation"]["kind"] == "apply_elimination"
    )
    active = report["active_elimination"]
    assert active["suspended"] is True
    assert active["active_interval"]["last_cohort_index"] == 0
    assert active["audited_boundary_reversals"]
    report_bytes = (options.run_dir / "report.json").read_bytes()
    (options.run_dir / "report.json").unlink()
    recording = ActiveProfileTarget(recovery=True)
    run_foreground(replace(options, resume=True), recording, model_proposer=ActiveProfileModel())
    assert not recording.calls
    assert (options.run_dir / "report.json").read_bytes() == report_bytes


def _mock_halving_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    def decide(manifest, state, cohort_index, prefix):  # type: ignore[no-untyped-def]
        candidates = current_active_candidates(state)
        prior_eliminated = {
            item.candidate_id
            for race in state.shadow_races
            if race.cohort_index == cohort_index and race.policy_kind == "successive_halving"
            for item in race.decisions
            if item.disposition == "eliminate"
        }
        ranked = sorted(candidates, key=lambda c: json.loads(c.canonical_config)["algorithm"])
        target = (len(ranked) + 1) // 2
        rank_of = {c.candidate_id: index + 1 for index, c in enumerate(ranked)}
        kept = {c.candidate_id for c in ranked[:target]}
        decisions = tuple(
            ShadowCandidateDecision(
                c.candidate_id,
                "continue" if c.candidate_id in kept else "eliminate",
                SuccessiveHalvingEvidence(
                    rank_of[c.candidate_id] if c.candidate_id not in prior_eliminated else None,
                    len(ranked),
                    target,
                    c.candidate_id not in kept and c.candidate_id not in prior_eliminated,
                ),
            )
            for c in candidates
        )
        return ShadowRaceDecision(
            cohort_index,
            prefix.prefix_id,
            tuple(
                item.observation_id
                for item in comparable_prefix_observations(state.observations, candidates, prefix)
            ),
            ranked[target - 1].candidate_id,
            decisions,
            "successive_halving",
            "successive-halving-spare-near-tie-v1",
        )

    monkeypatch.setattr("tuner_cli.continuation.decide_shadow_race", decide)
    monkeypatch.setattr("tuner_cli.replay.decide_shadow_race", decide)


def test_active_halving_run_completes_and_resumes_with_rank_tagged_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_halving_shadow(monkeypatch)
    options = _active_options(
        tmp_path,
        "active-halving",
        14,
        56,
        0.5,
        shadow_policy="successive_halving",
        shadow_halving_spare_margin=0.1,
    )
    run_foreground(options, ActiveProfileTarget(), model_proposer=ActiveProfileModel())
    events = [
        json.loads(line) for line in (options.run_dir / "evidence.jsonl").read_text().splitlines()
    ]
    actions = [
        action
        for event in events
        if event["type"] == "allocation_decided"
        and event["payload"]["allocation"]["kind"] == "apply_elimination"
        for action in event["payload"]["allocation"]["actions"]
    ]
    assert actions
    assert all(action["margin"]["kind"] == "successive_halving_rank" for action in actions)
    assert events[-1]["type"] == "run_completed"
    report_bytes = (options.run_dir / "report.json").read_bytes()
    (options.run_dir / "report.json").unlink()
    recording = ActiveProfileTarget()
    run_foreground(replace(options, resume=True), recording, model_proposer=ActiveProfileModel())
    assert not recording.calls
    assert (options.run_dir / "report.json").read_bytes() == report_bytes


def test_budget_admits_cohorts_at_exact_boundaries(tmp_path: Path) -> None:
    # With cohort_size=4, finalists=1, tuning_pairs=2: the initial cohort
    # costs 8 pairs and each challenger cohort costs (4-1)*2 = 6.
    for budget, expected_cohorts in ((8, 1), (14, 2), (20, 3)):
        _, events, _ = _run_and_load(_budgeted_options(tmp_path, budget, run_name=f"run-{budget}"))
        assert [c["cohort_index"] for c in _completed_cohorts(events)] == list(
            range(expected_cohorts)
        )
    # One pair less than the next challenger cohort begins validation and
    # reports the remainder.
    _, events, report = _run_and_load(_budgeted_options(tmp_path, 19, run_name="run-19"))
    assert [c["cohort_index"] for c in _completed_cohorts(events)] == [0, 1]
    assert report["compute"]["tuning"]["pair_attempts"] == 14
    assert report["compute"]["tuning"]["unspent_pair_attempts"] == 5
    assert report["compute"]["tuning"]["overrun_pair_attempts"] == 0


def test_repeated_cohorts_are_strict(tmp_path: Path) -> None:
    options = _budgeted_options(tmp_path, 20)
    _, events, report = _run_and_load(options)
    cohorts = _completed_cohorts(events)
    assert [c["cohort_index"] for c in cohorts] == [0, 1, 2]
    manifest = json.loads((options.run_dir / "manifest.json").read_text())
    challenger = manifest["proposer"]["challenger_source_schedule"]
    accepted = [e["payload"] for e in events if e["type"] == "proposal_accepted"]
    assert [p["source"] for p in accepted if p["cohort_index"] == 0] == manifest["proposer"][
        "source_schedule"
    ]
    for cohort_index in (1, 2):
        assert [p["source"] for p in accepted if p["cohort_index"] == cohort_index] == challenger
    retained = [
        e["payload"]["allocation"]
        for e in events
        if e["type"] == "allocation_decided"
        and e["payload"]["allocation"]["kind"] == "retain_elites"
    ]
    assert [a["cohort_index"] for a in retained] == [1, 2]
    # Each retention names exactly the top candidates of the latest completed
    # cohort; the next cohort runs with those elites and records them as its
    # own retained ids, leading its candidate list.
    assert cohorts[0]["retained_candidate_ids"] == []
    for allocation, cohort in zip(retained, cohorts[1:], strict=True):
        assert cohort["retained_candidate_ids"] == allocation["candidate_ids"]
        assert (
            cohort["candidate_ids"][: len(allocation["candidate_ids"])]
            == allocation["candidate_ids"]
        )
    # No pair is ever repeated, and every admitted challenger reaches the full
    # tuning prefix.
    pair_ids = [
        e["payload"]["pair_id"]
        for e in events
        if e["type"] == "pair_completed" and e["payload"]["phase"] == "tuning"
    ]
    assert len(pair_ids) == len(set(pair_ids)) == 20
    assert report["compute"]["tuning"]["pair_attempts"] == 20
    assert report["compute"]["tuning"]["unspent_pair_attempts"] == 0


def test_proposer_remembers_prior_candidates(tmp_path: Path) -> None:
    options = _budgeted_options(tmp_path, 20)
    _, events, report = _run_and_load(options)
    manifest = json.loads((options.run_dir / "manifest.json").read_text())
    block0_prefix = manifest["tuning_blocks"][0]["prefix"]["prefix_id"]
    observation_prefixes = {
        e["payload"]["observation_id"]: e["payload"]["prefix_id"]
        for e in events
        if e["type"] == "observation_completed"
    }
    observation_phases = {
        e["payload"]["observation_id"]: e["payload"]["phase"]
        for e in events
        if e["type"] == "observation_completed"
    }
    proposals = [e["payload"] for e in events if e["type"] == "proposal_created"]
    # Every frontier observation is an exact tuning block-0 observation;
    # validation and deeper prefixes never enter the frontier.
    for proposal in proposals:
        for observation_id in proposal["frontier_observation_ids"]:
            assert observation_phases[observation_id] == "tuning"
            assert observation_prefixes[observation_id] == block0_prefix
    # The first proposal of each later cohort binds every globally accepted
    # candidate's block-0 observation, including prior non-elites.
    first_of_cohort = {p["cohort_index"]: p for p in proposals if p["cohort_slot"] == 0}
    assert len(first_of_cohort[1]["frontier_observation_ids"]) == 4
    assert len(first_of_cohort[2]["frontier_observation_ids"]) == 7
    # Later proposals add earlier active challengers once their block-0
    # observations exist.
    cohort1 = [p for p in proposals if p["cohort_index"] == 1]
    assert [len(p["frontier_observation_ids"]) for p in cohort1] == [4, 5, 6]
    assert report["proposal_search"]["configured"]["cohorts"] == 3


def test_validation_budget_is_total(tmp_path: Path) -> None:
    # Two finalists share one total budget of 4: each receives exactly
    # 4 / 2 = 2 common held-out pairs.
    options = _budgeted_options(tmp_path, 12, finalists=2, validation_pair_budget=4)
    _, events, report = _run_and_load(options)
    manifest = json.loads((options.run_dir / "manifest.json").read_text())
    assert manifest["compute_budget"]["validation_pair_attempts"] == 4
    assert manifest["prefixes"]["validation"]["length"] == 2
    validation_pairs = [
        e["payload"]["candidate_id"]
        for e in events
        if e["type"] == "pair_completed" and e["payload"]["phase"] == "validation"
    ]
    assert len(validation_pairs) == 4
    for finalist_id in report["selection"]["finalist_ids"]:
        assert validation_pairs.count(finalist_id) == 2
    for entry in report["validation_order"]:
        assert entry["context"]["prefix_id"] == manifest["prefixes"]["validation"]["prefix_id"]
        assert entry["pairs"] == 2


def test_failed_attempt_consumes_budget_and_reports_overrun(tmp_path: Path) -> None:
    options = _budgeted_options(tmp_path, 14)
    failing = FailingTarget(fail_on_call=14)
    run_foreground(options, failing, model_proposer=FakeModel())
    report = json.loads((options.run_dir / "report.json").read_text())
    events = [
        json.loads(line) for line in (options.run_dir / "evidence.jsonl").read_text().splitlines()
    ]
    assert [c["cohort_index"] for c in _completed_cohorts(events)] == [0, 1]
    tuning = report["compute"]["tuning"]
    assert tuning["pair_attempts"] == 15
    assert tuning["completed_pairs"] == 14
    assert tuning["failed_attempts"] == 1
    assert tuning["censored_attempts"] == 0
    assert tuning["overrun_pair_attempts"] == 1
    assert tuning["unspent_pair_attempts"] == 0
    # The failed attempt never became a synthetic loss or a completed pair.
    assert report["compute"]["validation"]["pair_attempts"] == 2
    assert report["status"] == "complete"


def test_tampered_allocation_is_rejected_by_replay(tmp_path: Path) -> None:
    options = _budgeted_options(tmp_path, 14)
    run_dir, events, _ = _run_and_load(options)
    lines = (run_dir / "evidence.jsonl").read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if '"kind":"retain_elites"' in line)
    tampered = json.loads(lines[index])
    tampered["payload"]["policy_version"] = "tampered-v0"
    lines[index] = json.dumps(tampered)
    (run_dir / "evidence.jsonl").write_text("\n".join(lines) + "\n")
    manifest = read_manifest(run_dir / "manifest.json")
    with pytest.raises(ValueError):
        replay(manifest, read_events(run_dir / "evidence.jsonl"))


def test_budget_options_are_validated_up_front(tmp_path: Path) -> None:
    def _options(**overrides: object) -> RunOptions:
        base = _budgeted_options(tmp_path, 20, run_name="check")
        return replace(base, **overrides)  # type: ignore[arg-type]

    for bad in (
        _options(validation_pair_budget=3),
        _options(validation_pair_budget=0),
        _options(tuning_pair_budget=7),
        _options(tuning_pair_budget=True),
    ):
        with pytest.raises(ValueError):
            run_foreground(bad, FakeTarget(), model_proposer=FakeModel())


class SizedTarget(FakeTarget):
    """A fake game that advertises a bounded board ``size`` setup axis."""

    def __init__(self) -> None:
        super().__init__()
        self.game_configs: list[str] = []

    def describe(self) -> dict[str, object]:
        described = super().describe()
        schema = {
            "parameters": [{"name": "size", "type": "int", "bounds": [3, 19], "default": 5}],
            "conditions": [],
        }
        described["config_schema"] = schema
        tuning = described["tuning"]
        assert isinstance(tuning, dict)
        tuning["game_config_schema"] = schema
        return described

    def validate(self, candidates, opponent, game_config):  # type: ignore[no-untyped-def]
        self.game_configs.append(game_config)
        return super().validate(candidates, opponent, game_config)

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        self.game_configs.append(game_config)
        return super().evaluate(task, candidate, opponent, game_config, timeout_seconds)


def _sized_objective(tmp_path: Path, size: int | None) -> Path:
    path = tmp_path / "sized-objective.json"
    body = json.loads(_objective(tmp_path).read_text())
    if size is not None:
        body["game_config"] = {"size": size}
    path.write_text(json.dumps(body))
    return path


def test_game_config_override_threads_into_the_run(tmp_path: Path) -> None:
    target = SizedTarget()
    options = replace(
        _budgeted_options(tmp_path, 20, run_name="sized"),
        objective_file=_sized_objective(tmp_path, 9),
    )
    run_foreground(options, target, model_proposer=FakeModel())

    manifest = read_manifest(options.run_dir / "manifest.json")
    assert manifest.game_config == '{"size":9}'
    assert manifest.game_config_fingerprint == fingerprint({"size": 9})
    assert target.game_configs and set(target.game_configs) == {'{"size":9}'}

    # Editing the objective's game_config makes the frozen run incompatible.
    _sized_objective(tmp_path, 7)
    with pytest.raises(ValueError, match="differs from manifest"):
        run_foreground(replace(options, resume=True), target, model_proposer=FakeModel())


def test_worker_count_is_validated_before_creating_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tuner_cli.run.os.cpu_count", lambda: 1)
    options = _budgeted_options(tmp_path, 20, run_name="workers")
    with pytest.raises(ValueError):
        run_foreground(
            replace(options, evaluator_workers=2), FakeTarget(), model_proposer=FakeModel()
        )
    assert not options.run_dir.exists()


_CONSTRAINT = {"algorithm": {"choices": ["a", "b", "c", "d", "f", "g"]}}


def _built_manifest(options: RunOptions) -> tuple[dict, object]:
    """The manifest ``run_foreground`` would freeze for ``options`` -- built
    through the same authorities but without playing a game."""
    from tuner_cli.objective import resolve_objective
    from tuner_cli.run import game_spec, manifest_for, schema_default

    binary = options.game_binary.expanduser().resolve()
    spec = game_spec(FakeTarget(), binary)
    assert options.objective_file is not None
    objective = resolve_objective(
        options.objective_file,
        spec.kind,
        schema_default(spec, options.seed),
        spec.game_config_schema,
        spec.default_game_config,
    )
    manifest = manifest_for(options, options.run_dir, spec, objective)
    from tuner_cli.artifacts import manifest_json

    return manifest_json(manifest), manifest


def test_constraints_change_the_epoch_and_manifest(tmp_path: Path) -> None:
    plain = _budgeted_options(tmp_path, 16, run_name="plain")
    constrained = replace(
        _budgeted_options(tmp_path, 16, run_name="constrained"),
        constraints=decode_constraints(_CONSTRAINT),
    )
    plain_json, plain_manifest = _built_manifest(plain)
    constrained_json, constrained_manifest = _built_manifest(constrained)

    assert plain_manifest.constraints == ()
    assert encode_constraints(constrained_manifest.constraints) == [{"set": _CONSTRAINT}]
    assert constrained_manifest.epoch.fingerprint != plain_manifest.epoch.fingerprint
    assert constrained_json["constraints"] == [{"set": _CONSTRAINT}]
    assert "constraints" not in plain_json


def test_resume_rejects_a_constraint_change(tmp_path: Path) -> None:
    options = replace(
        _budgeted_options(tmp_path, 16, run_name="constrained"),
        constraints=decode_constraints(_CONSTRAINT),
    )
    frozen, _ = _built_manifest(options)
    widened = replace(
        options, constraints=decode_constraints({"algorithm": {"choices": ["a", "b"]}})
    )
    rebuilt, _ = _built_manifest(widened)
    assert frozen["fingerprint"] != rebuilt["fingerprint"]
