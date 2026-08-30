from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tuner_cli.artifacts import read_manifest
from tuner_cli.domain import (
    GameResult,
    ModelAttempt,
    ModelObservation,
    ObservationFrontier,
    PairResult,
    PairTask,
    ProposedConfiguration,
    StrategyMetrics,
    ValidationResult,
)
from tuner_cli.event_payloads import PairCompletedPayload
from tuner_cli.evidence import read_events, scientific_projection
from tuner_cli.identity import candidate_from_config, canonical_json, game_id
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
                        "config": {"source": "inline", "value": {"family": "b"}},
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
                        "name": "family",
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

    def evaluate(self, task, candidate, opponent, game_config, timeout_seconds):  # type: ignore[no-untyped-def]
        self.calls.append(task)
        outcome = (
            "candidate_win" if json.loads(candidate.canonical_config)["family"] == "b" else "draw"
        )
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
    def ask(
        self,
        observations: tuple[ModelObservation, ...],
        frontier: ObservationFrontier,
        excluded_fingerprints: frozenset[str],
        attempt: ModelAttempt,
    ) -> ProposedConfiguration:
        del observations, frontier, excluded_fingerprints
        return ProposedConfiguration(
            candidate_from_config({"family": f"model-{attempt.source_attempt}"}), None
        )


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
            tuning_max_iterations=3,
            validation_max_iterations=5,
            production_max_iterations=9,
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
    assert [item["budget"] for item in tuning_starts] == [3] * 14
    report = (run_dir / "report.json").read_bytes()
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
            tuning_max_iterations=3,
            validation_max_iterations=5,
            production_max_iterations=5,
        ),
        FakeTarget(),
        model_proposer=FakeModel(),
    )
    assert (
        json.loads((run_dir / "report.json").read_text())["validation_claim"]["claim"]
        == "production"
    )


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
        tuning_max_iterations=3,
        validation_max_iterations=5,
        production_max_iterations=9,
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
        tuning_max_iterations=3,
        validation_max_iterations=5,
        production_max_iterations=9,
    )


def _run_and_load(options: RunOptions, target: FakeTarget | None = None) -> tuple[Path, list, dict]:
    run_dir = options.run_dir
    run_foreground(options, target or FakeTarget(), model_proposer=FakeModel())
    events = [json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()]
    report = json.loads((run_dir / "report.json").read_text())
    return run_dir, events, report


def _completed_cohorts(events: list) -> list:
    return [event["payload"] for event in events if event["type"] == "cohort_completed"]


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
    with pytest.raises(PairExecutionError):
        run_foreground(options, failing, model_proposer=FakeModel())
    resumed = FakeTarget()
    run_foreground(replace(options, resume=True), resumed, model_proposer=FakeModel())
    assert resumed.calls[0].pair_id == failing.calls[-1].pair_id
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
