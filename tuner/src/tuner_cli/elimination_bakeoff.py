"""Strict three-arm elimination bake-off: spec, matched cell expansion, orchestration.

The three arms continue all candidates (``no_elimination``), enforce the landed
all-strata audited paired policy (``paired_elimination``), or enforce the
gate-approved audited spare-near-tie successive-halving policy
(``spare_near_tie``). Every arm records paired shadow evidence; only the active
arms enforce it, and both do so under the same deterministic prospective audit
and automatic suspension. The bake-off never changes a disposition formula,
prefix, threshold, or survivor quota; it only compares complete systems at equal
declared compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .active_audit import build_active_audit
from .artifacts import Manifest, read_manifest
from .codec import (
    JsonObject,
    JsonValue,
    elements,
    integer,
    json_object,
    number,
    object_fields,
    strict_json,
    string,
    strings,
)
from .domain import Observation, ReplayState, SearchEffort
from .effort import decode_effort, encode_effort, exceeds_same_kind
from .elimination_bakeoff_metrics import EliminationChildFact, EliminationDecision, aggregate
from .evidence import read_events
from .identity import canonical_json, fingerprint
from .proposer import ModelProposer
from .replay import replay
from .run import RunOptions, run_foreground
from .target import Target

SCHEMA_VERSION = 1
POLICIES: tuple[str, str, str] = ("no_elimination", "paired_elimination", "spare_near_tie")

# Landed policy constants. The spec's gate block documents this authorization; it
# is not a filesystem parser for a planning document.
GATE_DOCUMENT_ID = "task-11-successive-halving-shadow-gate.md"
GATE_DECISION = "PASS"
AUTHORIZED_POLICY_VERSION = "successive-halving-spare-near-tie-v1"
ACTIVE_AUDIT_PROBABILITY = 0.25
SPARE_NEAR_TIE_MARGIN = 0.10
BASELINE_PROPOSER = "smac_mixed"

_SHARED_RUN_FIELDS = {
    "proposer_policy",
    "cohort_size",
    "finalists",
    "bootstrap_candidates",
    "random_reserve_candidates",
    "tuning_pairs",
    "validation_pair_budget",
    "production_validation_pairs",
    "diagnostic_pair_budget",
    "tuning_effort",
    "validation_effort",
    "production_effort",
    "excluded_families",
    "evaluator_workers",
    "pair_timeout_seconds",
    "active_audit_probability",
}
_DECISION_FIELDS = {"score_practical_margin", "recall_noninferiority_margin", "top_set_k"}
_GATE_FIELDS = {"document_id", "decision", "authorized_policy_version"}
_SPEC_FIELDS = {
    "schema_version",
    "experiment_id",
    "game_binary",
    "objective_file",
    "policies",
    "proposal_seeds",
    "task_seed",
    "tuning_pair_budgets",
    "shared_run",
    "decision",
    "gate",
}


@dataclass(frozen=True, slots=True)
class EliminationSharedRun:
    proposer_policy: Literal["smac_mixed"]
    cohort_size: int
    finalists: int
    bootstrap_candidates: int
    random_reserve_candidates: int
    tuning_pairs: int
    validation_pair_budget: int
    production_validation_pairs: int
    diagnostic_pair_budget: int
    tuning_effort: SearchEffort
    validation_effort: SearchEffort
    production_effort: SearchEffort
    excluded_families: tuple[str, ...]
    evaluator_workers: int
    pair_timeout_seconds: int
    active_audit_probability: float


@dataclass(frozen=True, slots=True)
class EliminationGate:
    document_id: str
    decision: str
    authorized_policy_version: str


@dataclass(frozen=True, slots=True)
class EliminationBakeoffSpec:
    experiment_id: str
    game_binary: Path
    objective_file: Path
    proposal_seeds: tuple[int, ...]
    task_seed: int
    tuning_pair_budgets: tuple[int, ...]
    shared_run: EliminationSharedRun
    decision: EliminationDecision
    gate: EliminationGate


@dataclass(frozen=True, slots=True)
class EliminationCell:
    cell_id: str
    budget: int
    seed: int
    policy: str
    run_dir: Path


def _positive_integers(value: object, label: str, *, minimum: int) -> tuple[int, ...]:
    items = elements(value, label)
    if len(items) < minimum:
        raise ValueError(f"{label} needs at least {minimum} entries")
    return tuple(integer(item, label, positive=True) for item in items)


def _decode_shared_run(value: object) -> EliminationSharedRun:
    item = object_fields(value, _SHARED_RUN_FIELDS, "elimination shared run")
    if item["proposer_policy"] != BASELINE_PROPOSER:
        raise ValueError("elimination bake-off requires the smac_mixed proposer")
    audit = number(item["active_audit_probability"], "active audit probability")
    if audit != ACTIVE_AUDIT_PROBABILITY:
        raise ValueError(f"active audit probability must be {ACTIVE_AUDIT_PROBABILITY}")
    diagnostic = integer(item["diagnostic_pair_budget"], "diagnostic pair budget")
    if diagnostic != 0:
        raise ValueError("elimination bake-off diagnostic pair budget must be zero")
    shared = EliminationSharedRun(
        BASELINE_PROPOSER,
        integer(item["cohort_size"], "cohort size", positive=True),
        integer(item["finalists"], "finalists", positive=True),
        integer(item["bootstrap_candidates"], "bootstrap candidates", positive=True),
        integer(item["random_reserve_candidates"], "random reserve candidates", positive=True),
        integer(item["tuning_pairs"], "tuning pairs", positive=True),
        integer(item["validation_pair_budget"], "validation pair budget", positive=True),
        integer(item["production_validation_pairs"], "production validation pairs", positive=True),
        diagnostic,
        decode_effort(item["tuning_effort"], "tuning effort"),
        decode_effort(item["validation_effort"], "validation effort"),
        decode_effort(item["production_effort"], "production effort"),
        strings(item["excluded_families"], "excluded families"),
        integer(item["evaluator_workers"], "evaluator workers", positive=True),
        integer(item["pair_timeout_seconds"], "pair timeout seconds", positive=True),
        audit,
    )
    if shared.validation_pair_budget % shared.finalists:
        raise ValueError("validation pair budget must divide finalists")
    if shared.validation_pair_budget // shared.finalists != shared.production_validation_pairs:
        raise ValueError("elimination bake-off must validate the full production corpus")
    if exceeds_same_kind(shared.validation_effort, shared.production_effort) or exceeds_same_kind(
        shared.production_effort, shared.validation_effort
    ):
        raise ValueError("elimination bake-off must validate at full production effort")
    return shared


def _decode_decision(value: object) -> EliminationDecision:
    item = object_fields(value, _DECISION_FIELDS, "elimination decision")
    return EliminationDecision(
        number(item["score_practical_margin"], "score practical margin"),
        number(item["recall_noninferiority_margin"], "recall noninferiority margin"),
        integer(item["top_set_k"], "top set k", positive=True),
    )


def _decode_gate(value: object) -> EliminationGate:
    item = object_fields(value, _GATE_FIELDS, "elimination gate")
    document_id = string(item["document_id"], "gate document id", nonempty=True)
    decision = string(item["decision"], "gate decision", nonempty=True)
    version = string(item["authorized_policy_version"], "authorized policy version", nonempty=True)
    if (
        document_id != GATE_DOCUMENT_ID
        or decision != GATE_DECISION
        or version != AUTHORIZED_POLICY_VERSION
    ):
        raise ValueError("elimination gate block does not match the landed authorization")
    return EliminationGate(document_id, decision, version)


def read_elimination_spec(path: Path) -> EliminationBakeoffSpec:
    raw = json_object(
        strict_json(path.read_text(), "elimination bake-off spec"), "elimination spec"
    )
    if set(raw) != _SPEC_FIELDS:
        raise ValueError("invalid elimination bake-off specification fields")
    if raw["schema_version"] != SCHEMA_VERSION or raw["policies"] != list(POLICIES):
        raise ValueError("unsupported elimination bake-off schema or policy order")
    seeds = _positive_integers(raw["proposal_seeds"], "proposal seeds", minimum=4)
    if len(set(seeds)) < 4:
        raise ValueError("elimination bake-off needs four distinct positive proposal seeds")
    budgets = _positive_integers(raw["tuning_pair_budgets"], "tuning pair budgets", minimum=2)
    if list(budgets) != sorted(budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("elimination bake-off needs strictly increasing tuning budgets")
    shared_run = _decode_shared_run(raw["shared_run"])
    decision = _decode_decision(raw["decision"])
    if decision.top_set_k > shared_run.finalists:
        raise ValueError("elimination bake-off top_set_k must not exceed finalists")
    return EliminationBakeoffSpec(
        string(raw["experiment_id"], "experiment id", nonempty=True),
        Path(string(raw["game_binary"], "game binary", nonempty=True)),
        Path(string(raw["objective_file"], "objective file", nonempty=True)),
        seeds,
        integer(raw["task_seed"], "task seed", positive=True),
        budgets,
        shared_run,
        decision,
        _decode_gate(raw["gate"]),
    )


def _encode_shared_run(shared: EliminationSharedRun) -> JsonObject:
    return {
        "proposer_policy": shared.proposer_policy,
        "cohort_size": shared.cohort_size,
        "finalists": shared.finalists,
        "bootstrap_candidates": shared.bootstrap_candidates,
        "random_reserve_candidates": shared.random_reserve_candidates,
        "tuning_pairs": shared.tuning_pairs,
        "validation_pair_budget": shared.validation_pair_budget,
        "production_validation_pairs": shared.production_validation_pairs,
        "diagnostic_pair_budget": shared.diagnostic_pair_budget,
        "tuning_effort": encode_effort(shared.tuning_effort),
        "validation_effort": encode_effort(shared.validation_effort),
        "production_effort": encode_effort(shared.production_effort),
        "excluded_families": list(shared.excluded_families),
        "evaluator_workers": shared.evaluator_workers,
        "pair_timeout_seconds": shared.pair_timeout_seconds,
        "active_audit_probability": shared.active_audit_probability,
    }


def _spec_json(spec: EliminationBakeoffSpec) -> JsonObject:
    return {
        "game_binary": str(spec.game_binary),
        "objective_file": str(spec.objective_file),
        "policies": list(POLICIES),
        "proposal_seeds": list(spec.proposal_seeds),
        "task_seed": spec.task_seed,
        "tuning_pair_budgets": list(spec.tuning_pair_budgets),
        "shared_run": _encode_shared_run(spec.shared_run),
        "decision": {
            "score_practical_margin": spec.decision.score_practical_margin,
            "recall_noninferiority_margin": spec.decision.recall_noninferiority_margin,
            "top_set_k": spec.decision.top_set_k,
        },
        "gate": {
            "document_id": spec.gate.document_id,
            "decision": spec.gate.decision,
            "authorized_policy_version": spec.gate.authorized_policy_version,
        },
    }


def _cell_summary(cell: EliminationCell) -> JsonObject:
    return {
        "cell_id": cell.cell_id,
        "budget": cell.budget,
        "seed": cell.seed,
        "policy": cell.policy,
    }


def encode_experiment(spec: EliminationBakeoffSpec, cells: list[JsonObject]) -> str:
    raw: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "kind": "elimination-bakeoff",
        "experiment_id": spec.experiment_id,
        "spec": _spec_json(spec),
        "cells": list(cells),
    }
    fingerprinted: JsonObject = {**raw, "fingerprint": fingerprint(raw)}
    return canonical_json(fingerprinted) + "\n"


def read_experiment(text: str) -> JsonObject:
    raw = json_object(strict_json(text, "experiment"), "experiment")
    if "fingerprint" not in raw or raw.get("kind") != "elimination-bakeoff":
        raise ValueError("invalid elimination experiment manifest")
    stored = raw["fingerprint"]
    body: JsonObject = {key: value for key, value in raw.items() if key != "fingerprint"}
    if stored != fingerprint(body):
        raise ValueError("elimination experiment fingerprint mismatch")
    return raw


def experiment_fingerprint(text: str) -> str:
    value = read_experiment(text)["fingerprint"]
    return string(value, "experiment fingerprint", nonempty=True)


def _cells(spec: EliminationBakeoffSpec, directory: Path) -> list[EliminationCell]:
    result: list[EliminationCell] = []
    for budget in spec.tuning_pair_budgets:
        for seed in spec.proposal_seeds:
            for policy in POLICIES:
                run_dir = directory / "runs" / policy / f"budget-{budget}" / f"seed-{seed}"
                result.append(
                    EliminationCell(f"{budget}:{seed}:{policy}", budget, seed, policy, run_dir)
                )
    return result


def _options(spec: EliminationBakeoffSpec, cell: EliminationCell) -> RunOptions:
    shared = spec.shared_run
    audit = None if cell.policy == "no_elimination" else shared.active_audit_probability
    shadow_policy: Literal["paired_bootstrap", "successive_halving"] = (
        "successive_halving" if cell.policy == "spare_near_tie" else "paired_bootstrap"
    )
    spare = SPARE_NEAR_TIE_MARGIN if cell.policy == "spare_near_tie" else 0.0
    return RunOptions(
        spec.game_binary,
        cell.run_dir,
        objective_file=spec.objective_file,
        seed=cell.seed,
        task_seed=spec.task_seed,
        cohort_size=shared.cohort_size,
        finalists=shared.finalists,
        bootstrap_candidates=shared.bootstrap_candidates,
        random_reserve_candidates=shared.random_reserve_candidates,
        tuning_pairs=shared.tuning_pairs,
        tuning_pair_budget=cell.budget,
        validation_pair_budget=shared.validation_pair_budget,
        diagnostic_pair_budget=shared.diagnostic_pair_budget,
        production_validation_pairs=shared.production_validation_pairs,
        tuning_effort=shared.tuning_effort,
        validation_effort=shared.validation_effort,
        production_effort=shared.production_effort,
        pair_timeout_seconds=shared.pair_timeout_seconds,
        evaluator_workers=shared.evaluator_workers,
        shadow_policy=shadow_policy,
        shadow_halving_spare_margin=spare,
        active_elimination_audit_probability=audit,
        exclude_family=shared.excluded_families,
        proposer_policy=shared.proposer_policy,
        resume=cell.run_dir.exists(),
    )


def _score_key(observation: Observation, fingerprints: dict[str, str]) -> tuple[float, str]:
    return observation.estimate.mean, fingerprints[observation.candidate_id]


def _int_field(summary: JsonValue, key: str) -> int:
    assert isinstance(summary, dict)
    value = summary[key]
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _float_field(summary: JsonValue, key: str) -> float:
    assert isinstance(summary, dict)
    value = summary[key]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


@dataclass(frozen=True, slots=True)
class _HeldOut:
    finalist_fingerprints: tuple[str, ...]
    best_candidate_fingerprint: str
    best_score: float
    means: tuple[tuple[str, float], ...]


def _held_out(state: ReplayState) -> _HeldOut:
    assert state.finalists is not None
    finalist_ids = {candidate.candidate_id for candidate in state.finalists}
    fingerprints = {candidate.candidate_id: candidate.fingerprint for candidate in state.finalists}
    observations = [
        item
        for item in state.observations
        if item.phase == "validation" and item.candidate_id in finalist_ids
    ]
    if not observations:
        raise ValueError("elimination bake-off child has no finalist held-out observations")
    means_by_candidate: dict[str, float] = {}
    for item in observations:
        current = means_by_candidate.get(item.candidate_id)
        if current is None or item.estimate.mean > current:
            means_by_candidate[item.candidate_id] = item.estimate.mean
    best = max(observations, key=lambda item: _score_key(item, fingerprints))
    return _HeldOut(
        tuple(sorted(fingerprints.values())),
        fingerprints[best.candidate_id],
        best.estimate.mean,
        tuple(sorted((fingerprints[cid], mean) for cid, mean in means_by_candidate.items())),
    )


@dataclass(frozen=True, slots=True)
class _ActiveFacts:
    nominal_eliminations: int = 0
    pruned: int = 0
    audit_continued: int = 0
    audited_boundary_reversals: int = 0
    estimated_boundary_reversals: float = 0.0
    gross_nominal_suffix_unique_pairs: int = 0
    audit_continuation_suffix_unique_pairs: int = 0
    planned_unique_pair_savings: int = 0
    suspended: bool = False


def _active_facts(manifest: Manifest, state: ReplayState) -> _ActiveFacts:
    if manifest.active_elimination is None:
        return _ActiveFacts()
    audit = build_active_audit(manifest, state)
    summary = audit["summary"]
    return _ActiveFacts(
        nominal_eliminations=_int_field(summary, "nominal_eliminations"),
        pruned=_int_field(summary, "pruned"),
        audit_continued=_int_field(summary, "audit_continued"),
        audited_boundary_reversals=_int_field(summary, "audited_boundary_reversals"),
        estimated_boundary_reversals=_float_field(summary, "estimated_boundary_reversals"),
        gross_nominal_suffix_unique_pairs=_int_field(summary, "gross_nominal_suffix_unique_pairs"),
        audit_continuation_suffix_unique_pairs=_int_field(
            summary, "audit_continuation_suffix_unique_pairs"
        ),
        planned_unique_pair_savings=_int_field(summary, "planned_unique_pair_savings"),
        suspended=bool(audit["suspended"]),
    )


def _child_fact(cell: EliminationCell) -> EliminationChildFact:
    manifest = read_manifest(cell.run_dir / "manifest.json")
    state = replay(manifest, read_events(cell.run_dir / "evidence.jsonl"))
    if state.terminal_status != "complete" or not state.finalists:
        raise ValueError("elimination bake-off child is incomplete")
    held_out = _held_out(state)
    active = _active_facts(manifest, state)
    accepted_unique = len(
        {
            candidate.fingerprint
            for cohort in state.completed_cohorts
            for candidate in cohort.candidates
        }
    )
    budget = manifest.compute_budget.tuning_pair_attempts
    attempts = state.compute.tuning.pair_attempts
    return EliminationChildFact(
        cell_id=cell.cell_id,
        budget=cell.budget,
        seed=cell.seed,
        policy=cell.policy,
        manifest_fingerprint=manifest.fingerprint,
        best_candidate_fingerprint=held_out.best_candidate_fingerprint,
        finalist_fingerprints=held_out.finalist_fingerprints,
        held_out_means=held_out.means,
        held_out_best_score=held_out.best_score,
        completed_cohorts=len(state.completed_cohorts),
        accepted_unique_candidates=accepted_unique,
        terminal_candidate_failures=len(state.candidate_failures),
        censored_tuning_attempts=state.compute.tuning.censored_attempts,
        tuning_pair_attempts=attempts,
        tuning_physical_games=state.compute.tuning.physical_games,
        tuning_search_iterations=state.compute.tuning.search_iterations,
        tuning_wall_time_ms=state.compute.tuning.wall_time_ms,
        unspent_pair_attempts=max(0, budget - attempts),
        overrun_pair_attempts=max(0, attempts - budget),
        nominal_eliminations=active.nominal_eliminations,
        pruned=active.pruned,
        audit_continued=active.audit_continued,
        audited_boundary_reversals=active.audited_boundary_reversals,
        estimated_boundary_reversals=active.estimated_boundary_reversals,
        gross_nominal_suffix_unique_pairs=active.gross_nominal_suffix_unique_pairs,
        audit_continuation_suffix_unique_pairs=active.audit_continuation_suffix_unique_pairs,
        planned_unique_pair_savings=active.planned_unique_pair_savings,
        suspended=active.suspended,
    )


def run_elimination_bakeoff(
    spec: EliminationBakeoffSpec,
    experiment_dir: Path,
    resume: bool = False,
    target: Target | None = None,
    model_proposer: ModelProposer | None = None,
) -> Path:
    cells = _cells(spec, experiment_dir)
    manifest_path = experiment_dir / "experiment.json"
    encoded = encode_experiment(spec, [_cell_summary(cell) for cell in cells])
    if manifest_path.exists():
        if manifest_path.read_text() != encoded:
            raise ValueError("existing experiment does not match strict specification")
    elif resume:
        raise ValueError("cannot resume without experiment.json")
    else:
        experiment_dir.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(encoded)
    for cell in cells:
        run_foreground(_options(spec, cell), target=target, model_proposer=model_proposer)
    fingerprint_value = experiment_fingerprint(manifest_path.read_text())
    facts = [_child_fact(cell) for cell in cells]
    results = aggregate(facts, fingerprint_value, spec.decision)
    (experiment_dir / "results.json").write_text(results)
    return experiment_dir / "results.json"
