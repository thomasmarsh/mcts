"""Strict version-one codecs for proposer bake-off inputs and projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .codec import (
    JsonObject,
    elements,
    integer,
    json_object,
    number,
    object_fields,
    string,
)
from .constraints import Constraints, decode_constraints, encode_constraints
from .domain import SearchEffort
from .effort import decode_effort, encode_effort
from .identity import canonical_json, fingerprint
from .proposer import POLICIES

BAKEOFF_BASELINE = "smac_mixed"
BAKEOFF_CHALLENGER = "irace_generational"

_SHARED_RUN_FIELDS = {
    "cohort_size",
    "finalists",
    "bootstrap_candidates",
    "random_reserve_candidates",
    "tuning_pairs",
    "validation_pair_budget",
    "production_validation_pairs",
    "tuning_effort",
    "validation_effort",
    "production_effort",
    "constraints",
    "evaluator_workers",
    "pair_timeout_seconds",
}
_DECISION_FIELDS = {
    "baseline",
    "challenger",
    "score_practical_margin",
    "recall_noninferiority_margin",
    "top_set_k",
}
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
}


@dataclass(frozen=True, slots=True)
class SharedRun:
    cohort_size: int
    finalists: int
    bootstrap_candidates: int
    random_reserve_candidates: int
    tuning_pairs: int
    validation_pair_budget: int
    production_validation_pairs: int
    tuning_effort: SearchEffort
    validation_effort: SearchEffort
    production_effort: SearchEffort
    constraints: Constraints
    evaluator_workers: int
    pair_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class BakeoffDecision:
    baseline: Literal["smac_mixed"]
    challenger: Literal["irace_generational"]
    score_practical_margin: float
    recall_noninferiority_margin: float
    top_set_k: int


@dataclass(frozen=True, slots=True)
class BakeoffSpec:
    experiment_id: str
    game_binary: Path
    objective_file: Path
    proposal_seeds: tuple[int, ...]
    task_seed: int
    tuning_pair_budgets: tuple[int, ...]
    shared_run: SharedRun
    decision: BakeoffDecision


def _positive_integers(value: object, label: str, *, minimum: int) -> tuple[int, ...]:
    items = elements(value, label)
    if len(items) < minimum:
        raise ValueError(f"{label} needs at least {minimum} entries")
    return tuple(integer(item, label, positive=True) for item in items)


def _decode_shared_run(value: object) -> SharedRun:
    item = object_fields(value, _SHARED_RUN_FIELDS, "bakeoff shared run")
    return SharedRun(
        integer(item["cohort_size"], "cohort size", positive=True),
        integer(item["finalists"], "finalists", positive=True),
        integer(item["bootstrap_candidates"], "bootstrap candidates", positive=True),
        integer(item["random_reserve_candidates"], "random reserve candidates", positive=True),
        integer(item["tuning_pairs"], "tuning pairs", positive=True),
        integer(item["validation_pair_budget"], "validation pair budget", positive=True),
        integer(item["production_validation_pairs"], "production validation pairs", positive=True),
        decode_effort(item["tuning_effort"], "tuning effort"),
        decode_effort(item["validation_effort"], "validation effort"),
        decode_effort(item["production_effort"], "production effort"),
        decode_constraints(item["constraints"]),
        integer(item["evaluator_workers"], "evaluator workers", positive=True),
        integer(item["pair_timeout_seconds"], "pair timeout seconds", positive=True),
    )


def _decode_decision(value: object) -> BakeoffDecision:
    item = object_fields(value, _DECISION_FIELDS, "bakeoff decision")
    if item["baseline"] != BAKEOFF_BASELINE or item["challenger"] != BAKEOFF_CHALLENGER:
        raise ValueError("bakeoff decision must compare irace_generational against smac_mixed")
    top_set_k = integer(item["top_set_k"], "top set k", positive=True)
    return BakeoffDecision(
        BAKEOFF_BASELINE,
        BAKEOFF_CHALLENGER,
        number(item["score_practical_margin"], "score practical margin"),
        number(item["recall_noninferiority_margin"], "recall noninferiority margin"),
        top_set_k,
    )


def read_spec(path: Path) -> BakeoffSpec:
    from .codec import strict_json

    raw = json_object(strict_json(path.read_text(), "bakeoff spec"), "bakeoff spec")
    if set(raw) != _SPEC_FIELDS:
        raise ValueError("invalid proposer bake-off specification fields")
    if raw["schema_version"] != 1 or raw["policies"] != list(POLICIES):
        raise ValueError("unsupported bakeoff schema or policy order")
    seeds = _positive_integers(raw["proposal_seeds"], "proposal seeds", minimum=4)
    if len(set(seeds)) < 4:
        raise ValueError("bakeoff needs four distinct positive proposal seeds")
    budgets = _positive_integers(raw["tuning_pair_budgets"], "tuning pair budgets", minimum=2)
    if list(budgets) != sorted(budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("bakeoff needs strictly increasing tuning budgets")
    shared_run = _decode_shared_run(raw["shared_run"])
    decision = _decode_decision(raw["decision"])
    if decision.top_set_k > shared_run.finalists:
        raise ValueError("bakeoff top_set_k must not exceed finalists")
    return BakeoffSpec(
        string(raw["experiment_id"], "experiment id", nonempty=True),
        Path(string(raw["game_binary"], "game binary", nonempty=True)),
        Path(string(raw["objective_file"], "objective file", nonempty=True)),
        seeds,
        integer(raw["task_seed"], "task seed", positive=True),
        budgets,
        shared_run,
        decision,
    )


def _encode_shared_run(shared: SharedRun) -> JsonObject:
    return {
        "cohort_size": shared.cohort_size,
        "finalists": shared.finalists,
        "bootstrap_candidates": shared.bootstrap_candidates,
        "random_reserve_candidates": shared.random_reserve_candidates,
        "tuning_pairs": shared.tuning_pairs,
        "validation_pair_budget": shared.validation_pair_budget,
        "production_validation_pairs": shared.production_validation_pairs,
        "tuning_effort": encode_effort(shared.tuning_effort),
        "validation_effort": encode_effort(shared.validation_effort),
        "production_effort": encode_effort(shared.production_effort),
        "constraints": encode_constraints(shared.constraints),
        "evaluator_workers": shared.evaluator_workers,
        "pair_timeout_seconds": shared.pair_timeout_seconds,
    }


def _encode_decision(decision: BakeoffDecision) -> JsonObject:
    return {
        "baseline": decision.baseline,
        "challenger": decision.challenger,
        "score_practical_margin": decision.score_practical_margin,
        "recall_noninferiority_margin": decision.recall_noninferiority_margin,
        "top_set_k": decision.top_set_k,
    }


def _spec(spec: BakeoffSpec) -> JsonObject:
    return {
        "game_binary": str(spec.game_binary),
        "objective_file": str(spec.objective_file),
        "proposal_seeds": list(spec.proposal_seeds),
        "task_seed": spec.task_seed,
        "tuning_pair_budgets": list(spec.tuning_pair_budgets),
        "shared_run": _encode_shared_run(spec.shared_run),
        "decision": _encode_decision(spec.decision),
        "policies": list(POLICIES),
    }


def encode_experiment(spec: BakeoffSpec, cells: list[JsonObject]) -> str:
    raw: JsonObject = {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "spec": _spec(spec),
        "cells": list(cells),
    }
    fingerprinted: JsonObject = {**raw, "fingerprint": fingerprint(raw)}
    return canonical_json(fingerprinted) + "\n"


def read_experiment(text: str) -> JsonObject:
    from .codec import strict_json

    raw = json_object(strict_json(text, "experiment"), "experiment")
    if "fingerprint" not in raw:
        raise ValueError("invalid experiment manifest")
    stored = raw["fingerprint"]
    body: JsonObject = {key: value for key, value in raw.items() if key != "fingerprint"}
    if stored != fingerprint(body):
        raise ValueError("experiment fingerprint mismatch")
    return raw


def experiment_fingerprint(text: str) -> str:
    value = read_experiment(text)["fingerprint"]
    return string(value, "experiment fingerprint", nonempty=True)
