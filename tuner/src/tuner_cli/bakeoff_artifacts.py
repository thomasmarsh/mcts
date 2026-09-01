"""Strict version-one codecs for proposer bake-off inputs and projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .codec import strict_json
from .identity import canonical_json, fingerprint
from .proposer import POLICIES


@dataclass(frozen=True, slots=True)
class BakeoffSpec:
    experiment_id: str
    game_binary: Path
    objective_file: Path
    proposal_seeds: tuple[int, ...]
    task_seed: int
    tuning_pair_budgets: tuple[int, ...]
    shared_run: dict[str, object]
    decision: dict[str, object]


def read_spec(path: Path) -> BakeoffSpec:
    raw = strict_json(path.read_text(), "bakeoff spec")
    if not isinstance(raw, dict) or set(raw) != {
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
    }:
        raise ValueError("invalid proposer bake-off specification fields")
    if raw["schema_version"] != 1 or raw["policies"] != list(POLICIES):
        raise ValueError("unsupported bakeoff schema or policy order")
    seeds, budgets = raw["proposal_seeds"], raw["tuning_pair_budgets"]
    if (
        not isinstance(seeds, list)
        or len(set(seeds)) < 4
        or not all(type(x) is int and x > 0 for x in seeds)
    ):
        raise ValueError("bakeoff needs four distinct positive proposal seeds")
    if (
        not isinstance(budgets, list)
        or len(budgets) < 2
        or budgets != sorted(budgets)
        or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("bakeoff needs increasing tuning budgets")
    if (
        not all(type(x) is int and x > 0 for x in budgets)
        or type(raw["task_seed"]) is not int
        or raw["task_seed"] <= 0
    ):
        raise ValueError("invalid bakeoff numeric fields")
    if (
        not all(
            isinstance(raw[x], str) and raw[x]
            for x in ("experiment_id", "game_binary", "objective_file")
        )
        or not isinstance(raw["shared_run"], dict)
        or not isinstance(raw["decision"], dict)
    ):
        raise ValueError("invalid bakeoff specification values")
    return BakeoffSpec(
        raw["experiment_id"],
        Path(raw["game_binary"]),
        Path(raw["objective_file"]),
        tuple(seeds),
        raw["task_seed"],
        tuple(budgets),
        raw["shared_run"],
        raw["decision"],
    )


def encode_experiment(spec: BakeoffSpec, cells: list[dict[str, object]]) -> str:
    raw = {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "spec": _spec(spec),
        "cells": cells,
    }
    return canonical_json({**raw, "fingerprint": fingerprint(raw)}) + "\n"


def _spec(spec: BakeoffSpec) -> dict[str, object]:
    return {
        "game_binary": str(spec.game_binary),
        "objective_file": str(spec.objective_file),
        "proposal_seeds": list(spec.proposal_seeds),
        "task_seed": spec.task_seed,
        "tuning_pair_budgets": list(spec.tuning_pair_budgets),
        "shared_run": spec.shared_run,
        "decision": spec.decision,
        "policies": list(POLICIES),
    }
