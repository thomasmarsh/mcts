"""Canonical JSON and deterministic identities used by tuner artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

from .domain import (
    Candidate,
    Estimate,
    ObjectiveEpoch,
    Observation,
    ObservationFrontier,
    ObservationReference,
    Opponent,
    OpponentPanel,
    PairTask,
    Phase,
    SearchEffort,
    TaskCase,
    TaskCorpus,
    TaskPrefix,
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _normalize(value: object) -> JsonValue:
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value and isinstance(scalar, (type(None), bool, int, float, str)):
            return _normalize(scalar)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow non-finite floats")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON requires string mapping keys")
            normalized[key] = _normalize(child)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_normalize(child) for child in value]
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(kind: str, identity_payload: object) -> str:
    return f"{kind}-{fingerprint(identity_payload)}"


def derive_task_seed(root_seed: int, phase: str, ordinal: int) -> int:
    if phase not in {"tuning", "validation"} or ordinal < 0:
        raise ValueError("invalid task seed inputs")
    payload = {
        "namespace": "mcts-tuner-task-seed-v2",
        "root_seed": root_seed,
        "phase": phase,
        "ordinal": ordinal,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(payload).encode()).digest()[:8], "big") & (
        (1 << 53) - 1
    )


def candidate_from_config(config: object) -> Candidate:
    canonical = canonical_json(config)
    config_fingerprint = fingerprint(json.loads(canonical))
    return Candidate(f"candidate-{config_fingerprint}", config_fingerprint, canonical)


def candidate_from_canonical_config(canonical: str) -> Candidate:
    try:
        parsed = json.loads(canonical)
    except json.JSONDecodeError as error:
        raise ValueError("candidate configuration is not JSON") from error
    if canonical_json(parsed) != canonical:
        raise ValueError("candidate configuration is not canonical JSON")
    return candidate_from_config(parsed)


def panel_payload(opponents: tuple[Opponent, ...]) -> dict[str, object]:
    return {
        "version": "opponent-panel-v1",
        "opponents": [
            {
                "id": x.opponent_id,
                "source": x.source_id,
                "label": x.label,
                "role": x.role,
                "weight": x.weight,
                "config": x.canonical_config,
                "fingerprint": x.configuration_fingerprint,
            }
            for x in opponents
        ],
    }


def opponent_panel(opponents: tuple[Opponent, ...]) -> OpponentPanel:
    payload = panel_payload(opponents)
    digest = fingerprint(payload)
    return OpponentPanel(
        stable_id("panel", payload), digest, opponents, sum(x.weight for x in opponents)
    )


def task_case(
    phase: str,
    ordinal: int,
    root_seed: int,
    opponent: Opponent,
    panel: OpponentPanel,
    game_config_fingerprint: str,
) -> TaskCase:
    seed = derive_task_seed(root_seed, phase, ordinal)
    stratum_id = stable_id(
        "stratum",
        {
            "panel": panel.fingerprint,
            "opponent_id": opponent.opponent_id,
            "opponent_fingerprint": opponent.configuration_fingerprint,
            "start": "default",
        },
    )
    payload = {
        "phase": phase,
        "ordinal": ordinal,
        "seed": seed,
        "stratum_id": stratum_id,
        "opponent_id": opponent.opponent_id,
        "opponent_fingerprint": opponent.configuration_fingerprint,
        "panel_fingerprint": panel.fingerprint,
        "game_config_fingerprint": game_config_fingerprint,
        "start": "default",
    }
    return TaskCase(
        stable_id("task", payload),
        cast(Phase, phase),
        ordinal,
        seed,
        stratum_id,
        opponent.opponent_id,
        opponent.configuration_fingerprint,
        panel.fingerprint,
        game_config_fingerprint,
    )


def task_corpus(phase: str, cases: tuple[TaskCase, ...], panel: OpponentPanel) -> TaskCorpus:
    payload = {
        "phase": phase,
        "task_policy_version": "weighted-fair-prefix-v1",
        "panel_fingerprint": panel.fingerprint,
        "task_ids": [case.task_id for case in cases],
    }
    digest = fingerprint(payload)
    return TaskCorpus(
        stable_id("corpus", payload), digest, cast(Phase, phase), "weighted-fair-prefix-v1", cases
    )


def task_prefix(corpus: TaskCorpus, length: int) -> TaskPrefix:
    if length <= 0 or length > len(corpus.cases):
        raise ValueError("task prefix length is outside its corpus")
    ids = tuple(case.task_id for case in corpus.cases[:length])
    payload = {
        "corpus_id": corpus.corpus_id,
        "corpus_fingerprint": corpus.fingerprint,
        "task_ids": ids,
    }
    return TaskPrefix(stable_id("prefix", payload), corpus.corpus_id, length, ids)


def objective_epoch(payload: object) -> ObjectiveEpoch:
    digest = fingerprint(payload)
    return ObjectiveEpoch(stable_id("epoch", payload), digest)


def observation_reference(value: Observation) -> ObservationReference:
    context = value.context
    return ObservationReference(
        value.observation_id,
        value.candidate_id,
        context.objective_epoch_id,
        context.task_prefix.prefix_id,
        context.task_prefix.task_ids,
        context.search_effort,
    )


def observation_id(
    candidate_id: str,
    context: object,
    utilities: tuple[float, ...],
    estimate: Estimate,
) -> str:
    return stable_id(
        "observation",
        {
            "candidate_id": candidate_id,
            "context": context,
            "pair_utilities": utilities,
            "estimate": {
                "mean": estimate.mean,
                "lower": estimate.lower,
                "upper": estimate.upper,
            },
        },
    )


def observation_frontier(references: tuple[ObservationReference, ...]) -> ObservationFrontier:
    if not references:
        return ObservationFrontier("frontier-empty-v1", "", "", (), SearchEffort(1), ())
    first = references[0]
    common = (
        first.objective_epoch_id,
        first.prefix_id,
        first.task_ids,
        first.search_effort,
    )
    if any(
        (item.objective_epoch_id, item.prefix_id, item.task_ids, item.search_effort) != common
        for item in references
    ):
        raise ValueError("frontier observations do not share a tuning context")
    ids = tuple(item.observation_id for item in references)
    if len(ids) != len(set(ids)):
        raise ValueError("frontier repeats an observation")
    payload = {
        "version": "observation-frontier-v1",
        "objective_epoch_id": first.objective_epoch_id,
        "prefix_id": first.prefix_id,
        "task_ids": first.task_ids,
        "search_effort": first.search_effort.max_iterations,
        "observation_ids": ids,
    }
    return ObservationFrontier(stable_id("frontier", payload), *common, ids)


def pair_task(candidate: Candidate, case: TaskCase, budget: SearchEffort) -> PairTask:
    pair_id = stable_id(
        "pair",
        {
            "candidate_fingerprint": candidate.fingerprint,
            "task_id": case.task_id,
            "opponent_fingerprint": case.opponent_fingerprint,
            "max_iterations": budget.max_iterations,
        },
    )
    return PairTask(pair_id, candidate.candidate_id, case, budget)


def game_id(task: PairTask, candidate_side: str) -> str:
    if candidate_side not in {"first", "second"}:
        raise ValueError("invalid candidate side")
    return stable_id("game", {"pair_id": task.pair_id, "candidate_side": candidate_side})
