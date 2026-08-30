"""Versioned immutable run manifest encoding and strict artifact decoding."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from .domain import Candidate, TaskBlock, TaskCase
from .identity import candidate_from_config, canonical_json, fingerprint, task_block
from .schema import GameSpec, decode_game_spec
from .space import build_space, default_values

SCHEMA_VERSION = 2


def _constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("JSON object has duplicate keys")
    return result


def strict_json(text: str, label: str = "JSON") -> object:
    try:
        value = json.loads(text, parse_constant=_constant, object_pairs_hook=_unique)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict {label}: {error}") from error
    _finite(value, label)
    return value


def _finite(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child, label)
    elif isinstance(value, list):
        for child in value:
            _finite(child, label)


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} has invalid fields (missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)})"
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        raise ValueError(f"{label} must be{' a positive' if positive else ' an'} integer")
    return value


def _canonical_string(value: object, label: str) -> str:
    text = _string(value, label)
    parsed = strict_json(text, label)
    if canonical_json(parsed) != text:
        raise ValueError(f"{label} must contain canonical JSON")
    return text


def configspace_version() -> str:
    return version("ConfigSpace")


@dataclass(frozen=True, slots=True)
class Manifest:
    raw: dict[str, object]
    fingerprint: str
    spec: GameSpec
    opponent: Candidate
    tuning: TaskBlock
    validation: TaskBlock

    @property
    def seed(self) -> int:
        return self.raw["proposer"]["seed"]  # type: ignore[index,return-value]

    @property
    def cohort_size(self) -> int:
        return self.raw["proposer"]["cohort_size"]  # type: ignore[index,return-value]

    @property
    def finalists(self) -> int:
        return self.raw["proposer"]["finalists"]  # type: ignore[index,return-value]

    @property
    def budgets(self) -> dict[str, int]:
        return self.raw["budgets"]  # type: ignore[return-value]


def _case_dict(case: TaskCase) -> dict[str, object]:
    return {
        "task_id": case.task_id,
        "phase": case.phase,
        "ordinal": case.ordinal,
        "seed": case.seed,
        "opponent_id": case.opponent_id,
        "opponent_fingerprint": case.opponent_fingerprint,
        "game_config_fingerprint": case.game_config_fingerprint,
        "start": case.start,
    }


def _block_dict(block: TaskBlock) -> dict[str, object]:
    return {
        "block_id": block.block_id,
        "phase": block.phase,
        "cases": [_case_dict(c) for c in block.cases],
    }


def build_manifest(
    run_id: str,
    spec: GameSpec,
    seed: int,
    cohort_size: int,
    finalists: int,
    tuning_pairs: int,
    validation_pairs: int,
    tuning_max_iterations: int,
    validation_max_iterations: int,
    production_max_iterations: int,
) -> Manifest:
    opponent = candidate_from_config(default_values(build_space(spec.tuning, seed)))
    game_config_fingerprint = fingerprint(
        strict_json(spec.default_game_config, "game configuration")
    )
    tuning = task_block("tuning", tuning_pairs, seed, opponent, game_config_fingerprint)
    validation = task_block("validation", validation_pairs, seed, opponent, game_config_fingerprint)
    raw: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "command_policy_version": "generic-resumable-v2",
        "binary": {"path": str(spec.binary_path), "sha256": spec.binary_sha256},
        "engine_fingerprint": spec.engine_fingerprint,
        "description": spec.raw_description,
        "description_fingerprint": spec.description_fingerprint,
        "kind": spec.kind,
        "label": spec.label,
        "game_description": spec.description,
        "ai_presets": [asdict(item) for item in spec.ai_presets],
        "tuning": {
            "id": spec.tuning.id,
            "baselines": list(spec.tuning.baselines),
            "eval_rounds": spec.tuning.eval_rounds,
            "game_config": spec.tuning.game_config,
            "parameters": [asdict(item) for item in spec.tuning.parameters],
            "conditions": [asdict(item) for item in spec.tuning.conditions],
        },
        "tuning_schema_fingerprint": spec.schema_fingerprint,
        "game_config": spec.default_game_config,
        "game_config_fingerprint": game_config_fingerprint,
        "parameters": [asdict(item) for item in spec.tuning.parameters],
        "conditions": [asdict(item) for item in spec.tuning.conditions],
        "proposer": {
            "kind": "configspace_random",
            "version": "configspace-random-v1",
            "configspace_version": configspace_version(),
            "seed": seed,
            "cohort_size": cohort_size,
            "finalists": finalists,
        },
        "opponent": {
            "id": f"opponent-default-{opponent.fingerprint}",
            "canonical_config": opponent.canonical_config,
            "fingerprint": opponent.fingerprint,
        },
        "tuning_tasks": _block_dict(tuning),
        "validation_tasks": _block_dict(validation),
        "budgets": {
            "tuning": tuning_max_iterations,
            "validation": validation_max_iterations,
            "production": production_max_iterations,
        },
        "utility_formula_version": "pair_mean_v1",
        "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
        "interval_method": "hoeffding_pair_bound_v1",
        "confidence_level": 0.95,
        "tie_rule_version": "paired_hoeffding_v1",
        "limitations": [
            "one opponent",
            "default starting state",
            "sequential execution",
            "fixed iterations",
            "explicit resume",
        ],
    }
    return decode_manifest_object({**raw, "fingerprint": fingerprint(raw)})


_FIELDS = {
    "schema_version",
    "run_id",
    "command_policy_version",
    "binary",
    "engine_fingerprint",
    "description",
    "description_fingerprint",
    "kind",
    "label",
    "game_description",
    "ai_presets",
    "tuning",
    "tuning_schema_fingerprint",
    "game_config",
    "game_config_fingerprint",
    "parameters",
    "conditions",
    "proposer",
    "opponent",
    "tuning_tasks",
    "validation_tasks",
    "budgets",
    "utility_formula_version",
    "selection_rule_version",
    "interval_method",
    "confidence_level",
    "tie_rule_version",
    "limitations",
    "fingerprint",
}


def _decode_block(
    raw: object,
    expected: Literal["tuning", "validation"],
    manifest: dict[str, object],
    opponent: Candidate,
) -> TaskBlock:
    block = _object(raw, {"block_id", "phase", "cases"}, f"{expected} task block")
    if block["phase"] != expected or not isinstance(block["cases"], list):
        raise ValueError(f"invalid {expected} task block")
    expected_block = task_block(
        expected,
        len(block["cases"]),
        _integer(
            _object(
                manifest["proposer"],
                {"kind", "version", "configspace_version", "seed", "cohort_size", "finalists"},
                "proposer",
            )["seed"],
            "seed",
        ),
        opponent,
        _string(manifest["game_config_fingerprint"], "game configuration fingerprint"),
    )
    if _block_dict(expected_block) != block:
        raise ValueError(f"{expected} task identities do not match frozen inputs")
    return expected_block


def decode_manifest_object(value: object) -> Manifest:
    raw = _object(value, _FIELDS, "manifest")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema version: {raw['schema_version']!r}")
    if raw["command_policy_version"] != "generic-resumable-v2":
        raise ValueError("unsupported manifest command policy")
    stored = _string(raw["fingerprint"], "manifest fingerprint")
    unfingerprinted = {key: item for key, item in raw.items() if key != "fingerprint"}
    if fingerprint(unfingerprinted) != stored:
        raise ValueError("manifest fingerprint does not match content")
    binary = _object(raw["binary"], {"path", "sha256"}, "binary")
    path, digest = (
        Path(_string(binary["path"], "binary path")),
        _string(binary["sha256"], "binary hash"),
    )
    raw_description = _canonical_string(raw["description"], "description")
    description = strict_json(raw_description, "description")
    spec = decode_game_spec(description, path, digest)
    duplicated = {
        "description_fingerprint": spec.description_fingerprint,
        "kind": spec.kind,
        "label": spec.label,
        "game_description": spec.description,
        "ai_presets": [asdict(x) for x in spec.ai_presets],
        "tuning_schema_fingerprint": spec.schema_fingerprint,
        "game_config": spec.default_game_config,
        "parameters": [asdict(x) for x in spec.tuning.parameters],
        "conditions": [asdict(x) for x in spec.tuning.conditions],
        "engine_fingerprint": spec.engine_fingerprint,
    }
    for key, expected in duplicated.items():
        if canonical_json(raw[key]) != canonical_json(expected):
            raise ValueError(f"manifest {key} disagrees with description")
    expected_tuning = {
        "id": spec.tuning.id,
        "baselines": list(spec.tuning.baselines),
        "eval_rounds": spec.tuning.eval_rounds,
        "game_config": spec.tuning.game_config,
        "parameters": [asdict(x) for x in spec.tuning.parameters],
        "conditions": [asdict(x) for x in spec.tuning.conditions],
    }
    if canonical_json(raw["tuning"]) != canonical_json(expected_tuning):
        raise ValueError("manifest tuning disagrees with description")
    config = _canonical_string(raw["game_config"], "game configuration")
    if (
        config != spec.default_game_config
        or fingerprint(strict_json(config)) != raw["game_config_fingerprint"]
    ):
        raise ValueError("manifest game configuration is inconsistent")
    proposer = _object(
        raw["proposer"],
        {"kind", "version", "configspace_version", "seed", "cohort_size", "finalists"},
        "proposer",
    )
    if proposer["kind"] != "configspace_random" or proposer["version"] != "configspace-random-v1":
        raise ValueError("unsupported proposer")
    for key in ("seed", "cohort_size", "finalists"):
        _integer(proposer[key], f"proposer {key}", positive=True)
    if _integer(proposer["finalists"], "finalists", positive=True) > _integer(
        proposer["cohort_size"], "cohort size", positive=True
    ):
        raise ValueError("finalists exceeds cohort size")
    opponent_raw = _object(raw["opponent"], {"id", "canonical_config", "fingerprint"}, "opponent")
    opponent = candidate_from_config(
        strict_json(_canonical_string(opponent_raw["canonical_config"], "opponent configuration"))
    )
    if opponent_raw != {
        "id": f"opponent-default-{opponent.fingerprint}",
        "canonical_config": opponent.canonical_config,
        "fingerprint": opponent.fingerprint,
    }:
        raise ValueError("opponent identity is inconsistent")
    tuning = _decode_block(raw["tuning_tasks"], "tuning", raw, opponent)
    validation = _decode_block(raw["validation_tasks"], "validation", raw, opponent)
    if set(case.seed for case in tuning.cases) & set(case.seed for case in validation.cases):
        raise ValueError("task blocks have colliding seeds")
    budgets = _object(raw["budgets"], {"tuning", "validation", "production"}, "budgets")
    for name, item in budgets.items():
        _integer(item, f"{name} budget", positive=True)
    if (
        not isinstance(raw["confidence_level"], (float, int))
        or isinstance(raw["confidence_level"], bool)
        or not math.isfinite(float(raw["confidence_level"]))
    ):
        raise ValueError("confidence level must be finite")
    if not isinstance(raw["limitations"], list) or not all(
        isinstance(item, str) for item in raw["limitations"]
    ):
        raise ValueError("limitations must be strings")
    return Manifest(dict(raw), stored, spec, opponent, tuning, validation)


def read_manifest(path: Path) -> Manifest:
    return decode_manifest_object(strict_json(path.read_text(encoding="utf-8"), "manifest"))


def manifest_json(manifest: Manifest) -> dict[str, object]:
    return dict(manifest.raw)
