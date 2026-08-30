"""Versioned immutable manifest encoding and strict artifact decoding."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from .codec import integer, object_fields, strict_json, string
from .domain import (
    ObjectiveEpoch,
    Opponent,
    OpponentPanel,
    SearchEffort,
    TaskCorpus,
    TaskPrefix,
)
from .identity import (
    canonical_json,
    fingerprint,
    objective_epoch,
    opponent_panel,
    task_prefix,
)
from .objective import ResolvedObjective
from .schema import GameSpec, decode_game_spec
from .tasks import build_corpus, selected_prefix, validate_cycle_endpoint, verify_weighted_corpus

SCHEMA_VERSION = 3


def configspace_version() -> str:
    return version("ConfigSpace")


@dataclass(frozen=True, slots=True)
class Manifest:
    raw: dict[str, object]
    fingerprint: str
    spec: GameSpec
    objective_id: str
    objective_fingerprint: str
    panel: OpponentPanel
    tuning_corpus: TaskCorpus
    production_validation_corpus: TaskCorpus
    tuning_prefix: TaskPrefix
    validation_prefix: TaskPrefix
    epoch: ObjectiveEpoch

    @property
    def seed(self) -> int:
        return self.proposer["seed"]

    @property
    def task_seed(self) -> int:
        return self.proposer["task_seed"]

    @property
    def proposer(self) -> dict[str, int | str]:
        return self.raw["proposer"]

    @property
    def cohort_size(self) -> int:
        return self.proposer["cohort_size"]

    @property
    def finalists(self) -> int:
        return self.proposer["finalists"]

    @property
    def efforts(self) -> dict[str, SearchEffort]:
        raw = self.raw["fidelity"]
        return {
            name: SearchEffort(raw[name]["search_effort"]["max_iterations"])
            for name in ("tuning", "validation", "production")
        }

    @property
    def opponent(self) -> Opponent:
        return next(item for item in self.panel.opponents if item.role == "default")

    @property
    def tuning(self) -> TaskCorpus:
        return self.tuning_corpus

    @property
    def validation(self) -> TaskCorpus:
        return self.production_validation_corpus

    def prefix_cases(self, phase: str):
        corpus, prefix = (
            (self.tuning_corpus, self.tuning_prefix)
            if phase == "tuning"
            else (self.production_validation_corpus, self.validation_prefix)
        )
        return corpus.cases[: prefix.length]


def _opponent_dict(item: Opponent) -> dict[str, object]:
    return {
        "id": item.opponent_id,
        "source": item.source_id,
        "label": item.label,
        "role": item.role,
        "weight": item.weight,
        "canonical_config": item.canonical_config,
        "configuration_fingerprint": item.configuration_fingerprint,
    }


def _panel_dict(panel: OpponentPanel) -> dict[str, object]:
    return {
        "panel_id": panel.panel_id,
        "fingerprint": panel.fingerprint,
        "total_weight": panel.total_weight,
        "opponents": [_opponent_dict(item) for item in panel.opponents],
    }


def _case_dict(case) -> dict[str, object]:
    return {
        "task_id": case.task_id,
        "phase": case.phase,
        "ordinal": case.ordinal,
        "seed": case.seed,
        "stratum_id": case.stratum_id,
        "opponent_id": case.opponent_id,
        "opponent_fingerprint": case.opponent_fingerprint,
        "panel_fingerprint": case.panel_fingerprint,
        "game_config_fingerprint": case.game_config_fingerprint,
        "start": case.start,
    }


def _corpus_dict(corpus: TaskCorpus) -> dict[str, object]:
    return {
        "corpus_id": corpus.corpus_id,
        "fingerprint": corpus.fingerprint,
        "phase": corpus.phase,
        "task_policy_version": corpus.task_policy_version,
        "cases": [_case_dict(case) for case in corpus.cases],
    }


def _prefix_dict(prefix: TaskPrefix) -> dict[str, object]:
    return {
        "prefix_id": prefix.prefix_id,
        "corpus_id": prefix.corpus_id,
        "length": prefix.length,
        "task_ids": list(prefix.task_ids),
    }


def production_claim(
    validation_prefix: TaskPrefix,
    production_corpus: TaskCorpus,
    validation_effort: SearchEffort,
    production_effort: SearchEffort,
) -> tuple[str, tuple[str, ...]]:
    missing: list[str] = []
    if validation_prefix.length != len(
        production_corpus.cases
    ) or validation_prefix.task_ids != tuple(case.task_id for case in production_corpus.cases):
        missing.append("task_count")
    if validation_effort != production_effort:
        missing.append("search_effort")
    return ("production", ()) if not missing else ("mechanics_smoke", tuple(missing))


def _epoch_payload(
    spec: GameSpec,
    objective: ResolvedObjective,
    tuning: TaskCorpus,
    validation: TaskCorpus,
    production_effort: SearchEffort,
    game_config_fingerprint: str,
) -> dict[str, object]:
    return {
        "version": "objective-epoch-v1",
        "objective_id": objective.objective_id,
        "objective_fingerprint": objective.fingerprint,
        "game_kind": spec.kind,
        "engine_fingerprint": spec.engine_fingerprint,
        "schema_fingerprint": spec.schema_fingerprint,
        "game_config_fingerprint": game_config_fingerprint,
        "panel_fingerprint": objective.panel.fingerprint,
        "start_distribution_fingerprint": objective.start_distribution_fingerprint,
        "tuning_corpus_fingerprint": tuning.fingerprint,
        "production_validation_corpus_fingerprint": validation.fingerprint,
        "production_validation_pairs": len(validation.cases),
        "production_max_iterations": production_effort.max_iterations,
        "task_policy_version": "weighted-fair-prefix-v1",
        "utility_formula_version": "pair_mean_v1",
        "interval_method": "hoeffding_pair_bound_v1",
        "confidence_level": 0.95,
        "tie_rule_version": "paired_hoeffding_v1",
        "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
    }


def build_manifest(
    run_id: str,
    spec: GameSpec,
    objective: ResolvedObjective,
    seed: int,
    task_seed: int,
    cohort_size: int,
    finalists: int,
    tuning_pairs: int,
    validation_pairs: int,
    production_validation_pairs: int,
    tuning_max_iterations: int,
    validation_max_iterations: int,
    production_max_iterations: int,
) -> Manifest:
    for count, label in (
        (tuning_pairs, "tuning pairs"),
        (validation_pairs, "validation pairs"),
        (production_validation_pairs, "production validation pairs"),
    ):
        validate_cycle_endpoint(objective.panel, count, label)
    if validation_pairs > production_validation_pairs:
        raise ValueError("validation pairs cannot exceed production validation pairs")
    if (
        tuning_max_iterations > production_max_iterations
        or validation_max_iterations > production_max_iterations
    ):
        raise ValueError("observed search effort cannot exceed production effort")
    game_config_fingerprint = fingerprint(
        strict_json(spec.default_game_config, "game configuration")
    )
    tuning = build_corpus(
        "tuning", tuning_pairs, task_seed, objective.panel, game_config_fingerprint
    )
    production_validation = build_corpus(
        "validation",
        production_validation_pairs,
        task_seed,
        objective.panel,
        game_config_fingerprint,
    )
    tuning_prefix = selected_prefix(tuning, tuning_pairs)
    validation_prefix = selected_prefix(production_validation, validation_pairs)
    epoch = objective_epoch(
        _epoch_payload(
            spec,
            objective,
            tuning,
            production_validation,
            SearchEffort(production_max_iterations),
            game_config_fingerprint,
        )
    )
    raw: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "command_policy_version": "generic-resumable-v3",
        "binary": {"path": str(spec.binary_path), "sha256": spec.binary_sha256},
        "engine_fingerprint": spec.engine_fingerprint,
        "description": spec.raw_description,
        "description_fingerprint": spec.description_fingerprint,
        "kind": spec.kind,
        "label": spec.label,
        "game_description": spec.description,
        "tuning_schema_fingerprint": spec.schema_fingerprint,
        "game_config": spec.default_game_config,
        "game_config_fingerprint": game_config_fingerprint,
        "proposer": {
            "kind": "configspace_random",
            "version": "configspace-random-v1",
            "configspace_version": configspace_version(),
            "seed": seed,
            "task_seed": task_seed,
            "cohort_size": cohort_size,
            "finalists": finalists,
        },
        "objective": {
            "source_path": str(objective.source_path),
            "objective_id": objective.objective_id,
            "fingerprint": objective.fingerprint,
        },
        "opponent_panel": _panel_dict(objective.panel),
        "start_distribution": {
            "kind": "default_only",
            "fingerprint": objective.start_distribution_fingerprint,
        },
        "corpora": {
            "tuning": _corpus_dict(tuning),
            "production_validation": _corpus_dict(production_validation),
        },
        "prefixes": {
            "tuning": _prefix_dict(tuning_prefix),
            "validation": _prefix_dict(validation_prefix),
        },
        "fidelity": {
            "tuning": {
                "task_prefix_id": tuning_prefix.prefix_id,
                "search_effort": {"max_iterations": tuning_max_iterations},
            },
            "validation": {
                "task_prefix_id": validation_prefix.prefix_id,
                "search_effort": {"max_iterations": validation_max_iterations},
            },
            "production": {
                "task_prefix_id": selected_prefix(
                    production_validation, production_validation_pairs
                ).prefix_id,
                "search_effort": {"max_iterations": production_max_iterations},
            },
        },
        "epoch": {"epoch_id": epoch.epoch_id, "fingerprint": epoch.fingerprint},
        "utility_formula_version": "pair_mean_v1",
        "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
        "interval_method": "hoeffding_pair_bound_v1",
        "confidence_level": 0.95,
        "tie_rule_version": "paired_hoeffding_v1",
        "limitations": [
            "default-only start distribution",
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
    "tuning_schema_fingerprint",
    "game_config",
    "game_config_fingerprint",
    "proposer",
    "objective",
    "opponent_panel",
    "start_distribution",
    "corpora",
    "prefixes",
    "fidelity",
    "epoch",
    "utility_formula_version",
    "selection_rule_version",
    "interval_method",
    "confidence_level",
    "tie_rule_version",
    "limitations",
    "fingerprint",
}


def _decode_panel(value: object) -> OpponentPanel:
    raw = object_fields(
        value, {"panel_id", "fingerprint", "total_weight", "opponents"}, "opponent panel"
    )
    if not isinstance(raw["opponents"], list) or len(raw["opponents"]) < 2:
        raise ValueError("opponent panel needs at least two entries")
    opponents = []
    for item in raw["opponents"]:
        entry = object_fields(
            item,
            {
                "id",
                "source",
                "label",
                "role",
                "weight",
                "canonical_config",
                "configuration_fingerprint",
            },
            "panel opponent",
        )
        role, source = entry["role"], entry["source"]
        if role not in {"default", "historical_reference"} or source not in {
            "schema_default",
            "inline",
        }:
            raise ValueError("invalid panel opponent role or source")
        canonical = string(entry["canonical_config"], "panel configuration")
        if (
            canonical_json(strict_json(canonical, "panel configuration")) != canonical
            or fingerprint(strict_json(canonical)) != entry["configuration_fingerprint"]
        ):
            raise ValueError("panel configuration identity is invalid")
        opponents.append(
            Opponent(
                string(entry["id"], "opponent id", nonempty=True),
                source,
                string(entry["label"], "opponent label", nonempty=True),
                role,
                integer(entry["weight"], "opponent weight", positive=True),
                canonical,
                string(entry["configuration_fingerprint"], "configuration fingerprint"),
            )
        )
    panel = opponent_panel(tuple(opponents))
    if _panel_dict(panel) != raw:
        raise ValueError("opponent panel identity is inconsistent")
    return panel


def _decode_corpus(
    value: object, phase: str, panel: OpponentPanel, task_seed: int, game_config_fingerprint: str
) -> TaskCorpus:
    raw = object_fields(
        value,
        {"corpus_id", "fingerprint", "phase", "task_policy_version", "cases"},
        f"{phase} corpus",
    )
    if (
        raw["phase"] != phase
        or raw["task_policy_version"] != "weighted-fair-prefix-v1"
        or not isinstance(raw["cases"], list)
    ):
        raise ValueError(f"invalid {phase} corpus")
    expected = build_corpus(phase, len(raw["cases"]), task_seed, panel, game_config_fingerprint)
    if _corpus_dict(expected) != raw:
        raise ValueError(f"{phase} corpus identities do not match frozen inputs")
    verify_weighted_corpus(expected, panel)
    return expected


def _decode_prefix(value: object, corpus: TaskCorpus, label: str) -> TaskPrefix:
    raw = object_fields(value, {"prefix_id", "corpus_id", "length", "task_ids"}, f"{label} prefix")
    length = integer(raw["length"], f"{label} prefix length", positive=True)
    expected = task_prefix(corpus, length)
    if _prefix_dict(expected) != raw:
        raise ValueError(f"{label} prefix identity is inconsistent")
    return expected


def decode_manifest_object(value: object) -> Manifest:
    raw = object_fields(value, _FIELDS, "manifest")
    if (
        raw["schema_version"] != SCHEMA_VERSION
        or raw["command_policy_version"] != "generic-resumable-v3"
    ):
        raise ValueError("unsupported manifest schema version or command policy")
    stored = string(raw["fingerprint"], "manifest fingerprint")
    if fingerprint({key: item for key, item in raw.items() if key != "fingerprint"}) != stored:
        raise ValueError("manifest fingerprint does not match content")
    binary = object_fields(raw["binary"], {"path", "sha256"}, "binary")
    spec = decode_game_spec(
        strict_json(string(raw["description"], "description"), "description"),
        Path(string(binary["path"], "binary path")),
        string(binary["sha256"], "binary hash"),
    )
    duplicated = {
        "engine_fingerprint": spec.engine_fingerprint,
        "description_fingerprint": spec.description_fingerprint,
        "kind": spec.kind,
        "label": spec.label,
        "game_description": spec.description,
        "tuning_schema_fingerprint": spec.schema_fingerprint,
        "game_config": spec.default_game_config,
    }
    if any(
        canonical_json(raw[key]) != canonical_json(expected) for key, expected in duplicated.items()
    ):
        raise ValueError("manifest disagrees with game description")
    game_config_fingerprint = string(
        raw["game_config_fingerprint"], "game configuration fingerprint"
    )
    if fingerprint(strict_json(spec.default_game_config)) != game_config_fingerprint:
        raise ValueError("game configuration fingerprint is invalid")
    proposer = object_fields(
        raw["proposer"],
        {"kind", "version", "configspace_version", "seed", "task_seed", "cohort_size", "finalists"},
        "proposer",
    )
    if proposer["kind"] != "configspace_random" or proposer["version"] != "configspace-random-v1":
        raise ValueError("unsupported proposer")
    for key in ("seed", "task_seed", "cohort_size", "finalists"):
        integer(proposer[key], key, positive=True)
    if proposer["finalists"] > proposer["cohort_size"]:
        raise ValueError("finalists exceeds cohort size")
    panel = _decode_panel(raw["opponent_panel"])
    objective = object_fields(
        raw["objective"], {"source_path", "objective_id", "fingerprint"}, "objective"
    )
    string(objective["objective_id"], "objective id", nonempty=True)
    string(objective["fingerprint"], "objective fingerprint")
    start = object_fields(raw["start_distribution"], {"kind", "fingerprint"}, "start distribution")
    if start["kind"] != "default_only" or start["fingerprint"] != fingerprint(
        {"kind": "default_only"}
    ):
        raise ValueError("invalid start distribution")
    corpora = object_fields(raw["corpora"], {"tuning", "production_validation"}, "corpora")
    tuning = _decode_corpus(
        corpora["tuning"], "tuning", panel, proposer["task_seed"], game_config_fingerprint
    )
    validation = _decode_corpus(
        corpora["production_validation"],
        "validation",
        panel,
        proposer["task_seed"],
        game_config_fingerprint,
    )
    if set(case.seed for case in tuning.cases) & set(case.seed for case in validation.cases):
        raise ValueError("task corpora have colliding seeds")
    prefixes = object_fields(raw["prefixes"], {"tuning", "validation"}, "prefixes")
    tuning_prefix = _decode_prefix(prefixes["tuning"], tuning, "tuning")
    validation_prefix = _decode_prefix(prefixes["validation"], validation, "validation")
    fidelity = object_fields(raw["fidelity"], {"tuning", "validation", "production"}, "fidelity")
    efforts: dict[str, SearchEffort] = {}
    expected_prefixes = {
        "tuning": tuning_prefix,
        "validation": validation_prefix,
        "production": task_prefix(validation, len(validation.cases)),
    }
    for name, prefix in expected_prefixes.items():
        item = object_fields(
            fidelity[name], {"task_prefix_id", "search_effort"}, f"{name} fidelity"
        )
        effort = object_fields(item["search_effort"], {"max_iterations"}, f"{name} search effort")
        if item["task_prefix_id"] != prefix.prefix_id:
            raise ValueError(f"{name} fidelity prefix is inconsistent")
        efforts[name] = SearchEffort(
            integer(effort["max_iterations"], f"{name} max iterations", positive=True)
        )
    if (
        efforts["tuning"].max_iterations > efforts["production"].max_iterations
        or efforts["validation"].max_iterations > efforts["production"].max_iterations
    ):
        raise ValueError("observed search effort exceeds production effort")
    epoch = objective_epoch(
        {
            "version": "objective-epoch-v1",
            "objective_id": objective["objective_id"],
            "objective_fingerprint": objective["fingerprint"],
            "game_kind": spec.kind,
            "engine_fingerprint": spec.engine_fingerprint,
            "schema_fingerprint": spec.schema_fingerprint,
            "game_config_fingerprint": game_config_fingerprint,
            "panel_fingerprint": panel.fingerprint,
            "start_distribution_fingerprint": start["fingerprint"],
            "tuning_corpus_fingerprint": tuning.fingerprint,
            "production_validation_corpus_fingerprint": validation.fingerprint,
            "production_validation_pairs": len(validation.cases),
            "production_max_iterations": efforts["production"].max_iterations,
            "task_policy_version": "weighted-fair-prefix-v1",
            "utility_formula_version": "pair_mean_v1",
            "interval_method": "hoeffding_pair_bound_v1",
            "confidence_level": 0.95,
            "tie_rule_version": "paired_hoeffding_v1",
            "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
        }
    )
    if raw["epoch"] != {"epoch_id": epoch.epoch_id, "fingerprint": epoch.fingerprint}:
        raise ValueError("objective epoch is inconsistent")
    if (
        raw["utility_formula_version"] != "pair_mean_v1"
        or raw["selection_rule_version"] != "tuning_point_estimate_fingerprint_v1"
        or raw["interval_method"] != "hoeffding_pair_bound_v1"
        or raw["tie_rule_version"] != "paired_hoeffding_v1"
        or raw["confidence_level"] != 0.95
    ):
        raise ValueError("unsupported statistical policy")
    if not isinstance(raw["limitations"], list) or not all(
        isinstance(item, str) for item in raw["limitations"]
    ):
        raise ValueError("limitations must be strings")
    return Manifest(
        dict(raw),
        stored,
        spec,
        objective["objective_id"],
        objective["fingerprint"],
        panel,
        tuning,
        validation,
        tuning_prefix,
        validation_prefix,
        epoch,
    )


def read_manifest(path: Path) -> Manifest:
    return decode_manifest_object(strict_json(path.read_text(encoding="utf-8"), "manifest"))


def manifest_json(manifest: Manifest) -> dict[str, object]:
    return dict(manifest.raw)
