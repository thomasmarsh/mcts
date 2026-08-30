"""Strict, resolved deployment-objective input and frozen opponent panels."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Literal, cast

from .codec import integer, object_fields, strict_json, string
from .domain import Candidate, Opponent, OpponentPanel
from .identity import canonical_json, fingerprint, opponent_panel


@dataclass(frozen=True, slots=True)
class ResolvedObjective:
    objective_id: str
    game_kind: str
    fingerprint: str
    source_path: Path
    panel: OpponentPanel
    start_distribution_fingerprint: str


def resolve_objective(path: Path, game_kind: str, schema_default: Candidate) -> ResolvedObjective:
    resolved = path.expanduser().resolve()
    root = object_fields(
        strict_json(resolved.read_text(encoding="utf-8"), "objective"),
        {"schema_version", "objective_id", "game_kind", "opponents", "start_distribution"},
        "objective",
    )
    if integer(root["schema_version"], "objective schema version") != 1:
        raise ValueError("unsupported objective schema version")
    objective_id = string(root["objective_id"], "objective id", nonempty=True)
    if string(root["game_kind"], "objective game kind", nonempty=True) != game_kind:
        raise ValueError("objective game kind differs from discovered game")
    start = object_fields(root["start_distribution"], {"kind"}, "start distribution")
    if start["kind"] != "default_only":
        raise ValueError("only default_only start distribution is supported")
    if not isinstance(root["opponents"], list) or len(root["opponents"]) < 2:
        raise ValueError("objective needs at least two opponents")
    opponents = tuple(_opponent(item, schema_default) for item in root["opponents"])
    _validate_panel(opponents, schema_default)
    panel = opponent_panel(opponents)
    source = {
        "schema_version": 1,
        "objective_id": objective_id,
        "game_kind": game_kind,
        "panel_fingerprint": panel.fingerprint,
        "start_distribution": {"kind": "default_only"},
    }
    return ResolvedObjective(
        objective_id,
        game_kind,
        fingerprint(source),
        resolved,
        panel,
        fingerprint({"kind": "default_only"}),
    )


def _opponent(value: object, schema_default: Candidate) -> Opponent:
    raw = object_fields(value, {"id", "label", "role", "weight", "config"}, "objective opponent")
    opponent_id = string(raw["id"], "opponent id", nonempty=True)
    label = string(raw["label"], "opponent label", nonempty=True)
    role = raw["role"]
    if role not in {"default", "historical_reference"}:
        raise ValueError("objective opponent has invalid role")
    weight = integer(raw["weight"], "opponent weight", positive=True)
    if not isinstance(raw["config"], dict):
        raise ValueError("opponent config must be an object")
    fields = {"source"} if raw["config"].get("source") == "schema_default" else {"source", "value"}
    config = object_fields(raw["config"], fields, "opponent config")
    source = config["source"]
    if source == "schema_default":
        canonical = schema_default.canonical_config
    elif source == "inline":
        if not isinstance(config["value"], dict):
            raise ValueError("inline opponent value must be a JSON object")
        canonical = canonical_json(config["value"])
    else:
        raise ValueError("objective opponent has invalid config source")
    return Opponent(
        opponent_id,
        cast(Literal["schema_default", "inline"], source),
        label,
        cast(Literal["default", "historical_reference"], role),
        weight,
        canonical,
        fingerprint(strict_json(canonical, "opponent config")),
    )


def _validate_panel(opponents: tuple[Opponent, ...], schema_default: Candidate) -> None:
    if len({item.opponent_id for item in opponents}) != len(opponents):
        raise ValueError("objective opponent ids must be unique")
    if len({item.configuration_fingerprint for item in opponents}) != len(opponents):
        raise ValueError("objective opponents must have distinct effective configurations")
    defaults = [item for item in opponents if item.role == "default"]
    if (
        len(defaults) != 1
        or defaults[0].source_id != "schema_default"
        or defaults[0].canonical_config != schema_default.canonical_config
    ):
        raise ValueError("objective needs exactly one schema-default opponent")
    if any(
        item.role != "historical_reference" or item.source_id != "inline"
        for item in opponents
        if item.role != "default"
    ):
        raise ValueError("non-default opponents must be inline historical references")
    divisor = 0
    for item in opponents:
        divisor = gcd(divisor, item.weight)
    if divisor != 1:
        raise ValueError("objective opponent weights must be reduced")
