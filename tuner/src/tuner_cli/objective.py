"""Strict, resolved deployment-objective input and frozen opponent panels."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Literal

from .codec import (
    JsonValue,
    integer,
    json_object,
    literal,
    object_fields,
    strict_json,
    string,
)
from .domain import Candidate, Opponent, OpponentPanel, OpponentRole
from .identity import canonical_json, fingerprint, opponent_panel
from .schema import GameConfigSchema

_ROLES: tuple[OpponentRole, ...] = ("default", "historical_reference")
_SOURCES: tuple[Literal["schema_default", "inline"], ...] = ("schema_default", "inline")


@dataclass(frozen=True, slots=True)
class ResolvedObjective:
    objective_id: str
    game_kind: str
    fingerprint: str
    source_path: Path
    panel: OpponentPanel
    start_distribution_fingerprint: str
    game_config: str


def resolve_objective(
    path: Path,
    game_kind: str,
    schema_default: Candidate,
    game_config_schema: GameConfigSchema | None = None,
    game_config_default: str = "{}",
) -> ResolvedObjective:
    resolved = path.expanduser().resolve()
    root = json_object(strict_json(resolved.read_text(encoding="utf-8"), "objective"), "objective")
    required = {"schema_version", "objective_id", "game_kind", "opponents", "start_distribution"}
    missing, unknown = sorted(required - set(root)), sorted(set(root) - required - {"game_config"})
    if missing or unknown:
        raise ValueError(f"objective has invalid fields (missing={missing}, unknown={unknown})")
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
    game_config, game_config_override = _game_config(
        root.get("game_config"), game_config_schema, game_config_default
    )
    source = {
        "schema_version": 1,
        "objective_id": objective_id,
        "game_kind": game_kind,
        "panel_fingerprint": panel.fingerprint,
        "start_distribution": {"kind": "default_only"},
    }
    if game_config_override:
        source["game_config"] = game_config
    return ResolvedObjective(
        objective_id,
        game_kind,
        fingerprint(source),
        resolved,
        panel,
        fingerprint({"kind": "default_only"}),
        game_config,
    )


def _game_config(
    value: JsonValue | None,
    game_config_schema: GameConfigSchema | None,
    game_config_default: str,
) -> tuple[str, bool]:
    """Resolve the effective ``game_config`` -- an in-bounds override or the
    binary's default. Returns ``(canonical json, is an override)``."""
    if value is None:
        return canonical_json(strict_json(game_config_default, "default game config")), False
    if not isinstance(value, dict):
        raise ValueError("objective game_config must be a JSON object")
    canonical = canonical_json(value)
    if canonical == canonical_json(strict_json(game_config_default, "default game config")):
        raise ValueError("objective game_config equals the default -- omit it instead")
    schema = game_config_schema if game_config_schema is not None else GameConfigSchema((), ())
    errors = schema.validate_config(value)
    if errors:
        raise ValueError("; ".join(errors))
    return canonical, True


def _opponent(value: object, schema_default: Candidate) -> Opponent:
    raw = object_fields(value, {"id", "label", "role", "weight", "config"}, "objective opponent")
    opponent_id = string(raw["id"], "opponent id", nonempty=True)
    label = string(raw["label"], "opponent label", nonempty=True)
    role: OpponentRole = literal(raw["role"], _ROLES, "objective opponent role")
    weight = integer(raw["weight"], "opponent weight", positive=True)
    if not isinstance(raw["config"], dict):
        raise ValueError("opponent config must be an object")
    fields = {"source"} if raw["config"].get("source") == "schema_default" else {"source", "value"}
    config = object_fields(raw["config"], fields, "opponent config")
    source: Literal["schema_default", "inline"] = literal(
        config["source"], _SOURCES, "opponent config source"
    )
    if source == "schema_default":
        canonical = schema_default.canonical_config
    else:
        if not isinstance(config["value"], dict):
            raise ValueError("inline opponent value must be a JSON object")
        canonical = canonical_json(config["value"])
    return Opponent(
        opponent_id,
        source,
        label,
        role,
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
