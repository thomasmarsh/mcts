"""Structural validation of a deployment-objective file.

Reuses :func:`tuner_cli.objective.resolve_objective` -- the same authority the
foreground run path applies -- so the panel rules (exactly one schema-default
opponent, reduced weights, distinct effective configs, ...) live in one place.
Emits a single JSON line so a caller (the bench server's objective editor) can
report precise errors without re-implementing the schema.
"""

from __future__ import annotations

from pathlib import Path

from .codec import JsonObject
from .objective import resolve_objective
from .run import game_spec, schema_default
from .target import GameBinaryTarget


def validate_objective_file(game_binary: Path, objective_file: Path) -> JsonObject:
    """Return ``{"ok": True, "objective_id", "panel_fingerprint"}`` when the
    file is a well-formed objective for the game the binary describes, or
    ``{"ok": False, "errors": [...]}`` otherwise."""
    try:
        binary = game_binary.expanduser().resolve()
        spec = game_spec(GameBinaryTarget(binary), binary)
        default = schema_default(spec, 0)
        resolved = resolve_objective(
            objective_file,
            spec.kind,
            default,
            spec.game_config_schema,
            spec.default_game_config,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return {"ok": False, "errors": [str(error)]}
    return {
        "ok": True,
        "objective_id": resolved.objective_id,
        "panel_fingerprint": resolved.panel.fingerprint,
    }
