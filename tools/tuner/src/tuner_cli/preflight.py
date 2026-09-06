"""Dry-run validation of a foreground-tuner launch.

Runs exactly the checks `run_foreground` performs *before* it creates the run
directory or starts any search -- argument coherence (`validate_options`), the
game-spec / space-constraint checks, objective resolution, the
option-vs-panel cross-checks (`validate_objective_options`), and the
schema-default-vs-panel binary check (`preflight_default`) -- and reports what
fails as a JSON line. The bench server's launch form calls this so a launch
can never be started for a reason that was knowable up front.

Nothing here writes to disk or plays a game; the authorities are reused
verbatim from `run.py`, so this cannot drift from what a real launch enforces.
"""

from __future__ import annotations

from .codec import JsonObject, JsonValue
from .objective import resolve_objective
from .run import (
    RunOptions,
    game_spec,
    preflight_default,
    resolved_constraints,
    schema_default,
    validate_objective_options,
    validate_options,
)
from .target import GameBinaryTarget, Target


def preflight_launch(options: RunOptions, target: Target | None = None) -> JsonObject:
    """``{"ok": True}`` when a fresh run with these options would get past
    every pre-search check, else ``{"ok": False, "errors": [...]}``.

    Each stage raises on its first problem, so `errors` holds one message per
    stage that failed (argument coherence, then objective/panel); fix and
    re-check to surface the next.
    """
    errors: list[JsonValue] = []
    try:
        binary, _, objective_path = validate_options(options)
    except (OSError, ValueError) as error:
        return {"ok": False, "errors": [str(error)]}

    resolved_target = target or GameBinaryTarget(binary)
    try:
        spec = game_spec(resolved_target, binary)
        constraints = resolved_constraints(spec, options)
        objective = resolve_objective(
            objective_path,
            spec.kind,
            schema_default(spec, options.seed),
            spec.game_config_schema,
            spec.default_game_config,
        )
        validate_objective_options(options, objective)
        preflight_default(resolved_target, spec, objective, options.seed, constraints)
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(str(error))

    return {"ok": not errors, "errors": errors}
