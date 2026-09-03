"""Resolved-shape preview of a foreground-tuner launch.

Where :mod:`tuner_cli.preflight` answers *"is this launch legal?"*, this
answers *"what exactly would this launch do?"* -- the resolved opponent panel
(schema-default opponents expanded to their actual config), the tuning space
the proposer will explore after ``--exclude-family`` and ``space_overrides``,
the three phase efforts and pair budgets with the counts they buy, the
effective ``game_config``, and the objective-epoch fingerprint the run will
carry.

Like preflight, nothing here writes to disk or plays a game: every authority
(`validate_options`, `game_spec`, `resolve_objective`, `manifest_for`,
`constrained_schema`) is reused verbatim from :mod:`tuner_cli.run`, so a plan
can never drift from what a real launch resolves to. The returned object also
embeds the preflight ``ok`` / ``errors`` so a single call covers both.
"""

from __future__ import annotations

from pathlib import Path

from .codec import JsonObject, JsonValue, strict_json
from .effort import encode_effort
from .identity import canonical_json
from .objective import resolve_objective
from .preflight import preflight_launch
from .run import (
    RunOptions,
    game_spec,
    manifest_for,
    schema_default,
    validate_options,
)
from .schema import ActivationCondition, GameSpec, ParameterSpec, TuningSchema
from .space_overrides import constrained_schema
from .target import GameBinaryTarget, Target

_PLAN_RUN_ID = "plan-preview"


def plan_launch(options: RunOptions, target: Target | None = None) -> JsonObject:
    """A structured summary of what a fresh run with ``options`` resolves to.

    Always carries ``ok`` / ``errors`` from :func:`preflight_launch`. When
    resolution gets far enough, also carries ``opponents``, ``space``,
    ``efforts``, ``budgets``, ``game_config`` and ``epoch``; when an early
    stage raises, only the fields resolved so far are present and ``errors``
    explains the stop.
    """
    preflight = preflight_launch(options, target)
    raw_errors = preflight["errors"]
    errors: list[JsonValue] = list(raw_errors) if isinstance(raw_errors, list) else []
    result: JsonObject = {"ok": preflight["ok"], "errors": errors}
    result["budgets"] = _budgets(options)
    result["efforts"] = {
        "tuning": encode_effort(options.tuning_effort),
        "validation": encode_effort(options.validation_effort),
        "production": encode_effort(options.production_effort),
    }

    try:
        binary, _, objective_path = validate_options(options)
        resolved_target = target or GameBinaryTarget(binary)
        spec = game_spec(resolved_target, binary)
        objective = resolve_objective(
            objective_path,
            spec.kind,
            schema_default(spec, options.seed),
            spec.game_config_schema,
            spec.default_game_config,
        )
    except (OSError, RuntimeError, ValueError):
        # preflight already recorded the message; the plan just stays partial.
        return result

    result["game_kind"] = spec.kind
    result["objective_id"] = objective.objective_id
    result["objective_fingerprint"] = objective.fingerprint
    result["game_config"] = objective.game_config
    result["game_config_is_override"] = objective.game_config != _canonical_default(spec)
    result["opponents"] = [
        {
            "id": opponent.opponent_id,
            "label": opponent.label,
            "role": opponent.role,
            "weight": opponent.weight,
            "source": opponent.source_id,
            "config": opponent.canonical_config,
            "fingerprint": opponent.configuration_fingerprint,
        }
        for opponent in objective.panel.opponents
    ]
    result["panel_fingerprint"] = objective.panel.fingerprint
    result["space"] = _space_summary(spec.tuning, options)

    try:
        manifest = manifest_for(options, _plan_dir(options), spec, objective)
    except (OSError, RuntimeError, ValueError):
        return result
    result["epoch"] = {
        "epoch_id": manifest.epoch.epoch_id,
        "fingerprint": manifest.epoch.fingerprint,
    }
    return result


def _plan_dir(options: RunOptions) -> Path:
    # `manifest_for` only reads `directory.name` (for `manifest.run_id`); the
    # scientific fingerprint / epoch it computes is name-independent, so a
    # fixed placeholder keeps the preview from touching the real run dir.
    return options.run_dir.parent / _PLAN_RUN_ID


def _canonical_default(spec: GameSpec) -> str:
    return canonical_json(strict_json(spec.default_game_config, "default game config"))


def _budgets(options: RunOptions) -> JsonObject:
    validation_per_finalist = (
        options.validation_pair_budget // options.finalists if options.finalists else 0
    )
    return {
        "cohort_size": options.cohort_size,
        "finalists": options.finalists,
        "bootstrap_candidates": options.bootstrap_candidates,
        "random_reserve_candidates": options.random_reserve_candidates,
        "tuning_pairs": options.tuning_pairs,
        "tuning_pair_budget": options.tuning_pair_budget,
        "validation_pair_budget": options.validation_pair_budget,
        "diagnostic_pair_budget": options.diagnostic_pair_budget,
        "production_validation_pairs": options.production_validation_pairs,
        "proposer_policy": options.proposer_policy,
        "derived": {
            "initial_cohort_pairs": options.cohort_size * options.tuning_pairs,
            "validation_pairs_per_finalist": validation_per_finalist,
            "production_pairs": options.production_validation_pairs,
        },
    }


def _space_summary(schema: TuningSchema, options: RunOptions) -> JsonObject:
    constrained = constrained_schema(schema, options.space_overrides)
    conditioned = {
        child: condition for condition in constrained.conditions for child in condition.children
    }
    parameters: list[JsonValue] = [
        _parameter_summary(parameter, conditioned.get(parameter.name), options.excluded_families)
        for parameter in constrained.parameters
    ]
    return {
        "schema_id": constrained.id,
        "families": _effective_families(constrained, options.excluded_families),
        "excluded_families": list(options.excluded_families),
        "parameters": parameters,
    }


def _parameter_summary(
    parameter: ParameterSpec,
    condition: ActivationCondition | None,
    excluded_families: tuple[str, ...],
) -> JsonObject:
    choices: list[JsonValue] | None = None
    if parameter.choices is not None:
        choices = [
            choice
            for choice in parameter.choices
            if not (parameter.name == "family" and choice in excluded_families)
        ]
    active_when = (
        f"{condition.parent} in {list(condition.values)}" if condition is not None else None
    )
    return {
        "name": parameter.name,
        "kind": parameter.kind,
        "bounds": list(parameter.bounds) if parameter.bounds is not None else None,
        "choices": choices,
        "default": parameter.default,
        "constant_value": parameter.constant_value,
        "active_when": active_when,
    }


def _effective_families(
    schema: TuningSchema, excluded_families: tuple[str, ...]
) -> list[JsonValue]:
    for parameter in schema.parameters:
        if parameter.name != "family":
            continue
        if parameter.kind == "constant":
            return [parameter.constant_value]
        if parameter.choices is not None:
            return [c for c in parameter.choices if c not in excluded_families]
    return []
