"""Frozen named-family candidate-domain validation."""

from __future__ import annotations

from collections.abc import Iterable

from .codec import strict_json
from .domain import Candidate
from .schema import TuningSchema

FAMILY_EXCLUSION_POLICY_VERSION = "named-family-exclusions-v1"


def normalize_family_exclusions(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                "excluded family names must be nonempty strings without surrounding whitespace"
            )
        result.append(value)
    return tuple(sorted(set(result)))


def _family_choices(schema: TuningSchema) -> tuple[str, ...]:
    families = [parameter for parameter in schema.parameters if parameter.name == "family"]
    if len(families) != 1:
        raise ValueError("family exclusions require exactly one family parameter")
    family = families[0]
    if family.kind != "categorical" or family.choices is None:
        raise ValueError("family exclusions require a categorical family parameter")
    if any("family" in condition.children for condition in schema.conditions):
        raise ValueError("family exclusions require an unconditional family parameter")
    if not family.choices or not all(
        isinstance(choice, str) and choice for choice in family.choices
    ):
        raise ValueError("family exclusions require nonempty string family choices")
    return tuple(choice for choice in family.choices if isinstance(choice, str))


def validate_family_exclusions(schema: TuningSchema, excluded_families: tuple[str, ...]) -> None:
    if excluded_families != normalize_family_exclusions(excluded_families):
        raise ValueError("excluded families must be sorted and duplicate-free")
    if not excluded_families:
        return
    choices = _family_choices(schema)
    unknown = set(excluded_families) - set(choices)
    if unknown:
        raise ValueError(f"unknown excluded family: {sorted(unknown)[0]}")
    if len(excluded_families) == len(choices):
        raise ValueError("family exclusions cannot exclude every family")


def require_candidate_family_allowed(
    candidate: Candidate, excluded_families: tuple[str, ...]
) -> None:
    if not excluded_families:
        return
    raw = strict_json(candidate.canonical_config, "candidate configuration")
    family = raw.get("family") if isinstance(raw, dict) else None
    if not isinstance(family, str):
        raise ValueError("candidate configuration must contain a string family")
    if family in excluded_families:
        raise ValueError(f"candidate uses excluded family: {family}")
