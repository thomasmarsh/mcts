"""The shadow-race dispatcher records only the manifest-selected policy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tuner_cli.artifacts import Manifest, SuccessiveHalvingPolicySpecification, read_manifest
from tuner_cli.evidence import read_events
from tuner_cli.race_policy import decide_shadow_race, shadow_prefix_eligible
from tuner_cli.replay import replay

FIXTURES = Path(__file__).parent / "fixtures" / "version4"


def _halving(manifest: Manifest) -> Manifest:
    policy = SuccessiveHalvingPolicySpecification(
        "successive_halving",
        "successive-halving-common-prefix-eta2-v1",
        2,
        0.0,
        12,
        manifest.finalists,
        "tuning-point-estimate-fingerprint-v1",
    )
    return replace(manifest, shadow_policy=policy)


@pytest.fixture
def manifest() -> Manifest:
    return read_manifest(FIXTURES / "manifest.json")


def test_only_the_twelve_pair_non_final_prefix_is_eligible(manifest: Manifest) -> None:
    eligible = [
        block for block in manifest.tuning_blocks if shadow_prefix_eligible(manifest, block)
    ]
    assert [block.length for block in eligible] == [12]
    assert not shadow_prefix_eligible(manifest, manifest.tuning_prefix)
    # Eligibility is policy-neutral: swapping the tag does not move the boundary.
    assert [
        block.length
        for block in manifest.tuning_blocks
        if shadow_prefix_eligible(_halving(manifest), block)
    ] == [12]


def test_dispatch_follows_the_manifest_tag() -> None:
    events = read_events(FIXTURES / "evidence.jsonl")
    manifest = read_manifest(FIXTURES / "manifest.json")
    state = replay(manifest, events[:137])
    prefix = next(block for block in manifest.tuning_blocks if block.length == 12)

    paired = decide_shadow_race(manifest, state, 0, prefix)
    halving = decide_shadow_race(_halving(manifest), state, 0, prefix)

    assert paired.policy_kind == "paired_bootstrap"
    assert halving.policy_kind == "successive_halving"
    assert halving.policy_version == "successive-halving-common-prefix-eta2-v1"


def test_dispatch_rejects_an_ineligible_prefix() -> None:
    events = read_events(FIXTURES / "evidence.jsonl")
    manifest = read_manifest(FIXTURES / "manifest.json")
    state = replay(manifest, events[:137])

    with pytest.raises(ValueError, match="ineligible tuning prefix"):
        decide_shadow_race(manifest, state, 0, manifest.tuning_prefix)
