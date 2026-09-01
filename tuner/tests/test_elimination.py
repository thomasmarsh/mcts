from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tuner_cli.artifacts import ActiveEliminationSpecification, read_manifest
from tuner_cli.domain import ReplayState, ShadowCandidateDecision, ShadowRaceDecision
from tuner_cli.elimination import active_elimination_allocation


def test_active_elimination_sampling_is_deterministic_and_ignores_non_eliminations() -> None:
    manifest = replace(
        read_manifest(Path(__file__).parent / "fixtures" / "version4" / "manifest.json"),
        active_elimination=ActiveEliminationSpecification(0.5),
    )
    race = ShadowRaceDecision(
        0,
        manifest.tuning_blocks[0].prefix_id,
        (),
        "boundary",
        (
            ShadowCandidateDecision("candidate-a", 0, 4096, "eliminate"),
            ShadowCandidateDecision("candidate-b", 0, 4096, "continue"),
            ShadowCandidateDecision("boundary", 0, 4096, "eliminate"),
        ),
        manifest.shadow_policy.method_version,
    )
    state = ReplayState((), (), (), (), (), (), None, "open", 0, None)

    first = active_elimination_allocation(manifest, state, race)
    second = active_elimination_allocation(manifest, state, race)

    assert first == second
    assert [item.candidate_id for item in first.actions] == ["candidate-a"]
    assert first.actions[0].decision_margin == 0.05
