from __future__ import annotations

from tuner_cli.domain import Opponent
from tuner_cli.identity import opponent_panel
from tuner_cli.tasks import build_corpus, weighted_schedule


def test_weighted_schedule_has_exact_complete_cycles_and_disjoint_phase_seeds() -> None:
    panel = opponent_panel(
        (
            Opponent("a", "schema_default", "A", "default", 1, "{}", "a"),
            Opponent("b", "inline", "B", "historical_reference", 2, '{"b":1}', "b"),
            Opponent("c", "inline", "C", "historical_reference", 3, '{"c":1}', "c"),
        )
    )
    assert weighted_schedule(panel, 6) == (2, 1, 0, 2, 1, 2)
    tuning = build_corpus("tuning", 6, 9, panel, "game")
    validation = build_corpus("validation", 6, 9, panel, "game")
    assert [case.opponent_id for case in tuning.cases].count("a") == 1
    assert [case.opponent_id for case in tuning.cases].count("b") == 2
    assert not {case.seed for case in tuning.cases} & {case.seed for case in validation.cases}
    assert tuning.fingerprint != validation.fingerprint
