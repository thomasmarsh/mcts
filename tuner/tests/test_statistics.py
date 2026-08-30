from __future__ import annotations

from tuner_cli.domain import Estimate
from tuner_cli.statistics import marginal_interval, paired_difference, tie_relation


def test_intervals_and_tie_relation() -> None:
    interval = marginal_interval((0.5, 1.0))
    assert interval.mean == 0.75
    assert interval.lower == 0.0
    assert interval.upper == 1.0
    difference = paired_difference((1.0, 1.0), (0.0, 0.0))
    assert difference.mean == 1.0
    assert tie_relation(Estimate(0.1, 0.0, 0.2)) == "tie"
    assert tie_relation(Estimate(0.5, 0.1, 0.8)) == "better"
    assert tie_relation(Estimate(-0.5, -0.8, -0.1)) == "worse"
