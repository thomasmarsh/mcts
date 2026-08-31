from __future__ import annotations

import pytest

from tuner_cli.domain import ObservationContext, SearchEffort, TaskPrefix
from tuner_cli.observations import observation, paired_difference


def test_paired_difference_requires_full_context_identity() -> None:
    prefix = TaskPrefix("prefix", "corpus", 2, ("one", "two"))
    context = ObservationContext("epoch", "validation", prefix, SearchEffort("iterations", 10))
    left, right = (
        observation("left", context, (1.0, 0.5)),
        observation("right", context, (0.5, 0.0)),
    )
    assert paired_difference(left, right).mean == 0.5
    changed = observation(
        "right",
        ObservationContext("other", "validation", prefix, SearchEffort("iterations", 10)),
        (0.5, 0.0),
    )
    with pytest.raises(ValueError, match="epoch"):
        paired_difference(left, changed)
