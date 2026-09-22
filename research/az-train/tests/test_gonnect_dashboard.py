import math

from az_train.gonnect_dashboard import implied_elo


def test_implied_elo_chains_through_the_opponent():
    elo = implied_elo({1: (0, 0.5), 11: (1, 0.75), 12: (0, 0.75)})
    assert elo[1] == 0.0
    assert math.isclose(elo[12], 400 * math.log10(3))
    assert math.isclose(elo[11], elo[1] + 400 * math.log10(3))


def test_implied_elo_clamps_saturated_scores_and_skips_unreachable_opponents():
    elo = implied_elo({1: (0, 1.0), 2: (7, 0.6)})
    assert math.isclose(elo[1], 400 * math.log10(0.98 / 0.02))
    assert 2 not in elo
