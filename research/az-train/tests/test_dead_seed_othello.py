from __future__ import annotations

from az_train.dead_seed_othello import parse_seeds


def test_parse_seeds_accepts_ranges_lists_and_mixes() -> None:
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("3,7,9") == [3, 7, 9]
    assert parse_seeds("0-1,5,8-9") == [0, 1, 5, 8, 9]
