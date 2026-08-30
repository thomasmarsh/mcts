from __future__ import annotations

import json

from tuner_cli.domain import IterationBudget, PairTask, TaskCase
from tuner_cli.target import _splitmix_seed, parse_pair_output


def test_strict_pair_parser_decodes_ordered_games() -> None:
    case = TaskCase("task", "tuning", 0, 42, "opponent", "fingerprint", "game")
    task = PairTask("pair", "candidate", case, IterationBudget(10))
    metrics = {"iterations_total": 2, "iterations_first_half": 1, "move_time_ms": 3}
    records = [
        {
            "type": "configured_match_result",
            "seq": 1,
            "round": 1,
            "seed": _splitmix_seed(42),
            "candidate_side": "first",
            "outcome": "candidate_win",
            "trace_game_seq": None,
            "plies": 2,
            "elapsed_ms": 3,
            "candidate": metrics,
            "baseline": metrics,
        },
        {
            "type": "configured_match_result",
            "seq": 2,
            "round": 1,
            "seed": _splitmix_seed(42),
            "candidate_side": "second",
            "outcome": "draw",
            "trace_game_seq": None,
            "plies": 2,
            "elapsed_ms": 3,
            "candidate": metrics,
            "baseline": metrics,
        },
        {"type": "configured_comparison_summary", "games": 2, "wins": 1, "losses": 0, "draws": 1},
    ]
    result = parse_pair_output("\n".join(json.dumps(record) for record in records), task)
    assert [game.candidate_side for game in result.games] == ["first", "second"]
