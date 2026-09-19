"""Seat and reward bookkeeping: an agent playing itself scores exactly 0.500 over both colours."""

import numpy as np
import torch
from conftest import tiny_cfg

from ppo_train.match import (
    evaluate_vs_random,
    greedy_player,
    play_match,
    random_opening,
    random_player,
    score,
)
from ppo_train.net import build_net


def test_agent_vs_itself_scores_exactly_half(othello):
    net = build_net(tiny_cfg(net__arch="none").net)
    play = greedy_player(net, torch.device("cpu"))
    rng = np.random.default_rng(0)
    openings = random_opening(othello, 24, 4, rng)
    result = play_match(othello, play, play, openings)
    assert len(result) == 48
    assert (result != 0).any(), "games must reach decisive terminals for the check to mean anything"
    assert score(result) == 0.5
    # Same policy, same opening, colours swapped: the two games are the same game, opposite results.
    np.testing.assert_array_equal(result[:24], -result[24:])


def test_random_vs_random_is_balanced_and_terminates(othello):
    rng = np.random.default_rng(1)
    openings = random_opening(othello, 200, 4, rng)
    result = play_match(othello, random_player(rng), random_player(rng), openings)
    assert abs(score(result) - 0.5) < 0.1


def test_untrained_net_vs_random_reports_a_score(othello):
    net = build_net(tiny_cfg().net)
    out = evaluate_vs_random(othello, net, torch.device("cpu"), 32, 0, np.random.default_rng(2))
    assert abs(out["vs_random_win"] + out["vs_random_draw"] + out["vs_random_loss"] - 1) < 1e-9
