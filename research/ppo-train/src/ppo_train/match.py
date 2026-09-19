"""Paired matches on an `Env`: both colours from each opening, greedy or uniform-random players."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ppo_train.env import NUM_ACTIONS
from ppo_train.ppo import Env, infer

Player = Callable[[np.ndarray, np.ndarray], np.ndarray]


def greedy_player(net: torch.nn.Module, device: torch.device, chunk: int = 2048) -> Player:
    """Argmax of the legal-masked logits, literal orientation."""

    def play(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        net.eval()
        logits, _ = infer(net, obs, mask, chunk, device)
        return logits.argmax(axis=-1).astype(np.uint8)

    return play


def random_player(rng: np.random.Generator) -> Player:
    def play(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del obs
        return (rng.random(mask.shape) * mask).argmax(axis=-1).astype(np.uint8)

    return play


def random_opening(env: Env, n: int, plies: int, rng: np.random.Generator) -> np.ndarray:
    """`n` positions after exactly `plies` uniformly random legal plies from the start."""
    states = env.reset(n, 0, int(rng.integers(2**62)))
    play = random_player(rng)
    for _ in range(plies):
        obs, mask = env.observe(states)
        env.step(states, play(obs, mask))
    return states


def play_match(env: Env, player_a: Player, player_b: Player, openings: np.ndarray) -> np.ndarray:
    """Play every opening twice, A as Black then A as White.

    Returns a `(2 * len(openings),)` array of A's result in {-1, 0, +1}; the first half is A as
    Black. Reward attribution goes through the mover of the terminal ply, never the seat.
    """
    m = len(openings)
    states = np.concatenate([openings, openings]).copy()
    a_colour = np.concatenate([np.zeros(m, np.int64), np.ones(m, np.int64)])
    result = np.zeros(2 * m, dtype=np.float32)
    active = np.arange(2 * m)
    while active.size:
        sub = states[active]
        obs, mask = env.observe(sub)
        a_turn = env.mover(sub) == a_colour[active]
        actions = np.zeros(active.size, dtype=np.uint8)
        if a_turn.any():
            actions[a_turn] = player_a(obs[a_turn], mask[a_turn])
        if (~a_turn).any():
            actions[~a_turn] = player_b(obs[~a_turn], mask[~a_turn])
        reward, done = env.step(sub, actions)
        states[active] = sub
        finished = np.flatnonzero(done)
        result[active[finished]] = np.where(a_turn[finished], reward[finished], -reward[finished])
        active = active[~done]
    return result


def score(result: np.ndarray) -> float:
    """Mean of win = 1, draw = 0.5, loss = 0."""
    return float(((result > 0) + 0.5 * (result == 0)).mean())


def evaluate_vs_random(
    env: Env,
    net: torch.nn.Module,
    device: torch.device,
    games: int,
    opening_plies: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Greedy policy against uniform random over both colours (`games` total)."""
    assert games % 2 == 0 and NUM_ACTIONS == 65
    openings = random_opening(env, games // 2, opening_plies, rng)
    result = play_match(env, greedy_player(net, device), random_player(rng), openings)
    return {
        "vs_random_win": float((result > 0).mean()),
        "vs_random_draw": float((result == 0).mean()),
        "vs_random_loss": float((result < 0).mean()),
        "vs_random_score": score(result),
    }
