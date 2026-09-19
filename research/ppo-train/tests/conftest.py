"""Shared fixtures: tiny CPU nets (never a training target) and a scripted one-ply env."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppo_train.config import Config, load_config
from ppo_train.env import OthelloEnv

TINY = {
    "run.device": "cpu",
    "net.blocks": 1,
    "net.channels": 8,
    "net.value_hidden": 8,
    "rollout.num_envs": 64,
    "rollout.rollout_len": 4,
    "rollout.infer_chunk": 64,
    "ppo.minibatches": 2,
    "ppo.micro_batch": 32,
    "reset.pool": 128,
    "reset.max_depth": 30,
    "opponent.push_every": 2,
    "opponent.ring_size": 3,
    "eval.games": 32,
}


def tiny_cfg(**overrides: object) -> Config:
    named = {k.replace("__", "."): v for k, v in overrides.items()}
    return load_config(overrides={**TINY, **named})


class WinInOneEnv:
    """Every state is a one-ply game: exactly one of squares 0..7 wins (+1), the rest lose (-1).

    The winning square is the set bit of the own plane, so a policy has to read the board.
    """

    def reset(self, n: int, max_depth: int, seed: int) -> np.ndarray:
        del max_depth
        target = np.random.default_rng(seed).integers(0, 8, n)
        states = np.zeros((n, 3), dtype=np.uint64)
        states[:, 0] = np.uint64(1) << target.astype(np.uint64)
        return states

    @staticmethod
    def target(states: np.ndarray) -> np.ndarray:
        return np.log2(states[:, 0].astype(np.float64)).round().astype(np.int64)

    def observe(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(states)
        obs = np.zeros((n, 2, 64), dtype=np.float32)
        obs[np.arange(n), 0, self.target(states)] = 1.0
        mask = np.zeros((n, 65), dtype=np.bool_)
        mask[:, :8] = True
        return obs.reshape(n, 2, 8, 8), mask

    def step(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reward = np.where(actions == self.target(states), 1.0, -1.0).astype(np.float32)
        return reward, np.ones(len(states), dtype=np.bool_)

    def mover(self, states: np.ndarray) -> np.ndarray:
        return np.zeros(len(states), dtype=np.int64)


@pytest.fixture(scope="session")
def othello() -> OthelloEnv:
    return OthelloEnv(parallel=False)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
