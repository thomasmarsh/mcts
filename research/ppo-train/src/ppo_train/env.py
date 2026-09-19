"""Othello environment over the `othello-env` cdylib (rules are single-sourced in Rust).

The trainer and the match harness only touch the four methods of `OthelloEnv`, so tests can swap in
a scripted environment with the same shape.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

NUM_ACTIONS = 65
_WRAPPER = (
    Path(__file__).resolve().parents[4] / "crates" / "othello-env" / "python" / "othello_env.py"
)


def _load_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("othello_env", _WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OthelloEnv:
    def __init__(self, parallel: bool = True) -> None:
        self._lib = _load_wrapper()
        self._parallel = parallel

    def reset(self, n: int, max_depth: int, seed: int) -> np.ndarray:
        """`n` non-terminal states after depth ~ U[0, max_depth] random legal plies."""
        return self._lib.reset_random(n, max_depth, seed, self._parallel)

    def observe(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mover-relative `(n, 2, 8, 8)` float32 planes and an `(n, 65)` bool legal mask."""
        return self._lib.observe(states, self._parallel)

    def step(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """In-place step. Returns (reward for the mover who just moved, done)."""
        return self._lib.step(states, actions, self._parallel)

    @staticmethod
    def mover(states: np.ndarray) -> np.ndarray:
        """Colour to move: 0 = Black, 1 = White (flags bit 0)."""
        return (states[:, 2] & np.uint64(1)).astype(np.int64)
