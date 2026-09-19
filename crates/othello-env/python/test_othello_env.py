"""ABI smoke test for the ctypes wrapper. The rules themselves are tested in Rust
(`cargo test -p othello-env`); this checks dtypes, shapes, in-place stepping and errors.

Run: cargo build --release -p othello-env && \
     uv run --with numpy --with pytest pytest crates/othello-env/python
"""

import numpy as np
import pytest

import othello_env as env


def random_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scores = rng.random(mask.shape) * mask
    return scores.argmax(axis=1).astype(np.uint8)


def test_opening_observation():
    states = env.reset_random(2, 0, seed=1)
    obs, mask = env.observe(states)
    assert obs.shape == (2, 2, 8, 8) and obs.dtype == np.float32
    assert mask.shape == (2, 65) and mask.dtype == np.bool_
    assert obs[0].sum() == 4 and obs[0, 0].sum() == 2  # 2 own + 2 opponent discs
    assert set(np.flatnonzero(mask[0])) == {19, 26, 37, 44}  # Black's four opening moves


def test_random_games_finish_with_mover_relative_rewards():
    rng = np.random.default_rng(0)
    n = 128
    states = env.reset_random(n, 5, seed=2)
    finished = np.zeros(n, dtype=bool)
    reward_at_end = np.zeros(n, dtype=np.float32)
    for _ in range(200):
        _, mask = env.observe(states)
        live = ~finished
        # Park finished envs on the opening so the batch keeps stepping.
        states[finished] = env.reset_random(int(finished.sum()), 0, seed=3)
        _, mask = env.observe(states)
        reward, done = env.step(states, random_actions(mask, rng))
        newly = live & done
        reward_at_end[newly] = reward[newly]
        finished |= newly
        if finished.all():
            break
    assert finished.all()
    assert set(np.unique(reward_at_end)) <= {-1.0, 0.0, 1.0}
    assert (reward_at_end != 0).any()


def test_step_is_in_place_and_serial_matches_parallel():
    rng = np.random.default_rng(1)
    a = env.reset_random(256, 20, seed=4)
    b = a.copy()
    _, mask = env.observe(a)
    actions = random_actions(mask, rng)
    ra, da = env.step(a, actions, parallel=False)
    rb, db = env.step(b, actions, parallel=True)
    assert (a == b).all() and (ra == rb).all() and (da == db).all()
    assert not (a == env.reset_random(256, 20, seed=4)).all()


def test_illegal_action_raises_and_leaves_state_untouched():
    states = env.reset_random(1, 0, seed=5)
    before = states.copy()
    with pytest.raises(ValueError):
        env.step(states, np.array([0]))
    assert (states == before).all()
