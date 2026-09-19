"""Hand-computed negamax GAE on a 3-ply example (gamma 0.9, lambda 0.5)."""

import numpy as np

from ppo_train.ppo import compute_gae

V = np.array([[0.5], [0.2], [-0.3]], dtype=np.float32)


def test_terminal_reward_flips_sign_each_ply():
    # The last ply ends the game with reward +1 for its mover.
    #   t2: delta = 1 - (-0.3) = 1.3                      gae = 1.3
    #   t1: delta = 0.9*(+0.3) - 0.2 = 0.07               gae = 0.07 - 0.45*1.3   = -0.515
    #   t0: delta = 0.9*(-0.2) - 0.5 = -0.68              gae = -0.68 + 0.45*0.515 = -0.44825
    reward = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
    done = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
    adv, target = compute_gae(V, reward, done, np.array([0.4], dtype=np.float32), 0.9, 0.5)
    np.testing.assert_allclose(adv[:, 0], [-0.44825, -0.515, 1.3], atol=1e-6)
    np.testing.assert_allclose(target[:, 0], [0.05175, -0.315, 1.0], atol=1e-6)


def test_bootstrap_value_enters_negated():
    # No terminal: the bootstrap value 0.4 belongs to the opponent, so it enters as -0.4.
    #   t2: delta = 0.9*(-0.4) + 0.3 = -0.06              gae = -0.06
    #   t1: delta = 0.07                                   gae = 0.07 + 0.45*0.06  = 0.097
    #   t0: delta = -0.68                                  gae = -0.68 - 0.45*0.097 = -0.72365
    zeros = np.zeros((3, 1), dtype=np.float32)
    adv, _ = compute_gae(V, zeros, zeros, np.array([0.4], dtype=np.float32), 0.9, 0.5)
    np.testing.assert_allclose(adv[:, 0], [-0.72365, 0.097, -0.06], atol=1e-6)


def test_done_cuts_the_chain():
    # A terminal at t1 (reward -1 for its mover) must not leak t2 into t0/t1.
    reward = np.array([[0.0], [-1.0], [0.0]], dtype=np.float32)
    done = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    adv, _ = compute_gae(V, reward, done, np.array([0.4], dtype=np.float32), 0.9, 0.5)
    #   t2 is a fresh episode: delta = 0.9*(-0.4) + 0.3 = -0.06
    #   t1: done -> delta = -1 - 0.2 = -1.2, gae = -1.2
    #   t0: delta = 0.9*(-0.2) - 0.5 = -0.68, gae = -0.68 + 0.45*1.2 = -0.14
    np.testing.assert_allclose(adv[:, 0], [-0.14, -1.2, -0.06], atol=1e-6)
