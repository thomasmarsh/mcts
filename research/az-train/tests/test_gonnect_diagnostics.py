import numpy as np
import pytest

from az_train import gonnect_diagnostics as gd
from az_train.gonnect_records import Positions

SIZE, CELLS = 5, 25
ACTIONS = CELLS + 2


def positions(n, legal_count=4, seed=0):
    rng = np.random.default_rng(seed)
    planes = np.zeros((n, 7, SIZE, SIZE), dtype=np.uint8)
    legal = np.zeros((n, ACTIONS), dtype=bool)
    policy = np.zeros((n, ACTIONS), dtype=np.float32)
    for i in range(n):
        cells = rng.choice(CELLS, legal_count, replace=False)
        legal[i, cells] = True
        policy[i, cells] = 1.0 / legal_count
    planes[: n // 2, 2, 0, 0] = 1  # half the positions have a ko cell
    return Positions(planes, policy, legal, np.ones(n, dtype=np.float32), np.arange(n))


def test_a_calibrated_predictor_has_zero_ece_and_a_confident_wrong_one_a_large_one():
    n = 1000
    rng = np.random.default_rng(1)
    pred = np.full(n, 0.6)  # win probability 0.8
    outcome = np.where(rng.random(n) < 0.8, 1.0, -1.0)
    good = gd.calibration(pred, outcome)
    assert good["ece"] < 0.04
    assert sum(b["count"] for b in good["bins"]) == n
    bad = gd.calibration(np.full(n, 0.9), np.where(rng.random(n) < 0.5, 1.0, -1.0))
    assert bad["ece"] > 0.3 and bad["brier"] > good["brier"]


def test_uniform_targets_have_entropy_ln_k_and_a_matching_prior_has_zero_kl():
    n = 40
    shard = positions(n, legal_count=4)
    held = shard.slice(0, 10)
    logits = np.zeros((10, ACTIONS), dtype=np.float32)
    stats = {"games_started": 8, "games_finished": 6, "games_capped": 2, "mean_plies": 31.5,
             "black_wins": 3, "swaps": 1}  # fmt: skip
    out = gd.diagnostics(shard, stats, held, np.zeros(10), logits)
    assert out["policy"]["target_entropy"] == pytest.approx(np.log(4), abs=1e-6)
    assert out["policy"]["target_entropy_normalised"] == pytest.approx(1.0, abs=1e-6)
    assert out["policy"]["prior_entropy"] == pytest.approx(np.log(4), abs=1e-6)
    assert out["policy"]["kl_target_to_prior"] == pytest.approx(0.0, abs=1e-6)
    sp = out["selfplay"]
    assert sp["capped_rate"] == pytest.approx(0.25)
    assert sp["black_win_rate"] == pytest.approx(0.5) and sp["swap_rate"] == pytest.approx(1 / 6)
    assert sp["ko_position_rate"] == pytest.approx(0.5)
    assert sp["mean_legal_moves"] == 4.0


def test_a_confident_wrong_prior_has_positive_kl_and_low_entropy():
    shard = positions(20)
    held = shard.slice(0, 10)
    logits = np.full((10, ACTIONS), -20.0, dtype=np.float32)
    for i in range(10):
        logits[i, np.flatnonzero(held.legal[i])[0]] = 20.0
    out = gd.diagnostics(
        shard, {"games_started": 1, "games_finished": 1}, held, np.zeros(10), logits
    )
    assert out["policy"]["prior_entropy"] < 0.01
    assert out["policy"]["kl_target_to_prior"] > 1.0


def test_illegal_actions_get_no_prior_mass():
    held = positions(6)
    prior = gd.masked_softmax(np.random.default_rng(2).normal(size=(6, ACTIONS)), held.legal)
    assert (prior[~held.legal] == 0).all()
    np.testing.assert_allclose(prior.sum(axis=1), 1.0, atol=1e-6)
