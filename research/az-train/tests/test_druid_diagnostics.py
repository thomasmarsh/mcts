from pathlib import Path

import numpy as np

from az_train import druid_diagnostics as dd
from az_train import druid_records as dr

FIXTURES = Path(__file__).resolve().parents[3] / "games/druid/cnn/fixtures"


def _positions(size: int = 5) -> dr.Positions:
    return dr.load_positions(FIXTURES / f"encode-{size}.shard.bin")[1]


def test_phase_matches_the_pending_field_and_covers_all_three_phases():
    for size in (5, 7):
        _, records = dr.read_shard(FIXTURES / f"encode-{size}.shard.bin")
        legal = dr.decode_legal(records, size)
        phase = dd.phase_of(legal, size)
        pending = records["pending"]
        # A piece kind is chosen only with nothing pending; the orientation only after a lintel.
        np.testing.assert_array_equal(phase == 0, (pending == dr.NONE) & legal.any(axis=1))
        np.testing.assert_array_equal(phase == 1, pending == dr.LINTEL_CHOSEN)
        assert set(np.unique(phase)) == {0, 1, 2}


def test_value_fit_flags_an_anti_correlated_head():
    target = np.array([1, -1, 1, -1, 1, -1], dtype=np.float32)
    good = dd.value_fit(0.5 * target, target)
    assert good["pearson"] > 0.99 and good["mse_vs_constant"] < 1
    bad = dd.value_fit(-0.5 * target, target)
    assert bad["pearson"] < -0.99 and bad["mse_vs_constant"] > 1
    assert dd.value_fit(np.zeros(6, dtype=np.float32), target)["pearson"] is None


def test_policy_by_phase_uniform_logits_score_the_uniform_baseline():
    pos = _positions()
    logits = np.zeros(pos.policy.shape, dtype=np.float32)
    out = dd.policy_by_phase(pos, logits, 5)
    for name, row in out.items():
        if row["policy_ce"] is not None:
            # A uniform prior over the legal actions scores log(legal) against any target that
            # sums to one over them.
            assert abs(row["policy_ce"] - row["uniform_ce"]) < 1e-4, name
    # Forced positions are counted apart, never in the per-phase means.
    n_legal = pos.legal.sum(axis=1)
    assert sum(r["positions"] + r["forced"] for r in out.values()) == len(pos)
    assert sum(r["positions"] for r in out.values()) == int((n_legal > 1).sum())


def test_diagnostics_reports_draws_phases_and_the_value_gap():
    pos = _positions()
    n = len(pos)
    rng = np.random.default_rng(0)
    stats = {
        "games_started": 10,
        "games_finished": 10,
        "draws": 2,
        "black_wins": 5,
        "games_capped": 0,
    }
    train_pred = rng.uniform(-1, 1, 50).astype(np.float32)
    train_target = np.sign(train_pred).astype(np.float32)
    d = dd.diagnostics(
        pos, stats, pos, np.zeros(n, np.float32), np.zeros(pos.policy.shape, np.float32), 5,
        train_pred, train_target,
    )  # fmt: skip
    assert d["selfplay"]["draw_rate"] == 0.2
    assert abs(sum(d["selfplay"]["phase_share"].values()) - 1) < 1e-9
    assert set(d["policy"]["by_phase"]) == set(dd.PHASES)
    assert d["value"]["mse_gap"] == d["value"]["held"]["mse"] - d["value"]["train"]["mse"]
    assert d["value"]["train"]["pearson"] > 0.8


def test_value_by_progress_separates_early_noise_from_late_fit():
    from az_train.druid_diagnostics import game_progress, value_by_progress

    game = np.repeat([7, 3], 10)  # two games of 10 plies, the second listed after the first
    assert np.allclose(game_progress(game)[:10], np.arange(10) / 10)
    target = np.where(np.arange(20) % 2 == 0, 1.0, -1.0)
    # Perfect in the second half of each game, always wrong in the first half.
    pred = np.where(np.tile(np.arange(10), 2) < 5, -target, target)
    bins = value_by_progress(pred, target, game, bins=2)
    assert bins[0]["mse"] == 4.0 and bins[1]["mse"] == 0.0 and bins[0]["n"] == 10


def test_druid_warns_when_prior_entropy_collapses():
    from az_train import druid_cnn as dc

    def row(e):
        return {"diagnostics": {"policy": {"prior_entropy": e},
                                "selfplay": {"draw_rate": 0.0},
                                "value": {"held": {"pearson": 0.4, "mse_vs_constant": 0.8}}}}

    steady = [row(1.0)] * 30
    assert not any("rigid" in w for w in dc.druid_warnings(steady))
    assert any("rigid" in w for w in dc.druid_warnings([row(1.0)] * 10 + [row(0.5)] * 20))
