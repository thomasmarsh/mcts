import json
import math
from pathlib import Path

from az_train.druid_dashboard import implied_elo, summarize


def test_implied_elo_chains_through_the_opponent():
    elo = implied_elo({1: (0, 0.5), 11: (1, 0.75), 12: (0, 0.75)})
    assert elo[1] == 0.0
    assert math.isclose(elo[12], 400 * math.log10(3))
    assert math.isclose(elo[11], elo[1] + 400 * math.log10(3))


def test_summarize_reads_druid_diagnostics_and_survives_a_missing_gate(tmp_path: Path):
    phase = {"positions": 5, "forced": 1, "policy_ce": 2.9, "uniform_ce": 2.7, "policy_top1": 0.2}
    row = {
        "gen": 1,
        "selfplay": {"games_started": 4, "games_finished": 4, "draws": 1, "black_wins": 2},
        "train": {"fit_seconds": 3.0},
        "val_after": {
            "policy_ce": 1.3,
            "policy_top1": 0.4,
            "value_mse": 1.1,
            "value_pearson": -0.1,
        },
        "diagnostics": {
            "selfplay": {"phase_share": {"piece": 0.5, "orientation": 0.1, "cell": 0.4}},
            "policy": {"by_phase": {"cell": phase}},
            "value": {"held": {"mse": 1.1, "pearson": -0.1}, "train": {"mse": 0.6}, "mse_gap": 0.5},
            "calibration": {"ece": 0.1, "brier": 0.3, "bins": []},
        },
    }
    (tmp_path / "log.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "steps.jsonl").write_text(
        json.dumps({"gen": 0, "value_loss": 1.0, "policy_loss": 2.0}) + "\n"
    )
    g = summarize(tmp_path, 10)["gens"][0]
    assert g["gate_skipped"] and g["score"] is None
    assert g["draw"] == 0.25 and g["v_gap"] == 0.5 and g["vt_mse"] == 0.6
    assert g["phase_ce"] == {"cell": 2.9} and g["phase_uniform"] == {"cell": 2.7}
