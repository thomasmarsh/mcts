from pathlib import Path

import numpy as np
import torch

from az_train import druid_cnn as dc
from az_train import druid_records as dr

FIXTURES = Path(__file__).resolve().parents[3] / "games/druid/cnn/fixtures"


def test_augment_agrees_with_decoding_the_reflected_position():
    for size in (5, 7):
        _, records = dr.read_shard(FIXTURES / f"encode-{size}.shard.bin")
        pos = dr.load_positions(FIXTURES / f"encode-{size}.shard.bin")[1]
        src = torch.from_numpy(dr.symmetry_sources(size))
        x, pi, legal, kind = (
            torch.from_numpy(a)
            for a in (pos.planes, pos.policy, pos.legal.astype(np.uint8), pos.kind)
        )
        for s in range(dr.SYMMETRIES):
            flipped = records.copy()
            for name in ("heights", "owners"):
                g = records[name].reshape(len(records), size, size)
                g = g[:, ::-1, :] if s & 1 else g
                g = g[:, :, ::-1] if s & 2 else g
                flipped[name] = g.reshape(len(records), -1)
            ax, api, alegal = dc.augment(x, pi, legal, kind, src, size, torch.full((len(x),), s))
            np.testing.assert_array_equal(ax.numpy(), dr.decode_planes(flipped, size))
            np.testing.assert_array_equal(alegal.numpy() > 0, dr.decode_legal(flipped, size))
            np.testing.assert_allclose(api.numpy().sum(1), pos.policy.sum(1), rtol=1e-6)
            np.testing.assert_array_equal(api.numpy() > 0, alegal.numpy() > 0)


def _row(draw: float, pearson: float | None, over: float | None) -> dict:
    return {
        "diagnostics": {
            "selfplay": {"draw_rate": draw},
            "value": {"held": {"pearson": pearson, "mse_vs_constant": over}},
        }
    }


def test_druid_warnings_flag_draws_and_an_anti_correlated_value_head():
    healthy = [_row(0.02, 0.3, 0.8)] * 10
    assert dc.druid_warnings(healthy) == []
    sick = [_row(0.2, -0.2, 1.3)] * 10
    text = " | ".join(dc.druid_warnings(sick))
    assert "draw rate" in text and "Pearson" in text and "constant predictor" in text
    assert dc.druid_warnings([]) == []
    # Only the last 10 generations count, and a missing Pearson (constant head) is skipped.
    assert dc.druid_warnings([_row(0.9, -1.0, 2.0)] * 5 + healthy) == []
    assert dc.druid_warnings([_row(0.0, None, None)] * 3) == []


def test_geometry_of_reads_the_head_kind():
    net = {"size": 7, "channels": 8, "blocks": 1, "policy_planes": 2, "value_planes": 1}
    net["value_hidden"] = 4
    assert dc.geometry_of({"net": net}).head == "dense"
    agnostic = dc.geometry_of({"net": {**net, "head": "agnostic"}})
    assert agnostic.head == "agnostic" and agnostic.policy_out == 7 * 7 + 4
