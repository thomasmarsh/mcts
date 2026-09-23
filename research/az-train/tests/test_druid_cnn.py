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
