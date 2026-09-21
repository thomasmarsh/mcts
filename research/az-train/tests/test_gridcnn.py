import json
from pathlib import Path

import numpy as np
import torch

from az_train import gridcnn

FIXTURES = Path(__file__).resolve().parents[3] / "crates/grid-cnn/tests/fixtures"


def test_stored_fixture_matches_the_torch_model():
    g, flat, planes, values, logits = gridcnn.fixture()
    stored_g, stored_flat = gridcnn.read_weights(FIXTURES / "weights.bin")
    io = json.loads((FIXTURES / "io.json").read_text())
    assert stored_g == g
    np.testing.assert_allclose(stored_flat, flat, atol=1e-6)
    np.testing.assert_allclose(io["planes"], planes.ravel(), atol=0)
    np.testing.assert_allclose(io["values"], values, atol=1e-5)
    np.testing.assert_allclose(io["logits"], logits.ravel(), atol=1e-4)


def test_weights_file_round_trips(tmp_path: Path):
    g = gridcnn.FIXTURE_GEOMETRY
    flat = np.arange(g.n_weights(), dtype=np.float32) / 1000.0
    gridcnn.write_weights(tmp_path / "w.bin", g, flat)
    back_g, back = gridcnn.read_weights(tmp_path / "w.bin")
    assert back_g == g
    np.testing.assert_array_equal(back, flat)


def test_zero_model_exports_all_zeros_and_outputs_zero():
    g = gridcnn.FIXTURE_GEOMETRY
    model = gridcnn.zero_model(g).eval()
    assert not gridcnn.export_flat(model).any()
    value, logits = model(torch.ones(2, g.in_planes, g.size, g.size))
    assert not value.any() and not logits.any()


def test_d4_apply_yields_eight_distinct_images_forming_a_group():
    x = torch.arange(25.0).reshape(5, 5)
    images = [gridcnn.d4_apply(x, s) for s in range(8)]
    assert len({tuple(i.flatten().tolist()) for i in images}) == 8
    for a in images:
        for b in range(8):
            assert any(torch.equal(gridcnn.d4_apply(a, b), c) for c in images)
