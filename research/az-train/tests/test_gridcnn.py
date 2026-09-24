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


def test_dense_state_dict_keys_are_unchanged_by_the_agnostic_head():
    keys = list(gridcnn.GridCNN(gridcnn.FIXTURE_GEOMETRY).state_dict())
    assert [k.split(".")[0] for k in keys if not k.startswith(("stem", "blocks"))][:1] == [
        "policy_conv"
    ]
    assert "policy_dense.weight" in keys and "value_conv.conv.weight" in keys
    assert not any(k.startswith("policy_cell") for k in keys)


def test_stored_agnostic_fixture_matches_the_torch_model():
    g, flat, planes, values, logits = gridcnn.fixture(g=gridcnn.FIXTURE_GEOMETRY_AGNOSTIC)
    stored_g, stored_flat = gridcnn.read_weights(FIXTURES / "agnostic/weights.bin")
    io = json.loads((FIXTURES / "agnostic/io.json").read_text())
    assert stored_g == g and g.head == gridcnn.HEAD_AGNOSTIC
    np.testing.assert_allclose(stored_flat, flat, atol=1e-6)
    np.testing.assert_allclose(io["values"], values, atol=1e-5)
    np.testing.assert_allclose(io["logits"], logits.ravel(), atol=1e-4)


def test_agnostic_weight_count_does_not_depend_on_the_board_size():
    g = gridcnn.FIXTURE_GEOMETRY_AGNOSTIC
    bigger = gridcnn.Geometry(**{**g.__dict__, "size": 9, "policy_out": 83})
    assert bigger.n_weights() == g.n_weights()
    small, big = gridcnn.GridCNN(g), gridcnn.GridCNN(bigger)
    assert small.state_dict().keys() == big.state_dict().keys()
    assert all(v.shape == big.state_dict()[k].shape for k, v in small.state_dict().items())


def test_agnostic_weights_file_round_trips(tmp_path: Path):
    g = gridcnn.FIXTURE_GEOMETRY_AGNOSTIC
    flat = np.arange(g.n_weights(), dtype=np.float32) / 1000.0
    gridcnn.write_weights(tmp_path / "w.bin", g, flat)
    back_g, back = gridcnn.read_weights(tmp_path / "w.bin")
    assert back_g == g
    np.testing.assert_array_equal(back, flat)


def test_agnostic_net_runs_at_any_board_size_with_the_same_weights():
    g = gridcnn.FIXTURE_GEOMETRY_AGNOSTIC
    model = gridcnn.GridCNN(g).eval()
    for size in (5, 7, 9):
        x = torch.rand(3, g.in_planes, size, size)
        with torch.no_grad():
            value, logits = model(x)
        assert value.shape == (3,) and logits.shape == (3, size * size + (g.policy_out - g.cells))


def test_warm_start_trunk_copies_only_the_trunk_across_board_sizes():
    small = gridcnn.FIXTURE_GEOMETRY
    big = gridcnn.Geometry(**{**small.__dict__, "size": 9, "policy_out": 83})
    src, dst = gridcnn.GridCNN(small), gridcnn.GridCNN(big)
    gridcnn._fill_random(src, 5)
    copied = gridcnn.warm_start(dst, src.state_dict(), "trunk")
    assert copied and all(k.startswith(("stem.", "blocks.")) for k in copied)
    for k in copied:
        assert torch.equal(dst.state_dict()[k], src.state_dict()[k])
    assert not torch.equal(dst.policy_conv.conv.weight, src.policy_conv.conv.weight)


def test_warm_start_full_loads_an_agnostic_net_at_another_size():
    a = gridcnn.FIXTURE_GEOMETRY_AGNOSTIC
    bigger = gridcnn.Geometry(**{**a.__dict__, "size": 9, "policy_out": 83})
    src, dst = gridcnn.GridCNN(a), gridcnn.GridCNN(bigger)
    gridcnn._fill_random(src, 5)
    gridcnn.warm_start(dst, src.state_dict(), "full")
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)
