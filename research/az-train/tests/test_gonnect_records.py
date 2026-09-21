from pathlib import Path

import numpy as np

from az_train import gonnect_records as gr

FIXTURES = Path(__file__).resolve().parents[3] / "games/gonnect/cnn/fixtures"


def test_python_planes_match_the_rust_encoder():
    size, records = gr.read_shard(FIXTURES / "encode.shard.bin")
    assert size == 7
    expected = np.fromfile(FIXTURES / "encode.planes.bin", dtype="<f4").reshape(
        len(records), 7, 7, 7
    )
    got = gr.decode_planes(records, size)  # (N, C, H, W)
    np.testing.assert_array_equal(got.transpose(0, 2, 3, 1).astype(np.float32), expected)
    assert expected[..., 2].any(), "the fixture must contain a ko position"
    assert expected[..., 3].any(), "the fixture must contain a swap-available position"


def test_legal_mask_matches_the_policy_support_in_the_fixture():
    size, records = gr.read_shard(FIXTURES / "encode.shard.bin")
    legal = gr.decode_legal(records, size)
    assert legal.shape == (len(records), 51)
    np.testing.assert_array_equal(legal, records["policy"] > 0)
    assert (records["policy"].sum(axis=1) > 0.999).all()
