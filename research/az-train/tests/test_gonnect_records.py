from pathlib import Path

import numpy as np
import pytest

from az_train import gonnect_records as gr

FIXTURES = Path(__file__).resolve().parents[3] / "games/gonnect/cnn/fixtures"


SIZES = [7, 9]


@pytest.mark.parametrize("want", SIZES)
def test_python_planes_match_the_rust_encoder(want):
    size, records = gr.read_shard(FIXTURES / f"encode-{want}.shard.bin")
    assert size == want
    expected = np.fromfile(FIXTURES / f"encode-{want}.planes.bin", dtype="<f4").reshape(
        len(records), size, size, 7
    )
    got = gr.decode_planes(records, size)  # (N, C, H, W)
    np.testing.assert_array_equal(got.transpose(0, 2, 3, 1).astype(np.float32), expected)
    assert expected[..., 2].any(), "the fixture must contain a ko position"
    assert expected[..., 3].any(), "the fixture must contain a swap-available position"
    if size == 9:
        assert expected[..., 0][:, 7:, :].any() or expected[..., 1][:, 7:, :].any(), (
            "a 9x9 fixture must use cells past bit 63"
        )


@pytest.mark.parametrize("want", SIZES)
def test_legal_mask_matches_the_policy_support_in_the_fixture(want):
    size, records = gr.read_shard(FIXTURES / f"encode-{want}.shard.bin")
    legal = gr.decode_legal(records, size)
    assert legal.shape == (len(records), size * size + 2)
    np.testing.assert_array_equal(legal, records["policy"] > 0)
    assert (records["policy"].sum(axis=1) > 0.999).all()


@pytest.mark.parametrize("size", [5, 7, 9, 13, 19])
def test_bit_unpacking_covers_every_cell_of_every_supported_size(size):
    cells = size * size
    words = gr.mask_words(size)
    rng = np.random.default_rng(size)
    truth = rng.integers(0, 2, size=(4, cells), dtype=np.uint8)
    mask = np.zeros((4, words), dtype="<u8")
    for c in range(cells):
        mask[:, c // 64] |= truth[:, c].astype("<u8") << np.uint64(c % 64)
    np.testing.assert_array_equal(gr._bits(mask, cells), truth)
