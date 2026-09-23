from pathlib import Path

import numpy as np
import pytest

from az_train import druid_records as dr

FIXTURES = Path(__file__).resolve().parents[3] / "games/druid/cnn/fixtures"
SIZES = [5, 7]


@pytest.mark.parametrize("size", SIZES)
def test_python_planes_match_the_rust_encoder(size):
    got_size, records = dr.read_shard(FIXTURES / f"encode-{size}.shard.bin")
    assert got_size == size
    expected = np.fromfile(FIXTURES / f"encode-{size}.planes.bin", dtype="<f4").reshape(
        len(records), size, size, dr.IN_PLANES
    )
    got = dr.decode_planes(records, size)  # (N, C, H, W)
    np.testing.assert_array_equal(got.transpose(0, 2, 3, 1), expected)
    assert expected[..., 3].any(), "the fixture must contain a pending cell decision"
    assert (expected[..., 2] > 1 / 8).any(), "the fixture must contain stacks"
    assert set(np.unique(records["pending"])) == set(range(5))


@pytest.mark.parametrize("size", SIZES)
def test_derived_legality_matches_the_rust_policy_support(size):
    _, records = dr.read_shard(FIXTURES / f"encode-{size}.shard.bin")
    legal = dr.decode_legal(records, size)
    assert legal.shape == (len(records), dr.num_actions(size))
    np.testing.assert_array_equal(legal, records["policy"] > 0)
    assert (records["policy"].sum(axis=1) > 0.999).all()


def test_every_symmetry_is_a_permutation_and_an_involution():
    for size in (5, 7, 9):
        src = dr.symmetry_sources(size)
        for s in range(dr.SYMMETRIES):
            for k in range(3):
                assert sorted(src[s, k]) == list(range(size * size))
                np.testing.assert_array_equal(src[s, k][src[s, k]], np.arange(size * size))
        np.testing.assert_array_equal(src[0], np.tile(np.arange(size * size), (3, 1)))


def _flipped(records: np.ndarray, size: int, s: int) -> np.ndarray:
    """The records with the board reflected by symmetry ``s`` (fields only)."""
    out = records.copy()
    grid = lambda a: a.reshape(len(a), size, size)  # noqa: E731
    for name in ("heights", "owners"):
        g = grid(records[name])
        if s & 1:
            g = g[:, ::-1, :]
        if s & 2:
            g = g[:, :, ::-1]
        out[name] = g.reshape(len(g), -1)
    return out


@pytest.mark.parametrize("size", SIZES)
def test_symmetry_maps_planes_legality_and_policy_consistently(size):
    # The mapping of a position's legal set and planes is what training augments with. Derive
    # the legal set of the reflected board from scratch and check it equals the mapped one; the
    # Rust test `the_rules_are_invariant_under_axis_preserving_reflections` is what proves the
    # reflected game itself is the same game (legal moves, successors, winner).
    _, records = dr.read_shard(FIXTURES / f"encode-{size}.shard.bin")
    cells = size * size
    src = dr.symmetry_sources(size)
    kind = dr.phase_kind(records["pending"])
    for s in range(dr.SYMMETRIES):
        flipped = _flipped(records, size, s)
        idx = src[s][kind]  # (N, cells)
        legal, policy = dr.decode_legal(records, size), records["policy"]
        want_legal = np.concatenate(
            [np.take_along_axis(legal[:, :cells], idx, 1), legal[:, cells:]], 1
        )
        np.testing.assert_array_equal(dr.decode_legal(flipped, size), want_legal)
        want_policy = np.concatenate(
            [np.take_along_axis(policy[:, :cells], idx, 1), policy[:, cells:]], 1
        )
        np.testing.assert_allclose(want_policy > 0, want_legal)
        planes, planes_f = dr.decode_planes(records, size), dr.decode_planes(flipped, size)
        flat = planes.reshape(len(planes), dr.IN_PLANES, cells)
        got = np.take_along_axis(flat, src[s, 0][None, None, :].repeat(len(flat), 0), 2)
        got[:, 3] = np.take_along_axis(flat[:, 3], idx, 1)
        np.testing.assert_array_equal(got, planes_f.reshape(len(planes), dr.IN_PLANES, cells))


def test_reflection_is_not_a_no_op_on_lintel_anchors():
    src = dr.symmetry_sources(5)
    assert src[2, 1][0] == 2, "a horizontal lintel anchored at column 0 moves to column n-3"
    assert src[2, 2][0] == 4, "a vertical lintel's column reflects normally"
    assert src[1, 2][0] == 10, "a vertical lintel anchored at row 0 moves to row n-3"
