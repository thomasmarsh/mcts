import json
import struct
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
    # The connectivity encoding is the base planes plus six, matching the Rust fixture exactly.
    ext = np.fromfile(FIXTURES / f"encode-{size}.planes20.bin", dtype="<f4").reshape(
        len(records), size, size, dr.CONNECT_PLANES
    )
    got20 = dr.decode_planes(records, size, dr.CONNECT_PLANES)
    np.testing.assert_array_equal(got20.transpose(0, 2, 3, 1), ext)
    np.testing.assert_array_equal(ext[..., : dr.IN_PLANES], expected)
    assert ext[..., 14:].std() > 0.05, "the connectivity planes must vary across the fixture"
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


@pytest.mark.parametrize("in_planes", [dr.IN_PLANES, dr.CONNECT_PLANES])
@pytest.mark.parametrize("size", SIZES)
def test_symmetry_maps_planes_legality_and_policy_consistently(size, in_planes):
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
        planes = dr.decode_planes(records, size, in_planes)
        planes_f = dr.decode_planes(flipped, size, in_planes)
        flat = planes.reshape(len(planes), in_planes, cells)
        got = np.take_along_axis(flat, src[s, 0][None, None, :].repeat(len(flat), 0), 2)
        got[:, 3] = np.take_along_axis(flat[:, 3], idx, 1)
        np.testing.assert_array_equal(got, planes_f.reshape(len(planes), in_planes, cells))


def test_reflection_is_not_a_no_op_on_lintel_anchors():
    src = dr.symmetry_sources(5)
    assert src[2, 1][0] == 2, "a horizontal lintel anchored at column 0 moves to column n-3"
    assert src[2, 2][0] == 4, "a vertical lintel's column reflects normally"
    assert src[1, 2][0] == 10, "a vertical lintel anchored at row 0 moves to row n-3"


# --------------------------------------------------------------------------------- v1/v2 shards


def _v1_bytes(size: int, records: np.ndarray) -> bytes:
    """Hand-writes ``records`` (``dr.record_dtype_v1(size)``) in the legacy no-``q`` layout, byte
    for byte what ``shard.rs`` wrote before the root search value existed."""
    dtype = dr.record_dtype_v1(size)
    assert records.dtype == dtype
    header = dr.MAGIC + struct.pack("<3I", size, dr.num_actions(size), dtype.itemsize)
    return header + records.tobytes()


def test_a_v1_shard_reads_with_q_equal_to_value(tmp_path):
    size = 5
    n = 6
    rec = np.zeros(n, dtype=dr.record_dtype_v1(size))
    rec["value"] = np.linspace(-1, 1, n).astype("<f4")
    rec["game"] = np.arange(n)
    rec["policy"][:, 0] = 1.0

    path = tmp_path / "v1.bin"
    path.write_bytes(_v1_bytes(size, rec))
    got_size, got = dr.read_shard(path)
    assert got_size == size
    np.testing.assert_array_equal(got["q"], got["value"])
    np.testing.assert_array_equal(got["value"], rec["value"])
    np.testing.assert_array_equal(got["game"], rec["game"])


def test_a_v2_shard_round_trips_q_as_its_own_field(tmp_path):
    size = 5
    n = 6
    rec = np.zeros(n, dtype=dr.record_dtype(size))
    rec["value"] = np.linspace(-1, 1, n).astype("<f4")
    rec["q"] = np.linspace(1, -1, n).astype("<f4")  # deliberately not equal to value
    rec["policy"][:, 0] = 1.0

    path = tmp_path / "v2.bin"
    dr.write_shard(path, size, rec)
    got_size, got = dr.read_shard(path)
    assert got_size == size
    np.testing.assert_array_equal(got["q"], rec["q"])
    assert not np.array_equal(got["q"], got["value"]), "q must be stored separately from value"


# ------------------------------------------------------------------------------ terminal sidecar


def test_read_terminal_sidecar_decodes_heights_and_owners(tmp_path):
    lines = [
        {"game": 0, "winner": 0, "capped": False, "heights": [1, 0, 0, 0], "owners": [1, 0, 0, 0]},
        {
            "game": 1,
            "winner": None,
            "capped": True,
            "heights": [0, 0, 0, 0],
            "owners": [0, 0, 0, 0],
        },
    ]
    path = tmp_path / "shard.bin.terminal.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in lines) + "\n")
    games = dr.read_terminal_sidecar(path)
    assert [g["game"] for g in games] == [0, 1]
    assert games[0]["winner"] == 0 and games[0]["capped"] is False
    assert games[1]["winner"] is None and games[1]["capped"] is True
    assert games[0]["heights"].dtype == np.int64
    np.testing.assert_array_equal(games[0]["owners"], [1, 0, 0, 0])


# ------------------------------------------------------------------------------------- BFS winner


def test_bfs_winner_finds_a_black_column_connecting_top_to_bottom():
    size = 3
    owners = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
    winner, chain = dr.bfs_winner(owners, size)
    assert winner == 0
    assert chain == [0, 3, 6]


def test_bfs_winner_finds_a_white_row_connecting_left_to_right():
    size = 3
    owners = np.array([2, 2, 2, 0, 0, 0, 0, 0, 0])
    winner, chain = dr.bfs_winner(owners, size)
    assert winner == 1
    assert chain == [0, 1, 2]


def test_bfs_winner_follows_a_bent_black_chain():
    size = 3
    # (0,0)-(1,0)-(1,1)-(2,1), the only connected Black shape from row 0 to row 2.
    owners = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0])
    winner, chain = dr.bfs_winner(owners, size)
    assert winner == 0
    assert chain == [0, 3, 4, 7]


def test_bfs_winner_is_none_on_an_empty_or_unconnected_board():
    size = 3
    assert dr.bfs_winner(np.zeros(9, dtype=np.int64), size) == (None, [])
    # A single Black cell touching neither edge fully: no connection either color.
    owners = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    assert dr.bfs_winner(owners, size) == (None, [])
