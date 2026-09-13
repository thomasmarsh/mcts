# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""Byte-exact round-trip: the Python RecordV2 codec must reproduce Rust's
own ``game-othello dump --label gumbel`` bytes.

Shells out to the real binary over a tiny seeded dump.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from az_train.records_othello import (
    Positions,
    decode_records,
    encode_records,
    game_slices,
    load_positions,
    me_opp_bits,
    split_by_game,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _dump(tmp_path: Path, extra: list[str] | None = None) -> Path:
    bin_path = tmp_path / "positions.bin"
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "game-othello",
        "--bin",
        "game-othello",
        "--",
        "dump",
        "--label",
        "gumbel",
        "--games",
        "6",
        "--seed",
        "0",
        "--sims",
        "8",
        "--max-considered",
        "4",
        "--out",
        str(bin_path),
        *(extra or []),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            f"could not run `game-othello dump` (cargo exit {proc.returncode}); "
            f"build the workspace first.\n{proc.stderr[-2000:]}"
        )
    return bin_path


def test_round_trip_is_byte_exact(tmp_path: Path) -> None:
    raw = _dump(tmp_path).read_bytes()
    pos = decode_records(raw)
    assert len(pos) > 0
    assert encode_records(pos) == raw


def test_fields_are_in_range_and_policy_tail_sums_to_one(tmp_path: Path) -> None:
    pos = load_positions(_dump(tmp_path))
    assert set(pos.side.tolist()) <= {0, 1}
    assert set(float(v) for v in pos.value.tolist()) <= {-1.0, 0.0, 1.0}
    assert np.all((pos.black & pos.white) == 0)
    for entries in pos.policy:
        assert len(entries) > 0
        squares = [s for s, _ in entries]
        assert all(0 <= s <= 64 for s in squares)  # 64 == Move::PASS
        assert len(set(squares)) == len(squares)
        assert abs(sum(p for _, p in entries) - 1.0) < 1e-3


def test_me_opp_bits_track_side_to_move(tmp_path: Path) -> None:
    pos = load_positions(_dump(tmp_path))
    me, opp = me_opp_bits(pos)
    assert np.all((me & opp) == 0)
    black_to_move = pos.side == 0
    assert np.array_equal(me[black_to_move], pos.black[black_to_move])
    assert np.array_equal(opp[black_to_move], pos.white[black_to_move])
    assert np.array_equal(me[~black_to_move], pos.white[~black_to_move])


def test_ply_is_non_decreasing_within_a_game_including_across_passes(tmp_path: Path) -> None:
    pos = load_positions(_dump(tmp_path))
    for s in game_slices(pos):
        run = pos.ply[s].astype(np.int64)
        assert run[0] == 0
        assert np.all(np.diff(run) >= 0)


def _synthetic_games() -> Positions:
    # Three games; ply is non-decreasing within each (a pass repeats a ply,
    # but never at ply 0 -- the opening position always has a legal move in
    # real Othello, so ply 0 uniquely marks a game's first record).
    ply = np.asarray([0, 1, 1, 2, 0, 1, 0, 1, 1, 2], dtype=np.uint8)
    game_id = np.asarray([0, 0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.uint64)
    return Positions(
        black=game_id,
        white=np.zeros(len(ply), dtype=np.uint64),
        side=ply % 2,
        ply=ply,
        value=np.ones(len(ply), dtype=np.float32),
        policy=[[(0, 1.0)] for _ in ply],
    )


def test_game_slices_tolerates_a_repeated_ply_from_a_pass() -> None:
    slices = game_slices(_synthetic_games())
    assert [s.stop - s.start for s in slices] == [4, 2, 4]


def test_whole_games_never_straddle_the_split() -> None:
    train, validation, train_games, validation_games = split_by_game(_synthetic_games(), 0.34, 9)
    assert train_games + validation_games == 3
    assert set(train.black.tolist()).isdisjoint(validation.black.tolist())


def test_bad_ply_sequence_has_a_useful_error() -> None:
    pos = _synthetic_games()
    pos.ply[1] = 3  # first game's ply becomes [0, 3, 1, 2] -- a real decrease
    with pytest.raises(ValueError, match="less than the previous record's ply"):
        game_slices(pos)
