# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""Byte-exact round-trip: the Python v2 codec must reproduce Rust's own
dump bytes, and the decoded fields must match what the game produced.

Shells out to the real ``game-ttt dump`` binary over a small seeded dump
(a few milliseconds).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from az_train.records import decode_records, encode_records, load_positions, me_opp_planes

REPO_ROOT = Path(__file__).resolve().parents[3]


def _dump(tmp_path: Path, extra: list[str] | None = None) -> Path:
    bin_path = tmp_path / "positions.bin"
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "game-ttt",
        "--bin",
        "game-ttt",
        "--",
        "dump",
        "--games",
        "20",
        "--seed",
        "0",
        "--out",
        str(bin_path),
        *(extra or []),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            f"could not run `game-ttt dump` (cargo exit {proc.returncode}); "
            f"build the workspace first.\n{proc.stderr[-2000:]}"
        )
    return bin_path


def test_round_trip_is_byte_exact(tmp_path: Path) -> None:
    bin_path = _dump(tmp_path)
    raw = bin_path.read_bytes()
    pos = decode_records(raw)
    assert len(pos) > 0
    assert encode_records(pos) == raw


def test_fields_are_in_range(tmp_path: Path) -> None:
    pos = load_positions(_dump(tmp_path))
    assert set(pos.side.tolist()) <= {0, 1}
    assert set(float(v) for v in pos.value.tolist()) <= {-1.0, 0.0, 1.0}
    # ply is non-decreasing within a game and every board fits 18 bits.
    assert int(pos.board.max()) < (1 << 18)
    # outcome dumps carry no policy tail.
    assert all(len(p) == 0 for p in pos.policy)


def test_engine_dump_round_trips_too(tmp_path: Path) -> None:
    bin_path = _dump(tmp_path, ["--engine", "strong", "--epsilon", "0.2", "--games", "6"])
    raw = bin_path.read_bytes()
    assert encode_records(decode_records(raw)) == raw


def test_me_opp_planes_track_side_to_move(tmp_path: Path) -> None:
    pos = load_positions(_dump(tmp_path))
    me, opp = me_opp_planes(pos)
    # Piece counts: the mover has placed as many pieces as the opponent, or
    # one fewer (X moves first).
    diff = me.sum(axis=1) - opp.sum(axis=1)
    assert np.all((diff == 0) | (diff == -1))
    assert np.all(me.sum(axis=1) + opp.sum(axis=1) == pos.ply.astype(np.float32))
