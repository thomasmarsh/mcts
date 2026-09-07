# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""Byte-exact round-trip: the Python v2-connect4 codec must reproduce
Rust's own dump bytes, and the decoded planes must feed ``ntuple_c4``.

Shells out to the real ``game-connect4 dump`` binary over a small seeded
dump (a few milliseconds).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from az_train.ntuple_c4 import CELLS, N_WINDOWS, features
from az_train.records_c4 import (
    COLS,
    Positions,
    decode_records,
    encode_records,
    load_positions,
    me_opp_planes,
    split_by_game,
)
from az_train.train import train_cli, value_metrics


def test_reference_diagnostic_magic_is_not_replay() -> None:
    with pytest.raises(ValueError, match="not a v2-connect4 replay"):
        decode_records(b"C4REFD01" + b"\0" * 32)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _dump(tmp_path: Path, extra: list[str] | None = None) -> Path:
    bin_path = tmp_path / "positions.bin"
    cmd = [
        "cargo", "run", "-q", "-p", "game-connect4", "--bin", "game-connect4", "--",
        "dump", "--games", "20", "--seed", "0", "--out", str(bin_path),
        *(extra or []),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            f"could not run `game-connect4 dump` (cargo exit {proc.returncode}); "
            f"build the workspace first.\n{proc.stderr[-2000:]}"
        )
    return bin_path


def test_round_trip_is_byte_exact(tmp_path: Path) -> None:
    raw = _dump(tmp_path).read_bytes()
    pos = decode_records(raw)
    assert len(pos) > 0
    assert encode_records(pos) == raw


def test_fields_are_in_range(tmp_path: Path) -> None:
    pos = load_positions(_dump(tmp_path))
    assert set(pos.side.tolist()) <= {0, 1}
    assert set(float(v) for v in pos.value.tolist()) <= {-1.0, 0.0, 1.0}
    # Every board fits 42 bits and no cell holds both colors.
    assert int(pos.black.max()) < (1 << 42)
    assert int(pos.white.max()) < (1 << 42)
    assert np.all((pos.black & pos.white) == 0)
    # outcome dumps carry no policy tail.
    assert all(len(p) == 0 for p in pos.policy)


def test_me_opp_planes_track_side_to_move(tmp_path: Path) -> None:
    pos = load_positions(_dump(tmp_path))
    me, opp = me_opp_planes(pos)
    assert me.shape == (len(pos), CELLS)
    # Black moves first: the mover has placed as many discs as the opponent
    # (Black to move) or one fewer (White to move).
    diff = me.sum(axis=1) - opp.sum(axis=1)
    assert np.all((diff == 0) | (diff == -1))
    assert np.all(me.sum(axis=1) + opp.sum(axis=1) == pos.ply.astype(np.float32))
    assert np.all((me == 0) | (opp == 0))


def test_planes_feed_the_ntuple_design_matrix(tmp_path: Path) -> None:
    pos = load_positions(_dump(tmp_path))
    me, opp = me_opp_planes(pos)
    x = features(me, opp)
    # One bias column plus one selected column per window.
    assert np.allclose(x.sum(axis=1), 1 + N_WINDOWS)


def test_gumbel_dump_carries_a_policy_tail(tmp_path: Path) -> None:
    extra = ["--label", "gumbel", "--sims", "16", "--max-considered", "4"]
    raw = _dump(tmp_path, extra).read_bytes()
    pos = decode_records(raw)
    assert len(pos) > 0
    assert encode_records(pos) == raw
    for entries in pos.policy:
        assert len(entries) > 0, "gumbel positions carry a Sequential-Halving policy target"
        cols = [c for c, _ in entries]
        assert all(0 <= c < COLS for c in cols)
        assert len(set(cols)) == len(cols)
        assert abs(sum(p for _, p in entries) - 1.0) < 1e-4


def _synthetic_games() -> Positions:
    # Three games with distinct board words make leakage easy to detect.
    ply = np.asarray([0, 1, 2, 0, 1, 0, 1, 2, 3], dtype=np.uint8)
    return Positions(
        black=np.arange(len(ply), dtype=np.uint64), white=np.zeros(len(ply), dtype=np.uint64),
        side=ply % 2, ply=ply, value=np.ones(len(ply), dtype=np.float32),
        policy=[[] for _ in ply],
    )


def test_whole_games_never_straddle_the_split() -> None:
    train, validation, train_games, validation_games = split_by_game(_synthetic_games(), 0.34, 9)
    assert train_games + validation_games == 3
    assert set(train.black.tolist()).isdisjoint(validation.black.tolist())
    # A game marker identifies all of its records: every game lands together.
    assert {len(train), len(validation)} in ({2, 7}, {3, 6}, {4, 5})


def test_bad_ply_sequence_has_a_useful_error() -> None:
    pos = _synthetic_games()
    pos.ply[2] = 4
    with pytest.raises(ValueError, match="expected 2"):
        split_by_game(pos)


def test_constant_metric_cases_are_json_safe() -> None:
    metrics = value_metrics(np.zeros(3), np.zeros(3))
    assert metrics["pearson"] == 0.0
    assert metrics["sign_agreement"] == 0.0
    import json
    assert "NaN" not in json.dumps(metrics, allow_nan=False)


def test_connect4_cli_defaults_to_direct_and_writes_held_out_games(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(encode_records(_synthetic_games()))
    out = tmp_path / "weights.bin"
    held_out = tmp_path / "validation.bin"
    train_cli([
        "--game", "connect4", "--positions", str(source), "--out", str(out),
        "--l2", "1", "--validation-records-out", str(held_out),
    ])
    import json
    meta = json.loads(out.with_suffix(".bin.meta.json").read_text())
    assert meta["value_target"] == "direct"
    assert meta["metrics"]["train_games"] + meta["metrics"]["validation_games"] == 3
    assert len(decode_records(held_out.read_bytes())) > 0
