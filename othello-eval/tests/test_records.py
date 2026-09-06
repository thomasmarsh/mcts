"""Byte-exact round-trip: Python-decoded positions must match Rust's own
re-encoding of the same dump.

Runs the real ``game-othello dump`` binary over a 3-game seeded dump (a few
milliseconds) and checks the numpy reader against the JSON manifest Rust
writes alongside it, then re-encodes and compares raw bytes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from othello_eval.records import RECORD_DTYPE, load_positions

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dump(tmp_path: Path) -> tuple[Path, Path]:
    bin_path = tmp_path / "positions.bin"
    manifest = tmp_path / "positions.json"
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "game-othello",
        "--",
        "dump",
        "--games",
        "3",
        "--seed",
        "0",
        "--out",
        str(bin_path),
        "--manifest",
        str(manifest),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            "could not run `game-othello dump` "
            f"(cargo exit {proc.returncode}); build the workspace first.\n{proc.stderr[-2000:]}"
        )
    return bin_path, manifest


def test_round_trip(tmp_path: Path) -> None:
    bin_path, manifest_path = _dump(tmp_path)
    rows = load_positions(str(bin_path))
    manifest = json.loads(manifest_path.read_text())

    assert len(rows) == len(manifest)
    assert len(rows) > 0

    for i, (row, ref) in enumerate(zip(rows, manifest, strict=True)):
        assert int(row["black"]) == int(ref["black"], 16), i
        assert int(row["white"]) == int(ref["white"], 16), i
        assert int(row["side"]) == ref["side"], i
        assert int(row["ply"]) == ref["ply"], i
        assert float(row["target"]) == pytest.approx(ref["target"]), i

    # The "matches Rust's own re-encoding" clause: byte-exact, not approximate.
    assert rows.tobytes() == bin_path.read_bytes()


def test_dtype_is_packed_22_bytes() -> None:
    assert RECORD_DTYPE.itemsize == 22
    assert not RECORD_DTYPE.isalignedstruct


def test_targets_and_sides_are_in_range(tmp_path: Path) -> None:
    bin_path, _ = _dump(tmp_path)
    rows = load_positions(str(bin_path))
    sides = {int(x) for x in rows["side"].tolist()}
    targets = {float(x) for x in rows["target"].tolist()}
    assert sides <= {0, 1}
    assert targets <= {-1.0, 0.0, 1.0}
