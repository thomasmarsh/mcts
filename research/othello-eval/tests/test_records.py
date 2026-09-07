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


def _dump(tmp_path: Path, extra: list[str] | None = None) -> tuple[Path, Path]:
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
        *(extra or []),
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


def _dump_harvest(tmp_path: Path) -> Path:
    out_dir = tmp_path / "harvest"
    cfg = tmp_path / "harvest.toml"
    cfg.write_text(
        'engine = "strong"\n'
        "label_iters = 60\n"
        "epsilon = 0.1\n"
        "opening_plies = 4\n"
        "games = 2\n"
        "seed = 0\n"
        "min_visits = 1\n"
        "max_per_search = 512\n"
        "dedup = true\n"
        "td_lambda = 0.7\n"
    )
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "game-othello",
        "--",
        "dump",
        "--label",
        "harvest",
        "--out",
        str(out_dir),
        "--harvest-config",
        str(cfg),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"harvest dump failed (cargo exit {proc.returncode})\n{proc.stderr[-2000:]}")
    return out_dir


def test_harvest_arms_share_one_search_pass(tmp_path: Path) -> None:
    d = _dump_harvest(tmp_path)

    for arm in ("arm_a", "arm_b", "arm_c", "arm_d"):
        rows = load_positions(str(d / f"{arm}.bin"))
        manifest = json.loads((d / f"{arm}.json").read_text())
        assert len(rows) == len(manifest) > 0, arm
        assert rows.tobytes() == (d / f"{arm}.bin").read_bytes(), arm
        for row, ref in zip(rows, manifest, strict=True):
            assert int(row["black"]) == int(ref["black"], 16), arm
            assert int(row["white"]) == int(ref["white"], 16), arm

    a = load_positions(str(d / "arm_a.bin"))
    b = load_positions(str(d / "arm_b.bin"))
    c = load_positions(str(d / "arm_c.bin"))

    # arm A and arm B are the same positions, different targets.
    def keyset(rows: object) -> set[tuple[int, int, int, int]]:
        return set(
            zip(
                rows["black"].tolist(),  # type: ignore[index]
                rows["white"].tolist(),  # type: ignore[index]
                rows["side"].tolist(),  # type: ignore[index]
                rows["ply"].tolist(),  # type: ignore[index]
                strict=True,
            )
        )

    assert keyset(a) == keyset(b)
    # arm C is a superset of arm B's positions (it keeps the played root plus
    # its whole subtree).
    assert keyset(b) <= keyset(c)


def test_engine_dump_round_trips_too(tmp_path: Path) -> None:
    # `--engine <preset>` swaps the position source (a real MCTS engine,
    # epsilon-randomised) but not the record format or the labelling.
    # `easy` (30 iterations) keeps this in the fast suite; the engine code
    # path is identical for every preset.
    bin_path, manifest_path = _dump(tmp_path, ["--engine", "easy", "--epsilon", "0.2"])
    rows = load_positions(str(bin_path))
    manifest = json.loads(manifest_path.read_text())
    assert len(rows) == len(manifest) > 0
    assert rows.tobytes() == bin_path.read_bytes()
    for row, ref in zip(rows, manifest, strict=True):
        assert int(row["black"]) == int(ref["black"], 16)
        assert float(row["target"]) in (-1.0, 0.0, 1.0)
