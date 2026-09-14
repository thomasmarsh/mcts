# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false, reportMissingTypeStubs=false
"""Checks for ``az_train.convnet_othello``: the bit-plane conversion (hand-
verifiable, deterministic) and an end-to-end CLI smoke test against a real,
tiny ``game-othello dump --label gumbel --head cnn`` corpus (the same
shell-out-and-skip-if-unavailable pattern ``test_records_othello.py`` uses).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from othello_eval import convnet

from az_train import convnet_othello
from az_train.records_othello import Positions

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_me_opp_planes_matches_known_bit_pattern() -> None:
    # Black to move, black holds squares 0 and 63, white holds square 1.
    pos = Positions(
        black=np.asarray([1 | (1 << 63)], dtype=np.uint64),
        white=np.asarray([1 << 1], dtype=np.uint64),
        side=np.asarray([0], dtype=np.uint8),
        ply=np.asarray([4], dtype=np.uint8),
        value=np.asarray([0.0], dtype=np.float32),
        policy=[[(0, 1.0)]],
    )
    me, opp = convnet_othello.me_opp_planes(pos)
    assert me.shape == (1, 64)
    assert opp.shape == (1, 64)
    expected_me = np.zeros(64, dtype=np.float32)
    expected_me[[0, 63]] = 1.0
    expected_opp = np.zeros(64, dtype=np.float32)
    expected_opp[1] = 1.0
    np.testing.assert_array_equal(me[0], expected_me)
    np.testing.assert_array_equal(opp[0], expected_opp)


def test_me_opp_planes_swaps_perspective_for_white_to_move() -> None:
    pos = Positions(
        black=np.asarray([1], dtype=np.uint64),
        white=np.asarray([1 << 1], dtype=np.uint64),
        side=np.asarray([1], dtype=np.uint8),
        ply=np.asarray([4], dtype=np.uint8),
        value=np.asarray([0.0], dtype=np.float32),
        policy=[[(1, 1.0)]],
    )
    me, opp = convnet_othello.me_opp_planes(pos)
    assert me[0, 1] == 1.0 and me[0, 0] == 0.0
    assert opp[0, 0] == 1.0 and opp[0, 1] == 0.0


def _dump_cnn(tmp_path: Path) -> Path:
    # `--head cnn` self-play costs ~60s/game under an unoptimized debug build
    # (this hand-rolled numpy-style convolution has no `--release`-only fast
    # path); shell out to a `--release` build instead of `cargo run`'s debug
    # default, matching every coordinator script's own convention.
    build = subprocess.run(
        ["cargo", "build", "--release", "-q", "-p", "game-othello", "--bin", "game-othello"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if build.returncode != 0:
        pytest.skip(
            f"could not build game-othello --release (exit {build.returncode}): "
            f"{build.stderr[-800:]}"
        )
    bin_path = tmp_path / "gen0.bin"
    cmd = [
        str(REPO_ROOT / "target" / "release" / "game-othello"),
        "dump", "--label", "gumbel", "--head", "cnn",
        "--games", "3", "--seed", "0", "--sims", "4", "--max-considered", "2",
        "--out", str(bin_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            f"could not run `game-othello dump --head cnn` (exit {proc.returncode}): "
            f"{proc.stderr[-800:]}"
        )
    return bin_path


def test_train_cli_fits_and_writes_a_loadable_checkpoint(tmp_path: Path) -> None:
    dump = _dump_cnn(tmp_path)
    out = tmp_path / "gen1.cnn.bin"
    convnet_othello.train_cli(
        [
            "--positions", str(dump), "--out", str(out),
            "--validation-fraction", "0.3", "--epochs", "1",
            "--batch-size", "8", "--report-every", "1", "--validate-every", "1",
        ]
    )
    assert out.exists()
    weights = convnet.read_weights(str(out))
    assert weights.shape == (convnet.N_WEIGHTS,)

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["model"] == "OTCNN001"
    assert meta["n_weights"] == convnet.N_WEIGHTS
    assert meta["train"]["positions"] > 0
    assert "final_validation_metrics" in meta["metrics"]
