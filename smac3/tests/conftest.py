"""Shared fixtures for the ``smac3_cli`` test suite."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """Walk up from this file to the workspace root (has ``Cargo.lock``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.lock").is_file():
            return parent
    raise RuntimeError("could not locate workspace root (no Cargo.lock found)")


@pytest.fixture(scope="session")
def game_nim_binary() -> Path:
    """Path to a debug build of ``game-nim``, `mcts-tune`'s own small fixture
    game (see its ``[dev-dependencies]``), built on demand.

    A debug build is deliberate -- `tune describe` is pure metadata (no MCTS
    search runs), so a release build buys nothing here and would make this
    test slower than the rest of the suite for no benefit.
    """
    root = _repo_root()
    subprocess.run(
        ["cargo", "build", "-p", "game-nim", "--bin", "game-nim"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    binary = root / "target" / "debug" / "game-nim"
    assert binary.is_file(), f"expected cargo build to produce {binary}"
    return binary
