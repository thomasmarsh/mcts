"""The coordinator's resume check: a generation's shard is reused only when a
marker written right after its self-play still describes it exactly."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

COORDINATOR = Path(__file__).parent.parent / "coordinator_othello_cnn.sh"


def _functions() -> str:
    text = COORDINATOR.read_text()
    match = re.search(r"^shard_key\(\) \{.*?^shard_is_complete\(\) \{.*?^\}\n", text, re.M | re.S)
    assert match is not None
    return match.group(0)


def _run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", _functions() + script],
        env={
            "PATH": "/usr/bin:/bin",
            "GAMES": "200",
            "SIMS": "32",
            "SELFPLAY_ENGINE": "batched",
            "MAX_CONSIDERED": "8",
            "TEMP_MOVES": "12",
            "FORCED_OPENING_PLIES": "6",
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _complete(tmp_path: Path, seed: int = 1000, **env: str) -> bool:
    shard, weights = tmp_path / "gen0.bin", tmp_path / "gen0.cnn.bin"
    result = _run(f'shard_is_complete "{shard}" "{weights}" {seed}', env)
    return result.returncode == 0


def _finish_selfplay(tmp_path: Path) -> None:
    shard, weights = tmp_path / "gen0.bin", tmp_path / "gen0.cnn.bin"
    shard.write_bytes(b"records" * 100)
    weights.write_bytes(b"weights")
    _run(f'shard_key "{shard}" "{weights}" 1000 > "{shard}.complete"', {})


def test_a_shard_with_a_matching_marker_is_reused(tmp_path: Path) -> None:
    _finish_selfplay(tmp_path)
    assert _complete(tmp_path)


def test_a_shard_without_a_marker_is_played_again(tmp_path: Path) -> None:
    _finish_selfplay(tmp_path)
    (tmp_path / "gen0.bin.complete").unlink()
    assert not _complete(tmp_path)


def test_a_truncated_shard_is_played_again(tmp_path: Path) -> None:
    _finish_selfplay(tmp_path)
    shard = tmp_path / "gen0.bin"
    shard.write_bytes(shard.read_bytes()[:-1])
    assert not _complete(tmp_path)


def test_changed_settings_seed_or_weights_invalidate_the_shard(tmp_path: Path) -> None:
    _finish_selfplay(tmp_path)
    assert not _complete(tmp_path, SIMS="16")
    assert not _complete(tmp_path, GAMES="800")
    assert not _complete(tmp_path, seed=2000)
    (tmp_path / "gen0.cnn.bin").write_bytes(b"other weights")
    assert not _complete(tmp_path)
