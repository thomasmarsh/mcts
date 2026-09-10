# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: E501
"""Fast checks for the gen0-reservoir training-row resample (no CNN fit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from az_train.mixture_selfplay_c4 import (
    gen0_reservoir_resample,
    load_split_replay,
)
from az_train.records_c4 import Positions, encode_records, game_slices, load_positions
from az_train.trainer_hygiene_c4 import replay_game_split_indices


def _write_shard(path: Path, n_games: int, plies_per_game: int, value: float) -> int:
    plies = list(range(plies_per_game)) * n_games
    n = len(plies)
    pos = Positions(
        black=np.zeros(n, dtype=np.uint64),
        white=np.zeros(n, dtype=np.uint64),
        side=np.array([p % 2 for p in plies], dtype=np.uint8),
        ply=np.array(plies, dtype=np.uint8),
        value=np.full(n, value, dtype=np.float32),
        policy=[[(3, 1.0)] for _ in plies],
    )
    path.write_bytes(encode_records(pos))
    return n


def _shards(tmp_path: Path) -> tuple[list[Path], list[Path], int]:
    gen0 = tmp_path / "gen0.bin"
    gen1 = tmp_path / "gen1.bin"
    n0 = _write_shard(gen0, n_games=20, plies_per_game=6, value=1.0)
    _write_shard(gen1, n_games=20, plies_per_game=6, value=-1.0)
    sv0 = tmp_path / "gen0.f32"
    sv1 = tmp_path / "gen1.f32"
    sv0.write_bytes(np.zeros(n0, dtype="<f4").tobytes())
    sv1.write_bytes(np.zeros(n0, dtype="<f4").tobytes())
    return [gen0, gen1], [sv0, sv1], n0


def _canonical_train_idx(positions_paths: list[Path], *, validation_fraction: float, split_seed: int) -> np.ndarray:
    from az_train.fitability_c4 import _concat

    pos = _concat([load_positions(p) for p in positions_paths])
    game_bounds = [(int(s.start), int(s.stop)) for s in game_slices(pos)]
    train_idx, _held, _tg, _hg = replay_game_split_indices(game_bounds, validation_fraction, split_seed)
    return train_idx


def test_fraction_zero_is_the_plain_whole_game_train_split(tmp_path: Path) -> None:
    pos_paths, sv_paths, _n0 = _shards(tmp_path)
    canonical = np.sort(_canonical_train_idx(pos_paths, validation_fraction=0.25, split_seed=0))

    train, _held, counts = load_split_replay(
        pos_paths, sv_paths, validation_fraction=0.25, split_seed=0, gen0_reservoir_fraction=0.0
    )
    reservoir: dict[str, object] = counts["gen0_reservoir"]  # type: ignore[assignment]
    assert reservoir["fraction"] == 0.0
    assert reservoir["resampled_train_records"] == reservoir["canonical_train_records"] == canonical.size
    # Every training row carries one policy entry, so the packed train rows line
    # up one-for-one with the canonical whole-game split, in sorted order.
    assert train["outcome"].size == canonical.size


def test_resample_fraction_half_gives_even_gen0_share(tmp_path: Path) -> None:
    pos_paths, _sv, n0 = _shards(tmp_path)
    canonical = _canonical_train_idx(pos_paths, validation_fraction=0.25, split_seed=0)
    rng = np.random.default_rng(20260909)
    resampled = gen0_reservoir_resample(canonical, n0, 0.5, rng)

    assert resampled.size == canonical.size
    assert np.array_equal(resampled, np.sort(resampled))
    gen0_share = float(np.mean(resampled < n0))
    assert abs(gen0_share - 0.5) < 0.05


def test_single_generation_replay_is_a_no_op(tmp_path: Path) -> None:
    # Generation 0's fit sees only gen0.bin; the reservoir has nothing to hold
    # against and must fall back to the plain whole-game split, not raise.
    gen0 = tmp_path / "gen0.bin"
    n0 = _write_shard(gen0, n_games=20, plies_per_game=6, value=1.0)
    sv0 = tmp_path / "gen0.f32"
    sv0.write_bytes(np.zeros(n0, dtype="<f4").tobytes())
    canonical = np.sort(_canonical_train_idx([gen0], validation_fraction=0.25, split_seed=0))

    train, _held, counts = load_split_replay(
        [gen0], [sv0], validation_fraction=0.25, split_seed=0, gen0_reservoir_fraction=0.5
    )
    reservoir: dict[str, object] = counts["gen0_reservoir"]  # type: ignore[assignment]
    assert reservoir["fraction"] == 0.5
    assert reservoir["resampled_train_records"] == reservoir["canonical_train_records"] == canonical.size
    assert train["outcome"].size == canonical.size

    resampled = gen0_reservoir_resample(canonical, n0, 0.5, np.random.default_rng(0))
    assert np.array_equal(resampled, canonical)


def test_resample_fraction_one_draws_only_gen0(tmp_path: Path) -> None:
    pos_paths, _sv, n0 = _shards(tmp_path)
    canonical = _canonical_train_idx(pos_paths, validation_fraction=0.25, split_seed=0)
    resampled = gen0_reservoir_resample(canonical, n0, 1.0, np.random.default_rng(1))
    assert resampled.size == canonical.size
    assert np.all(resampled < n0)


def test_held_out_split_is_unchanged_by_the_reservoir(tmp_path: Path) -> None:
    pos_paths, sv_paths, _n0 = _shards(tmp_path)
    _t0, held0, _c0 = load_split_replay(
        pos_paths, sv_paths, validation_fraction=0.25, split_seed=0, gen0_reservoir_fraction=0.0
    )
    _t1, held1, _c1 = load_split_replay(
        pos_paths, sv_paths, validation_fraction=0.25, split_seed=0, gen0_reservoir_fraction=0.5
    )
    np.testing.assert_array_equal(held0["outcome"], held1["outcome"])
    np.testing.assert_array_equal(held0["me"], held1["me"])


def test_resample_rejects_out_of_range_fraction(tmp_path: Path) -> None:
    pos_paths, _sv, n0 = _shards(tmp_path)
    canonical = _canonical_train_idx(pos_paths, validation_fraction=0.25, split_seed=0)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            gen0_reservoir_resample(canonical, n0, bad, np.random.default_rng(0))
