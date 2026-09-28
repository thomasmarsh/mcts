import json
from pathlib import Path

import numpy as np
import pytest

from az_train import druid_value_bench as dvb
from az_train.druid_records import num_actions, read_shard, record_dtype, write_shard

SIZE = 5


def _records(n: int, *, heights=None, game=None, value=None) -> np.ndarray:
    """``n`` zero-filled ``record_dtype(SIZE)`` rows with the given fields overridden; the fields
    the bench never reads (owners, hands, pending, player, ply, policy) stay at their zero
    default."""
    rec = np.zeros(n, dtype=record_dtype(SIZE))
    if heights is not None:
        rec["heights"] = heights
    if game is not None:
        rec["game"] = game
    if value is not None:
        rec["value"] = value
    return rec


# --------------------------------------------------------------------------------- shard writer


def test_write_shard_round_trips_every_field(tmp_path):
    rng = np.random.default_rng(0)
    rec = _records(6)
    rec["heights"] = rng.integers(0, 8, rec["heights"].shape).astype("<u2")
    rec["owners"] = rng.integers(0, 3, rec["owners"].shape).astype("u1")
    rec["game"] = [0, 0, 1, 1, 1, 2]
    rec["value"] = rng.uniform(-1, 1, 6).astype("<f4")
    rec["policy"] = rng.uniform(0, 1, rec["policy"].shape).astype("<f4")

    path = tmp_path / "shard.bin"
    write_shard(path, SIZE, rec)
    got_size, got = read_shard(path)
    assert got_size == SIZE
    assert len(got) == len(rec)
    for name in rec.dtype.names:
        np.testing.assert_array_equal(got[name], rec[name])


def test_write_shard_accepts_a_reordered_or_filtered_subset(tmp_path):
    rec = _records(4, game=[3, 1, 2, 0])
    subset = rec[[3, 1]]  # game ids 0, 1: out of original order, a strict subset
    path = tmp_path / "shard.bin"
    write_shard(path, SIZE, subset)
    _, got = read_shard(path)
    np.testing.assert_array_equal(got["game"], [0, 1])


# --------------------------------------------------------------------------- stratified sampling


def test_stratified_sample_splits_evenly_across_progress_bins():
    # Bin 0 has plenty of positions, bin 4 (the last fifth) is scarce: the sampler must not ask
    # for more than a scarce bin has, and must still hit the per-bin target elsewhere.
    progress = np.concatenate([np.full(1000, 0.05), np.full(1000, 0.45), np.full(3, 0.95)])
    chosen = dvb._stratified_sample(progress, positions=50, seed=1)
    slot = np.minimum((progress[chosen] * 5).astype(int), 4)
    counts = np.bincount(slot, minlength=5)
    assert counts[0] == 10 and counts[2] == 10
    assert counts[4] == 3  # capped by the pool, not the 10-per-bin target
    assert len(chosen) == len(np.unique(chosen)), "sampling without replacement"


def test_stratified_sample_is_deterministic_given_the_same_seed():
    progress = np.random.default_rng(2).uniform(0, 1, 400)
    a = dvb._stratified_sample(progress, positions=40, seed=7)
    b = dvb._stratified_sample(progress, positions=40, seed=7)
    np.testing.assert_array_equal(a, b)


# -------------------------------------------------------------------------------- point metrics


def test_point_metrics_on_hand_computed_values():
    predicted = np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32)
    label = np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float32)
    outcome = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    m = dvb._point_metrics(predicted, label, outcome)
    assert m["n"] == 4
    np.testing.assert_allclose(m["label_var"], label.var())
    np.testing.assert_allclose(m["mse_vs_label"], np.mean((predicted - label) ** 2))
    np.testing.assert_allclose(m["r2_vs_label"], 1.0 - m["mse_vs_label"] / m["label_var"])
    np.testing.assert_allclose(m["mse_vs_outcome"], np.mean((predicted - outcome) ** 2))
    assert m["pearson_vs_label"] == pytest.approx(np.corrcoef(predicted, label)[0, 1])


def test_point_metrics_r2_is_none_for_a_constant_label():
    m = dvb._point_metrics(np.array([0.1, 0.2]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert m["r2_vs_label"] is None
    assert m["label_var"] == 0.0


def test_a_perfect_predictor_scores_zero_mse_and_r2_one():
    label = np.array([-1.0, 0.2, 0.7, 1.0], dtype=np.float32)
    m = dvb._point_metrics(label.copy(), label, label)
    assert m["mse_vs_label"] == pytest.approx(0.0, abs=1e-12)
    assert m["r2_vs_label"] == pytest.approx(1.0)
    assert m["pearson_vs_label"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------------- bootstrap


def test_bootstrap_ci_brackets_the_point_estimate_and_is_seed_reproducible():
    rng = np.random.default_rng(1)
    n_games = 30
    game = np.repeat(np.arange(n_games), 10)
    predicted = np.random.default_rng(3).normal(size=len(game)).astype(np.float32)
    noise = np.random.default_rng(4).normal(scale=0.3, size=len(game)).astype(np.float32)
    label = predicted + noise
    outcome = np.sign(label)
    point = dvb._point_metrics(predicted, label, outcome)
    ci = dvb._bootstrap_ci(predicted, label, outcome, game, rng)
    for key in ("mse_vs_label", "pearson_vs_label", "r2_vs_label"):
        lo, hi = ci[key]
        assert lo <= point[key] <= hi, f"{key}: point {point[key]} outside [{lo}, {hi}]"

    ci_again = dvb._bootstrap_ci(
        predicted, label, outcome, game, np.random.default_rng(1)
    )
    assert ci == ci_again


def test_bootstrap_ci_is_none_with_fewer_than_two_games():
    ci = dvb._bootstrap_ci(
        np.array([0.1, 0.2]), np.array([0.1, 0.3]), np.array([1.0, 1.0]),
        np.array([0, 0]), np.random.default_rng(0),
    )
    assert all(v is None for v in ci.values())


# ------------------------------------------------------------------------------------- bucketed


def test_bucketed_progress_and_height_bins_partition_every_position():
    n = 200
    rng = np.random.default_rng(5)
    predicted = rng.normal(size=n).astype(np.float32)
    label = predicted + rng.normal(scale=0.2, size=n).astype(np.float32)
    outcome = np.sign(label)
    game = np.repeat(np.arange(20), 10)
    progress = rng.uniform(0, 1, n)
    max_height = rng.integers(0, 7, n)  # includes 0 (empty board) and 6 (past the 4+ bucket)

    report = dvb._bucketed(predicted, label, outcome, game, progress, max_height, rng)
    assert report["overall"]["n"] == n
    assert sum(b["n"] for b in report["by_progress"]) == n
    assert sum(b["n"] for b in report["by_height"]) == n
    assert [b["bin"] for b in report["by_progress"]] == [0, 1, 2, 3, 4]
    assert [b["bin"] for b in report["by_height"]] == [1, 2, 3, 4]
    # Heights 0 and 1 both fall in the "1" bucket, 6 is folded into "4 or more".
    assert report["by_height"][0]["n"] == int(np.isin(max_height, [0, 1]).sum())
    assert report["by_height"][-1]["n"] == int((max_height >= 4).sum())


# --------------------------------------------------------------------------------- held-out set


def _write_gen_shard(path: Path, size: int, games: list[int], plies: list[int]) -> None:
    rec = np.zeros(len(games), dtype=record_dtype(size))
    rec["game"] = games
    rec["ply"] = plies
    # Vary the stack height with ply so max-height buckets 1..4+ are all populated by tests that
    # exercise the height split.
    rec["heights"][:, 0] = (np.asarray(plies) % 5) + 1
    write_shard(path, size, rec)


def test_held_out_positions_takes_the_first_validation_games_ids_per_generation(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    # Generation 0: games 0, 1, 2 (2 plies each); generation 1: games 0, 1 (2 plies each).
    _write_gen_shard(shards / "gen0.bin", SIZE, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1])
    _write_gen_shard(shards / "gen1.bin", SIZE, [0, 0, 1, 1], [0, 1, 0, 1])

    size, records, progress, game = dvb._held_out_positions(
        tmp_path, range(0, 2), validation_games=1
    )
    assert size == SIZE
    # Only game 0 of each generation is held out (validation_games=1), offset by generation.
    np.testing.assert_array_equal(np.unique(game), [0, 1_000_000])
    assert len(records) == 4
    np.testing.assert_allclose(progress, [0.0, 0.5, 0.0, 0.5])


def test_num_actions_matches_the_record_dtype_used_by_the_fixture():
    # Sanity check the test helper's dtype agrees with the real layout before trusting the rest.
    assert record_dtype(SIZE)["policy"].shape == (num_actions(SIZE),)


# --------------------------------------------------------------------------------- build/score


def _write_config(path: Path, validation_games: int) -> None:
    path.write_text(f"[net]\nsize = {SIZE}\n\n[train]\nvalidation_games = {validation_games}\n")


def test_build_writes_a_bench_and_refuses_to_overwrite_it(tmp_path):
    # 3 games of 5 plies each per generation: progress fractions 0, 0.2, 0.4, 0.6, 0.8 land one
    # in each of the 5 bins, so with validation_games=2 every bin's pool is the same size (4) and
    # a positions=4 request (remainder 4, per_bin 0) takes exactly one from each of the first 4.
    games = [g for g in range(3) for _ in range(5)]
    plies = list(range(5)) * 3
    run_dir = tmp_path / "run"
    (run_dir / "shards").mkdir(parents=True)
    _write_config(run_dir / "config.effective.toml", validation_games=2)
    _write_gen_shard(run_dir / "shards" / "gen0.bin", SIZE, games, plies)
    _write_gen_shard(run_dir / "shards" / "gen1.bin", SIZE, games, plies)

    out_dir = tmp_path / "bench"
    dvb.build(out_dir, run_dir, range(0, 2), positions=4, seed=1)
    assert (out_dir / "positions.bin").exists()
    meta = np.load(out_dir / "meta.npz")
    assert len(meta["outcome"]) == 4
    assert len(meta["game"]) == 4
    got_size, positions = read_shard(out_dir / "positions.bin")
    assert got_size == SIZE and len(positions) == 4

    with pytest.raises(SystemExit):
        dvb.build(out_dir, run_dir, range(0, 2), positions=4, seed=1)


def test_score_and_compare_read_cached_files_without_invoking_rust(tmp_path):
    # 3 games of 5 plies each, all held out (validation_games=3): a full 5-bin, 3-games-per-bin
    # spread, so every bucket in the score/compare report has more than one game in it.
    games = [g for g in range(3) for _ in range(5)]
    plies = list(range(5)) * 3
    run_dir = tmp_path / "run"
    (run_dir / "shards").mkdir(parents=True)
    _write_config(run_dir / "config.effective.toml", validation_games=3)
    _write_gen_shard(run_dir / "shards" / "gen0.bin", SIZE, games, plies)

    out_dir = tmp_path / "bench"
    dvb.build(out_dir, run_dir, range(0, 1), positions=15, seed=1)
    n = len(np.load(out_dir / "meta.npz")["outcome"])
    assert n == 15

    rng = np.random.default_rng(9)
    deep = rng.normal(size=n).astype("<f4")
    shallow = deep + rng.normal(scale=0.1, size=n).astype("<f4")
    (out_dir / "labels").mkdir()
    deep.tofile(out_dir / "labels" / "deep.f32")
    shallow.tofile(out_dir / "labels" / "shallow.f32")
    (out_dir / "raw").mkdir()
    checkpoint = deep + rng.normal(scale=0.5, size=n).astype("<f4")
    checkpoint.tofile(out_dir / "raw" / "A.f32")

    dvb.score(out_dir, "deep", [("A", Path("unused.bin"))])
    report = json.loads((out_dir / "report-deep.json").read_text())
    assert report["checkpoints"]["A"]["overall"]["n"] == n

    dvb.compare(out_dir, "shallow", "deep")
    cmp_report = json.loads((out_dir / "report-shallow-vs-deep.json").read_text())
    assert cmp_report["overall"]["n"] == n
