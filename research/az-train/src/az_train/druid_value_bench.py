# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""A frozen value-head bench for the Druid CNN track: a fixed set of held-out positions, scored
against a deep-search label rather than only the noisy final outcome.

``build`` samples the bench once from a run's held-out (never trained) games, using exactly the
coordinator's own held-out rule (``druid_cnn.split_validation``): the first ``validation_games``
distinct game ids of each named generation's shard. Positions are stratified by game progress
(``druid_diagnostics.game_progress``, in fifths) so early, mid and late game are represented evenly
regardless of how many long or short games happened to land in the sample. The bench is written
once as a shard (``druid_records.write_shard``, the same ``DRDSHRD1`` layout ``druid_value_label``
reads) plus a sidecar of each position's outcome, unique game id, progress fraction and max stack
height, so later slices can re-score it without re-sampling.

``label`` runs ``druid_value_label`` (see ``games/druid/examples/druid_value_label.rs``) at a given
search depth and stores the result under ``labels/``; a deep label (many simulations) stands in for
the "true" value of a position, since replaying the actual game to the end would need move
selection the bench cannot reconstruct. ``score`` runs the same labeller at ``--sims 0`` (the net's
raw value, cached under ``raw/``) for one or more checkpoints and reports, against both the deep
label and the outcome: MSE, Pearson and R^2 = 1 - MSE / Var(label), overall and split by progress
fifth and by max-height bucket (1, 2, 3, 4+), each with a 95% interval from resampling games (not
positions -- positions of the same game are not independent). ``compare`` runs the same report
between two already-labelled files, for measuring label noise (e.g. a shallow label against a deep
one) without touching a checkpoint at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from az_train.druid_diagnostics import PROGRESS_BINS, _pearson, game_progress
from az_train.druid_records import read_shard, write_shard
from az_train.gonnect_cnn import EXAMPLES, ROOT, load_config, rust_env, write_json

HEIGHT_BUCKETS = (1, 2, 3, 4)  # the last one is "4 or more"
N_BOOT = 1000
BOOT_SEED = 20260927


# --------------------------------------------------------------------------------------- build


def _held_out_positions(
    run_dir: Path, gens: range, validation_games: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Held-out records of ``gens`` (each generation's first ``validation_games`` distinct game
    ids, ``druid_cnn.split_validation``'s own rule) plus each record's game progress fraction and a
    game id unique across generations (``gen * 1_000_000 + the shard's own id``, matching
    ``druid_cnn.load_positions``'s ``game_offset`` convention)."""
    size = None
    parts, progress_parts, game_parts = [], [], []
    for g in gens:
        got_size, rec = read_shard(run_dir / "shards" / f"gen{g}.bin")
        if size is None:
            size = got_size
        elif got_size != size:
            raise ValueError(
                f"gen{g}.bin is a {got_size}x{got_size} shard, gen{gens[0]} was {size}"
            )
        held_games = np.unique(rec["game"])[:validation_games]
        held = np.isin(rec["game"], held_games)
        rec_held = rec[held]
        parts.append(rec_held)
        progress_parts.append(game_progress(rec_held["game"]))
        game_parts.append(rec_held["game"].astype(np.int64) + g * 1_000_000)
    assert size is not None, "gens must be non-empty"
    records = np.concatenate(parts)
    progress = np.concatenate(progress_parts)
    game = np.concatenate(game_parts)
    return size, records, progress, game  # type: ignore[return-value]


def _stratified_sample(
    progress: np.ndarray, positions: int, seed: int, bins: int = PROGRESS_BINS
) -> np.ndarray:
    """Indices of about ``positions`` positions, split as evenly as possible across ``bins``
    equal slices of game progress, sampled without replacement within each slice."""
    rng = np.random.default_rng(seed)
    slot = np.minimum((progress * bins).astype(int), bins - 1)
    per_bin, remainder = divmod(positions, bins)
    chosen = []
    for b in range(bins):
        pool = np.flatnonzero(slot == b)
        take = min(per_bin + (1 if b < remainder else 0), len(pool))
        chosen.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def build(out_dir: Path, run_dir: Path, gens: range, positions: int, seed: int) -> None:
    bench_shard = out_dir / "positions.bin"
    if bench_shard.exists():
        raise SystemExit(f"{out_dir} already has a bench (remove {bench_shard} to rebuild)")
    config_path = run_dir / "config.effective.toml"
    cfg = load_config(config_path)
    validation_games = cfg["train"]["validation_games"]

    size, records, progress, game = _held_out_positions(run_dir, gens, validation_games)
    chosen = _stratified_sample(progress, positions, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_shard(bench_shard, size, records[chosen])
    np.savez(
        out_dir / "meta.npz",
        outcome=records["value"][chosen],
        game=game[chosen],
        progress=progress[chosen],
        max_height=records["heights"][chosen].max(axis=1).astype(np.int32),
    )
    write_json(
        out_dir / "meta.json",
        {
            "run_dir": str(run_dir),
            "config": str(config_path),
            "size": size,
            "gens": [gens[0], gens[-1]],
            "validation_games": validation_games,
            "positions": int(len(chosen)),
            "seed": seed,
        },
    )
    print(f"{len(chosen)} positions from {run_dir.name} gens {gens[0]}-{gens[-1]} -> {out_dir}")


# --------------------------------------------------------------------------------------- label


def _run_value_label(
    config: str, weights: Path, shard: Path, out: Path, sims: int, workers: int
) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(EXAMPLES / "druid_value_label"),
        "--config", str(ROOT / config), "--weights", str(weights),
        "--shard", str(shard), "--out", str(out),
        "--sims", str(sims), "--workers", str(workers),
    ]  # fmt: skip
    done = subprocess.run(cmd, cwd=ROOT, env=rust_env(), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"druid_value_label failed ({done.returncode}): {done.stderr[-2000:]}")
    return json.loads(Path(f"{out}.json").read_text())


def label(bench_dir: Path, weights: Path, sims: int, name: str, workers: int = 8) -> None:
    meta = json.loads((bench_dir / "meta.json").read_text())
    sidecar = _run_value_label(
        meta["config"], weights, bench_dir / "positions.bin", bench_dir / "labels" / f"{name}.f32",
        sims, workers,
    )  # fmt: skip
    rate = sidecar["count"] / max(sidecar["seconds"], 1e-9)
    print(f"{name}: {sidecar['count']} positions at {sims} sims, {rate:.1f} positions/s")


def _load_label(bench_dir: Path, name: str) -> np.ndarray:
    path = bench_dir / "labels" / f"{name}.f32"
    if not path.exists():
        raise SystemExit(f"no label {name!r} in {bench_dir} (run `label` first)")
    return np.fromfile(path, dtype="<f4")


def _raw_values(
    bench_dir: Path, meta: dict[str, Any], name: str, weights: Path, workers: int
) -> np.ndarray:
    """The net's raw (``--sims 0``) value on the bench, cached under ``raw/`` keyed by ``name``."""
    out = bench_dir / "raw" / f"{name}.f32"
    if not out.exists():
        _run_value_label(meta["config"], weights, bench_dir / "positions.bin", out, 0, workers)
    return np.fromfile(out, dtype="<f4")


# --------------------------------------------------------------------------------------- score


def _r2(mse: float, label_var: float) -> float | None:
    return None if label_var < 1e-12 else 1.0 - mse / label_var


def _point_metrics(
    predicted: np.ndarray, label: np.ndarray, outcome: np.ndarray
) -> dict[str, Any]:
    label_var = float(np.var(label))
    mse_label = float(np.mean((predicted - label) ** 2))
    return {
        "n": int(len(predicted)),
        "label_var": label_var,
        "label_mean": float(np.mean(label)),
        "mse_vs_label": mse_label,
        "pearson_vs_label": _pearson(predicted, label),
        "r2_vs_label": _r2(mse_label, label_var),
        "mse_vs_outcome": float(np.mean((predicted - outcome) ** 2)),
        "pearson_vs_outcome": _pearson(predicted, outcome),
    }


_CI_KEYS = (
    "label_var", "mse_vs_label", "pearson_vs_label",
    "r2_vs_label", "mse_vs_outcome", "pearson_vs_outcome",
)  # fmt: skip


def _bootstrap_ci(
    predicted: np.ndarray,
    label: np.ndarray,
    outcome: np.ndarray,
    game: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float] | None]:
    """95% interval of each ``_point_metrics`` key by resampling games with replacement, so
    positions of the same game (which are not independent draws) move together."""
    unique_games = np.unique(game)
    if len(unique_games) < 2:
        return {k: None for k in _CI_KEYS}
    by_game = {g: np.flatnonzero(game == g) for g in unique_games}
    draws: dict[str, list[float]] = {k: [] for k in _CI_KEYS}
    # A resample of a small bucket can duplicate a single game enough times to make a draw's
    # values numerically constant; that draw's Pearson is meaningless and dropped below, but the
    # near-zero-variance division on the way there is not itself an error.
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(N_BOOT):
            games = rng.choice(unique_games, len(unique_games))
            idx = np.concatenate([by_game[g] for g in games])
            m = _point_metrics(predicted[idx], label[idx], outcome[idx])
            for k in _CI_KEYS:
                v = m[k]
                if v is not None and not np.isnan(v):
                    draws[k].append(v)
    return {
        k: (float(np.percentile(vs, 2.5)), float(np.percentile(vs, 97.5))) if len(vs) >= 2 else None
        for k, vs in draws.items()
    }


def _bucketed(
    predicted: np.ndarray,
    label: np.ndarray,
    outcome: np.ndarray,
    game: np.ndarray,
    progress: np.ndarray,
    max_height: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    def entry(sel: np.ndarray, n: int) -> dict[str, Any]:
        return {
            **_point_metrics(predicted[sel], label[sel], outcome[sel]),
            "ci": _bootstrap_ci(predicted[sel], label[sel], outcome[sel], game[sel], rng),
            "bin": n,
        }

    fifth = np.minimum((progress * PROGRESS_BINS).astype(int), PROGRESS_BINS - 1)
    # An empty board (height 0) only ever occurs at ply 0; fold it into the "1" bucket rather
    # than giving it one of its own.
    height = np.clip(max_height, HEIGHT_BUCKETS[0], HEIGHT_BUCKETS[-1])
    return {
        "overall": entry(np.ones(len(predicted), dtype=bool), -1),
        "by_progress": [entry(fifth == b, b) for b in range(PROGRESS_BINS)],
        "by_height": [entry(height == b, b) for b in HEIGHT_BUCKETS],
    }


def _fmt(v: Any, fmt: str) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        ("name", "{}"), ("n", "{}"), ("mse_vs_label", "{:.4f}"), ("r2_vs_label", "{:.3f}"),
        ("pearson_vs_label", "{:.3f}"), ("mse_vs_outcome", "{:.3f}"),
        ("pearson_vs_outcome", "{:.3f}"),
    ]  # fmt: skip
    cells = [[_fmt(r[k], fmt) for k, fmt in cols] for r in rows]
    widths = [max(len(k), *(len(c[i]) for c in cells)) for i, (k, _) in enumerate(cols)]
    print("  ".join(k.ljust(w) for (k, _), w in zip(cols, widths, strict=True)))
    for c in cells:
        print("  ".join(v.ljust(w) for v, w in zip(c, widths, strict=True)))


def score(
    bench_dir: Path, label_name: str, entries: list[tuple[str, Path]], workers: int = 8
) -> None:
    meta = json.loads((bench_dir / "meta.json").read_text())
    npz = np.load(bench_dir / "meta.npz")
    outcome, game, progress = npz["outcome"], npz["game"], npz["progress"]
    max_height = npz["max_height"]
    deep_label = _load_label(bench_dir, label_name)
    assert len(deep_label) == len(outcome), f"label {label_name!r} does not match the bench"

    rng = np.random.default_rng(BOOT_SEED)
    rows, reports = [], {}
    for name, weights in entries:
        predicted = _raw_values(bench_dir, meta, name, weights, workers)
        report = _bucketed(predicted, deep_label, outcome, game, progress, max_height, rng)
        reports[name] = report
        rows.append({"name": name, **report["overall"]})

    _print_table(rows)
    report = {"label": label_name, "checkpoints": reports}
    write_json(bench_dir / f"report-{label_name}.json", report)


def compare(bench_dir: Path, name_a: str, name_b: str) -> None:
    """Label ``a`` scored against label ``b`` as if ``b`` were the deep label -- for measuring the
    noise between two labels of the same bench (e.g. a shallow search against a deep one)."""
    npz = np.load(bench_dir / "meta.npz")
    outcome, game, progress = npz["outcome"], npz["game"], npz["progress"]
    max_height = npz["max_height"]
    a, b = _load_label(bench_dir, name_a), _load_label(bench_dir, name_b)
    rng = np.random.default_rng(BOOT_SEED)
    report = _bucketed(a, b, outcome, game, progress, max_height, rng)
    _print_table([{"name": f"{name_a} vs {name_b}", **report["overall"]}])
    out = bench_dir / f"report-{name_a}-vs-{name_b}.json"
    write_json(out, {"a": name_a, "b": name_b, **report})


# ------------------------------------------------------------------------------------------ CLI


def _parse_gens(text: str) -> range:
    lo, _, hi = text.partition("-")
    return range(int(lo), int(hi) + 1)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("run_dir")
    b.add_argument("--gens", required=True, help="LO-HI, inclusive")
    b.add_argument("--positions", type=int, default=3000)
    b.add_argument("--seed", type=int, default=1)

    l = sub.add_parser("label")  # noqa: E741
    l.add_argument("bench_dir")
    l.add_argument("--weights", required=True)
    l.add_argument("--sims", type=int, required=True)
    l.add_argument("--name", required=True)
    l.add_argument("--workers", type=int, default=8)

    s = sub.add_parser("score")
    s.add_argument("bench_dir")
    s.add_argument("--label", required=True)
    s.add_argument("entries", nargs="+", help="NAME=WEIGHTS")
    s.add_argument("--workers", type=int, default=8)

    c = sub.add_parser("compare")
    c.add_argument("bench_dir")
    c.add_argument("name_a")
    c.add_argument("name_b")

    args = p.parse_args()
    if args.cmd == "build":
        build(
            _resolve(args.out_dir), _resolve(args.run_dir), _parse_gens(args.gens),
            args.positions, args.seed,
        )  # fmt: skip
    elif args.cmd == "label":
        label(_resolve(args.bench_dir), _resolve(args.weights), args.sims, args.name, args.workers)
    elif args.cmd == "score":
        entries = []
        for e in args.entries:
            name, _, weights = e.partition("=")
            entries.append((name, _resolve(weights)))
        score(_resolve(args.bench_dir), args.label, entries, args.workers)
    else:
        compare(_resolve(args.bench_dir), args.name_a, args.name_b)


if __name__ == "__main__":
    main()
