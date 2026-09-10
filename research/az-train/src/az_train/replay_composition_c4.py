# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false
# ruff: noqa: E501
"""Offline diagnostic: which replay-composition scheme arrests value-head dilution.

The graded Connect Four Gumbel self-play loop
(``local/output/az/connect4/recovery/graded-cnn-800/``) improves
generation-over-generation in head-to-head play, but its value head degrades as
the retained replay grows: train value MSE rises and frozen reference-corpus
proven-value Pearson slips across generations. The coordinator keeps the *full*
replay (every shard gen0..genN, always including gen0) every generation, a
deliberate deviation from the sliding-window replay the AlphaZero/Gumbel
literature uses.

This driver refits the compact ``C4CNN001`` value+policy head from scratch for a
chosen "current generation" g, composing the training replay several different
ways from the frozen shards gen0..geng, and scores each refit the same way
``mixture_selfplay_c4.run_generation`` does (in-replay held-out mixed-target
Pearson, value MSE, policy cross-entropy, and the frozen proven reference
corpus through ``value_report``). The reference corpus never influences fitting
or model selection; it is measurement only.

The fit itself is unchanged -- every scheme calls
``mixture_selfplay_c4.fit_and_diagnose`` with identical seed, L2, learning rate,
epoch budget and mixture weight ``b``. Only the training rows differ.

Schemes (``g`` is the current generation, shards gen0..geng available):

* ``full`` -- every shard (the coordinator's current behaviour; baseline).
* ``window2`` -- only gen{g-1}, geng.
* ``window3`` -- only gen{g-2}, gen{g-1}, geng.
* ``window2_plus_gen0`` -- gen0 plus gen{g-1}, geng (keeps the diverse zero-net
  data, drops the middle generations).
* ``recency_weighted`` -- every shard, but rows are drawn with probability
  proportional to ``gamma ** (g - shard_index)`` (gamma=0.5 by default).
* ``anti_recency`` -- every shard, but rows are drawn with probability
  proportional to ``gamma_old ** shard_index`` (gamma_old=0.5 by default), so the
  oldest, most diverse shards are up-weighted -- the mirror image of
  ``recency_weighted``.
* ``gen0_reservoir`` -- every shard, but the per-row weights are set so that in
  expectation half the sampled rows come from gen0 alone and half come uniformly
  from the pooled gen1..geng rows.

``fit_value_policy_with_diagnostics`` takes no per-sample weight, so the weighted
schemes are realised by *resampling* training rows with replacement in proportion
to the shard weights -- no change to the fit code.

To keep the sweep affordable (the instrumented pure-numpy CNN fit costs roughly
0.07 s per training row for a 25-epoch fit, so a native full-replay g=4 fit is
well over an hour), every scheme -- ``full`` included -- resamples its training
rows with replacement to a fixed ``--budget`` count. This holds fit compute
constant across schemes and generations and isolates replay *composition* from
replay *volume* (the volume effect -- more data is better -- is already
established and is not the question here). Pass ``--budget 0`` to disable
resampling and fit each scheme on its native row count instead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from az_train.convnet_c4 import N_WEIGHTS
from az_train.fitability_c4 import _concat
from az_train.mirror_diagnostic_c4 import _sha256, read_reference_corpus
from az_train.mixture_selfplay_c4 import fit_and_diagnose, read_concat_searched_values
from az_train.records_c4 import game_slices, load_positions
from az_train.trainer_hygiene_c4 import _pack_rows, replay_game_split_indices

SCHEMES: tuple[str, ...] = (
    "full",
    "window2",
    "window3",
    "window2_plus_gen0",
    "recency_weighted",
    "anti_recency",
    "gen0_reservoir",
)

# Sentinel shard weight for ``gen0_reservoir``: the gen0 weight is not known until
# ``build_training_indices`` sees the actual per-shard row counts of the training
# pool, so ``scheme_shards`` emits this marker and the resampler resolves it.
RESERVOIR_GEN0_WEIGHT = -1.0


def scheme_shards(
    scheme: str, g: int, *, gamma: float = 0.5, gamma_old: float = 0.5
) -> list[tuple[int, float]]:
    """Return ``(shard_index, relative_sample_weight)`` for a scheme at generation ``g``.

    Shard indices are generation numbers 0..g. Weights are relative; only their
    ratios matter to the resampler. Uniform schemes return weight 1.0 for every
    kept shard.
    """
    if g < 1:
        raise ValueError("current generation g must be >= 1")
    every = list(range(g + 1))
    if scheme == "full":
        kept = every
    elif scheme == "window2":
        kept = every[-2:]
    elif scheme == "window3":
        kept = every[-3:]
    elif scheme == "window2_plus_gen0":
        kept = sorted({0, *every[-2:]})
    elif scheme == "recency_weighted":
        return [(i, float(gamma ** (g - i))) for i in every]
    elif scheme == "anti_recency":
        return [(i, float(gamma_old**i)) for i in every]
    elif scheme == "gen0_reservoir":
        return [(0, RESERVOIR_GEN0_WEIGHT)] + [(i, 1.0) for i in every[1:]]
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return [(i, 1.0) for i in kept]


def shard_of_row(shard_record_counts: list[int]) -> np.ndarray:
    """Map every concatenated record index to the shard (generation) it came from."""
    bounds = np.cumsum(shard_record_counts)
    return np.searchsorted(bounds, np.arange(int(bounds[-1])), side="right")


def build_training_indices(
    spec: list[tuple[int, float]],
    canonical_train_idx: np.ndarray,
    row_shard: np.ndarray,
    *,
    budget: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Training record indices for one scheme.

    Starts from the canonical whole-game train split, keeps only rows whose
    shard the scheme retains, then either returns them as-is (uniform scheme,
    ``budget`` disabled) or resamples with replacement -- to ``budget`` rows, or
    to the kept-pool size when ``budget`` is ``None`` -- with per-row probability
    proportional to the scheme's shard weight.
    """
    weight_by_shard = {int(s): float(w) for s, w in spec}
    kept = np.array(sorted(weight_by_shard), dtype=np.int64)
    pool_shard = row_shard[canonical_train_idx]
    keep_mask = np.isin(pool_shard, kept)
    pool = canonical_train_idx[keep_mask]
    if pool.size == 0:
        raise ValueError("scheme kept no training rows")
    kept_row_shard = pool_shard[keep_mask]
    if weight_by_shard.get(0) == RESERVOIR_GEN0_WEIGHT:
        # Resolve the gen0 weight so gen0's total sampling mass equals the pooled
        # mass of every later shard: expected 50/50 split, gen0 vs gen1..geng.
        n_gen0 = int((kept_row_shard == 0).sum())
        n_rest = int(pool.size - n_gen0)
        if n_gen0 == 0 or n_rest == 0:
            raise ValueError("gen0_reservoir needs gen0 rows and at least one later shard")
        weight_by_shard = {**weight_by_shard, 0: n_rest / n_gen0}
    weights = np.array([weight_by_shard[int(s)] for s in kept_row_shard], dtype=np.float64)
    uniform = bool(np.allclose(weights, weights[0]))
    if budget in (0, None) and uniform:
        return np.sort(pool)
    draw_n = int(budget) if budget else pool.size
    probs = weights / weights.sum()
    picked = rng.choice(pool.size, size=draw_n, replace=True, p=probs)
    return np.sort(pool[picked])


def _row(scheme: str, g: int, spec: list[tuple[int, float]], counts: dict[str, object], res: dict[str, object]) -> dict[str, object]:
    proven: dict[str, object] = res["held_out_proven"]  # type: ignore[assignment]
    early: dict[str, float] = proven["principled_early_stop"]  # type: ignore[assignment]
    final: dict[str, float] = proven["final_epoch"]  # type: ignore[assignment]
    in_replay: dict[str, object] = res["in_replay_mixed_target_pearson"]  # type: ignore[assignment]
    ir_early: dict[str, float] = in_replay["principled_early_stop"]  # type: ignore[assignment]
    fit_metrics: dict[str, float] = res["fit_metrics"]  # type: ignore[assignment]
    monitor: list[float] = res["monitor_proven_pearson_by_epoch"]  # type: ignore[assignment]
    return {
        "scheme": scheme,
        "g": g,
        "shards": [int(s) for s, _ in spec],
        "shard_weights": [round(float(w), 4) for _, w in spec],
        "replay_split": counts,
        "principled_early_stop_epoch": res["principled_early_stop_epoch"],
        "reference_proven_value_pearson_early_stop": early["value_pearson"],
        "reference_proven_value_pearson_final": final["value_pearson"],
        "reference_balanced_sign_accuracy_early_stop": early["balanced_sign_accuracy"],
        "reference_proven_value_pearson_oracle": round(float(max(monitor)), 4),
        "in_replay_mixed_pearson_train_early_stop": round(float(ir_early["train"]), 4),
        "in_replay_mixed_pearson_held_out_early_stop": round(float(ir_early["held_out"]), 4),
        "train_value_mse_final": fit_metrics["train_value_mse"],
        "held_out_value_mse_final": fit_metrics["validation_value_mse"],
        "train_policy_cross_entropy_final": fit_metrics["train_policy_cross_entropy"],
        "held_out_policy_cross_entropy_final": fit_metrics["validation_policy_cross_entropy"],
        "fit_wall_seconds": res["fit_wall_seconds"],
    }


def run_sweep(
    corpus_path: Path,
    run_dir: Path,
    out_dir: Path,
    *,
    generations: tuple[int, ...] = (1, 2, 3, 4),
    schemes: tuple[str, ...] = SCHEMES,
    budget: int | None = 12000,
    gamma: float = 0.5,
    gamma_old: float = 0.5,
    b: float = 0.75,
    l2: float = 1e-4,
    learning_rate: float = 2e-3,
    epochs: int = 25,
    seed: int = 20260907,
    validation_fraction: float = 0.2,
    split_seed: int = 20260908,
) -> dict[str, object]:
    corpus = read_reference_corpus(corpus_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "config": {
            "run_dir": str(run_dir),
            "generations": list(generations),
            "schemes": list(schemes),
            "budget": budget,
            "gamma": gamma,
            "gamma_old": gamma_old,
            "b": b, "l2": l2, "learning_rate": learning_rate, "epochs": epochs,
            "seed": seed, "validation_fraction": validation_fraction, "split_seed": split_seed,
            "n_weights": int(N_WEIGHTS),
        },
        "inputs": {"reference_corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)}},
        "rows": [],
    }
    rows: list[dict[str, object]] = result["rows"]  # type: ignore[assignment]

    started = time.perf_counter()
    for g in generations:
        pos_paths = [run_dir / f"gen{h}.bin" for h in range(g + 1)]
        sv_paths = [run_dir / f"gen{h}.sv.f32" for h in range(g + 1)]
        shards = [load_positions(p) for p in pos_paths]
        all_pos = _concat(shards)
        searched_all = read_concat_searched_values(sv_paths, len(all_pos))
        row_shard = shard_of_row([len(s) for s in shards])
        game_bounds = [(int(s.start), int(s.stop)) for s in game_slices(all_pos)]
        c_train, c_held, c_train_games, c_held_games = replay_game_split_indices(
            game_bounds, validation_fraction, split_seed
        )
        held_pack = _pack_rows(all_pos, searched_all, c_held)

        by_spec: dict[tuple[tuple[int, float], ...], dict[str, object]] = {}
        for scheme in schemes:
            spec = scheme_shards(scheme, g, gamma=gamma, gamma_old=gamma_old)
            key = tuple(spec)
            if key not in by_spec:
                rng = np.random.default_rng([seed, g, SCHEMES.index(scheme)])
                train_idx = build_training_indices(spec, c_train, row_shard, budget=budget, rng=rng)
                train_pack = _pack_rows(all_pos, searched_all, train_idx)
                counts: dict[str, object] = {
                    "train_rows": int(train_pack["outcome"].size),
                    "held_out_rows": int(held_pack["outcome"].size),
                    "held_out_games": c_held_games,
                    "canonical_train_games": c_train_games,
                    "shards": [int(s) for s, _ in spec],
                    "shard_weights": [round(float(w), 4) for _, w in spec],
                    "native_kept_rows": int(np.isin(row_shard[c_train], [s for s, _ in spec]).sum()),
                }
                scheme_out = out_dir / f"g{g}" / scheme
                res = fit_and_diagnose(
                    corpus, corpus_path, pos_paths, sv_paths,
                    train_pack, held_pack, counts, scheme_out, g,
                    b=b, l2=l2, learning_rate=learning_rate, epochs=epochs, seed=seed,
                    validation_fraction=validation_fraction, split_seed=split_seed,
                )
                by_spec[key] = _row(scheme, g, spec, counts, res)
                print(
                    f"g{g} {scheme}: shards {counts['shards']} "
                    f"ref r(early)={by_spec[key]['reference_proven_value_pearson_early_stop']} "
                    f"train_mse={by_spec[key]['train_value_mse_final']} "
                    f"held_mse={by_spec[key]['held_out_value_mse_final']} "
                    f"({by_spec[key]['fit_wall_seconds']}s)",
                    flush=True,
                )
            row = dict(by_spec[key])
            computed_as = row["scheme"]
            row["scheme"] = scheme
            if computed_as != scheme:
                row["identical_replay_as"] = computed_as
            rows.append(row)
            (out_dir / "sweep.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    (out_dir / "sweep.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _summary_table(rows: list[dict[str, Any]]) -> str:
    head = f"{'scheme':<20} {'g':>2}  {'ref_r_early':>11} {'ref_r_final':>11} {'train_mse':>9} {'held_mse':>9} {'pol_ce_ho':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{str(r['scheme']):<20} {int(r['g']):>2}  "
            f"{float(r['reference_proven_value_pearson_early_stop']):>11.4f} "
            f"{float(r['reference_proven_value_pearson_final']):>11.4f} "
            f"{float(r['train_value_mse_final']):>9.4f} "
            f"{float(r['held_out_value_mse_final']):>9.4f} "
            f"{float(r['held_out_policy_cross_entropy_final']):>9.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.replay_composition_c4")
    parser.add_argument("--reference-corpus", required=True, help="frozen corpus-v2.c4ref")
    parser.add_argument("--run-dir", required=True, help="graded run dir holding gen{N}.bin / gen{N}.sv.f32")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generations", default="1,2,3,4")
    parser.add_argument("--schemes", default=",".join(SCHEMES))
    parser.add_argument("--budget", type=int, default=12000, help="resample every scheme to this many training rows; 0 disables resampling")
    parser.add_argument("--gamma", type=float, default=0.5, help="recency_weighted decay: shard i weight = gamma ** (g - i)")
    parser.add_argument("--gamma-old", type=float, default=0.5, help="anti_recency decay: shard i weight = gamma_old ** i")
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--replay-validation-fraction", type=float, default=0.2)
    parser.add_argument("--replay-split-seed", type=int, default=20260908)
    args = parser.parse_args(argv)

    result = run_sweep(
        Path(args.reference_corpus), Path(args.run_dir), Path(args.out_dir),
        generations=tuple(int(t) for t in args.generations.split(",")),
        schemes=tuple(args.schemes.split(",")),
        budget=args.budget or None,
        gamma=args.gamma, gamma_old=args.gamma_old, b=args.b, l2=args.l2, learning_rate=args.learning_rate,
        epochs=args.epochs, seed=args.seed,
        validation_fraction=args.replay_validation_fraction,
        split_seed=args.replay_split_seed,
    )
    rows: list[dict[str, Any]] = result["rows"]  # type: ignore[assignment]
    table = _summary_table(rows)
    (Path(args.out_dir) / "summary.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
