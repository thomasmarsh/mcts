# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false
# ruff: noqa: E501
"""Offline refit that recency-weights the policy loss alone over a frozen replay.

The graded Connect Four Gumbel loop records each position's improved-policy
target once, at self-play time, using the value net of the generation that
played the game. Generation zero's targets therefore come from a flat zero net
and carry little information. Under the coordinator's full uniform replay those
rows keep the same per-row weight in every later generation's policy
cross-entropy, so the policy head is asked to reproduce the oldest, weakest
targets as faithfully as the newest ones.

This driver refits the compact ``C4CNN001`` head from scratch for a chosen
generation ``g`` over the exact frozen shards gen0..geng the coordinator would
have used, changing one thing: each training row's *policy* cross-entropy term
is scaled by ``gamma ** (g - shard_index)`` (normalized to mean one over the
training rows). The value term keeps every row at weight one, so the value head
still sees the full uniform replay that is its established best treatment and
the comparison is not confounded by a simultaneous change to both heads.

``--gamma 1.0`` reproduces the unweighted baseline fit and is useful as a
control. Weights, per-generation result JSON and ``metrics.jsonl`` are written
by :func:`mixture_selfplay_c4.fit_and_diagnose`, so the refit ``gen<g>.c4cnn``
drops straight into the Rust diagnostics that score a net.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from az_train.fitability_c4 import _concat
from az_train.mirror_diagnostic_c4 import read_reference_corpus
from az_train.mixture_selfplay_c4 import fit_and_diagnose, read_concat_searched_values
from az_train.records_c4 import game_slices, load_positions
from az_train.replay_composition_c4 import shard_of_row
from az_train.trainer_hygiene_c4 import _pack_rows, replay_game_split_indices


def recency_policy_weights(row_shard: np.ndarray, generation: int, gamma: float) -> np.ndarray:
    """Mean-one policy-loss weights decaying geometrically into older shards.

    A row from shard ``i`` of a replay whose current generation is ``g`` gets
    raw weight ``gamma ** (g - i)``, so ``gamma == 1.0`` is uniform and smaller
    values discount the older, flatter recorded policy targets. The vector is
    rescaled to mean one so the policy term keeps the same overall magnitude
    against the value term regardless of ``gamma``.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    raw = np.power(float(gamma), generation - np.asarray(row_shard, dtype=np.float64))
    mean = float(raw.mean())
    if mean <= 0.0:
        raise ValueError("recency weights collapsed to zero")
    return raw / mean


def run_refit(
    corpus_path: Path,
    run_dir: Path,
    out_dir: Path,
    generation: int,
    *,
    gamma: float = 0.5,
    b: float = 0.75,
    l2: float = 1e-4,
    learning_rate: float = 2e-3,
    epochs: int = 25,
    seed: int = 20260907,
    validation_fraction: float = 0.2,
    split_seed: int = 20260908,
) -> dict[str, object]:
    positions_paths = [run_dir / f"gen{h}.bin" for h in range(generation + 1)]
    searched_paths = [run_dir / f"gen{h}.sv.f32" for h in range(generation + 1)]
    shards = [load_positions(path) for path in positions_paths]
    pos = _concat(shards)
    searched_all = read_concat_searched_values(searched_paths, len(pos))
    row_shard = shard_of_row([len(shard) for shard in shards])
    game_bounds = [(int(s.start), int(s.stop)) for s in game_slices(pos)]
    train_idx, held_out_idx, train_games, held_out_games = replay_game_split_indices(
        game_bounds, validation_fraction, split_seed
    )
    train = _pack_rows(pos, searched_all, train_idx)
    held_out = _pack_rows(pos, searched_all, held_out_idx)
    weights = recency_policy_weights(row_shard[train_idx], generation, gamma)
    counts: dict[str, object] = {
        "games": len(game_bounds),
        "train_games": train_games,
        "held_out_games": held_out_games,
        "train_rows": int(train["outcome"].size),
        "held_out_rows": int(held_out["outcome"].size),
        "total_records": int(len(pos)),
        "policy_recency": {
            "gamma": float(gamma),
            "train_rows_by_shard": [
                int((row_shard[train_idx] == h).sum()) for h in range(generation + 1)
            ],
            "policy_weight_by_shard": [
                round(float(weights[row_shard[train_idx] == h][0]), 6)
                if int((row_shard[train_idx] == h).sum()) else 0.0
                for h in range(generation + 1)
            ],
        },
    }
    return fit_and_diagnose(
        read_reference_corpus(corpus_path), corpus_path, positions_paths, searched_paths,
        train, held_out, counts, out_dir, generation,
        b=b, l2=l2, learning_rate=learning_rate, epochs=epochs, seed=seed,
        validation_fraction=validation_fraction, split_seed=split_seed,
        policy_row_weight=weights,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.policy_recency_c4")
    parser.add_argument("--reference-corpus", required=True, help="frozen corpus-v2.c4ref")
    parser.add_argument("--run-dir", required=True, help="graded run dir holding gen{N}.bin / gen{N}.sv.f32")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.5, help="policy-loss weight of shard i is gamma ** (g - i); 1.0 is the unweighted baseline")
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--replay-validation-fraction", type=float, default=0.2)
    parser.add_argument("--replay-split-seed", type=int, default=20260908)
    args = parser.parse_args(argv)

    result = run_refit(
        Path(args.reference_corpus), Path(args.run_dir), Path(args.out_dir), args.generation,
        gamma=args.gamma, b=args.b, l2=args.l2, learning_rate=args.learning_rate,
        epochs=args.epochs, seed=args.seed,
        validation_fraction=args.replay_validation_fraction,
        split_seed=args.replay_split_seed,
    )
    print(json.dumps({
        "generation": result["generation"],
        "replay_split": result["replay_split"],
        "policy_row_weight_summary": result["policy_row_weight_summary"],
        "principled_early_stop_epoch": result["principled_early_stop_epoch"],
        "held_out_proven": result["held_out_proven"],
        "fit_metrics": result["fit_metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
