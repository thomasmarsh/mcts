# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
# ruff: noqa: E501
"""Per-generation fit driver for the diverse-opening Connect Four self-play smoke loop.

This is not wired into ``coordinator_c4.sh`` or the replay trainer. It fits one
generation of the compact ``C4CNN001`` value+policy head on the retained
self-play replay (every shard so far, generation zero always included) with the
value target

    target(b) = (1 - b) * self_play_outcome + b * searched_value

where ``searched_value`` is the offline single-threaded ``MaterialBlind``
bounded-depth negamax scalar from ``connect4_replay_searched_value``. The policy
target is the recorded completed-Q improved policy.

The replay is split into train and held-out games by whole game (never by
position). The principled early stop uses only the in-replay held-out split's
mixed-target Pearson. The frozen ``C4REFD02`` proven validation split is scored
once per epoch through ``epoch_monitor`` purely for measurement, so the
reference corpus never influences model selection.

Each generation writes its principled-early-stop weights as ``gen<N>.c4cnn``
(the weights the next generation's self-play uses), its final-epoch weights as
``gen<N>-final.c4cnn``, a per-generation ``gen<N>.result.json``, and one line
appended to ``metrics.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from az_train.convnet_c4 import (
    N_WEIGHTS,
    _pearson,
    fit_value_policy_with_diagnostics,
    predict,
    read_weights,
    write_weights,
)
from az_train.fitability_c4 import _concat
from az_train.mirror_diagnostic_c4 import ReferenceCorpus, read_reference_corpus
from az_train.records_c4 import game_slices, load_positions
from az_train.reference_label_fit_c4 import _proven_split, _sha256, value_report
from az_train.target_mixture_c4 import mixed_target
from az_train.trainer_hygiene_c4 import (
    _pack_rows,
    _proven_headline,
    oracle_epoch,
    replay_game_split_indices,
)


def read_concat_searched_values(paths: list[Path], expected: int) -> np.ndarray:
    """Concatenate per-shard ``f32`` searched-value arrays and check the total length."""
    arrays = [np.frombuffer(Path(p).read_bytes(), dtype="<f4").astype(np.float64) for p in paths]
    values = np.concatenate(arrays) if arrays else np.zeros(0)
    if values.size != expected:
        raise ValueError(f"searched values hold {values.size} entries, expected {expected}")
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
        raise ValueError("searched values must be finite and within [-1, 1]")
    return values


def load_split_replay(
    positions_paths: list[Path],
    searched_paths: list[Path],
    *,
    validation_fraction: float,
    split_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Whole-game train / held-out replay packs plus their searched-value arrays."""
    pos = _concat([load_positions(path) for path in positions_paths])
    searched_all = read_concat_searched_values(searched_paths, len(pos))
    game_bounds = [(int(s.start), int(s.stop)) for s in game_slices(pos)]
    train_idx, held_out_idx, train_games, held_out_games = replay_game_split_indices(
        game_bounds, validation_fraction, split_seed
    )
    train = _pack_rows(pos, searched_all, train_idx)
    held_out = _pack_rows(pos, searched_all, held_out_idx)
    counts: dict[str, object] = {
        "games": len(game_bounds),
        "train_games": train_games,
        "held_out_games": held_out_games,
        "train_rows": int(train["outcome"].size),
        "held_out_rows": int(held_out["outcome"].size),
        "total_records": int(len(pos)),
    }
    return train, held_out, counts


def _in_replay_pearson(weights: np.ndarray, pack: dict[str, np.ndarray], target: np.ndarray) -> float:
    return _pearson(predict(weights, pack["me"], pack["opp"])[0], target)


def run_generation(
    corpus_path: Path,
    positions_paths: list[Path],
    searched_paths: list[Path],
    out_dir: Path,
    generation: int,
    *,
    b: float = 0.75,
    l2: float = 1e-4,
    learning_rate: float = 2e-3,
    epochs: int = 30,
    seed: int = 20260907,
    validation_fraction: float = 0.2,
    split_seed: int = 20260908,
) -> dict[str, object]:
    corpus = read_reference_corpus(corpus_path)
    train, held_out, counts = load_split_replay(
        positions_paths, searched_paths,
        validation_fraction=validation_fraction, split_seed=split_seed,
    )
    return fit_and_diagnose(
        corpus, corpus_path, positions_paths, searched_paths,
        train, held_out, counts, out_dir, generation,
        b=b, l2=l2, learning_rate=learning_rate, epochs=epochs, seed=seed,
        validation_fraction=validation_fraction, split_seed=split_seed,
    )


def fit_and_diagnose(
    corpus: ReferenceCorpus,
    corpus_path: Path,
    positions_paths: list[Path],
    searched_paths: list[Path],
    train: dict[str, np.ndarray],
    held_out: dict[str, np.ndarray],
    counts: dict[str, object],
    out_dir: Path,
    generation: int,
    *,
    b: float = 0.75,
    l2: float = 1e-4,
    learning_rate: float = 2e-3,
    epochs: int = 30,
    seed: int = 20260907,
    validation_fraction: float = 0.2,
    split_seed: int = 20260908,
) -> dict[str, object]:
    """Fit the head on a prepared train/held-out replay pair and assemble the result record.

    Split out of :func:`run_generation` so a diagnostic driver can supply an
    alternately composed ``train`` pack while holding every fit knob and the
    reporting path fixed.
    """
    ref = _proven_split(corpus, 1)

    train_target = mixed_target(train["outcome"], train["searched"], b).astype(np.float32)
    held_out_target = mixed_target(held_out["outcome"], held_out["searched"], b).astype(np.float32)
    val_pack = (held_out["me"], held_out["opp"], held_out_target, held_out["policy"], held_out["legal"])

    def monitor(weights: np.ndarray) -> dict[str, float]:
        report = value_report(
            predict(weights, ref["me"], ref["opp"])[0],
            ref["value"], ref["exact"], ref["ply"], ref["side"],
        )
        overall = report["overall"]  # type: ignore[index]
        bands = report["by_ply_band"]  # type: ignore[index]
        return {
            "value_pearson": float(overall["value_pearson"]),  # type: ignore[index]
            "sign_agreement": float(overall["sign_agreement"]),  # type: ignore[index]
            "balanced_sign_accuracy": float(overall["balanced_sign_accuracy"]),  # type: ignore[index]
            "middle_pearson": float(bands["middle"]["value_pearson"]),  # type: ignore[index]
            "late_pearson": float(bands["late"]["value_pearson"]),  # type: ignore[index]
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    early_path = out_dir / f"gen{generation}.c4cnn"
    final_path = out_dir / f"gen{generation}-final.c4cnn"

    started = time.perf_counter()
    weights, meta = fit_value_policy_with_diagnostics(
        train["me"], train["opp"], train_target, train["policy"], train["legal"],
        val_pack, l2=l2, seed=seed, epochs=epochs, learning_rate=learning_rate,
        selected_validation_checkpoint_out=str(early_path),
        epoch_monitor=monitor,
    )
    write_weights(str(final_path), weights)
    early_weights = read_weights(str(early_path))

    monitor_trace: list[dict[str, float]] = list(meta["monitor_epoch_trace"])  # type: ignore[arg-type]
    principled_epoch = int(meta["selected_validation_epoch"])  # type: ignore[arg-type]
    oracle = oracle_epoch(monitor_trace)  # type: ignore[arg-type]

    def proven(w: np.ndarray) -> dict[str, float]:
        return _proven_headline(value_report(
            predict(w, ref["me"], ref["opp"])[0],
            ref["value"], ref["exact"], ref["ply"], ref["side"],
        ))

    result: dict[str, object] = {
        "generation": generation,
        "config": {
            "b": b, "l2": l2, "learning_rate": learning_rate, "epochs": epochs,
            "seed": seed, "validation_fraction": validation_fraction,
            "split_seed": split_seed, "n_weights": int(N_WEIGHTS),
        },
        "replay_split": counts,
        "reference_proven_counts": {
            "train": int(_proven_split(corpus, 0)["value"].size),
            "validation": int(ref["value"].size),
            "validation_wins": int(np.count_nonzero(ref["value"] > 0.0)),
            "validation_losses": int(np.count_nonzero(ref["value"] < 0.0)),
        },
        "searched_value_summary": {
            "train_proven_fraction": float(np.mean(train["searched"] != 0.0)),
            "held_out_proven_fraction": float(np.mean(held_out["searched"] != 0.0)),
            "outcome_vs_searched_pearson": float(np.corrcoef(
                np.concatenate([train["outcome"], held_out["outcome"]]),
                np.concatenate([train["searched"], held_out["searched"]]),
            )[0, 1]) if np.std(np.concatenate([train["searched"], held_out["searched"]])) > 0.0 else 0.0,
        },
        "principled_early_stop_epoch": principled_epoch,
        "oracle_epoch": oracle,
        "held_out_proven": {
            "principled_early_stop": proven(early_weights),
            "final_epoch": proven(weights),
        },
        "in_replay_mixed_target_pearson": {
            "principled_early_stop": {
                "train": _in_replay_pearson(early_weights, train, train_target),
                "held_out": _in_replay_pearson(early_weights, held_out, held_out_target),
            },
            "final_epoch": {
                "train": _in_replay_pearson(weights, train, train_target),
                "held_out": _in_replay_pearson(weights, held_out, held_out_target),
            },
        },
        "monitor_proven_pearson_by_epoch": [round(float(m["value_pearson"]), 4) for m in monitor_trace],
        "weights": {
            "principled_early_stop": early_path.name,
            "principled_early_stop_sha256": _sha256(early_path),
            "final_epoch": final_path.name,
            "final_epoch_sha256": _sha256(final_path),
        },
        "fit_metrics": {
            "train_value_mse": round(float(meta["train_value_mse"]), 5),  # type: ignore[arg-type]
            "validation_value_mse": round(float(meta["validation_value_mse"]), 5),  # type: ignore[arg-type]
            "train_policy_cross_entropy": round(float(meta["train_policy_cross_entropy"]), 5),  # type: ignore[arg-type]
            "validation_policy_cross_entropy": round(float(meta["validation_policy_cross_entropy"]), 5),  # type: ignore[arg-type]
        },
        "fit_wall_seconds": round(float(meta["fit_wall_seconds"]), 1),  # type: ignore[arg-type]
        "peak_rss_bytes": int(meta["peak_rss_bytes"]),  # type: ignore[arg-type]
        "driver_wall_seconds": round(time.perf_counter() - started, 1),
        "inputs": {
            "reference_corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
            "positions": [{"path": str(p), "sha256": _sha256(p)} for p in positions_paths],
            "searched_values": [{"path": str(p), "sha256": _sha256(p)} for p in searched_paths],
        },
    }
    (out_dir / f"gen{generation}.result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (out_dir / "metrics.jsonl").open("a") as fh:
        fh.write(json.dumps(result, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.mixture_selfplay_c4")
    parser.add_argument("--reference-corpus", required=True, help="corrected C4REFD02 artifact")
    parser.add_argument("--positions", required=True, help="comma-separated v2-connect4 replay shards, generation zero first")
    parser.add_argument("--searched-values", required=True, help="comma-separated per-shard f32 arrays, same order as --positions")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--replay-validation-fraction", type=float, default=0.2)
    parser.add_argument("--replay-split-seed", type=int, default=20260908)
    args = parser.parse_args(argv)

    positions_paths = [Path(p) for p in args.positions.split(",")]
    searched_paths = [Path(p) for p in args.searched_values.split(",")]
    if len(positions_paths) != len(searched_paths):
        parser.error("--positions and --searched-values must list the same number of files")

    result = run_generation(
        Path(args.reference_corpus), positions_paths, searched_paths,
        Path(args.out_dir), args.generation,
        b=args.b, l2=args.l2, learning_rate=args.learning_rate, epochs=args.epochs,
        seed=args.seed, validation_fraction=args.replay_validation_fraction,
        split_seed=args.replay_split_seed,
    )
    print(json.dumps({
        "generation": result["generation"],
        "replay_split": result["replay_split"],
        "searched_value_summary": result["searched_value_summary"],
        "principled_early_stop_epoch": result["principled_early_stop_epoch"],
        "held_out_proven": result["held_out_proven"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
