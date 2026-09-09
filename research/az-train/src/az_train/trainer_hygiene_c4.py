# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
# ruff: noqa: E501
"""Diagnostic-only trainer-hygiene sweep for the compact Connect Four value head.

This is not a production training path.  It answers a single question: how much
of the gap between the self-play-outcome value head's held-out proven Pearson
(~0.11) and the raw outcome/best-play alignment (~0.39, ceiling 0.66) on the
existing small `compact-conv-smoke-gradient-fix` replay is recoverable from
trainer hygiene alone -- epoch budget, a principled early stop, and L2 -- with no
new self-play.

For each `(b, l2, learning_rate)` the literal `C4CNN001` value head is fitted on
the replay with the value target

    target(b) = (1 - b) * self_play_outcome + b * searched_value

The replay is split into train and held-out games by whole game (never by
position).  Model selection -- the principled early stop -- uses only the
in-replay held-out split's mixed-target Pearson.  The frozen `C4REFD02` proven
validation split is scored once per epoch through `epoch_monitor` purely for
measurement, so the reference corpus never influences selection.  Every arm is
also reported at the oracle epoch (the reference-corpus argmax) to bound how much
a perfect early stop could add.

Policy targets stay the recorded completed-Q improved policies; value is the
object of study.  Weights land under `trainer-hygiene/` and are never loaded by
`coordinator_c4.sh` or the replay trainer.
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
from az_train.fitability_c4 import _concat, _dense_policy
from az_train.mirror_diagnostic_c4 import read_reference_corpus
from az_train.records_c4 import Positions, game_slices, load_positions, me_opp_planes
from az_train.reference_label_fit_c4 import _proven_split, _sha256, value_report
from az_train.target_mixture_c4 import mixed_target, read_searched_values

DEFAULT_B = (0.0, 0.75)
DEFAULT_L2 = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2)
DEFAULT_LR = (2e-3,)


def replay_game_split_indices(
    game_bounds: list[tuple[int, int]], validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Assign whole games to train / held-out.

    Returns the record indices of each split plus the train and held-out game
    counts.

    No record from one game ever straddles the split.  The assignment is a
    seeded permutation so it is reproducible and independent of game order.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if len(game_bounds) < 2:
        raise ValueError("need at least two complete games for a train/held-out split")
    n_val = min(len(game_bounds) - 1, max(1, round(len(game_bounds) * validation_fraction)))
    order = np.random.default_rng(seed).permutation(len(game_bounds))
    val_games = {int(i) for i in order[:n_val]}
    train = np.concatenate(
        [np.arange(start, stop) for i, (start, stop) in enumerate(game_bounds) if i not in val_games]
    )
    held_out = np.concatenate(
        [np.arange(start, stop) for i, (start, stop) in enumerate(game_bounds) if i in val_games]
    )
    return train, held_out, len(game_bounds) - n_val, n_val


def oracle_epoch(monitor_trace: list[dict[str, float]]) -> int:
    """One-based epoch with the highest monitored proven Pearson, earliest on a tie."""
    if not monitor_trace:
        raise ValueError("cannot pick an oracle epoch from an empty monitor trace")
    return 1 + max(
        range(len(monitor_trace)),
        key=lambda epoch: (monitor_trace[epoch]["value_pearson"], -epoch),
    )


def _pack_rows(pos: Positions, searched_all: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Planes, outcome, searched value, and completed-Q policy for the policy-bearing rows."""
    idx = np.array(
        [int(r) for r in indices if len(pos.policy[int(r)]) > 0], dtype=np.int64
    )
    if not idx.size:
        raise ValueError("replay split contains no completed-Q policy rows")
    subset = Positions(
        black=pos.black[idx], white=pos.white[idx], side=pos.side[idx],
        ply=pos.ply[idx], value=pos.value[idx],
        policy=[pos.policy[int(r)] for r in idx],
    )
    me, opp = me_opp_planes(subset)
    policy, legal = _dense_policy([pos.policy[int(r)] for r in idx])
    return {
        "me": me.astype(np.float32),
        "opp": opp.astype(np.float32),
        "outcome": pos.value[idx].astype(np.float64),
        "searched": searched_all[idx],
        "policy": policy,
        "legal": legal,
        "ply": pos.ply[idx],
    }


def load_split_replay(
    positions_paths: list[Path], searched_path: Path, *, validation_fraction: float, split_seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Whole-game train / held-out replay packs plus their searched-value arrays."""
    pos = _concat([load_positions(path) for path in positions_paths])
    searched_all = read_searched_values(searched_path, len(pos))
    game_bounds = [(int(s.start), int(s.stop)) for s in game_slices(pos)]
    train_idx, held_out_idx, train_games, held_out_games = replay_game_split_indices(
        game_bounds, validation_fraction, split_seed
    )
    train = _pack_rows(pos, searched_all, train_idx)
    held_out = _pack_rows(pos, searched_all, held_out_idx)
    counts = {
        "games": len(game_bounds),
        "train_games": train_games,
        "held_out_games": held_out_games,
        "train_rows": int(train["outcome"].size),
        "held_out_rows": int(held_out["outcome"].size),
    }
    return train, held_out, counts


def _value_fit_metrics(weights: np.ndarray, me: np.ndarray, opp: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = predict(weights, me, opp)[0]
    return {
        "value_pearson": _pearson(prediction, target),
        "value_mse": float(np.mean((prediction - target) ** 2)),
    }


def _proven_headline(report: dict[str, object]) -> dict[str, float]:
    overall = report["overall"]  # type: ignore[index]
    bands = report["by_ply_band"]  # type: ignore[index]
    return {
        "value_pearson": round(float(overall["value_pearson"]), 4),  # type: ignore[index]
        "sign_agreement": round(float(overall["sign_agreement"]), 4),  # type: ignore[index]
        "balanced_sign_accuracy": round(float(overall["balanced_sign_accuracy"]), 4),  # type: ignore[index]
        "value_mse": round(float(overall["value_mse"]), 4),  # type: ignore[index]
        "mean_abs_prediction": round(float(overall["mean_abs_prediction"]), 4),  # type: ignore[index]
        "middle_pearson": round(float(bands["middle"]["value_pearson"]), 4),  # type: ignore[index]
        "middle_mse": round(float(bands["middle"]["value_mse"]), 4),  # type: ignore[index]
        "late_pearson": round(float(bands["late"]["value_pearson"]), 4),  # type: ignore[index]
        "late_mse": round(float(bands["late"]["value_mse"]), 4),  # type: ignore[index]
    }


def run_sweep(
    corpus_path: Path,
    positions_paths: list[Path],
    searched_path: Path,
    out_dir: Path,
    *,
    b_values: tuple[float, ...] = DEFAULT_B,
    l2_values: tuple[float, ...] = DEFAULT_L2,
    lr_values: tuple[float, ...] = DEFAULT_LR,
    seed: int = 20260907,
    epochs: int = 80,
    validation_fraction: float = 0.2,
    split_seed: int = 20260908,
) -> dict[str, object]:
    corpus = read_reference_corpus(corpus_path)
    ref = _proven_split(corpus, 1)
    train, held_out, counts = load_split_replay(
        positions_paths, searched_path, validation_fraction=validation_fraction, split_seed=split_seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "config": {
            "b_values": list(b_values), "l2_values": list(l2_values), "lr_values": list(lr_values),
            "seed": seed, "epochs": epochs, "validation_fraction": validation_fraction,
            "split_seed": split_seed, "n_weights": int(N_WEIGHTS),
        },
        "replay_split": counts,
        "reference_proven_counts": {
            "train": int((_proven_split(corpus, 0))["value"].size),
            "validation": int(ref["value"].size),
            "validation_wins": int(np.count_nonzero(ref["value"] > 0.0)),
            "validation_losses": int(np.count_nonzero(ref["value"] < 0.0)),
        },
        "arms": {},
    }

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
            "value_mse": float(overall["value_mse"]),  # type: ignore[index]
            "middle_pearson": float(bands["middle"]["value_pearson"]),  # type: ignore[index]
            "late_pearson": float(bands["late"]["value_pearson"]),  # type: ignore[index]
        }

    overall_started = time.perf_counter()
    for b in b_values:
        train_target = mixed_target(train["outcome"], train["searched"], b).astype(np.float32)
        held_out_target = mixed_target(held_out["outcome"], held_out["searched"], b).astype(np.float32)
        val_pack = (held_out["me"], held_out["opp"], held_out_target, held_out["policy"], held_out["legal"])
        for l2 in l2_values:
            for lr in lr_values:
                key = f"b{b:g}_l2{l2:g}_lr{lr:g}"
                early_path = out_dir / f"{key}-earlystop.c4cnn"
                weights, meta = fit_value_policy_with_diagnostics(
                    train["me"], train["opp"], train_target, train["policy"], train["legal"],
                    val_pack, l2=l2, seed=seed, epochs=epochs, learning_rate=lr,
                    selected_validation_checkpoint_out=str(early_path),
                    epoch_monitor=monitor,
                )
                final_path = out_dir / f"{key}.c4cnn"
                write_weights(str(final_path), weights)
                early_weights = read_weights(str(early_path))

                monitor_trace: list[dict[str, float]] = list(meta["monitor_epoch_trace"])  # type: ignore[arg-type]
                principled_epoch = int(meta["selected_validation_epoch"])  # type: ignore[arg-type]
                oracle = oracle_epoch(monitor_trace)  # type: ignore[arg-type]

                oracle_monitor: dict[str, float] = {
                    k: round(float(v), 4) for k, v in monitor_trace[oracle - 1].items()
                }

                final_report = value_report(
                    predict(weights, ref["me"], ref["opp"])[0],
                    ref["value"], ref["exact"], ref["ply"], ref["side"],
                )
                early_report = value_report(
                    predict(early_weights, ref["me"], ref["opp"])[0],
                    ref["value"], ref["exact"], ref["ply"], ref["side"],
                )

                final_headline = _proven_headline(final_report)
                early_headline = _proven_headline(early_report)
                arm: dict[str, object] = {
                    "b": b, "l2": l2, "learning_rate": lr,
                    "principled_early_stop_epoch": principled_epoch,
                    "oracle_epoch": oracle,
                    "weights_final": final_path.name,
                    "weights_final_sha256": _sha256(final_path),
                    "weights_early_stopped": early_path.name,
                    "weights_early_stopped_sha256": _sha256(early_path),
                    "held_out_proven": {
                        "final_epoch": final_headline,
                        "principled_early_stop": early_headline,
                        "oracle_epoch_monitor": oracle_monitor,
                    },
                    "in_replay_gap": {
                        "final_epoch": {
                            "train": _value_fit_metrics(weights, train["me"], train["opp"], train_target),
                            "held_out": _value_fit_metrics(weights, held_out["me"], held_out["opp"], held_out_target),
                        },
                        "principled_early_stop": {
                            "train": _value_fit_metrics(early_weights, train["me"], train["opp"], train_target),
                            "held_out": _value_fit_metrics(early_weights, held_out["me"], held_out["opp"], held_out_target),
                        },
                    },
                    "monitor_proven_pearson_by_epoch": [round(float(m["value_pearson"]), 4) for m in monitor_trace],
                    "fit_wall_seconds": round(float(meta["fit_wall_seconds"]), 1),  # type: ignore[arg-type]
                    "peak_rss_bytes": int(meta["peak_rss_bytes"]),  # type: ignore[arg-type]
                }
                arms: dict[str, object] = result["arms"]  # type: ignore[assignment]
                arms[key] = arm
                (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                print(f"{key}: early-stop ep {principled_epoch} proven r "
                      f"{early_headline['value_pearson']}, final r {final_headline['value_pearson']}, "
                      f"oracle ep {oracle} r {oracle_monitor['value_pearson']} "
                      f"({round(float(meta['fit_wall_seconds']), 1)}s)", flush=True)  # type: ignore[arg-type]

    result["wall_seconds"] = round(time.perf_counter() - overall_started, 1)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.trainer_hygiene_c4")
    parser.add_argument("--reference-corpus", required=True, help="corrected C4REFD02 artifact")
    parser.add_argument("--positions", required=True, help="comma-separated v2-connect4 replay files")
    parser.add_argument("--searched-values", required=True, help="f32 array from connect4_replay_searched_value")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--b", default=",".join(f"{b:g}" for b in DEFAULT_B), help="comma-separated mixture weights")
    parser.add_argument("--l2", default=",".join(f"{v:g}" for v in DEFAULT_L2), help="comma-separated L2 weights")
    parser.add_argument("--learning-rate", default=",".join(f"{v:g}" for v in DEFAULT_LR), help="comma-separated learning rates")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--replay-validation-fraction", type=float, default=0.2)
    parser.add_argument("--replay-split-seed", type=int, default=20260908)
    args = parser.parse_args(argv)

    corpus_path = Path(args.reference_corpus)
    positions_paths = [Path(path) for path in args.positions.split(",")]
    searched_path = Path(args.searched_values)
    out_dir = Path(args.out_dir)

    result = run_sweep(
        corpus_path, positions_paths, searched_path, out_dir,
        b_values=tuple(float(t) for t in args.b.split(",")),
        l2_values=tuple(float(t) for t in args.l2.split(",")),
        lr_values=tuple(float(t) for t in args.learning_rate.split(",")),
        seed=args.seed, epochs=args.epochs,
        validation_fraction=args.replay_validation_fraction,
        split_seed=args.replay_split_seed,
    )
    result["inputs"] = {
        "reference_corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
        "searched_values": {"path": str(searched_path), "sha256": _sha256(searched_path)},
        "positions": [{"path": str(path), "sha256": _sha256(path)} for path in positions_paths],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arms: dict[str, dict[str, object]] = result["arms"]  # type: ignore[assignment]
    print(json.dumps({key: arm["held_out_proven"] for key, arm in arms.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
