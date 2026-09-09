# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
# ruff: noqa: E501
"""Diagnostic-only searched-value / outcome target-mixture sweep for the compact Connect Four value head.

This is not a production training path. It fits the literal ``C4CNN001`` value
head on the existing self-play replay shards with the value target replaced by

    target(b) = (1 - b) * self_play_outcome + b * searched_value

for a sweep of ``b`` in ``[0, 1]``, where ``searched_value`` is the offline
single-threaded ``MaterialBlind`` bounded-depth negamax scalar produced by the
``connect4_replay_searched_value`` Rust example (proven win/loss -> +/-1, exact
draw and unresolved cutoff -> 0, side-to-move perspective). Every arm is scored
on the *same* frozen ``C4REFD02`` held-out proven validation split, so ``b = 0``
reproduces the self-play-outcome arm and ``b = 1`` is a pure searched-value
target.

Policy targets stay the recorded completed-Q improved policies; value is the
object of study. Weights land under ``target-mixture/`` and are never loaded by
``coordinator_c4.sh`` or the replay trainer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from az_train.convnet_c4 import (
    N_WEIGHTS,
    fit_value_policy_with_diagnostics,
    predict,
    read_weights,
    write_weights,
)
from az_train.fitability_c4 import _concat, _dense_policy
from az_train.mirror_diagnostic_c4 import read_reference_corpus
from az_train.records_c4 import Positions, load_positions, me_opp_planes
from az_train.reference_label_fit_c4 import _proven_split, _sha256, value_report

DEFAULT_MIXTURE = (0.0, 0.25, 0.5, 0.75, 1.0)


def mixed_target(outcome: np.ndarray, searched: np.ndarray, b: float) -> np.ndarray:
    """Return ``(1 - b) * outcome + b * searched`` with matching-shape guards.

    ``b = 0`` returns the outcome target unchanged and ``b = 1`` the searched
    value; both inputs must be one-dimensional and the same length.
    """
    outcome = np.asarray(outcome, dtype=np.float64)
    searched = np.asarray(searched, dtype=np.float64)
    if outcome.ndim != 1 or outcome.shape != searched.shape:
        raise ValueError(f"outcome and searched must be matching 1-D arrays, got {outcome.shape} and {searched.shape}")
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"mixture weight b must be in [0, 1], got {b}")
    return (1.0 - b) * outcome + b * searched


def read_searched_values(path: str | Path, expected: int) -> np.ndarray:
    """Load the bare little-endian ``f32`` searched-value array and check its length."""
    raw = Path(path).read_bytes()
    if len(raw) != expected * 4:
        raise ValueError(f"{path}: holds {len(raw) // 4} searched values, expected {expected}")
    values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
        raise ValueError(f"{path}: searched values must be finite and within [-1, 1]")
    return values


def _replay_rows_with_searched(
    positions_paths: list[Path], searched_path: Path
) -> dict[str, np.ndarray]:
    """Every legal completed-Q replay row, with its outcome target and searched value.

    The searched-value array is one entry per input record in concatenation
    order, so it is indexed with the same policy-bearing row mask.
    """
    pos = _concat([load_positions(path) for path in positions_paths])
    searched_all = read_searched_values(searched_path, len(pos))
    rows = np.flatnonzero(np.array([len(entry) > 0 for entry in pos.policy]))
    if not rows.size:
        raise ValueError("self-play replay has no completed-Q policy rows")
    me, opp = me_opp_planes(Positions(
        black=pos.black[rows], white=pos.white[rows], side=pos.side[rows],
        ply=pos.ply[rows], value=pos.value[rows],
        policy=[pos.policy[int(row)] for row in rows],
    ))
    policy, legal = _dense_policy([pos.policy[int(row)] for row in rows])
    return {
        "me": me.astype(np.float32),
        "opp": opp.astype(np.float32),
        "outcome": pos.value[rows].astype(np.float64),
        "searched": searched_all[rows],
        "policy": policy,
        "legal": legal,
    }


def run_sweep(
    corpus_path: Path,
    positions_paths: list[Path],
    searched_path: Path,
    out_dir: Path,
    *,
    mixture: tuple[float, ...] = DEFAULT_MIXTURE,
    seed: int = 20260907,
    epochs: int = 60,
    l2: float = 1e-4,
    learning_rate: float = 2e-3,
) -> dict[str, object]:
    """Fit the literal value head at each mixture weight and score it on the proven split."""
    corpus = read_reference_corpus(corpus_path)
    replay = _replay_rows_with_searched(positions_paths, searched_path)
    val = _proven_split(corpus, 1)
    val_pack = (val["me"], val["opp"], val["value"], val["policy"], val["legal"])

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    arms: dict[str, object] = {}
    for b in mixture:
        target = mixed_target(replay["outcome"], replay["searched"], b).astype(np.float32)
        stem = f"b{b:0.2f}".replace(".", "_")
        early_path = out_dir / f"{stem}-earlystop.c4cnn"
        weights, meta = fit_value_policy_with_diagnostics(
            replay["me"], replay["opp"], target, replay["policy"], replay["legal"],
            val_pack, l2=l2, seed=seed, epochs=epochs, learning_rate=learning_rate,
            selected_validation_checkpoint_out=str(early_path),
        )
        final_path = out_dir / f"{stem}.c4cnn"
        write_weights(str(final_path), weights)
        early_weights = read_weights(str(early_path))
        arms[f"{b:.2f}"] = {
            "b": b,
            "weights_final": final_path.name,
            "weights_final_sha256": _sha256(final_path),
            "weights_early_stopped": early_path.name,
            "weights_early_stopped_sha256": _sha256(early_path),
            "selected_validation_epoch": meta["selected_validation_epoch"],
            "target_mean_abs": float(np.mean(np.abs(target))),
            "validation_report": value_report(
                predict(weights, val["me"], val["opp"])[0],
                val["value"], val["exact"], val["ply"], val["side"],
            ),
            "validation_report_early_stopped": value_report(
                predict(early_weights, val["me"], val["opp"])[0],
                val["value"], val["exact"], val["ply"], val["side"],
            ),
        }

    return {
        "config": {
            "mixture": list(mixture), "seed": seed, "epochs": epochs, "l2": l2,
            "learning_rate": learning_rate, "n_weights": int(N_WEIGHTS),
        },
        "reference_proven_counts": {
            "train": int((_proven_split(corpus, 0))["value"].size),
            "validation": int(val["value"].size),
            "validation_wins": int(np.count_nonzero(val["value"] > 0.0)),
            "validation_losses": int(np.count_nonzero(val["value"] < 0.0)),
        },
        "self_play_replay_rows": int(replay["outcome"].size),
        "searched_value_summary": {
            "mean_abs": float(np.mean(np.abs(replay["searched"]))),
            "proven_fraction": float(np.mean(replay["searched"] != 0.0)),
            "outcome_vs_searched_pearson": float(np.corrcoef(replay["outcome"], replay["searched"])[0, 1])
            if np.std(replay["searched"]) > 0.0 else 0.0,
        },
        "arms": arms,
        "wall_seconds": time.perf_counter() - started,
    }


def _headline(report: dict[str, object]) -> dict[str, float]:
    overall = report["overall"]  # type: ignore[index]
    return {k: round(float(overall[k]), 4) for k in ("value_mse", "value_pearson", "sign_agreement", "balanced_sign_accuracy", "mean_abs_prediction")}  # type: ignore[index]


def _band_pearson(report: dict[str, object]) -> dict[str, float]:
    return {
        band: round(float(report["by_ply_band"][band]["value_pearson"]), 4)  # type: ignore[index]
        for band in ("middle", "late")
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.target_mixture_c4")
    parser.add_argument("--reference-corpus", required=True, help="frozen C4REFD02 artifact")
    parser.add_argument("--positions", required=True, help="comma-separated v2-connect4 replay files")
    parser.add_argument("--searched-values", required=True, help="f32 array from connect4_replay_searched_value")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mixture", default=",".join(f"{b:g}" for b in DEFAULT_MIXTURE),
                        help="comma-separated mixture weights b in [0, 1]")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    args = parser.parse_args(argv)

    corpus_path = Path(args.reference_corpus)
    positions_paths = [Path(path) for path in args.positions.split(",")]
    searched_path = Path(args.searched_values)
    mixture = tuple(float(token) for token in args.mixture.split(","))
    out_dir = Path(args.out_dir)

    result = run_sweep(
        corpus_path, positions_paths, searched_path, out_dir,
        mixture=mixture, seed=args.seed, epochs=args.epochs, l2=args.l2,
        learning_rate=args.learning_rate,
    )
    result["inputs"] = {
        "reference_corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
        "searched_values": {"path": str(searched_path), "sha256": _sha256(searched_path)},
        "positions": [{"path": str(path), "sha256": _sha256(path)} for path in positions_paths],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    summary = {
        "reference_proven_counts": result["reference_proven_counts"],
        "searched_value_summary": result["searched_value_summary"],
        "held_out_proven_final_epoch": {
            b: _headline(arm["validation_report"]) for b, arm in result["arms"].items()  # type: ignore[union-attr,index]
        },
        "held_out_proven_early_stopped": {
            b: _headline(arm["validation_report_early_stopped"]) for b, arm in result["arms"].items()  # type: ignore[union-attr,index]
        },
        "held_out_proven_pearson_by_band_early_stopped": {
            b: _band_pearson(arm["validation_report_early_stopped"]) for b, arm in result["arms"].items()  # type: ignore[union-attr,index]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
