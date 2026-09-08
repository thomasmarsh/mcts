# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
# ruff: noqa: E501
"""Diagnostic-only fit of the compact Connect Four value head on proven reference labels.

This is not a production training path.  It fits the ``C4CNN001`` value-and-policy
head directly against the frozen ``C4REFD01`` strong-reference proven labels, held
out by whole source game, to separate two hypotheses that every prior Slice 4
head confounded: is the self-play-outcome value *target* the thing that fails to
generalize, or is it the representation / corpus?

Three arms are fitted and scored on the *same* proven reference validation split:

* the mirror-equivariant value head trained on proven reference labels;
* the literal value head trained on proven reference labels (control);
* the literal value head trained on the self-play replay outcome target,
  evaluated on the proven reference validation split (the direct A/B).

The weights written here are diagnostic artifacts.  They use the same container
as production weights but live under ``reference-label-fit/`` and are never loaded
by ``coordinator_c4.sh`` or the replay trainer.  Policy targets are uniform over
legal columns for the reference arms because value is the object of study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from az_train.convnet_c4 import (
    N_WEIGHTS,
    _pearson,
    _predict_value_equivariant,
    fit_equivariant_value_policy,
    fit_value_policy_with_diagnostics,
    predict,
    read_weights,
    write_weights,
)
from az_train.fitability_c4 import _concat, _dense_policy
from az_train.mirror_diagnostic_c4 import ReferenceCorpus, read_reference_corpus
from az_train.records_c4 import Positions, load_positions, me_opp_planes

ROWS = 6
COLS = 7
_TOP_ROW_BASE = (ROWS - 1) * COLS
_PLY_BAND_EDGES = (15, 23)
_PLY_BAND_NAMES = ("opening", "middle", "late")


def legal_columns_from_boards(black: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Return an ``(N, 7)`` bool mask of columns that still accept a disc.

    A column is full when its top cell (row 5, bit ``35 + col``) is occupied by
    either player.
    """
    occ = np.asarray(black, dtype=np.uint64) | np.asarray(white, dtype=np.uint64)
    cols = np.arange(COLS, dtype=np.uint64)
    top_filled = ((occ[:, None] >> (np.uint64(_TOP_ROW_BASE) + cols[None, :])) & np.uint64(1)).astype(bool)
    return ~top_filled


def uniform_legal_policy(legal: np.ndarray) -> np.ndarray:
    """Return an ``(N, 7)`` float32 policy that is uniform over the legal columns."""
    legal = np.asarray(legal, dtype=bool)
    counts = legal.sum(axis=1, keepdims=True)
    if not np.all(counts > 0):
        raise ValueError("every position must have at least one legal column")
    return (legal / counts).astype(np.float32)


def ply_band(ply: np.ndarray) -> np.ndarray:
    """Map each disc count to an ``opening`` / ``middle`` / ``late`` band label."""
    ply = np.asarray(ply)
    index = np.digitize(ply, _PLY_BAND_EDGES)
    return np.asarray(_PLY_BAND_NAMES, dtype=object)[index]


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Finite value metrics over proven win/loss labels.

    Neutral documented values are returned for empty, singleton, constant-vector,
    and one-class inputs: correlation 0.0, sign accuracy 0.0.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    nonzero = target != 0.0
    signs = np.sign(prediction[nonzero]) == np.sign(target[nonzero])
    per_class = [signs[np.sign(target[nonzero]) == side] for side in (-1.0, 1.0)]
    present = [float(np.mean(group)) for group in per_class if group.size]
    return {
        "count": int(prediction.size),
        "value_mse": float(np.mean((prediction - target) ** 2)) if prediction.size else 1.0,
        "value_pearson": _pearson(prediction, target),
        "sign_agreement": float(np.mean(signs)) if signs.size else 0.0,
        "balanced_sign_accuracy": float(np.mean(present)) if present else 0.0,
        "mean_abs_prediction": float(np.mean(np.abs(prediction))) if prediction.size else 0.0,
    }


def value_report(
    prediction: np.ndarray,
    target: np.ndarray,
    exact: np.ndarray,
    ply: np.ndarray,
    side: np.ndarray,
) -> dict[str, object]:
    """Overall proven-label metrics plus exact/bounded, ply-band, and side breakouts."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    exact = np.asarray(exact, dtype=bool)
    bands = ply_band(ply)
    side = np.asarray(side)

    def slice_metrics(mask: np.ndarray) -> dict[str, float]:
        mask = np.asarray(mask, dtype=bool)
        return _metrics(prediction[mask], target[mask])

    return {
        "overall": _metrics(prediction, target),
        "zero_predictor": _metrics(np.zeros_like(prediction), target),
        "by_proof": {
            "exact": slice_metrics(exact),
            "bounded": slice_metrics(~exact),
        },
        "by_ply_band": {name: slice_metrics(bands == name) for name in _PLY_BAND_NAMES},
        "by_side_to_move": {
            "black_to_move": slice_metrics(side == 0),
            "white_to_move": slice_metrics(side == 1),
        },
    }


def _proven_split(corpus: ReferenceCorpus, split: int) -> dict[str, np.ndarray]:
    """Proven-label planes, targets, and strata for one frozen group split."""
    keep = (corpus.split == split) & np.isfinite(corpus.label_sign)
    pos = corpus.positions
    subset = Positions(
        black=pos.black[keep], white=pos.white[keep], side=pos.side[keep],
        ply=pos.ply[keep], value=pos.value[keep], policy=[],
    )
    me, opp = me_opp_planes(subset)
    legal = legal_columns_from_boards(pos.black[keep], pos.white[keep])
    return {
        "me": me.astype(np.float32),
        "opp": opp.astype(np.float32),
        "value": corpus.label_sign[keep].astype(np.float32),
        "policy": uniform_legal_policy(legal),
        "legal": legal,
        "exact": corpus.exact[keep],
        "ply": pos.ply[keep],
        "side": pos.side[keep],
        "source_outcome": corpus.source_outcome[keep].astype(np.float64),
        "group": corpus.group[keep],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay_rows(positions_paths: list[Path]) -> dict[str, np.ndarray]:
    """Every legal self-play replay row with a completed-Q policy target."""
    pos = _concat([load_positions(path) for path in positions_paths])
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
        "value": pos.value[rows].astype(np.float32),
        "policy": policy,
        "legal": legal,
    }


def run_arms(
    corpus: ReferenceCorpus,
    replay: dict[str, np.ndarray],
    out_dir: Path,
    *,
    seed: int,
    epochs: int,
    l2: float,
    equivariant_epochs: int | None = None,
    equivariant_lr: float = 2e-3,
    literal_lr: float = 2e-3,
) -> dict[str, object]:
    """Fit the three arms and score every one on the proven reference validation split."""
    equivariant_epochs = equivariant_epochs if equivariant_epochs is not None else epochs
    train = _proven_split(corpus, 0)
    val = _proven_split(corpus, 1)
    overlap = np.intersect1d(np.unique(train["group"]), np.unique(val["group"]))
    if overlap.size:
        raise ValueError(f"reference split leaks {overlap.size} source groups across train/validation")

    ref_train_pack = (train["me"], train["opp"], train["value"], train["policy"], train["legal"])
    val_pack = (val["me"], val["opp"], val["value"], val["policy"], val["legal"])

    out_dir.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, object]] = {}
    fit_metadata: dict[str, object] = {}

    started = time.perf_counter()
    eq_weights, eq_meta = fit_equivariant_value_policy(
        *ref_train_pack, val_pack, l2=l2, seed=seed, epochs=equivariant_epochs,
        learning_rate=equivariant_lr,
    )
    eq_path = out_dir / "equivariant-proven.c4cnn"
    write_weights(str(eq_path), eq_weights)
    arms["equivariant_trained_on_proven"] = {
        "weights_final": eq_path.name,
        "weights_final_sha256": _sha256(eq_path),
        "train_report": value_report(
            _predict_value_equivariant(eq_weights, train["me"], train["opp"]),
            train["value"], train["exact"], train["ply"], train["side"],
        ),
        "validation_report": value_report(
            _predict_value_equivariant(eq_weights, val["me"], val["opp"]),
            val["value"], val["exact"], val["ply"], val["side"],
        ),
    }
    fit_metadata["equivariant_trained_on_proven"] = eq_meta

    def literal_arm(
        name: str,
        train_pack: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        stem: str,
        report_train: bool,
    ) -> None:
        early_path = out_dir / f"{stem}-earlystop.c4cnn"
        weights, meta = fit_value_policy_with_diagnostics(
            *train_pack, val_pack, l2=l2, seed=seed, epochs=epochs,
            learning_rate=literal_lr,
            selected_validation_checkpoint_out=str(early_path),
        )
        final_path = out_dir / f"{stem}.c4cnn"
        write_weights(str(final_path), weights)
        early_weights = read_weights(str(early_path))
        arm: dict[str, object] = {
            "weights_final": final_path.name,
            "weights_final_sha256": _sha256(final_path),
            "weights_early_stopped": early_path.name,
            "weights_early_stopped_sha256": _sha256(early_path),
            "selected_validation_epoch": meta["selected_validation_epoch"],
            "validation_report": value_report(
                predict(weights, val["me"], val["opp"])[0],
                val["value"], val["exact"], val["ply"], val["side"],
            ),
            "validation_report_early_stopped": value_report(
                predict(early_weights, val["me"], val["opp"])[0],
                val["value"], val["exact"], val["ply"], val["side"],
            ),
        }
        if report_train:
            arm["train_report"] = value_report(
                predict(weights, train["me"], train["opp"])[0],
                train["value"], train["exact"], train["ply"], train["side"],
            )
        arms[name] = arm
        fit_metadata[name] = meta

    literal_arm(
        "literal_trained_on_proven", ref_train_pack, "literal-proven", report_train=True,
    )
    literal_arm(
        "literal_trained_on_selfplay_outcome",
        (replay["me"], replay["opp"], replay["value"], replay["policy"], replay["legal"]),
        "literal-selfplay", report_train=False,
    )

    # Direct measurement of the diagnostic-target noise: the source-game outcome
    # versus the proven best-play label on the same validation positions.
    source_outcome_vs_proven = _metrics(val["source_outcome"], val["value"])

    return {
        "config": {
            "seed": seed, "epochs": epochs, "l2": l2, "n_weights": int(N_WEIGHTS),
            "equivariant_epochs": equivariant_epochs, "equivariant_lr": equivariant_lr,
            "literal_lr": literal_lr,
        },
        "reference_proven_counts": {
            "train": int(train["value"].size),
            "validation": int(val["value"].size),
            "validation_wins": int(np.count_nonzero(val["value"] > 0.0)),
            "validation_losses": int(np.count_nonzero(val["value"] < 0.0)),
            "validation_black_to_move": int(np.count_nonzero(val["side"] == 0)),
            "validation_white_to_move": int(np.count_nonzero(val["side"] == 1)),
        },
        "self_play_replay_rows": int(replay["value"].size),
        "source_outcome_vs_proven_label_validation": source_outcome_vs_proven,
        "arms": arms,
        "fit_metadata": fit_metadata,
        "wall_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.reference_label_fit_c4")
    parser.add_argument("--reference-corpus", required=True, help="frozen C4REFD01 artifact")
    parser.add_argument("--positions", required=True, help="comma-separated v2-connect4 replay files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--equivariant-epochs", type=int, default=None,
                        help="epoch budget for the equivariant arm (defaults to --epochs)")
    parser.add_argument("--equivariant-lr", type=float, default=2e-3)
    parser.add_argument("--literal-lr", type=float, default=2e-3)
    args = parser.parse_args(argv)

    corpus_path = Path(args.reference_corpus)
    positions_paths = [Path(path) for path in args.positions.split(",")]
    corpus = read_reference_corpus(corpus_path)
    replay = _replay_rows(positions_paths)

    out_dir = Path(args.out_dir)
    result = run_arms(
        corpus, replay, out_dir, seed=args.seed, epochs=args.epochs, l2=args.l2,
        equivariant_epochs=args.equivariant_epochs, equivariant_lr=args.equivariant_lr,
        literal_lr=args.literal_lr,
    )
    result["inputs"] = {
        "reference_corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
        "positions": [{"path": str(path), "sha256": _sha256(path)} for path in positions_paths],
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    def headline(report: dict[str, object]) -> dict[str, float]:
        overall = report["overall"]  # type: ignore[index]
        return {k: round(float(overall[k]), 4) for k in ("value_mse", "value_pearson", "sign_agreement", "balanced_sign_accuracy", "mean_abs_prediction")}  # type: ignore[index]

    summary = {
        "reference_proven_counts": result["reference_proven_counts"],
        "source_outcome_vs_proven_label_validation": {
            k: round(float(v), 4) for k, v in result["source_outcome_vs_proven_label_validation"].items()  # type: ignore[union-attr]
        },
        "held_out_validation_final_epoch": {
            name: headline(arm["validation_report"])  # type: ignore[index]
            for name, arm in result["arms"].items()  # type: ignore[union-attr]
        },
        "held_out_validation_early_stopped": {
            name: headline(arm["validation_report_early_stopped"])  # type: ignore[index]
            for name, arm in result["arms"].items()  # type: ignore[union-attr]
            if "validation_report_early_stopped" in arm  # type: ignore[operator]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
