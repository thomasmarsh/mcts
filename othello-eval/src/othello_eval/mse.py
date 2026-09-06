"""Held-out regression diagnostic for trained n-tuple weights.

The bake-off kill gate leans on this when the Edax ladder
saturates at its floor: does arm C's model predict a deep-search value more
accurately than arm A's, by a margin whose bootstrap CI excludes zero?

Value prediction matches the Rust `NTupleEval::evaluate` hot path exactly:
``tanh(sum of selected weights)``, in ``[-1, 1]``, side-to-move perspective.
The held-out targets are themselves searched values (a deep label search on
independent seeds -- the best cheap proxy for ground truth).
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from othello_eval.ntuple import ModelGeometry, featurize, load_model_toml
from othello_eval.records import load_positions


def predict(weights: np.ndarray, feat_idx: np.ndarray) -> np.ndarray:
    """``tanh(sum of selected weights)`` per position -- the Rust evaluator's
    value, in ``[-1, 1]``."""
    return np.tanh(weights[feat_idx].sum(axis=1))


def load_weights(weights_dir: str | Path) -> tuple[np.ndarray, ModelGeometry]:
    d = Path(weights_dir)
    geom = load_model_toml(d / "model.toml")
    meta = json.loads((d / "weights.meta.json").read_text())
    if meta["model_toml_sha256"] != geom.sha256_hex:
        raise ValueError(f"{d}: weights.meta.json geometry hash mismatch")
    w = np.fromfile(d / "weights.bin", dtype="<f4").astype(np.float64)
    if w.size != geom.n_weights:
        raise ValueError(f"{d}: {w.size} weights, geometry wants {geom.n_weights}")
    return w, geom


def squared_errors(weights_dir: str | Path, positions: np.ndarray) -> np.ndarray:
    w, geom = load_weights(weights_dir)
    feat_idx = featurize(positions, geom)
    pred = predict(w, feat_idx)
    target = positions["target"].astype(np.float64)
    return (pred - target) ** 2


def sign_accuracy(weights_dir: str | Path, positions: np.ndarray) -> float:
    w, geom = load_weights(weights_dir)
    pred = predict(w, featurize(positions, geom))
    target = positions["target"].astype(np.float64)
    decisive = np.abs(target) > 1e-6
    if not decisive.any():
        return float("nan")
    return float(np.mean(np.sign(pred[decisive]) == np.sign(target[decisive])))


def bootstrap_diff_ci(
    se_a: np.ndarray, se_b: np.ndarray, iters: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """95% percentile bootstrap CI for ``MSE(a) - MSE(b)`` (paired resample
    over positions). A CI strictly above 0 means model *b* is the more
    accurate one."""
    rng = np.random.default_rng(seed)
    n = len(se_a)
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        diffs[i] = se_a[idx].mean() - se_b[idx].mean()
    point = float(se_a.mean() - se_b.mean())
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    return point, lo, hi


def mse_cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="othello-eval-mse")
    ap.add_argument("--held", required=True, help="held-out positions .bin (deep-search labelled)")
    ap.add_argument(
        "--weights",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="a trained weights directory, tagged; repeat for a pairwise comparison",
    )
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--json-out", help="also write the report as JSON here")
    args = ap.parse_args(argv)

    positions = load_positions(args.held)
    print(f"held-out: {len(positions)} positions from {args.held}", flush=True)

    models: dict[str, str] = {}
    for spec in args.weights:
        name, _, path = spec.partition("=")
        models[name or path] = path or name

    models_report: dict[str, object] = {}
    pairwise: dict[str, object] = {}
    se: dict[str, np.ndarray] = {}
    for name, path in models.items():
        se[name] = squared_errors(path, positions)
        mse = float(se[name].mean())
        acc = sign_accuracy(path, positions)
        models_report[name] = {"mse": mse, "sign_accuracy": acc, "dir": path}
        print(f"  {name:<12} MSE {mse:.5f}   sign-acc {acc:.4f}", flush=True)

    names = list(models)
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                point, lo, hi = bootstrap_diff_ci(se[a], se[b], args.bootstrap_iters)
                excl = lo > 0 or hi < 0
                pairwise[f"{a}-vs-{b}"] = {
                    "mse_diff": point,
                    "ci95": [lo, hi],
                    "ci_excludes_zero": bool(excl),
                }
                print(
                    f"  MSE({a}) - MSE({b}) = {point:+.5f}  ci95 [{lo:+.5f}, {hi:+.5f}]  "
                    f"{'CI excludes 0' if excl else 'CI includes 0'}",
                    flush=True,
                )

    if args.json_out:
        report = {
            "held_out": args.held,
            "n": int(len(positions)),
            "models": models_report,
            "pairwise": pairwise,
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    mse_cli()
