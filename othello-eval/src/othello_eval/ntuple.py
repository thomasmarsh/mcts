"""Numpy n-tuple evaluator trainer for Othello.

An n-tuple is an ordered list of board squares. A position maps each tuple
to one base-3 *feature index* (digit 0 empty, 1 side-to-move disc, 2
opponent disc, least-significant digit first), and each tuple owns a table
of ``3 ** k`` weights. The score is the plain sum of the selected weights
over every tuple and all 8 D4 board orientations, which share one table.

Geometry comes from ``model.toml`` (the same file the Rust evaluator in
``games/othello/src/ntuple.rs`` reads); its SHA-256 is written alongside the
trained weights so a geometry/weights skew is caught on load.

Train from the command line::

    othello-eval-train --positions a.bin,b.bin --model games/othello/ntuple/model.toml \\
        --config games/othello/ntuple/train.toml --out /tmp/oth-ntuple
"""

# numpy's strict-mode stubs widen most array-returning calls (`np.where`,
# `Generator.permutation`, `np.add.at`, `np.concatenate`, ...) to Unknown,
# which then propagates through this numeric module. This is a stub gap, not
# a real type-safety issue, so it's suppressed here rather than loosening
# the project's strict mode (same call the tuner's ConfigSpace bridge makes).
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from othello_eval.records import load_positions

BOARD = 8


def d4_index_table() -> np.ndarray:
    """``(8, 64)`` table: entry ``[k, i]`` is the image of square ``i`` under
    D4 group element ``k`` (``k == 0`` identity), matching
    ``game_core::symmetry::D4Symmetry::index_symmetries``' element order."""

    def transpose(j: int) -> int:
        r, c = divmod(j, BOARD)
        return c * BOARD + r

    def flip_cols(j: int) -> int:
        r, c = divmod(j, BOARD)
        return r * BOARD + (BOARD - 1 - c)

    def flip_rows(j: int) -> int:
        r, c = divmod(j, BOARD)
        return (BOARD - 1 - r) * BOARD + c

    table = np.zeros((8, 64), dtype=np.int64)
    for i in range(64):
        fc, fr = flip_cols(i), flip_rows(i)
        table[:, i] = [
            i,
            fc,
            fr,
            transpose(i),
            flip_rows(fc),
            transpose(fc),
            transpose(fr),
            transpose(flip_rows(fc)),
        ]
    return table


D4 = d4_index_table()


@dataclass
class ModelGeometry:
    """Parsed tuple geometry."""

    # Per tuple: an ``(8, k)`` int array of the tuple's square list under
    # each of the 8 D4 orientations.
    tuple_syms: list[np.ndarray]
    offsets: np.ndarray  # (n_tuples,) start of each tuple's table
    n_weights: int
    sha256_hex: str
    names: list[str]

    @property
    def n_features(self) -> int:
        """Columns produced by :func:`featurize` -- one per (tuple, orientation)."""
        return len(self.tuple_syms) * 8


def load_model_toml(path: str | Path) -> ModelGeometry:
    raw = Path(path).read_bytes()
    doc = tomllib.loads(raw.decode("utf-8"))
    tuples = doc.get("tuple", [])
    if not tuples:
        raise ValueError(f"{path}: no [[tuple]] entries")

    tuple_syms: list[np.ndarray] = []
    names: list[str] = []
    offsets: list[int] = []
    off = 0
    for t in tuples:
        squares = np.asarray(t["squares"], dtype=np.int64)
        if squares.ndim != 1 or squares.size == 0 or squares.size > 12:
            raise ValueError(f"tuple {t.get('name')!r}: 1..=12 squares required")
        if squares.min() < 0 or squares.max() >= 64:
            raise ValueError(f"tuple {t.get('name')!r}: square out of range")
        tuple_syms.append(D4[:, squares])  # (8, k)
        names.append(str(t.get("name", f"tuple{len(names)}")))
        offsets.append(off)
        off += 3 ** int(squares.size)

    return ModelGeometry(
        tuple_syms=tuple_syms,
        offsets=np.asarray(offsets, dtype=np.int64),
        n_weights=off,
        sha256_hex=hashlib.sha256(raw).hexdigest(),
        names=names,
    )


def _me_opp(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    black = positions["black"].astype(np.uint64)
    white = positions["white"].astype(np.uint64)
    side0 = positions["side"] == 0
    me = np.where(side0, black, white)
    opp = np.where(side0, white, black)
    return me, opp


def featurize(positions: np.ndarray, geom: ModelGeometry) -> np.ndarray:
    """``(N, geom.n_features)`` int64 array of global weight indices, one
    column per (tuple, orientation) pair, tuple-major then orientation."""
    me, opp = _me_opp(positions)
    cols: list[np.ndarray] = []
    for ti, syms in enumerate(geom.tuple_syms):
        k = syms.shape[1]
        place = (3 ** np.arange(k, dtype=np.int64))[None, :]
        for s in range(8):
            sq = syms[s].astype(np.uint64)  # (k,)
            mbit = ((me[:, None] >> sq[None, :]) & np.uint64(1)).astype(np.int64)
            obit = ((opp[:, None] >> sq[None, :]) & np.uint64(1)).astype(np.int64)
            trit = mbit + 2 * obit  # (N, k)
            feat = (trit * place).sum(axis=1)  # (N,)
            cols.append(geom.offsets[ti] + feat)
    return np.stack(cols, axis=1).astype(np.int64)  # (N, F)


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 0.05
    l2: float = 1e-6
    batch: int = 262144
    val_fraction: float = 0.05
    seed: int = 0
    report_every: int = 20

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainConfig:
        doc = tomllib.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in doc.items() if k in known})


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def bce_and_acc(w: np.ndarray, feat: np.ndarray, y01: np.ndarray) -> tuple[float, float]:
    logit = w[feat].sum(axis=1)
    p = _sigmoid(logit)
    eps = 1e-7
    bce = float(-np.mean(y01 * np.log(p + eps) + (1 - y01) * np.log(1 - p + eps)))
    # Accuracy ignores exact draws (y01 == 0.5).
    decisive = y01 != 0.5
    if decisive.any():
        acc = float(np.mean((p[decisive] > 0.5) == (y01[decisive] > 0.5)))
    else:
        acc = float("nan")
    return bce, acc


def fit(feat_idx: np.ndarray, y01: np.ndarray, cfg: TrainConfig, n_weights: int) -> np.ndarray:
    """Full logistic regression by minibatch gradient descent with L2.

    ``feat_idx`` is ``(N, F)`` global indices, ``y01`` is ``(N,)`` in
    ``{0, 0.5, 1}``. Returns the flat ``float32`` weight vector.
    """
    rng = np.random.default_rng(cfg.seed)
    n = len(y01)
    perm = rng.permutation(n)
    feat_idx = feat_idx[perm]
    y01 = y01[perm].astype(np.float64)

    n_val = int(round(n * cfg.val_fraction))
    val_f, val_y = feat_idx[:n_val], y01[:n_val]
    tr_f, tr_y = feat_idx[n_val:], y01[n_val:]
    n_tr = len(tr_y)

    w = np.zeros(n_weights, dtype=np.float64)
    print(
        f"fit: {n_tr} train / {n_val} val positions, {feat_idx.shape[1]} features/pos, "
        f"{n_weights} weights",
        flush=True,
    )
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(n_tr)
        for b0 in range(0, n_tr, cfg.batch):
            idx = order[b0 : b0 + cfg.batch]
            bf, by = tr_f[idx], tr_y[idx]
            logit = w[bf].sum(axis=1)
            resid = (_sigmoid(logit) - by) / len(by)  # (B,)
            grad = np.zeros_like(w)
            np.add.at(grad, bf, resid[:, None])
            grad += 2.0 * cfg.l2 * w
            w -= cfg.lr * grad
        if epoch % cfg.report_every == 0 or epoch == cfg.epochs:
            tb, ta = bce_and_acc(w, tr_f, tr_y)
            vb, va = bce_and_acc(w, val_f, val_y) if n_val else (float("nan"), float("nan"))
            print(
                f"  epoch {epoch:4d}  train bce {tb:.4f} acc {ta:.4f}  "
                f"val bce {vb:.4f} acc {va:.4f}  ({time.time() - start:.1f}s)",
                flush=True,
            )
    return w.astype(np.float32)


def write_weights(out_dir: str | Path, w: np.ndarray, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w.astype("<f4").tofile(out / "weights.bin")
    (out / "weights.meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _val_accuracy(w: np.ndarray, feat_idx: np.ndarray, y01: np.ndarray, cfg: TrainConfig) -> float:
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(y01))
    n_val = int(round(len(y01) * cfg.val_fraction))
    if n_val == 0:
        return float("nan")
    vf, vy = feat_idx[perm][:n_val], y01[perm][:n_val]
    return bce_and_acc(w, vf, vy)[1]


def train_cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="othello-eval-train")
    ap.add_argument("--positions", required=True, help="comma-separated dump .bin files")
    ap.add_argument("--model", required=True, help="model.toml geometry")
    ap.add_argument("--config", required=True, help="train.toml hyperparameters")
    ap.add_argument("--out", required=True, help="output directory for weights.bin")
    args = ap.parse_args(argv)

    geom = load_model_toml(args.model)
    cfg = TrainConfig.from_toml(args.config)

    paths = [p.strip() for p in args.positions.split(",") if p.strip()]
    parts = [load_positions(p) for p in paths]
    positions = np.concatenate(parts) if len(parts) > 1 else parts[0]
    print(f"loaded {len(positions)} positions from {len(paths)} file(s)", flush=True)

    y01 = (positions["target"].astype(np.float64) + 1.0) / 2.0
    feat_idx = featurize(positions, geom)
    w = fit(feat_idx, y01, cfg, geom.n_weights)

    val_acc = _val_accuracy(w, feat_idx, y01, cfg)
    final_bce = bce_and_acc(w, feat_idx, y01)[0]
    meta = {
        "model_toml_sha256": geom.sha256_hex,
        "n_weights": int(geom.n_weights),
        "train": {
            "positions": int(len(positions)),
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "l2": cfg.l2,
            "final_loss": final_bce,
            "val_accuracy": val_acc,
            "sources": paths,
        },
    }
    write_weights(args.out, w, meta)
    # Copy the geometry in so $OTHELLO_NTUPLE_WEIGHTS is a self-contained
    # directory (model.toml + weights.bin + weights.meta.json), as the Rust
    # loader expects.
    (Path(args.out) / "model.toml").write_bytes(Path(args.model).read_bytes())
    print(f"wrote {args.out}/weights.bin ({geom.n_weights} f32)  val_acc={val_acc:.4f}", flush=True)


if __name__ == "__main__":
    train_cli()
