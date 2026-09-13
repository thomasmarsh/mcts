# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""N-tuple value head trainer for the Othello Gumbel self-play loop.

Geometry (tuple square lists, weight-table layout) comes from ``model.toml``,
identical to ``games/othello/src/ntuple.rs`` and its static-fit counterpart
``othello_eval.ntuple`` -- this module duplicates that geometry code rather
than importing across packages (``az_train`` and ``othello-eval`` are
independent ``uv`` projects, matching the existing ``_c4``-suffixed
modules' self-contained convention) but keeps the same fit method
(minibatch logistic regression against the win/draw/loss outcome, BCE loss)
``othello_eval.ntuple`` uses for this model class. This module's only real
difference is its data source: ``research/az-train``'s ``RecordV2``
self-play dumps via ``az_train.records_othello``, carrying a policy tail,
rather than ``othello_eval.records``'s outcome-only v1 dumps.
"""

from __future__ import annotations

import hashlib
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from az_train.records_othello import Positions, me_opp_bits

BOARD = 8
SQUARES = 64


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

    table = np.zeros((8, SQUARES), dtype=np.int64)
    for i in range(SQUARES):
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

    @property
    def n_tuples(self) -> int:
        return len(self.tuple_syms)


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
        if squares.min() < 0 or squares.max() >= SQUARES:
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


def featurize(pos: Positions, geom: ModelGeometry) -> np.ndarray:
    """``(N, geom.n_features)`` int64 global weight indices, one column per
    (tuple, orientation) pair, tuple-major then orientation -- the exact
    layout ``ModelGeometry::feature_indices`` (Rust) and
    ``othello_eval.ntuple.featurize`` produce."""
    me, opp = me_opp_bits(pos)
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
    w = np.zeros(n_weights, dtype=np.float64)
    print(
        f"fit: {n} positions, {feat_idx.shape[1]} features/pos, {n_weights} weights",
        flush=True,
    )
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(n)
        for b0 in range(0, n, cfg.batch):
            idx = order[b0 : b0 + cfg.batch]
            bf, by = feat_idx[idx], y01[idx]
            logit = w[bf].sum(axis=1)
            resid = (_sigmoid(logit) - by) / len(by)  # (B,)
            grad = np.zeros_like(w)
            np.add.at(grad, bf, resid[:, None])
            grad += 2.0 * cfg.l2 * w
            w -= cfg.lr * grad
        if epoch % cfg.report_every == 0 or epoch == cfg.epochs:
            tb, ta = bce_and_acc(w, feat_idx, y01)
            elapsed = time.time() - start
            print(f"  epoch {epoch:4d}  bce {tb:.4f} acc {ta:.4f}  ({elapsed:.1f}s)", flush=True)
    return w.astype(np.float32)


def write_weights(out_dir: str | Path, w: np.ndarray, model_toml: str | Path, meta: dict) -> None:
    """Write ``<out_dir>/weights.bin`` + ``weights.meta.json`` and copy in
    ``model.toml``, so ``<out_dir>`` is a self-contained checkpoint directory
    matching ``NTupleModel::from_dir``'s expected layout."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w.astype("<f4").tofile(out / "weights.bin")
    (out / "weights.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "model.toml").write_bytes(Path(model_toml).read_bytes())
