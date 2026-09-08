# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
# ruff: noqa: E501
"""Isolate whether mandatory mirror-averaging blocks the compact Connect Four value head.

``convnet_c4.predict`` averages the literal and column-mirrored board for both
outputs.  Connect Four value is genuinely left-right mirror invariant, so this is
sound in principle.  This module fits the head as today and then scores its value
predictions three ways -- literal orientation only, the mirror-averaged baseline,
and mirror-averaged with the (identity) mirror map applied to the target -- on the
balanced replay fitability subset and on the frozen strong-reference corpus train
and validation splits.  It optionally fits and scores a structurally
mirror-equivariant value head as the principled fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from az_train.convnet_c4 import (
    _mirror_planes,
    _pearson,
    _predict_literal,
    _predict_value_equivariant,
    fit_equivariant_value_policy,
    fit_value_policy_with_diagnostics,
    write_weights,
)
from az_train.fitability_c4 import _concat, _dense_policy, select_balanced_unique_rows
from az_train.records_c4 import Positions, load_positions, me_opp_planes

_REF_MAGIC = b"C4REFD01"
_REF_VERSION = 1
_REF_HEADER_BYTES = 23
_REF_RECORD_BYTES = 30
_REF_RECORD = struct.Struct("<QQBBIBBBBf")
# label ordering matches games/connect4/src/reference_diagnostic.rs ReferenceLabel
_LABEL_SIGN = (1.0, -1.0, 0.0, 1.0, -1.0, float("nan"))
_LABEL_EXACT = (True, True, True, False, False, False)


@dataclass(frozen=True)
class ReferenceCorpus:
    positions: Positions
    split: np.ndarray  # (N,) uint8, 0 = train, 1 = validation
    label_sign: np.ndarray  # (N,) float32, proven value, NaN when unresolved
    exact: np.ndarray  # (N,) bool
    group: np.ndarray  # (N,) uint32, frozen source-game group id
    source_outcome: np.ndarray  # (N,) float32, source-game outcome, side-to-move perspective


def read_reference_corpus(path: str | Path) -> ReferenceCorpus:
    """Decode a ``C4REFD01`` strong-reference diagnostic artifact."""
    raw = Path(path).read_bytes()
    if len(raw) < _REF_HEADER_BYTES or raw[:8] != _REF_MAGIC:
        raise ValueError(f"{path}: not a C4REFD01 reference corpus")
    version, count = struct.unpack_from("<II", raw, 8)
    if version != _REF_VERSION or raw[16] != 0xFF:
        raise ValueError(f"{path}: unsupported C4REFD01 header")
    if len(raw) != _REF_HEADER_BYTES + count * _REF_RECORD_BYTES:
        raise ValueError(f"{path}: length does not match {count} records")
    black, white, side, ply, split = [], [], [], [], []
    label_sign, exact, group, source_outcome = [], [], [], []
    for i in range(count):
        off = _REF_HEADER_BYTES + i * _REF_RECORD_BYTES
        b, w, s, p, grp, sp, label, _pd, _md, outcome = _REF_RECORD.unpack_from(raw, off)
        if sp > 1 or label > 5:
            raise ValueError(f"{path}: record {i} has an invalid split or label")
        black.append(b)
        white.append(w)
        side.append(s)
        ply.append(p)
        split.append(sp)
        label_sign.append(_LABEL_SIGN[label])
        exact.append(_LABEL_EXACT[label])
        group.append(grp)
        source_outcome.append(outcome)
    positions = Positions(
        black=np.asarray(black, dtype=np.uint64),
        white=np.asarray(white, dtype=np.uint64),
        side=np.asarray(side, dtype=np.uint8),
        ply=np.asarray(ply, dtype=np.uint8),
        value=np.asarray(label_sign, dtype=np.float32),
        policy=[[] for _ in range(count)],
    )
    return ReferenceCorpus(
        positions=positions,
        split=np.asarray(split, dtype=np.uint8),
        label_sign=np.asarray(label_sign, dtype=np.float32),
        exact=np.asarray(exact, dtype=bool),
        group=np.asarray(group, dtype=np.uint32),
        source_outcome=np.asarray(source_outcome, dtype=np.float32),
    )


def _value_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """MSE, Pearson, plain and class-balanced sign accuracy over proven labels."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    nonzero = target != 0.0
    signs = np.sign(prediction[nonzero]) == np.sign(target[nonzero])
    classes = [
        signs[np.sign(target[nonzero]) == side]
        for side in (-1.0, 1.0)
    ]
    present = [float(np.mean(group)) for group in classes if group.size]
    return {
        "count": int(prediction.size),
        "value_mse": float(np.mean((prediction - target) ** 2)) if prediction.size else 0.0,
        "value_pearson": _pearson(prediction, target),
        "sign_agreement": float(np.mean(signs)) if signs.size else 0.0,
        "balanced_sign_accuracy": float(np.mean(present)) if present else 0.0,
    }


def _mirror_value_target(target: np.ndarray) -> np.ndarray:
    """The left-right reflection of a Connect Four value target -- the identity.

    The board's value does not change under a column flip, so a harness that
    mirrors the board must leave the target alone.  This makes variant (c) equal
    to the mirror-averaged baseline by construction and isolates any residual gap
    to the learned head rather than a missing target transform.
    """
    return np.asarray(target, dtype=np.float64)


def evaluate_value_orientations(
    weights: np.ndarray, me: np.ndarray, opp: np.ndarray, target: np.ndarray,
) -> dict[str, object]:
    """Score the fitted value head literal-only, mirror-averaged, and target-mirrored."""
    literal_value, _ = _predict_literal(weights, me, opp)
    mm, om = _mirror_planes(me, opp)
    reflected_value, _ = _predict_literal(weights, mm, om)
    averaged = 0.5 * (literal_value + reflected_value)
    return {
        "literal": _value_metrics(literal_value, target),
        "mirror_averaged": _value_metrics(averaged, target),
        "mirror_averaged_target_mirrored": _value_metrics(averaged, _mirror_value_target(target)),
        "literal_vs_reflected_remapped_value_pearson": _pearson(literal_value, reflected_value),
        "equivariant": _value_metrics(_predict_value_equivariant(weights, me, opp), target),
    }


def _reference_planes(
    corpus: ReferenceCorpus, split: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return proven-label ``me``/``opp`` planes and value targets for one split."""
    keep = (corpus.split == split) & np.isfinite(corpus.label_sign)
    pos = corpus.positions
    subset = Positions(
        black=pos.black[keep], white=pos.white[keep], side=pos.side[keep],
        ply=pos.ply[keep], value=pos.value[keep], policy=[],
    )
    me, opp = me_opp_planes(subset)
    return me, opp, corpus.label_sign[keep].astype(np.float64)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.mirror_diagnostic_c4")
    parser.add_argument("--positions", required=True, help="comma-separated v2-connect4 replay files")
    parser.add_argument("--reference-corpus", required=True, help="frozen C4REFD01 artifact")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-outcome", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--fit-scope", choices=("subset", "full"), default="subset",
        help="fit the balanced 48-row memorization subset or every legal replay row",
    )
    parser.add_argument("--equivariant", action="store_true", help="also fit and score the equivariant value head")
    args = parser.parse_args(argv)

    input_paths = [Path(path) for path in args.positions.split(",")]
    pos = _concat([load_positions(path) for path in input_paths])
    if args.fit_scope == "full":
        rows = np.flatnonzero(np.array([len(entry) > 0 for entry in pos.policy]))
    else:
        rows = select_balanced_unique_rows(pos, args.per_outcome, args.seed)
    subset = Positions(
        black=pos.black[rows], white=pos.white[rows], side=pos.side[rows],
        ply=pos.ply[rows], value=pos.value[rows],
        policy=[pos.policy[int(row)] for row in rows],
    )
    me, opp = me_opp_planes(subset)
    value = pos.value[rows].astype(np.float64)
    policy, legal = _dense_policy([pos.policy[int(row)] for row in rows])

    corpus = read_reference_corpus(args.reference_corpus)
    ref_train = _reference_planes(corpus, 0)
    ref_val = _reference_planes(corpus, 1)

    weights, fit_metadata = fit_value_policy_with_diagnostics(
        me.astype(np.float32), opp.astype(np.float32), value.astype(np.float32),
        policy, legal, (me.astype(np.float32), opp.astype(np.float32), value.astype(np.float32), policy, legal),
        seed=args.seed, epochs=args.epochs,
    )

    corpora = {
        "fitability_subset": evaluate_value_orientations(weights, me, opp, value),
        "reference_train": evaluate_value_orientations(weights, *ref_train),
        "reference_validation": evaluate_value_orientations(weights, *ref_val),
    }

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = output / "baseline.c4cnn"
    write_weights(str(baseline_path), weights)

    artifacts: dict[str, str] = {
        "baseline_weights": baseline_path.name,
        "baseline_weights_sha256": _sha256(baseline_path),
    }
    equivariant_value: dict[str, dict[str, float]] = {}
    result: dict[str, object] = {
        "inputs": [{"path": str(p), "sha256": _sha256(p)} for p in input_paths],
        "reference_corpus": {"path": str(args.reference_corpus), "sha256": _sha256(Path(args.reference_corpus))},
        "selection": {"seed": args.seed, "per_outcome": args.per_outcome, "epochs": args.epochs,
                      "fit_scope": args.fit_scope, "fit_row_count": int(rows.size),
                      "source_rows": rows.tolist()},
        "reference_split_proven_counts": {
            "train": int(ref_train[2].size), "validation": int(ref_val[2].size),
        },
        "baseline_fit": fit_metadata,
        "baseline_value_orientations": corpora,
        "artifacts": artifacts,
    }

    if args.equivariant:
        eq_weights, eq_metadata = fit_equivariant_value_policy(
            me.astype(np.float32), opp.astype(np.float32), value.astype(np.float32),
            policy, legal, (me.astype(np.float32), opp.astype(np.float32), value.astype(np.float32), policy, legal),
            seed=args.seed, epochs=args.epochs,
        )
        eq_path = output / "equivariant.c4cnn"
        write_weights(str(eq_path), eq_weights)
        artifacts["equivariant_weights"] = eq_path.name
        artifacts["equivariant_weights_sha256"] = _sha256(eq_path)
        equivariant_value = {
            "fitability_subset": _value_metrics(_predict_value_equivariant(eq_weights, me, opp), value),
            "reference_train": _value_metrics(_predict_value_equivariant(eq_weights, ref_train[0], ref_train[1]), ref_train[2]),
            "reference_validation": _value_metrics(_predict_value_equivariant(eq_weights, ref_val[0], ref_val[1]), ref_val[2]),
        }
        result["equivariant_fit"] = eq_metadata
        result["equivariant_value"] = equivariant_value

    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary: dict[str, object] = {"baseline_value_orientations": corpora}
    if args.equivariant:
        summary["equivariant_value"] = equivariant_value
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
