# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""``az-train`` -- fit one generation's value head from self-play dumps.

    az-train --positions gen_0/a.bin,gen_0/b.bin --out weights/gen_1.bin

Fits a value head only (no policy head) by least-squares.

``--game ttt`` (the default) reads tic-tac-toe dumps (``az_train.records``)
and fits either the n-tuple head (``--head ntuple``, 541 weights,
``az_train.ntuple``) or the 19-weight linear head (``--head linear``,
``az_train.model``).

``--game connect4`` reads v2-connect4 dumps (``az_train.records_c4``) and
fits the 6x7 n-tuple head (``az_train.ntuple_c4``, 5590 weights). Only
``--head ntuple`` is defined for connect4.

Writes ``<out>`` (flat ``f32``) and ``<out>.meta.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from az_train.model import fit_value_head as fit_linear
from az_train.model import predict as predict_linear
from az_train.model import write_weights as write_linear
from az_train.ntuple import fit_value_head as fit_ntuple
from az_train.ntuple import predict as predict_ntuple
from az_train.ntuple import write_weights as write_ntuple
from az_train.ntuple_c4 import fit_value_head_with_diagnostics as fit_ntuple_c4_with_diagnostics
from az_train.ntuple_c4 import predict as predict_ntuple_c4
from az_train.ntuple_c4 import write_weights as write_ntuple_c4
from az_train.records import Positions, load_positions
from az_train.records_c4 import Positions as PositionsC4
from az_train.records_c4 import encode_records as encode_records_c4
from az_train.records_c4 import load_positions as load_positions_c4
from az_train.records_c4 import me_opp_planes as me_opp_planes_c4
from az_train.records_c4 import split_by_game as split_c4_by_game

_TTT_HEADS = {
    "linear": (fit_linear, predict_linear, write_linear),
    "ntuple": (fit_ntuple, predict_ntuple, write_ntuple),
}


def _concat_ttt(parts: list[Positions]) -> Positions:
    if len(parts) == 1:
        return parts[0]
    return Positions(
        board=np.concatenate([p.board for p in parts]),
        side=np.concatenate([p.side for p in parts]),
        ply=np.concatenate([p.ply for p in parts]),
        value=np.concatenate([p.value for p in parts]),
        policy=[e for p in parts for e in p.policy],
    )


def _concat_c4(parts: list[PositionsC4]) -> PositionsC4:
    if len(parts) == 1:
        return parts[0]
    return PositionsC4(
        black=np.concatenate([p.black for p in parts]),
        white=np.concatenate([p.white for p in parts]),
        side=np.concatenate([p.side for p in parts]),
        ply=np.concatenate([p.ply for p in parts]),
        value=np.concatenate([p.value for p in parts]),
        policy=[e for p in parts for e in p.policy],
    )


def _fit_ttt(paths: list[str], head: str, l2: float | None) -> tuple[np.ndarray, float, int]:
    fit, predict, _ = _TTT_HEADS[head]
    pos = _concat_ttt([load_positions(p) for p in paths])
    print(f"loaded {len(pos)} positions from {len(paths)} file(s)", flush=True)
    w = fit(pos) if l2 is None else fit(pos, l2=l2)
    mse = float(np.mean((predict(w, pos) - pos.value) ** 2))
    return w, mse, len(pos)


def _pearson(prediction: np.ndarray, value: np.ndarray) -> float:
    """Pearson correlation, with 0.0 as the neutral constant-vector value."""
    if len(prediction) < 2 or np.std(prediction) == 0.0 or np.std(value) == 0.0:
        return 0.0
    return float(np.corrcoef(prediction, value)[0, 1])


def value_metrics(
    prediction: np.ndarray, value: np.ndarray
) -> dict[str, float | int | dict[str, int]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    nonzero = value != 0.0
    return {
        "mse": float(np.mean((prediction - value) ** 2)),
        "zero_mse": float(np.mean(value ** 2)),
        "pearson": _pearson(prediction, value),
        "sign_agreement": float(np.mean(np.sign(prediction[nonzero]) == np.sign(value[nonzero])))
        if np.any(nonzero) else 0.0,
        "mean_absolute_prediction": float(np.mean(np.abs(prediction))),
        "fraction_abs_prediction_gt_0_95": float(np.mean(np.abs(prediction) > 0.95)),
        "labels": {
            "minus_one": int(np.count_nonzero(value == -1.0)),
            "zero": int(np.count_nonzero(value == 0.0)),
            "plus_one": int(np.count_nonzero(value == 1.0)),
        },
        "positions": int(len(value)),
    }


def _fit_c4(
    paths: list[str], head: str, l2: float | None, validation_fraction: float,
    split_seed: int, value_target: str,
) -> tuple[np.ndarray, dict[str, Any], PositionsC4]:
    if head != "ntuple":
        raise SystemExit(f"--game connect4 only supports --head ntuple (got {head})")
    pos = _concat_c4([load_positions_c4(p) for p in paths])
    print(f"loaded {len(pos)} positions from {len(paths)} file(s)", flush=True)
    train, validation, train_games, validation_games = split_c4_by_game(
        pos, validation_fraction, split_seed
    )
    train_me, train_opp = me_opp_planes_c4(train)
    validation_me, validation_opp = me_opp_planes_c4(validation)
    w, fit_diagnostics = fit_ntuple_c4_with_diagnostics(
        train_me, train_opp, train.value, 1e-3 if l2 is None else l2, value_target
    )
    metrics: dict[str, Any] = {
        "train": value_metrics(predict_ntuple_c4(w, train_me, train_opp), train.value),
        "validation": value_metrics(
            predict_ntuple_c4(w, validation_me, validation_opp), validation.value
        ),
        "train_games": train_games,
        "validation_games": validation_games,
        **fit_diagnostics,
    }
    return w, metrics, validation


def train_cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="az-train")
    ap.add_argument("--positions", required=True, help="comma-separated dump .bin files")
    ap.add_argument("--out", required=True, help="output weights.bin path")
    ap.add_argument("--game", choices=("ttt", "connect4"), default="ttt")
    ap.add_argument("--head", choices=("linear", "ntuple"), default="ntuple")
    ap.add_argument("--l2", type=float, default=None,
                    help="ridge penalty (default: the head's own default)")
    ap.add_argument("--validation-fraction", type=float, default=0.2,
                    help="whole-game validation fraction (default: 0.2)")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed for deterministic whole-game split (default: 0)")
    ap.add_argument("--validation-records-out",
                    help="write the exact held-out Connect Four games here")
    ap.add_argument("--value-target", choices=("direct", "atanh"),
                    help="Connect Four pre-tanh target (default: direct)")
    args = ap.parse_args(argv)

    paths = [p.strip() for p in args.positions.split(",") if p.strip()]
    validation: PositionsC4 | None = None
    value_target: str | None = None
    c4_metrics: dict[str, Any] | None = None
    if args.game == "connect4":
        c4_value_target = args.value_target or "direct"
        value_target = c4_value_target
        w, c4_metrics, validation = _fit_c4(
            paths, args.head, args.l2, args.validation_fraction, args.split_seed, c4_value_target
        )
        n_pos = int(c4_metrics["train"]["positions"]) + int(c4_metrics["validation"]["positions"])
        mse = float(c4_metrics["train"]["mse"])
        write_weights = write_ntuple_c4
    else:
        if args.validation_records_out:
            raise SystemExit("--validation-records-out is only supported for --game connect4")
        if args.value_target:
            raise SystemExit("--value-target is currently only supported for --game connect4")
        w, mse, n_pos = _fit_ttt(paths, args.head, args.l2)
        write_weights = _TTT_HEADS[args.head][2]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_weights(str(out), w)
    if args.game == "connect4" and args.validation_records_out:
        assert validation is not None
        validation_path = Path(args.validation_records_out)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_bytes(encode_records_c4(validation))
    meta = {
        "game": args.game,
        "head": args.head,
        "n_weights": int(w.shape[0]),
        "positions": n_pos,
        "sources": paths,
        "l2": args.l2,
        "train_mse": mse,
    }
    if args.game == "connect4":
        assert value_target is not None and c4_metrics is not None
        meta.update({
            "value_target": value_target,
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
            "metrics": c4_metrics,
        })
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} ({w.shape[0]} f32)  train_mse={mse:.4f}", flush=True)


if __name__ == "__main__":
    train_cli()
