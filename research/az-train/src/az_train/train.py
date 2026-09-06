# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""``az-train`` -- fit one generation's value head from self-play dumps.

    az-train --positions gen_0/a.bin,gen_0/b.bin --out weights/gen_1.bin

Fits the linear value head only (``az_train.model``) by least-squares; no
policy head. Writes ``<out>`` (flat ``f32``) and ``<out>.meta.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from az_train.model import fit_value_head, predict, write_weights
from az_train.records import Positions, load_positions


def _concat(parts: list[Positions]) -> Positions:
    if len(parts) == 1:
        return parts[0]
    return Positions(
        board=np.concatenate([p.board for p in parts]),
        side=np.concatenate([p.side for p in parts]),
        ply=np.concatenate([p.ply for p in parts]),
        value=np.concatenate([p.value for p in parts]),
        policy=[e for p in parts for e in p.policy],
    )


def train_cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="az-train")
    ap.add_argument("--positions", required=True, help="comma-separated dump .bin files")
    ap.add_argument("--out", required=True, help="output weights.bin path")
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args(argv)

    paths = [p.strip() for p in args.positions.split(",") if p.strip()]
    pos = _concat([load_positions(p) for p in paths])
    print(f"loaded {len(pos)} positions from {len(paths)} file(s)", flush=True)

    w = fit_value_head(pos, l2=args.l2)
    pred = predict(w, pos)
    mse = float(np.mean((pred - pos.value) ** 2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_weights(str(out), w)
    meta = {
        "n_weights": int(w.shape[0]),
        "positions": int(len(pos)),
        "sources": paths,
        "l2": args.l2,
        "train_mse": mse,
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} ({w.shape[0]} f32)  train_mse={mse:.4f}", flush=True)


if __name__ == "__main__":
    train_cli()
