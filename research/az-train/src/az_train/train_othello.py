# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""``az-train-othello`` -- fit one Gumbel self-play generation's value and
policy heads for Othello.

    az-train-othello --positions gen0.bin,gen1.bin --model games/othello/ntuple/model.toml \\
        --train-config games/othello/ntuple/train.toml --out local/output/az/othello/run0/gen2

Reads ``RecordV2`` dumps (``az_train.records_othello``, produced by
``game-othello dump --label gumbel``) rather than ``othello-eval``'s v1
outcome-only dumps -- this is the self-play training loop, not the static
bake-off tooling in ``research/othello-eval``, which this leaves unchanged.
The value head's fit method (minibatch logistic regression against the
outcome label) is the same one ``othello_eval.ntuple`` uses; only the data
source and the added policy sidecar are new here.

Writes a self-contained checkpoint directory: ``model.toml`` (copied in),
``weights.bin`` + ``weights.meta.json`` (value head), ``policy.bin`` +
``policy.meta.json`` (policy sidecar) -- exactly the layout
``NTupleModel::from_dir`` / ``NTuplePolicyNet::from_dir`` expect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from az_train import ntuple_othello, policy_othello
from az_train.records_othello import Positions, concat, load_positions, split_by_game


def train_cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="az-train-othello")
    ap.add_argument("--positions", required=True, help="comma-separated RecordV2 dump .bin files")
    ap.add_argument("--model", required=True, help="model.toml geometry")
    ap.add_argument(
        "--train-config", required=True, help="train.toml value-head hyperparameters"
    )
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--policy-l2", type=float, default=1e-4)
    ap.add_argument("--policy-epochs", type=int, default=60)
    ap.add_argument("--policy-batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    paths = [p.strip() for p in args.positions.split(",") if p.strip()]
    parts: list[Positions] = [load_positions(p) for p in paths]
    pos = concat(parts)
    print(f"loaded {len(pos)} positions from {len(paths)} file(s)", flush=True)

    train, validation, train_games, validation_games = split_by_game(
        pos, args.validation_fraction, args.split_seed
    )

    geom = ntuple_othello.load_model_toml(args.model)
    train_cfg = ntuple_othello.TrainConfig.from_toml(args.train_config)

    print("=== value head ===", flush=True)
    train_feat = ntuple_othello.featurize(train, geom)
    y01 = (train.value.astype(np.float64) + 1.0) / 2.0
    w = ntuple_othello.fit(train_feat, y01, train_cfg, geom.n_weights)
    va_feat = ntuple_othello.featurize(validation, geom)
    va_y01 = (validation.value.astype(np.float64) + 1.0) / 2.0
    train_bce, train_acc = ntuple_othello.bce_and_acc(w, train_feat, y01)
    val_bce, val_acc = ntuple_othello.bce_and_acc(w, va_feat, va_y01)
    value_metrics = {
        "train_bce": train_bce,
        "train_accuracy": train_acc,
        "validation_bce": val_bce,
        "validation_accuracy": val_acc,
    }
    print(
        f"  train bce {train_bce:.4f} acc {train_acc:.4f}  "
        f"val bce {val_bce:.4f} acc {val_acc:.4f}",
        flush=True,
    )

    print("=== policy sidecar ===", flush=True)
    has_train_targets = all(entries for entries in train.policy)
    has_validation_targets = all(entries for entries in validation.policy)
    if has_train_targets and has_validation_targets:
        policy_w, policy_metrics = policy_othello.fit(
            train,
            geom,
            validation,
            l2=args.policy_l2,
            seed=args.split_seed,
            epochs=args.policy_epochs,
            batch_size=args.policy_batch_size,
        )
        print(
            f"  train ce {policy_metrics['train']['cross_entropy']:.4f} "
            f"val ce {policy_metrics['validation']['cross_entropy']:.4f} "
            f"(uniform {policy_metrics['validation']['uniform_baseline_cross_entropy']:.4f})",
            flush=True,
        )
    else:
        policy_w = np.zeros(geom.n_weights * policy_othello.SQUARES, dtype=np.float32)
        policy_metrics = {"status": "no_policy_targets"}

    out = Path(args.out)
    meta = {
        "model_toml_sha256": geom.sha256_hex,
        "n_weights": int(geom.n_weights),
        "train": {
            "positions": int(len(pos)),
            "train_games": train_games,
            "validation_games": validation_games,
            "epochs": train_cfg.epochs,
            "lr": train_cfg.lr,
            "l2": train_cfg.l2,
            "sources": paths,
        },
        "metrics": value_metrics,
    }
    ntuple_othello.write_weights(out, w, args.model, meta)

    policy_meta = {
        "model_toml_sha256": geom.sha256_hex,
        "n_weights": int(geom.n_weights),
        "metrics": policy_metrics,
    }
    policy_othello.write_weights(str(out), policy_w, geom.n_weights, policy_meta)

    print(f"wrote {out}/{{weights,policy}}.bin  n_weights={geom.n_weights}", flush=True)


if __name__ == "__main__":
    train_cli()
