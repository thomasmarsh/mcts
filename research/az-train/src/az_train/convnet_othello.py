# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false, reportMissingTypeStubs=false
"""``az-train-othello-cnn`` -- fit one Gumbel self-play generation's OTCNN001
value+policy head for Othello.

    az-train-othello-cnn --positions gen0.bin,gen1.bin \\
        --out local/output/az/othello-cnn/run0/gen2.bin

Reads ``RecordV2`` dumps (``az_train.records_othello``, produced by
``game-othello dump --label gumbel --head cnn``) the same way
``az_train.train_othello`` does for the n-tuple head. Unlike the n-tuple
head, this module **imports** its model class
(``othello_eval.convnet.OTCNN001``) rather than reimplementing it, breaking
with the established ``_c4``/``_othello`` per-package-duplication convention
(``ntuple_othello.py``/``policy_othello.py`` both reimplement rather than
import their ``othello_eval`` counterparts). That convention's own
justification was cheapness: the n-tuple fit is a ~30-line ridge-style
minibatch logistic regression, cheap enough that two independent copies cost
little and buy package independence. The CNN's forward/backward pass is the
opposite case -- a ~300-line hand-derived convolutional backward pass
(stem, two residual blocks, split value/policy heads, D4-PASS-as-mean
gradient routing), already built and finite-difference-verified in
``othello_eval.convnet`` (Sessions 4.8/4.9). Re-deriving and re-verifying
that by hand a second time in ``az_train`` would not buy independence worth
having -- it would create two numerically-delicate backward passes that can
silently drift apart, exactly the risk hand-derived-gradient code is most
exposed to. So this module adds an ``othello-eval`` path dependency
(``pyproject.toml``'s ``[tool.uv.sources]``) instead, and only supplies what
``othello_eval.convnet`` does not already have: a ``RecordV2`` reader (this
package's own ``records_othello``, which ``othello_eval`` doesn't need for
its static-corpus role) and the checkpoint-directory/CLI glue.

Checkpoint layout: unlike the n-tuple head, the CNN's architecture is fixed
code, not data (no ``model.toml`` geometry to carry), so there is no
checkpoint *directory* -- ``--out`` is a single ``OTCNN001``-format file
(``othello_eval.convnet.write_weights``'s exact byte layout, the same one
``games/othello/src/convnet.rs::CnnValueNet::load`` reads directly with no
conversion) plus a ``<out>.meta.json`` sidecar carrying training metadata --
a single file per generation rather than a checkpoint directory, since there
is no separate geometry to carry alongside the weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from othello_eval import convnet

from az_train import policy_othello
from az_train.records_othello import Positions, concat, load_positions, me_opp_bits, split_by_game

SQUARES = 64


def me_opp_planes(pos: Positions) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 64)`` float32 0/1 occupancy planes from a decoded ``Positions``,
    the ``RecordV2``-sourced analogue of ``othello_eval.convnet.me_opp_planes``
    (which reads a structured v1-record array instead)."""
    me_bits, opp_bits = me_opp_bits(pos)
    squares = np.arange(SQUARES, dtype=np.uint64)
    me = ((me_bits[:, None] >> squares[None, :]) & np.uint64(1)).astype(np.float32)
    opp = ((opp_bits[:, None] >> squares[None, :]) & np.uint64(1)).astype(np.float32)
    return me, opp


def train_cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="az-train-othello-cnn")
    ap.add_argument("--positions", required=True, help="comma-separated RecordV2 dump .bin files")
    ap.add_argument("--out", required=True, help="output OTCNN001 checkpoint file")
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--learning-rate", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--validate-every", type=int, default=1)
    ap.add_argument("--report-every", type=int, default=1)
    args = ap.parse_args(argv)

    paths = [p.strip() for p in args.positions.split(",") if p.strip()]
    parts: list[Positions] = [load_positions(p) for p in paths]
    pos = concat(parts)
    print(f"loaded {len(pos)} positions from {len(paths)} file(s)", flush=True)

    train, validation, train_games, validation_games = split_by_game(
        pos, args.validation_fraction, args.split_seed
    )

    has_train_targets = all(entries for entries in train.policy)
    has_validation_targets = all(entries for entries in validation.policy)
    if not (has_train_targets and has_validation_targets):
        raise ValueError(
            "az-train-othello-cnn requires a completed-Q policy target on every "
            "position -- every record must come from `dump --label gumbel`"
        )

    train_me, train_opp = me_opp_planes(train)
    va_me, va_opp = me_opp_planes(validation)
    train_policy, train_legal = policy_othello.targets(train.policy)
    va_policy, va_legal = policy_othello.targets(validation.policy)

    print(
        f"=== OTCNN001 fit: {len(train)} train / {len(validation)} validation positions ===",
        flush=True,
    )
    weights, metadata = convnet.fit(
        train_me,
        train_opp,
        train.value.astype(np.float64),
        (va_me, va_opp, validation.value.astype(np.float64)),
        l2=args.l2,
        seed=args.split_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validate_every=args.validate_every,
        report_every=args.report_every,
        policy=train_policy,
        legal=train_legal,
        validation_policy=va_policy,
        validation_legal=va_legal,
    )
    final: dict[str, float] = metadata["final_validation_metrics"]  # pyright: ignore[reportAssignmentType]
    print(
        f"  final: value mse {final['value_mse']:.4f} pearson {final['value_pearson']:.4f} "
        f"sign-acc {final['value_sign_agreement']:.4f} policy ce "
        f"{final.get('masked_policy_cross_entropy', float('nan')):.4f}",
        flush=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    convnet.write_weights(str(out), weights)

    meta = {
        "model": convnet.MAGIC.decode(), "version": convnet.VERSION, "n_weights": convnet.N_WEIGHTS,
        "train": {
            "positions": int(len(pos)), "train_games": train_games,
            "validation_games": validation_games, "sources": paths,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "l2": args.l2,
        },
        "metrics": metadata,
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} ({convnet.N_WEIGHTS} weights) + {out.name}.meta.json", flush=True)


if __name__ == "__main__":
    train_cli()
