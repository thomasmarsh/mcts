"""Dead-seed tracing for the production Othello CNN fit.

At the production geometry and batch size, many seeded inits leave the
network dead (constant outputs, validation pearson exactly 0.0) from the very
first updates. This tool fits many seeds on a real shard, probing validation
pearson at a grid of early optimizer steps, and writes one JSON line per seed
so a :class:`az_train.convnet_othello_torch.StallCheck` (step, threshold) can
be chosen from data: the earliest step and the threshold that separate dead
seeds from live ones with no false stalls.

    az-train-othello-dead-seeds --positions shard.bin --seeds 0-31 --epochs 2 \\
        --out traces.jsonl

By default fits run to completion (no stall check) so a seed that looks live
early but dies later is visible. With ``--stall-check-step`` a dead seed is
abandoned at that step (recorded ``stalled_at``), which makes measuring a
dead-seed rate over many seeds cheap: only live seeds pay a full run.
``lr_decay`` is off because a short trace never reaches the production
schedule's decay half.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from az_train.convnet_othello_torch import (
    StallCheck,
    TrainingStalledError,
    fit_torch,
    prepare_fit_data,
)

#: Roughly geometric grid of optimizer steps to probe: dense early (where a
#: dead net dies), sparse later (to catch a seed that dies late).
DEFAULT_TRACE_STEPS = (
    1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
)

#: A run whose final full-validation pearson reaches this is labeled live.
LIVE_PEARSON = 0.2


def parse_seeds(spec: str) -> list[int]:
    """``"0-31"`` -> 0..31 inclusive; ``"3,7,9"`` -> those seeds; both may mix."""
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="az-train-othello-dead-seeds")
    ap.add_argument("--positions", required=True, help="comma-separated RecordV2 dump .bin files")
    ap.add_argument("--seeds", default="0-15")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--init", default="fixed_normal")
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--grad-clip-norm", type=float, default=None)
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stall-check-step", type=int, default=None)
    ap.add_argument("--stall-check-min-pearson", type=float, default=0.05)
    ap.add_argument("--out", required=True, help="JSONL, one line per seed (appended)")
    args = ap.parse_args(argv)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    paths = [p.strip() for p in args.positions.split(",") if p.strip()]
    data = prepare_fit_data(paths, args.validation_fraction, args.split_seed)
    config = {
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "init": args.init,
        "warmup_epochs": args.warmup_epochs, "grad_clip_norm": args.grad_clip_norm,
        "epochs": args.epochs, "train_positions": len(data.train),
        "stall_check_step": args.stall_check_step,
    }
    stall_check = (
        None if args.stall_check_step is None
        else StallCheck(args.stall_check_step, args.stall_check_min_pearson)
    )
    print(f"config {config} device={device}", flush=True)

    live = 0
    seeds = parse_seeds(args.seeds)
    with open(args.out, "a") as out:
        for seed in seeds:
            started = time.perf_counter()
            try:
                _, meta = fit_torch(
                    data.train_me, data.train_opp, data.train.value.astype(np.float64),
                    (data.va_me, data.va_opp, data.validation.value.astype(np.float64)),
                    l2=args.l2, seed=seed, batch_size=args.batch_size, epochs=args.epochs,
                    learning_rate=args.learning_rate, device=device, lr_decay=False,
                    init=args.init, grad_clip_norm=args.grad_clip_norm,
                    warmup_epochs=args.warmup_epochs, trace_steps=DEFAULT_TRACE_STEPS,
                    stall_check=stall_check,
                    policy=data.train_policy, legal=data.train_legal,
                    validation_policy=data.va_policy, validation_legal=data.va_legal,
                )
            except TrainingStalledError as e:
                is_live = False
                stalled_at: int | None = e.step
                epochs: list[float] = []
                step_trace: list[dict[str, float]] = []
            else:
                epochs = [m["value_pearson"] for m in meta["validation_epoch_trace"]]
                is_live = epochs[-1] >= LIVE_PEARSON
                stalled_at = None
                step_trace = meta["step_trace"]
            live += is_live
            line = {
                "seed": seed, "config": config, "live": bool(is_live), "stalled_at": stalled_at,
                "epoch_pearson": epochs, "step_trace": step_trace,
                "wall_seconds": time.perf_counter() - started,
            }
            out.write(json.dumps(line) + "\n")
            out.flush()
            print(
                f"seed {seed}: {'LIVE' if is_live else 'dead'}  epoch pearson "
                f"{[round(e, 3) for e in epochs]}  ({line['wall_seconds']:.0f}s)",
                flush=True,
            )
    print(f"live {live}/{len(seeds)}", flush=True)


if __name__ == "__main__":
    main()
