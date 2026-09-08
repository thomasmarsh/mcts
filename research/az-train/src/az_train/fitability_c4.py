# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Deterministic small-set fitability controls for Connect Four replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from az_train.convnet_c4 import (
    _literal_loss_gradient,
    fit_value_policy_with_diagnostics,
    initial_weights,
    predict,
    write_weights,
)
from az_train.records_c4 import Positions, load_positions, me_opp_planes


def _mirror_bits(bits: int) -> int:
    """Return the left-right reflection of one 6x7 bitboard word."""
    reflected = 0
    for row in range(6):
        for col in range(7):
            if bits & (1 << (row * 7 + col)):
                reflected |= 1 << (row * 7 + (6 - col))
    return reflected


def _canonical_board_key(black: int, white: int, side: int) -> tuple[int, int, int]:
    """Use the lexicographically smaller literal/reflected board identity."""
    literal = (black, white, side)
    mirrored = (_mirror_bits(black), _mirror_bits(white), side)
    return min(literal, mirrored)


def select_balanced_unique_rows(pos: Positions, per_outcome: int, seed: int) -> np.ndarray:
    """Select equal +/- outcome rows with no literal-or-mirror duplicate."""
    if per_outcome <= 0:
        raise ValueError("per_outcome must be positive")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    seen: set[tuple[int, int, int]] = set()
    for outcome in (-1.0, 1.0):
        candidates = np.flatnonzero(pos.value == outcome)
        taken = 0
        for row in rng.permutation(candidates):
            index = int(row)
            if not pos.policy[index]:
                continue
            key = _canonical_board_key(
                int(pos.black[index]), int(pos.white[index]), int(pos.side[index]),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(index)
            taken += 1
            if taken == per_outcome:
                break
        if taken != per_outcome:
            raise ValueError(f"only found {taken} unique outcome {outcome:g} rows")
    return np.asarray(selected, dtype=np.intp)


def _dense_policy(entries: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((len(entries), 7), dtype=np.float32)
    legal = np.zeros((len(entries), 7), dtype=bool)
    for row, sparse in enumerate(entries):
        for column, probability in sparse:
            target[row, column] = probability
            legal[row, column] = True
    if not np.all(legal.any(axis=1)):
        raise ValueError("fitability control requires completed-Q policies")
    return target, legal


def _concat(parts: list[Positions]) -> Positions:
    return Positions(
        black=np.concatenate([part.black for part in parts]),
        white=np.concatenate([part.white for part in parts]),
        side=np.concatenate([part.side for part in parts]),
        ply=np.concatenate([part.ply for part in parts]),
        value=np.concatenate([part.value for part in parts]),
        policy=[entry for part in parts for entry in part.policy],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.fitability_c4")
    parser.add_argument(
        "--positions", required=True, help="comma-separated v2-connect4 replay files"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-outcome", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args(argv)
    input_paths = [Path(path) for path in args.positions.split(",")]
    pos = _concat([load_positions(path) for path in input_paths])
    rows = select_balanced_unique_rows(pos, args.per_outcome, args.seed)
    me, opp = me_opp_planes(Positions(
        black=pos.black[rows], white=pos.white[rows], side=pos.side[rows],
        ply=pos.ply[rows], value=pos.value[rows], policy=[pos.policy[int(row)] for row in rows],
    ))
    value = pos.value[rows]
    policy, legal = _dense_policy([pos.policy[int(row)] for row in rows])
    weights, metadata = fit_value_policy_with_diagnostics(
        me, opp, value, policy, legal, (me, opp, value, policy, legal),
        seed=args.seed, epochs=args.epochs,
    )
    prediction, logits = predict(weights, me, opp)
    initial = initial_weights(args.seed)
    initial_loss, _ = _literal_loss_gradient(initial, me, opp, value, policy, legal, 1e-4)
    final_loss, _ = _literal_loss_gradient(weights, me, opp, value, policy, legal, 1e-4)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    weight_path = output / "fitability.c4cnn"
    write_weights(str(weight_path), weights)
    np.savez_compressed(
        output / "selected.npz",
        rows=rows,
        black=pos.black[rows],
        white=pos.white[rows],
        side=pos.side[rows],
        ply=pos.ply[rows],
        value=value,
        policy=policy,
        legal=legal,
    )
    result = {
        "inputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in input_paths
        ],
        "selection": {
            "seed": args.seed,
            "per_outcome": args.per_outcome,
            "source_rows": rows.tolist(),
            "outcome_counts": {
                "minus_one": int(np.count_nonzero(value == -1.0)),
                "plus_one": int(np.count_nonzero(value == 1.0)),
            },
        },
        "fit": metadata,
        "same_subset": {
            "initial_joint_objective": initial_loss,
            "final_joint_objective": final_loss,
            "objective_ratio": final_loss / initial_loss,
            "value_min": float(prediction.min()),
            "value_max": float(prediction.max()),
            "value_std": float(np.std(prediction)),
            "positive_prediction_count": int(np.count_nonzero(prediction > 0.0)),
            "negative_prediction_count": int(np.count_nonzero(prediction < 0.0)),
            "value_sign_agreement": float(np.mean(np.sign(prediction) == np.sign(value))),
            "policy_logit_standard_deviation": float(np.std(logits)),
        },
        "artifacts": {
            "weights": weight_path.name,
            "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
