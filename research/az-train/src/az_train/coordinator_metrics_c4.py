# ruff: noqa: E501
"""Merge one generation's mixture-fit result and its two equal-budget gates into a
single JSON metrics line for the graded Connect Four CNN coordinator.

The mixture fit driver (``az_train.mixture_selfplay_c4``) already writes a rich
``gen<N>.result.json`` (replay counts, label balance, held-out proven value
Pearson, in-replay mixed-target Pearson, fit value MSE and policy cross-entropy,
artifact hashes). This module adds the search-strength half: it parses the
``connect4_cnn_smoke_gate`` stdout for the gen-vs-zero and gen-vs-gen0 matches
and emits one flat line per generation so the whole curve is reconstructable
without scraping console text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_GATE_RE = re.compile(
    r"(?P<wins>\d+)-(?P<draws>\d+)-(?P<losses>\d+)\s*\(W-D-L\).*?"
    r"score share (?P<share>[0-9.]+),\s*games=(?P<games>\d+),\s*sims=(?P<sims>\d+)"
)


def wilson_lower_bound(successes: float, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    ``successes`` may be fractional (a draw counts as half a success), matching the
    gate's score-share convention.
    """
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def parse_gate_line(text: str) -> dict[str, float | int]:
    """Extract the W-D-L record, score share, Wilson lower bound, games and sims
    from ``connect4_cnn_smoke_gate`` output."""
    match = _GATE_RE.search(text)
    if match is None:
        raise ValueError(f"no gate result line found in:\n{text}")
    wins = int(match["wins"])
    draws = int(match["draws"])
    losses = int(match["losses"])
    games = int(match["games"])
    successes = wins + 0.5 * draws
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": games,
        "sims": int(match["sims"]),
        "score_share": round(successes / games, 4) if games else 0.0,
        "wilson_lower_bound": round(wilson_lower_bound(successes, games), 4),
    }


def merge_generation_metrics(
    result: dict[str, object],
    gate_vs_zero_text: str,
    gate_vs_gen0_text: str,
    *,
    generation: int,
    wall_seconds: float,
) -> dict[str, object]:
    """One flat metrics record for generation ``generation``."""
    held_out = result["held_out_proven"]["principled_early_stop"]  # type: ignore[index]
    in_replay = result["in_replay_mixed_target_pearson"]["principled_early_stop"]  # type: ignore[index]
    fit = result.get("fit_metrics", {})  # type: ignore[assignment]
    split = result["replay_split"]  # type: ignore[index]
    ref_counts = result["reference_proven_counts"]  # type: ignore[index]
    searched = result["searched_value_summary"]  # type: ignore[index]
    return {
        "generation": generation,
        "coordinator_wall_seconds": round(float(wall_seconds), 1),
        "replay_split": split,
        "reference_proven_counts": ref_counts,
        "searched_value_summary": searched,
        "principled_early_stop_epoch": result.get("principled_early_stop_epoch"),
        "value_pearson_reference_corpus": held_out["value_pearson"],  # type: ignore[index]
        "value_sign_agreement_reference_corpus": held_out["sign_agreement"],  # type: ignore[index]
        "value_pearson_selfplay_heldout": round(float(in_replay["held_out"]), 4),  # type: ignore[index]
        "value_pearson_selfplay_train": round(float(in_replay["train"]), 4),  # type: ignore[index]
        "fit_metrics": fit,
        "gate_vs_zero": parse_gate_line(gate_vs_zero_text),
        "gate_vs_gen0": parse_gate_line(gate_vs_gen0_text),
        "fit_wall_seconds": result.get("fit_wall_seconds"),
        "peak_rss_bytes": result.get("peak_rss_bytes"),
        "weights": result.get("weights"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m az_train.coordinator_metrics_c4")
    parser.add_argument("--result", required=True, help="gen<N>.result.json from mixture_selfplay_c4")
    parser.add_argument("--gate-vs-zero", required=True, help="connect4_cnn_smoke_gate stdout, gen vs zero net")
    parser.add_argument("--gate-vs-gen0", required=True, help="connect4_cnn_smoke_gate stdout, gen vs gen0 head")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    args = parser.parse_args(argv)

    result = json.loads(Path(args.result).read_text())
    line = merge_generation_metrics(
        result,
        Path(args.gate_vs_zero).read_text(),
        Path(args.gate_vs_gen0).read_text(),
        generation=args.generation,
        wall_seconds=args.wall_seconds,
    )
    print(json.dumps(line, sort_keys=True))


if __name__ == "__main__":
    main()
