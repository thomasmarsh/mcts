"""Render a Gonnect CNN run (``log.jsonl`` + ``steps.jsonl``) as one self-contained HTML page.

    uv run --project research/az-train python -m az_train.gonnect_dashboard \\
        RUN_DIR OUT.html [TOTAL [CONFIG]]

The page embeds a snapshot of the run; rerun and republish to refresh it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from az_train.gonnect_cnn import evaluate_run, load_config, read_jsonl

TEMPLATE = Path(__file__).with_name("gonnect_dashboard.html")


def summarize(run_dir: Path, total: int, config: Path | None = None) -> dict:
    rows = read_jsonl(run_dir / "log.jsonl")
    steps = defaultdict(list)
    for r in read_jsonl(run_dir / "steps.jsonl"):
        steps[r["gen"]].append(r)
    gens = []
    for r in rows:
        sp = r["selfplay"]
        finished = max(1, sp.get("games_finished", 1))
        s = steps.get(r["gen"] - 1, [])
        tail = s[-100:]
        lag = r.get("gate", {}).get("lag", {})
        best = r.get("gate", {}).get("best", {})
        diag = r.get("diagnostics", {})
        gens.append(
            {
                "gen": r["gen"],
                "score": lag.get("score"),
                "lo": lag.get("wilson_lo"),
                "hi": lag.get("wilson_hi"),
                "lag_opponent": lag.get("opponent_gen"),
                "best_score": best.get("score"),
                "champion": r.get("gate", {}).get("champion"),
                "promoted": r.get("gate", {}).get("promoted", False),
                "cnn_ms": lag.get("ms_per_move"),
                "value_loss": sum(x["value_loss"] for x in tail) / max(1, len(tail)),
                "policy_loss": sum(x["policy_loss"] for x in tail) / max(1, len(tail)),
                "val_policy_ce": r["val_after"]["policy_ce"],
                "val_policy_top1": r["val_after"]["policy_top1"],
                "val_value_mse": r["val_after"]["value_mse"],
                "prior_entropy": diag.get("policy", {}).get("prior_entropy"),
                "target_entropy": diag.get("policy", {}).get("target_entropy"),
                "ece": diag.get("calibration", {}).get("ece"),
                "capped": sp.get("games_capped", 0) / max(1, sp.get("games_started", 1)),
                "plies": sp.get("mean_plies"),
                "black": sp.get("black_wins", 0) / finished,
                "swap": sp.get("swaps", 0) / finished,
                "positions": sp.get("positions"),
                "t_selfplay": r.get("selfplay_seconds", 0),
                "t_fit": r.get("train", {}).get("fit_seconds", 0),
                "t_gate": r.get("gate_seconds", 0),
                "wall": r.get("wall_seconds", 0),
            }
        )
    report = run_dir / "ratings" / "report.json"
    svg = run_dir / "ratings" / "curve.svg"
    verdict = None
    if config is not None:
        verdict = evaluate_run(load_config(config), run_dir, total)[0]
    return {
        "total": total,
        "gens": gens,
        "ratings": json.loads(report.read_text()) if report.exists() else None,
        "ratings_svg": svg.read_text() if svg.exists() else None,
        "verdict": verdict,
        "steps_done": sum(len(v) for v in steps.values()),
    }


def main() -> None:
    run_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    config = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    data = summarize(run_dir, total, config)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data))
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes, {len(data['gens'])} generations)")


if __name__ == "__main__":
    main()
