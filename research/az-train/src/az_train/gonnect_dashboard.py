"""Render a Gonnect CNN run (``log.jsonl`` + ``steps.jsonl``) as one self-contained HTML page.

    uv run --project research/az-train python -m az_train.gonnect_dashboard RUN_DIR OUT.html

The page embeds a snapshot of the run; rerun and republish to refresh it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from az_train.gonnect_cnn import curve_report, read_jsonl

TEMPLATE = Path(__file__).with_name("gonnect_dashboard.html")


def summarize(run_dir: Path, total: int) -> dict:
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
        gens.append(
            {
                "gen": r["gen"],
                "score": r["gate"]["score"],
                "lo": r["gate"]["wilson_lo"],
                "hi": r["gate"]["wilson_hi"],
                "cnn_ms": r["gate"]["cnn_ms_per_move"],
                "value_loss": sum(x["value_loss"] for x in tail) / max(1, len(tail)),
                "policy_loss": sum(x["policy_loss"] for x in tail) / max(1, len(tail)),
                "val_policy_ce": r["val_after"]["policy_ce"],
                "val_policy_top1": r["val_after"]["policy_top1"],
                "val_value_mse": r["val_after"]["value_mse"],
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
    return {
        "total": total,
        "gens": gens,
        "curve": curve_report(rows),
        "steps_done": sum(len(v) for v in steps.values()),
    }


def main() -> None:
    run_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    data = summarize(run_dir, total)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data))
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes, {len(data['gens'])} generations)")


if __name__ == "__main__":
    main()
