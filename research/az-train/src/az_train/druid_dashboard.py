"""Render a Druid CNN run (``log.jsonl`` + ``steps.jsonl``) as one self-contained HTML page.

    uv run --project research/az-train python -m az_train.druid_dashboard \\
        RUN_DIR OUT.html [TOTAL [CONFIG]]

The page embeds a snapshot of the run; rerun and republish to refresh it.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from az_train.druid_cnn import evaluate_run, load_config, read_jsonl
from az_train.yardstick import Pair, fit_bradley_terry

ELO_PER_LOGIT = 400 / 2.302585092994046

TEMPLATE = Path(__file__).with_name("druid_dashboard.html")


def implied_elo(scores: dict[int, tuple[int, float]]) -> dict[int, float]:
    """Chain lag-gate scores into approximate Elo:
    ``Elo(g) = Elo(opponent) + 400 log10(s / (1 - s))`` with ``s`` clamped to [0.02, 0.98];
    generation 0 is 0. ``scores`` maps generation to (opponent generation, score). Generations
    whose opponent has no Elo yet are left out."""
    elo = {0: 0.0}
    for g in sorted(scores):
        opp, s = scores[g]
        if opp not in elo:
            continue
        s = min(0.98, max(0.02, s))
        elo[g] = elo[opp] + ELO_PER_LOGIT * math.log(s / (1 - s))
    return elo


def milestone_elo(milestones: list[dict]) -> None:
    """Add ``elo`` (relative to the reference net, 0) to each milestone row. The first milestone is
    placed by its score against the reference; each later one by its score against the previous
    milestone added to that milestone's Elo, so the curve does not saturate the way the reference
    score does once a net wins nearly every game (scores clamped to [0.02, 0.98])."""
    elo: dict[int, float] = {}
    for m in milestones:
        prev = m.get("previous_gen")
        if prev in elo and m.get("previous"):
            base, s = elo[prev], m["previous"]["score"]
        else:
            base, s = 0.0, next(iter(m["references"].values()))["score"]
        s = min(0.98, max(0.02, s))
        elo[m["gen"]] = m["elo"] = base + ELO_PER_LOGIT * math.log(s / (1 - s))


def gate_pairs(rows: list[dict]) -> list[Pair]:
    """Every gate result as a pair ``gen{g}`` vs ``gen{opponent}`` (lag and champion gates). A
    pairing appearing as both a lag and a champion gate (the same opponent) is counted once."""
    pairs = []
    for r in rows:
        seen = set()
        for kind in ("lag", "best"):
            m = r.get("gate", {}).get(kind)
            if not m or "wins" not in m:
                continue
            key = (m["opponent_gen"], m["wins"], m["losses"], m["draws"])
            if key in seen:
                continue
            seen.add(key)
            name, opp = f"gen{r['gen']}", f"gen{m['opponent_gen']}"
            pairs.append(Pair(name, opp, m["wins"], m["losses"], m["draws"]))
    return pairs


def bt_ratings(rows: list[dict]) -> dict[int, tuple[float, float]]:
    """Joint Bradley-Terry Elo (generation 0 = 0) over all gate results, with standard errors.
    Unlike chaining lag scores, the champion gates tie the ten lag-gate chains together."""
    pairs = gate_pairs(rows)
    if not pairs:
        return {}
    fit = fit_bradley_terry(pairs, order=["gen0"])
    rows_ = zip(fit["players"], fit["elo"], fit["se"], strict=True)
    return {int(n[3:]): (e, se) for n, e, se in rows_}


def summarize(run_dir: Path, total: int, config: Path | None = None) -> dict:
    rows = read_jsonl(run_dir / "log.jsonl")
    steps = defaultdict(list)
    for r in read_jsonl(run_dir / "steps.jsonl"):
        steps[r["gen"]].append(r)
    gens = []
    for r in rows:
        # A generation whose checkpoint survived a crash but whose log line did not is logged on
        # resume with only its gate, so self-play and validation stats may be missing.
        sp = r.get("selfplay", {})
        val_after = r.get("val_after", {})
        finished = max(1, sp.get("games_finished", 1))
        s = steps.get(r["gen"] - 1, [])
        tail = s[-100:]
        lag = r.get("gate", {}).get("lag", {})
        best = r.get("gate", {}).get("best", {})
        diag = r.get("diagnostics", {})
        sp_diag = diag.get("selfplay", {})
        pol = diag.get("policy", {})
        cal = diag.get("calibration", {})
        val = diag.get("value", {})
        held, train = val.get("held", {}), val.get("train", {})
        phases = pol.get("by_phase", {})
        gens.append(
            {
                "gen": r["gen"],
                "score": lag.get("score"),
                "lo": lag.get("wilson_lo"),
                "hi": lag.get("wilson_hi"),
                "lag_opponent": lag.get("opponent_gen"),
                "lag_games": lag.get("games"),
                "best_score": best.get("score"),
                "best_lo": best.get("wilson_lo"),
                "best_hi": best.get("wilson_hi"),
                "best_opponent": best.get("opponent_gen"),
                "gate_skipped": "gate" not in r,
                "champion": r.get("gate", {}).get("champion"),
                "promoted": r.get("gate", {}).get("promoted", False),
                "cnn_ms": lag.get("ms_per_move"),
                "value_loss": sum(x["value_loss"] for x in tail) / max(1, len(tail)),
                "policy_loss": sum(x["policy_loss"] for x in tail) / max(1, len(tail)),
                "val_policy_ce": val_after.get("policy_ce"),
                "val_policy_top1": val_after.get("policy_top1"),
                "val_value_mse": val_after.get("value_mse"),
                "val_value_pearson": val_after.get("value_pearson"),
                "prior_entropy": pol.get("prior_entropy"),
                "target_entropy": pol.get("target_entropy"),
                "kl": pol.get("kl_target_to_prior"),
                "ece": cal.get("ece"),
                "brier": cal.get("brier"),
                "bins": [
                    [b["count"], b.get("mean_pred"), b.get("mean_outcome")]
                    for b in cal.get("bins", [])
                ],
                "legal_moves": sp_diag.get("mean_legal_moves"),
                "phase_share": sp_diag.get("phase_share", {}),
                "phase_ce": {k: v.get("policy_ce") for k, v in phases.items()},
                "phase_uniform": {k: v.get("uniform_ce") for k, v in phases.items()},
                "phase_top1": {k: v.get("policy_top1") for k, v in phases.items()},
                "vh_mse": held.get("mse"),
                "vh_mse_vs_constant": held.get("mse_vs_constant"),
                "vh_pearson": held.get("pearson"),
                "vh_progress": [b.get("mse") for b in held.get("by_progress", [])],
                "vh_sign": held.get("sign_agreement"),
                "vt_mse": train.get("mse"),
                "vt_pearson": train.get("pearson"),
                "v_gap": val.get("mse_gap"),
                "capped": sp.get("games_capped", 0) / max(1, sp.get("games_started", 1)),
                "plies": sp.get("mean_plies"),
                "black": sp.get("black_wins", 0) / finished,
                "draw": sp.get("draws", 0) / finished,
                "positions": sp.get("positions"),
                "t_selfplay": r.get("selfplay_seconds", 0),
                "t_fit": r.get("train", {}).get("fit_seconds", 0),
                "t_gate": r.get("gate_seconds", 0),
                "wall": r.get("wall_seconds", 0),
            }
        )
    elo = implied_elo(
        {g["gen"]: (g["lag_opponent"], g["score"]) for g in gens if g["score"] is not None}
    )
    bt = bt_ratings(rows)
    for g in gens:
        g["implied_elo"] = elo.get(g["gen"])
        g["bt_elo"], g["bt_se"] = bt.get(g["gen"], (None, None))
    report = run_dir / "ratings" / "report.json"
    verdict = None
    settings = {}
    warm = run_dir / "warm-start.json"
    if config is not None:
        # The run's effective config carries any --set overrides (net size, optimizer).
        effective = run_dir / "config.effective.toml"
        cfg = load_config(effective if effective.exists() else config)
        verdict = evaluate_run(cfg, run_dir, total)[0]
        settings = {
            "size": cfg["net"]["size"],
            "channels": cfg["net"]["channels"],
            "blocks": cfg["net"]["blocks"],
            "optimizer": cfg["train"].get("optimizer", "adam"),
            "weight_decay": cfg["train"].get("weight_decay"),
            "learning_rate": cfg["train"]["learning_rate"],
            "warm_learning_rate": cfg["train"].get("warm_learning_rate"),
            "warm_start": json.loads(warm.read_text()) if warm.exists() else None,
            "lag": cfg["gate"]["lag"],
            "promote_score": cfg["gate"]["promote_score"],
            "gate_games": cfg["gate"]["games"],
            "max_capped_rate": cfg["rules"]["max_capped_rate"],
            "early_gen": cfg["rules"]["early_gen"],
            "early_min": cfg["rules"]["early_min_mean_lag_score"],
        }
    milestones = read_jsonl(run_dir / "milestones.jsonl")
    milestone_elo(milestones)
    return {
        "milestones": milestones,
        "total": total,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "settings": settings,
        "gens": gens,
        "ratings": json.loads(report.read_text()) if report.exists() else None,
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
