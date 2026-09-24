# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Compare Druid CNN runs that differ in one setting (net size, head kind, warm start).

``summary RUN...`` prints, per run: parameter count, wall cost per generation, cumulative wall
time, lag-gate score, held-out value MSE / Pearson, and the cell head's cross entropy against the
uniform one. ``tournament OUT NAME=RUN:GEN ...`` plays every pair of the named checkpoints (paired
openings, both seats) with the ``druid_gate`` binary, and prints each score with its Wilson 95%
interval: an interval containing 0.5 is a tie. ``--at-wall SECONDS`` in ``summary`` reports the
generation each run had reached when its cumulative wall time first exceeded ``SECONDS``.

Single seed, 40-game gates: every number here is noisy, and the tournament's interval is the only
honest statement of what a score difference means.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from az_train.druid_cnn import geometry_of, run_gate_binary
from az_train.gonnect_cnn import ROOT, load_config, match_config_text, read_jsonl


def run_rows(run: Path) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(run / "log.jsonl") if "wall_seconds" in r]


def diagnostics_by_gen(run: Path) -> dict[int, dict[str, Any]]:
    return {r["gen"]: r for r in read_jsonl(run / "diagnostics.jsonl")}


def cumulative_wall(rows: list[dict[str, Any]]) -> list[float]:
    return list(np.cumsum([r["wall_seconds"] for r in rows]))


def gen_at_wall(rows: list[dict[str, Any]], seconds: float) -> int:
    """The last generation whose cumulative wall time is within ``seconds`` (0 if none)."""
    cum = cumulative_wall(rows)
    return max([r["gen"] for r, c in zip(rows, cum, strict=True) if c <= seconds], default=0)


def cell_ce(diag: list[dict[str, Any]]) -> tuple[float, float]:
    """Pooled (cell CE, uniform CE) over generations, weighted by non-forced cell positions."""
    w = np.array([d["policy"]["by_phase"]["cell"]["positions"] for d in diag], dtype=float)
    ce = np.array([d["policy"]["by_phase"]["cell"]["policy_ce"] for d in diag])
    un = np.array([d["policy"]["by_phase"]["cell"]["uniform_ce"] for d in diag])
    return float((ce * w).sum() / w.sum()), float((un * w).sum() / w.sum())


def summarize(run: Path, lo: int, hi: int) -> dict[str, Any]:
    """Metrics of generations ``lo..hi`` (inclusive) of one run."""
    rows = [r for r in run_rows(run) if lo <= r["gen"] <= hi]
    diag = diagnostics_by_gen(run)
    picked = [diag[r["gen"]] for r in rows if r["gen"] in diag]
    cfg = load_config(run / "config.effective.toml")
    lag = [r["gate"]["lag"]["score"] for r in rows if "gate" in r]
    ce, uniform = cell_ce(picked)
    pear = [
        d["value"]["held"]["pearson"] for d in picked if d["value"]["held"]["pearson"] is not None
    ]
    return {
        "run": run.name,
        "params": geometry_of(cfg).n_weights(),
        "gens": f"{lo}-{hi}",
        "wall_per_gen": float(np.mean([r["wall_seconds"] for r in rows])),
        "selfplay_s": float(np.mean([r["selfplay_seconds"] for r in rows])),
        "fit_s": float(np.mean([r["train"]["fit_seconds"] for r in rows])),
        "gate_s": float(np.mean([r["gate_seconds"] for r in rows])),
        "lag_score": float(np.mean(lag)),
        "value_mse": float(np.mean([d["value"]["held"]["mse"] for d in picked])),
        "value_pearson": float(np.mean(pear)),
        "cell_ce": ce,
        "cell_uniform": uniform,
        "cell_gain_pct": 100.0 * (uniform - ce) / uniform,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        ("run", "{}"), ("params", "{:,}"), ("gens", "{}"), ("wall_per_gen", "{:.0f}"),
        ("selfplay_s", "{:.0f}"), ("fit_s", "{:.0f}"), ("gate_s", "{:.0f}"),
        ("lag_score", "{:.3f}"), ("value_mse", "{:.3f}"), ("value_pearson", "{:.3f}"),
        ("cell_ce", "{:.3f}"), ("cell_uniform", "{:.3f}"), ("cell_gain_pct", "{:.1f}"),
    ]  # fmt: skip
    cells = [[fmt.format(r[k]) for k, fmt in cols] for r in rows]
    widths = [max(len(k), *(len(c[i]) for c in cells)) for i, (k, _) in enumerate(cols)]
    print("  ".join(k.ljust(w) for (k, _), w in zip(cols, widths, strict=True)))
    for c in cells:
        print("  ".join(v.ljust(w) for v, w in zip(c, widths, strict=True)))


def summary(runs: list[Path], lo: int, hi: int, at_wall: float | None) -> None:
    print_table([summarize(r, lo, hi) for r in runs])
    print()
    for r in runs:
        rows = run_rows(r)
        cum = cumulative_wall(rows)
        at12 = cum[11] if len(cum) > 11 else float("nan")
        line = f"{r.name}: cumulative wall at gen 12 = {at12:.0f} s"
        if at_wall is not None:
            line += f", generation reached within {at_wall:.0f} s = {gen_at_wall(rows, at_wall)}"
        print(line)


def tournament(out: Path, entries: list[tuple[str, Path, int]], games: int, workers: int) -> None:
    """Every pair of checkpoints plays ``games`` paired games; prints score, Wilson interval, and
    whether the pair is a tie (interval contains 0.5)."""
    cfg = load_config(entries[0][1] / "config.effective.toml")
    agents = [
        (name, run / f"gen{gen}.bin", cfg["gate"]["simulations"]) for name, run, gen in entries
    ]
    pairs = list(itertools.combinations([n for n, _, _ in entries], 2))
    out.mkdir(parents=True, exist_ok=True)
    text = match_config_text(
        cfg,
        out=out / "tournament.jsonl",
        agents=agents,
        pairs=pairs,
        openings=games // 2,
        opening_plies=cfg["gate"]["opening_plies"],
        seed=cfg["gate"]["seed"] + 100,
        workers=workers,
        max_plies=cfg["gate"]["max_plies"],
    )
    rows = run_gate_binary(out / "tournament.toml", text, out / "tournament.jsonl", resume=True)
    for r in rows:
        lo, hi = r["wilson_lo"], r["wilson_hi"]
        verdict = (
            "TIE" if lo <= 0.5 <= hi else (f"{r['a']} better" if lo > 0.5 else f"{r['b']} better")
        )
        print(
            f"{r['a']} vs {r['b']}: {r['score_a']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"({r['a_wins']}-{r['b_wins']}-{r['draws']}, {r['games']} games)  {verdict}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("runs", nargs="+")
    s.add_argument("--from-gen", type=int, default=5)
    s.add_argument("--to-gen", type=int, default=12)
    s.add_argument("--at-wall", type=float)
    t = sub.add_parser("tournament")
    t.add_argument("out")
    t.add_argument("entries", nargs="+", help="NAME=RUN_DIR:GEN")
    t.add_argument("--games", type=int, default=200)
    t.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    def path(x: str) -> Path:
        q = Path(x)
        return q if q.is_absolute() else ROOT / q

    if args.cmd == "summary":
        summary([path(r) for r in args.runs], args.from_gen, args.to_gen, args.at_wall)
    else:
        entries = []
        for e in args.entries:
            name, _, rest = e.partition("=")
            run, _, gen = rest.rpartition(":")
            entries.append((name, path(run), int(gen)))
        tournament(path(args.out), entries, args.games, args.workers)


if __name__ == "__main__":
    main()
