"""Roll a `bakeoff.sh` work directory up into the bake-off deliverable: a
strength-vs-cumulative-CPU-seconds table (+ PNG if matplotlib is present).

Reads, from the work dir:
  cpu_seconds.tsv     stage -> wall seconds (harvest, train_*, edax_*, h2h_*)
  harvest/harvest.json  per-arm Edax label CPU breakout (ORACLE=edax runs)
  mse.json            per-arm held-out MSE + pairwise bootstrap CI
  edax_<arm>.txt      "secondary N = <level>" ladder placement
  h2h_<a>_<b>.txt     per-depth A-vs-B win rate + CI

Writes bakeoff_results.csv and bakeoff_results.md into the work dir and
prints the table. Nothing here asserts the kill gate -- it lays the numbers
out so the operator can apply the kill gate by eye.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import re
import sys
from pathlib import Path

ARMS = ["a", "b", "c", "d"]


def read_cpu(work: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    tsv = work / "cpu_seconds.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            if "\t" in line:
                name, secs = line.split("\t")
                out[name] = out.get(name, 0.0) + float(secs)
    return out


def edax_level(work: Path, arm: str) -> int | None:
    f = work / f"edax_{arm}.txt"
    if not f.exists():
        return None
    m = re.findall(r"secondary N = (\d+)", f.read_text())
    return int(m[-1]) if m else None


def h2h_rows(work: Path) -> list[tuple[str, str, int, float, float, float]]:
    rows: list[tuple[str, str, int, float, float, float]] = []
    for f in sorted(work.glob("h2h_*_*.txt")):
        _, a, b = f.stem.split("_")
        for m in re.finditer(
            r"D=(\d+): A win_rate=([\d.]+) ci=\[([\d.]+), ([\d.]+)\]", f.read_text()
        ):
            d, wr, lo, hi = m.groups()
            rows.append((a.upper(), b.upper(), int(d), float(wr), float(lo), float(hi)))
    return rows


def main() -> None:
    work = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    cpu = read_cpu(work)
    mse = json.loads((work / "mse.json").read_text()) if (work / "mse.json").exists() else {}
    mse_models = mse.get("models", {})

    harvest = (
        json.loads((work / "harvest" / "harvest.json").read_text())
        if (work / "harvest" / "harvest.json").exists()
        else {}
    )
    # Per-arm label-generation CPU. For ORACLE=mcts the arms share the whole
    # `harvest` stage. For ORACLE=edax the stage's `cpu` block breaks the
    # Edax bill out: arm A pays only the self-play search, B/D add the root
    # evals, C adds the per-node harvest (attributed to C, never amortised).
    stage_secs = cpu.get("harvest", 0.0)
    hc = harvest.get("cpu") if harvest.get("oracle") == "edax" else None
    if hc:
        selfplay = hc["selfplay_s"]
        label_gen = {
            "a": selfplay,
            "b": selfplay + hc["edax_root_s"],
            "c": selfplay + hc["edax_harvest_s"],
            "d": selfplay + hc["edax_root_s"],
        }
    else:
        label_gen = {arm: stage_secs for arm in ARMS}

    csv_lines = ["arm,label_gen_cpu_s,train_cpu_s,cumulative_cpu_s,edax_level,held_out_mse"]
    table: list[tuple[str, ...]] = []
    for arm in ARMS:
        train_s = cpu.get(f"train_{arm}", 0.0)
        shared_label_secs = label_gen[arm]
        cum = shared_label_secs + train_s
        lvl = edax_level(work, arm)
        m = mse_models.get(arm.upper(), {}).get("mse")
        csv_lines.append(
            f"{arm},{shared_label_secs:.0f},{train_s:.0f},{cum:.0f},"
            f"{'' if lvl is None else lvl},{'' if m is None else f'{m:.5f}'}"
        )
        table.append(
            (
                arm.upper(),
                f"{shared_label_secs:.0f}",
                f"{train_s:.0f}",
                f"{cum:.0f}",
                "-" if lvl is None else f"L{lvl}",
                "-" if m is None else f"{m:.5f}",
            )
        )

    (work / "bakeoff_results.csv").write_text("\n".join(csv_lines) + "\n")

    md = ["# Bake-off results", "", "| arm | label CPU s | train CPU s | cumulative CPU s | Edax level | held-out MSE |", "|---|---|---|---|---|---|"]
    for r in table:
        md.append("| " + " | ".join(r) + " |")
    md += ["", "## Head-to-head (A-vs-B win rate, A's perspective)", "", "| A | B | depth | A win rate | CI95 |", "|---|---|---|---|---|"]
    for a, b, depth, wr, lo, hi in h2h_rows(work):
        md.append(f"| {a} | {b} | {depth} | {wr:.3f} | [{lo:.3f}, {hi:.3f}] |")
    for name, pw in mse.get("pairwise", {}).items():
        md.append("")
        md.append(
            f"held-out MSE {name}: diff {pw['mse_diff']:+.5f} "
            f"ci95 [{pw['ci95'][0]:+.5f}, {pw['ci95'][1]:+.5f}] "
            f"({'excludes' if pw['ci_excludes_zero'] else 'includes'} 0)"
        )
    (work / "bakeoff_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [label_gen[a] + cpu.get(f"train_{a}", 0.0) for a in ARMS]
        ys = [edax_level(work, a) or 0 for a in ARMS]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(xs, ys, "o-")
        for a, x, y in zip(ARMS, xs, ys, strict=True):
            ax1.annotate(a.upper(), (x, y))
        ax1.set_xlabel("cumulative Rust CPU-seconds")
        ax1.set_ylabel("Edax level")
        ax1.set_title("strength vs CPU-seconds")
        ms = [mse_models.get(a.upper(), {}).get("mse", float("nan")) for a in ARMS]
        ax2.bar(ARMS, ms)
        ax2.set_ylabel("held-out MSE")
        ax2.set_title("held-out regression MSE")
        fig.tight_layout()
        fig.savefig(work / "bakeoff_results.png", dpi=120)
        print(f"wrote {work / 'bakeoff_results.png'}")
    except ImportError:
        print("(matplotlib not installed -- CSV + markdown only)")


if __name__ == "__main__":
    main()
