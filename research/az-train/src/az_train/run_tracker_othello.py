"""Turn an Othello CNN coordinator ``RUN_DIR`` into the data behind a live
tracking page, and render that page.

``research/az-train/coordinator_othello_cnn.sh`` leaves everything a monitor
needs on disk while it runs: ``launch.log`` (phase banners with timestamps),
``gen<N>.epochs.jsonl`` (one line per validated epoch, written during the fit),
``gen<N>.cnn.bin.meta.json`` (seed attempts, retry cost, best epoch; written when
the fit finishes), ``gen<N>.gate-vs-*.txt`` (gate output, written as it prints),
``log.jsonl`` (one line per finished generation) and ``watch.log`` (the memory
watchdog's samples). This module derives every number the page shows from those
files, so the page itself only draws. Anything not yet on disk is reported as
missing, with the reason, instead of being left blank.

    uv run --project research/az-train az-train-run-tracker \
        --run-dir local/output/az/othello-cnn/run0 --out page.html --planned-gens 2
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from az_train.coordinator_metrics_othello import parse_gate_output

# Same value as the trainer's default stall threshold (`--stall-check-min-pearson`);
# a network whose validation |pearson| is below it is treated as dead.
DEAD_PEARSON = 0.05
# The per-generation wall-clock the launch script's sizing predicted at the
# replay-window cap, before retries.
ESTIMATE_GENERATION_SECONDS = 5400.0
DEFAULT_FLOOR_MB = 900

_BANNER_RE = re.compile(
    r"^=== (?P<what>.+?) @ (?P<ts>\w{3} \w{3}\s+\d+ \d\d:\d\d:\d\d) \S+ (?P<year>\d{4}) ===$",
    re.MULTILINE,
)
_SELFPLAY_RE = re.compile(
    r"^generation (?P<g>\d+): self-play \((?P<games>\d+) games, (?P<sims>\d+) sims, "
    r"engine=(?P<engine>[\w-]+)\)$"
)
_TRAIN_RE = re.compile(r"^generation (?P<g>\d+) -> (?P<n>\d+): train$")
_REUSED_RE = re.compile(r"^generation (?P<g>\d+): self-play reused \(")
_GATES_RE = re.compile(r"^generation (?P<n>\d+): gates \((?P<games>\d+) games\)$")
_WATCH_RE = re.compile(
    r"^(?P<t>\d\d:\d\d:\d\d) available_mb=(?P<avail>\d+) group_rss_mb=(?P<rss>\d+)"
)


def _parse_time(ts: str, year: str) -> datetime:
    return datetime.strptime(f"{' '.join(ts.split())} {year}", "%a %b %d %H:%M:%S %Y")


def parse_launch_log(text: str) -> dict[str, Any]:
    """Phase banners of one coordinator run: self-play/train/gates start times per
    produced generation (self-play of generation g produces generation g + 1), the
    ``done`` banner, and the run configuration the banners carry."""
    selfplay: dict[int, datetime] = {}
    reused: dict[int, float] = {}
    train: dict[int, datetime] = {}
    gates: dict[int, datetime] = {}
    done: datetime | None = None
    config: dict[str, Any] = {}
    for m in _BANNER_RE.finditer(text):
        when = _parse_time(m["ts"], m["year"])
        what = m["what"]
        if what == "done":
            done = when
        elif ru := _REUSED_RE.match(what):
            n = int(ru["g"]) + 1
            # Self-play the earlier (killed) run already finished: its banner and the
            # train banner that followed it give the original duration.
            reused[n] = _seconds(selfplay.get(n), train.get(n)) or 0.0
            selfplay[n] = when
            train.pop(n, None)
            gates.pop(n, None)
        elif sp := _SELFPLAY_RE.match(what):
            selfplay[int(sp["g"]) + 1] = when
            train.pop(int(sp["g"]) + 1, None)
            gates.pop(int(sp["g"]) + 1, None)
            reused.pop(int(sp["g"]) + 1, None)
            config.update(games=int(sp["games"]), sims=int(sp["sims"]), engine=sp["engine"])
        elif tr := _TRAIN_RE.match(what):
            train[int(tr["n"])] = when
        elif ga := _GATES_RE.match(what):
            gates[int(ga["n"])] = when
            config["gate_games"] = int(ga["games"])
    return {
        "selfplay": selfplay,
        "reused": reused,
        "train": train,
        "gates": gates,
        "done": done,
        "config": config,
        "all_seeds_stalled": "AllSeedsStalledError" in text,
        "traceback": "Traceback (most recent call last)" in text,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every complete JSON line; a half-written trailing line is skipped."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def last_fit_attempt(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The epoch rows of the most recent fit attempt (epoch numbers restart at 1
    with every retry) and how many attempts wrote any epoch line."""
    segments: list[list[dict[str, Any]]] = []
    previous = 0
    for row in rows:
        if not segments or row["epoch"] <= previous:
            segments.append([])
        segments[-1].append(row)
        previous = row["epoch"]
    return (segments[-1] if segments else []), len(segments)


def best_epoch(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The epoch the fit saves its checkpoint from: lowest validation value MSE,
    first occurrence (the trainer's strict ``<`` comparison)."""
    best: dict[str, Any] | None = None
    for row in rows:
        if best is None or row["value_mse"] < best["value_mse"]:
            best = row
    return best


def _watch_summary(text: str, floor_mb: int) -> dict[str, Any]:
    samples = [m for line in text.splitlines() if (m := _WATCH_RE.match(line))]
    killed = [line for line in text.splitlines() if "WATCHDOG" in line]
    if not samples:
        return {"samples": 0, "floor_mb": floor_mb, "killed": bool(killed), "kill_lines": killed}
    lowest = min(samples, key=lambda m: int(m["avail"]))
    return {
        "samples": len(samples),
        "floor_mb": floor_mb,
        "min_available_mb": int(lowest["avail"]),
        "min_available_at": lowest["t"],
        "last_available_mb": int(samples[-1]["avail"]),
        "last_sample_at": samples[-1]["t"],
        "killed": bool(killed),
        "kill_lines": killed,
    }


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _gate_leg(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text()
    try:
        checks = parse_gate_output(text)
    except ValueError:
        checks = {}
    return {"checks": checks, "finished": "GATE:" in text}


def _generation(
    n: int,
    run_dir: Path,
    banners: dict[str, Any],
    log_line: dict[str, Any] | None,
    now: datetime,
    finished_run: bool,
    config_epochs: int,
) -> dict[str, Any]:
    sp_start = banners["selfplay"].get(n)
    tr_start = banners["train"].get(n)
    ga_start = banners["gates"].get(n)
    wall = log_line["wall_seconds"] if log_line else None
    gen_end = None
    if sp_start is not None and wall is not None:
        gen_end = datetime.fromtimestamp(sp_start.timestamp() + wall)

    original_selfplay = banners["reused"].get(n)
    bounds = [
        ("selfplay", sp_start, tr_start),
        ("fit", tr_start, ga_start),
        ("gates", ga_start, gen_end),
    ]
    phases: dict[str, Any] = {}
    running = None
    for name, start, end in bounds:
        if start is None:
            phases[name] = {"state": "pending", "seconds": None}
        elif end is None and not finished_run:
            phases[name] = {
                "state": "running",
                "seconds": _seconds(start, now),
                "started": start.isoformat(),
            }
            running = name
        else:
            phases[name] = {
                "state": "done",
                "seconds": _seconds(start, end),
                "started": start.isoformat(),
            }
    if original_selfplay is not None:
        phases["selfplay"] = {
            "state": "reused",
            "seconds": 0.0,
            "original_seconds": original_selfplay,
            "started": sp_start.isoformat() if sp_start else None,
        }
        if tr_start is None and sp_start is not None and not finished_run:
            phases["fit"] = {
                "state": "running",
                "seconds": _seconds(sp_start, now),
                "started": sp_start.isoformat(),
            }
            running = "fit"
        elif running == "selfplay":
            running = None
    state = "done" if log_line else (running or "pending")

    rows, segments = last_fit_attempt(read_jsonl(run_dir / f"gen{n}.epochs.jsonl"))
    best = best_epoch(rows)
    dead_epochs = [r["epoch"] for r in rows if abs(r["value_pearson"]) < DEAD_PEARSON]

    meta_path = run_dir / f"gen{n}.cnn.bin.meta.json"
    fit: dict[str, Any] | None = None
    live_network: bool | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        metrics = meta["metrics"]
        train = meta["train"]
        retry = float(metrics["retry_wall_seconds"])
        fit_wall = float(metrics["fit_wall_seconds"])
        saved = metrics["best_checkpoint_validation_metrics"]
        live_network = abs(saved["value_pearson"]) >= DEAD_PEARSON
        fit = {
            "seed_used": train["seed_used"],
            "seed_attempts": train["seed_attempts"],
            "stalled_attempts": len(train["seed_attempts"]),
            "retry_wall_seconds": retry,
            "fit_wall_seconds": fit_wall,
            "retry_fraction_of_fit": retry / (retry + fit_wall) if retry + fit_wall else 0.0,
            "retry_fraction_of_generation": retry / wall if wall else None,
            "positions": train["positions"],
            "train_games": train["train_games"],
            "validation_games": train["validation_games"],
            "sources": len(train["sources"]),
            "checkpoint_epoch": metrics["best_checkpoint_epoch"],
            "checkpoint_metrics": saved,
            "final_metrics": metrics["final_validation_metrics"],
            "epochs_configured": train["epochs"],
            "batch_size": train["batch_size"],
            "learning_rate": train["learning_rate"],
        }

    missing: list[str] = []
    if state == "pending":
        missing.append("Not started.")
    if phases["selfplay"]["state"] == "running":
        missing.append(
            "Self-play progress is not observable: the shard is written only when every game ends."
        )
    if phases["fit"]["state"] == "running" and not rows:
        missing.append("No epoch has been validated yet, so there are no curves.")
    if fit is None and phases["fit"]["state"] != "pending":
        missing.append(
            "Seed attempts, stall count and retry cost are written to meta.json when the fit "
            "finishes, so they cannot be shown while it runs."
        )
    if phases["gates"]["state"] == "running":
        missing.append("Gate legs are not timed separately; only the whole gate phase is timed.")
    if (
        state != "done"
        and phases["fit"]["state"] == "done"
        and not (run_dir / f"gen{n}.gate-vs-gen0.txt").exists()
    ):
        missing.append("No gate output yet.")

    return {
        "generation": n,
        "state": state,
        "phases": phases,
        "wall_seconds": wall,
        "wall_seconds_incl_reused_selfplay": None
        if wall is None
        else wall + (original_selfplay or 0.0),
        "wall_vs_estimate": (wall + (original_selfplay or 0.0)) / ESTIMATE_GENERATION_SECONDS
        if wall
        else None,
        "epochs": rows,
        "epochs_configured": config_epochs,
        "epoch_log_segments": segments,
        "best_epoch": None if best is None else best["epoch"],
        "best_metrics": best,
        "dead_epochs": dead_epochs,
        "late_death": bool(rows) and rows[0]["value_pearson"] >= DEAD_PEARSON and bool(dead_epochs),
        "fit": fit,
        "live_network": live_network,
        "gate_vs_gen0": _gate_leg(run_dir / f"gen{n}.gate-vs-gen0.txt"),
        "gate_vs_prev": _gate_leg(run_dir / f"gen{n}.gate-vs-prev.txt") if n >= 2 else None,
        "missing": missing,
    }


def build_run_data(
    run_dir: Path, now: datetime, planned_gens: int | None = None, floor_mb: int = DEFAULT_FLOOR_MB
) -> dict[str, Any]:
    launch = run_dir / "launch.log"
    banners = parse_launch_log(launch.read_text() if launch.exists() else "")
    log_lines = {row["generation"]: row for row in read_jsonl(run_dir / "log.jsonl")}
    watch_path = run_dir / "watch.log"
    watch = _watch_summary(watch_path.read_text() if watch_path.exists() else "", floor_mb)

    seen = set(banners["selfplay"]) | set(log_lines)
    gens = max(seen | {planned_gens or 0}) if seen or planned_gens else 0
    finished = banners["done"] is not None
    stopped = watch["killed"] or banners["traceback"]
    config_epochs = 30
    for path in run_dir.glob("gen*.cnn.bin.meta.json"):
        config_epochs = json.loads(path.read_text())["train"]["epochs"]

    generations = [
        _generation(n, run_dir, banners, log_lines.get(n), now, finished or stopped, config_epochs)
        for n in range(1, gens + 1)
    ]
    if finished:
        status = "finished"
    elif watch["killed"]:
        status = "killed by the memory watchdog"
    elif banners["traceback"]:
        status = "crashed"
    else:
        status = "running"
    current = next((g for g in generations if g["state"] not in ("done", "pending")), None)

    newest = max((p.stat().st_mtime for p in run_dir.iterdir() if p.is_file()), default=None)
    return {
        "run_dir": str(run_dir),
        "snapshot": now.isoformat(),
        "status": status,
        "current": None
        if current is None
        else {
            "generation": current["generation"],
            "phase": next(k for k, v in current["phases"].items() if v["state"] == "running"),
        },
        "config": {**banners["config"], "epochs": config_epochs, "planned_gens": gens},
        "estimate_generation_seconds": ESTIMATE_GENERATION_SECONDS,
        "dead_pearson": DEAD_PEARSON,
        "all_seeds_stalled": banners["all_seeds_stalled"],
        "seconds_since_last_write": None if newest is None else max(0.0, now.timestamp() - newest),
        "memory": watch,
        "earlier_kills": [
            line
            for path in sorted(run_dir.glob("watch.killed-*.log"))
            for line in path.read_text().splitlines()
            if "WATCHDOG" in line
        ],
        "generations": generations,
    }


def render_page(data: dict[str, Any]) -> str:
    template = (Path(__file__).parent / "run_tracker_template.html").read_text()
    payload = json.dumps(data, sort_keys=True).replace("</", "<\\/")
    return template.replace("/*__RUN_DATA__*/null", payload)


def main(argv: list[str] | None = None) -> None:
    summary = (__doc__ or "").split("\n\n")[0]
    ap = argparse.ArgumentParser(prog="az-train-run-tracker", description=summary)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="HTML page to write")
    ap.add_argument("--data-out", type=Path, default=None, help="also write the derived JSON")
    ap.add_argument("--planned-gens", type=int, default=None, help="show unstarted generations")
    ap.add_argument("--floor-mb", type=int, default=DEFAULT_FLOOR_MB, help="watchdog floor in use")
    ap.add_argument("--now", default=None, help="ISO time to treat as now (default: the clock)")
    args = ap.parse_args(argv)

    now = datetime.fromisoformat(args.now) if args.now else datetime.now()
    data = build_run_data(args.run_dir, now, args.planned_gens, args.floor_mb)
    if args.data_out is not None:
        args.data_out.write_text(json.dumps(data, indent=1, sort_keys=True))
    args.out.write_text(render_page(data))
    print(f"wrote {args.out} (status: {data['status']}, generations: {len(data['generations'])})")


if __name__ == "__main__":
    main()
