# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Trainer and coordinator for the Gonnect CNN track (``games/gonnect/cnn/az-7x7.toml``).

One generation is: Gumbel self-play with the current weights (Rust, MLX) -> a fixed number of
batch-32 Adam steps on a sliding replay window, warm-started from the previous generation (torch,
MPS) -> the progress gate, the new net against an MCTS preset (Rust, MLX). Everything a generation
produces is written before the next starts, so a killed run resumes from ``latest.pt``.

Files under the run directory: ``gen<N>.bin`` (exported weights, ``crates/grid-cnn`` format),
``shards/gen<N>.bin`` (self-play), ``latest.pt`` (model + optimizer of the newest generation),
``steps.jsonl`` (one line per optimizer step, written as it happens), ``log.jsonl`` (one line per
generation), ``gate/gen<N>.jsonl`` (the progress gate's games).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import subprocess
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from az_train import gridcnn
from az_train.gonnect_records import (
    IN_PLANES,
    Positions,
    decode_planes,
    load_positions,
    num_actions,
    read_shard,
)

ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "target" / "release" / "examples"


# --------------------------------------------------------------------------------------- config


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = tomllib.loads(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        value = tomllib.loads(f"v = {raw}")["v"]
        cfg[section][name] = value
    return cfg


def geometry_of(cfg: dict[str, Any]) -> gridcnn.Geometry:
    n = cfg["net"]
    return gridcnn.Geometry(
        size=n["size"],
        in_planes=IN_PLANES,
        channels=n["channels"],
        blocks=n["blocks"],
        policy_planes=n["policy_planes"],
        policy_out=num_actions(n["size"]),
        value_planes=n["value_planes"],
        value_hidden=n["value_hidden"],
    )


def train_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("the Gonnect CNN trains on MPS; there is no CPU fallback")
    return torch.device("mps")


# ------------------------------------------------------------------------------------- training


def d4_sources(size: int, device: torch.device) -> torch.Tensor:
    """``(8, cells)`` gather indices: ``x_flat[..., src[s]]`` is symmetry ``s`` applied to a
    row-major ``size x size`` grid."""
    cells = torch.arange(size * size).reshape(size, size)
    return torch.stack([gridcnn.d4_apply(cells, s).flatten() for s in range(8)]).to(device)


class Data:
    """A replay window as device tensors. ``legal`` is stored as uint8 (gather is not defined for
    bool on every backend)."""

    def __init__(self, positions: Positions, device: torch.device) -> None:
        self.planes = torch.from_numpy(positions.planes).to(device)
        self.policy = torch.from_numpy(positions.policy).to(device)
        self.legal = torch.from_numpy(positions.legal.astype(np.uint8)).to(device)
        self.value = torch.from_numpy(positions.value).to(device)

    def __len__(self) -> int:
        return len(self.value)


def concat_positions(parts: list[Positions]) -> Positions:
    return Positions(
        np.concatenate([p.planes for p in parts]),
        np.concatenate([p.policy for p in parts]),
        np.concatenate([p.legal for p in parts]),
        np.concatenate([p.value for p in parts]),
        np.concatenate([p.game for p in parts]),
    )


def split_validation(positions: Positions, games: int) -> tuple[Positions, Positions]:
    """Positions of the first ``games`` distinct game ids of a shard are held out."""
    held = np.isin(positions.game, np.unique(positions.game)[:games])
    return positions.take(~held), positions.take(held)


def _augment(
    x: torch.Tensor, pi: torch.Tensor, legal: torch.Tensor, src: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One random board symmetry per sample, applied consistently to the planes, the policy
    target and the legal mask (the swap and no-move entries are symmetry independent)."""
    b, c = x.shape[0], x.shape[1]
    cells = size * size
    idx = src[torch.randint(0, 8, (b,), device=x.device)]
    x = (
        x.reshape(b, c, cells)
        .gather(2, idx[:, None, :].expand(-1, c, -1))
        .reshape(b, c, size, size)
    )
    pi = torch.cat([pi[:, :cells].gather(1, idx), pi[:, cells:]], dim=1)
    legal = torch.cat([legal[:, :cells].gather(1, idx), legal[:, cells:]], dim=1)
    return x, pi, legal


def policy_loss(logits: torch.Tensor, target: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Cross entropy of the improved-policy target under the legal-masked softmax."""
    logp = F.log_softmax(logits.masked_fill(~legal, -1e9), dim=1)
    return -(target * logp).sum(dim=1).mean()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate(
    model: gridcnn.GridCNN, positions: Positions, device: torch.device, chunk: int = 512
) -> dict[str, float]:
    """Fit metrics on positions with no augmentation, batch-norm in eval mode."""
    was_training = model.training
    model.eval()
    size = model.geometry.size
    values, ce, top1 = [], [], []
    for start in range(0, len(positions), chunk):
        part = Data(positions.slice(start, start + chunk), device)
        v, logits = model(part.planes.float())
        values.append(v.cpu().numpy())
        legal = part.legal > 0
        logp = F.log_softmax(logits.masked_fill(~legal, -1e9), dim=1)
        ce.append((-(part.policy * logp).sum(dim=1)).cpu().numpy())
        pred = logits.masked_fill(~legal, -1e9).argmax(dim=1)
        top1.append((pred == part.policy.argmax(dim=1)).float().cpu().numpy())
    pred_v = np.concatenate(values)
    target = positions.value
    model.train(was_training)
    assert model.geometry.size == size
    return {
        "value_std": float(pred_v.std()),
        "value_mse": float(np.mean((pred_v - target) ** 2)),
        "value_pearson": _pearson(pred_v, target),
        "value_sign_agreement": float(np.mean(np.sign(pred_v) == np.sign(target))),
        "policy_ce": float(np.concatenate(ce).mean()),
        "policy_top1": float(np.concatenate(top1).mean()),
    }


class Stalled(RuntimeError):
    """The value head outputs a constant at the stall check (a dead initialization)."""


def fit(
    model: gridcnn.GridCNN,
    opt: torch.optim.Optimizer,
    window: Data,
    val: Positions,
    tcfg: dict[str, Any],
    *,
    gen: int,
    global_step: int,
    device: torch.device,
    step_log: Callable[[dict[str, Any]], None],
    stall_check: bool,
) -> tuple[int, dict[str, Any]]:
    """``steps_per_generation`` Adam steps at ``batch_size`` on ``window``; returns the new global
    step and the generation's training summary."""
    size = model.geometry.size
    src = d4_sources(size, device)
    weights = model.weight_tensors()
    batch, steps, l2 = tcfg["batch_size"], tcfg["steps_per_generation"], tcfg["l2"]
    model.train()
    losses: list[float] = []
    val_trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = torch.randint(0, len(window), (batch,), device=device)
        x, pi, legal = _augment(
            window.planes[idx].float(), window.policy[idx], window.legal[idx], src, size
        )
        value, logits = model(x)
        value_loss = F.mse_loss(value, window.value[idx])
        pol_loss = policy_loss(logits, pi, legal > 0)
        reg = l2 * sum((w**2).sum() for w in weights)
        loss = value_loss + pol_loss + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        global_step += 1
        losses.append(loss.item())
        step_log(
            {
                "gen": gen,
                "step": step,
                "global_step": global_step,
                "loss": losses[-1],
                "value_loss": value_loss.item(),
                "policy_loss": pol_loss.item(),
                "reg": reg.item(),
                "step_seconds": time.perf_counter() - t0,
            }
        )
        checkpoint = step == tcfg["stall_check_step"] and stall_check
        if checkpoint or step % tcfg["validate_every"] == 0 or step == steps:
            metrics = evaluate(model, val, device)
            val_trace.append({"step": step, **metrics})
            if checkpoint and metrics["value_std"] < tcfg["stall_check_min_value_std"]:
                raise Stalled(
                    f"step {step}: value predictions are constant (std {metrics['value_std']:.5f})"
                )
    k = max(1, min(50, steps // 4))
    return global_step, {
        "steps": steps,
        "loss_first": float(np.mean(losses[:k])),
        "loss_last": float(np.mean(losses[-k:])),
        "fit_seconds": time.perf_counter() - started,
        "validation": val_trace,
    }


# ----------------------------------------------------------------------------------- run state


def atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _log_step(path: Path, extra: dict[str, Any], row: dict[str, Any]) -> None:
    append_jsonl(path, {**row, **extra})


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def export_weights(model: gridcnn.GridCNN, path: Path) -> None:
    model.eval()
    tmp = path.with_suffix(".tmp")
    gridcnn.write_weights(tmp, model.geometry, gridcnn.export_flat(model))
    os.replace(tmp, path)
    model.train()


def new_model(g: gridcnn.Geometry, seed: int, device: torch.device) -> gridcnn.GridCNN:
    torch.manual_seed(seed)
    return gridcnn.GridCNN(g).to(device)


def make_optimizer(model: gridcnn.GridCNN, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)


# ------------------------------------------------------------------------------ Rust binaries


def rust_env() -> dict[str, str]:
    return {**os.environ, "LIBRARY_PATH": os.environ.get("LIBRARY_PATH", "/opt/homebrew/lib")}


def build_binaries() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "-p", "game-gonnect", "--example", "gonnect_selfplay"]
        + ["--example", "gonnect_gate", "--example", "gonnect_check"],
        cwd=ROOT,
        env=rust_env(),
        check=True,
    )


def run_selfplay(config: Path, weights: Path, shard: Path, seed: int) -> dict[str, Any]:
    cmd = [str(EXAMPLES / "gonnect_selfplay"), "--config", str(config), "--weights", str(weights)]
    out = subprocess.run(
        cmd + ["--out", str(shard), "--seed", str(seed)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def gate_config_text(cfg: dict[str, Any], run_dir: Path, gen: int, weights: Path) -> str:
    gate, play = cfg["gate"], cfg["play"]
    return f"""size = {cfg["net"]["size"]}
openings = {gate["games"] // 2}
opening_plies = 2
max_plies = {gate["max_plies"]}
seed = {gate["seed"]}
workers = {gate["workers"]}
out = "{run_dir / "gate" / f"gen{gen}.jsonl"}"
presets = "{gate["presets"]}"
pairs = ["cnn:{gate["opponent"]}"]

[[agent]]
name = "{gate["opponent"]}"
kind = "preset"
preset = "{gate["opponent_preset"]}"
iterations = {gate["opponent_iterations"]}

[[agent]]
name = "cnn"
kind = "cnn-gumbel"
weights = "{weights}"
iterations = {play["simulations"]}
considered_actions = {play["considered_actions"]}
value_scale = {play["value_scale"]}
max_visit_init = {play["max_visit_init"]}
chunk_size = {play["chunk_size"]}
"""


def run_progress_gate(
    cfg: dict[str, Any], run_dir: Path, gen: int, weights: Path
) -> dict[str, Any]:
    (run_dir / "gate").mkdir(exist_ok=True)
    out = run_dir / "gate" / f"gen{gen}.jsonl"
    out.unlink(missing_ok=True)
    conf = run_dir / "gate" / f"gen{gen}.toml"
    conf.write_text(gate_config_text(cfg, run_dir, gen, weights))
    subprocess.run(
        [str(EXAMPLES / "gonnect_gate"), "--config", str(conf)],
        cwd=ROOT,
        env=rust_env(),
        check=True,
        capture_output=True,
    )
    rows = [r for r in read_jsonl(out) if r.get("type") == "pairing"]
    return rows[-1]


# ------------------------------------------------------------------------------ coordinator


def wilson(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games == 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return centre - half, centre + half


def load_window(
    run_dir: Path, gen: int, cfg: dict[str, Any], device: torch.device
) -> tuple[Data, Positions, int]:
    """Replay window for generation ``gen``'s fit (train parts of the last ``replay_generations``
    shards) and the held-out games of shard ``gen``."""
    tcfg = cfg["train"]
    parts: list[Positions] = []
    val = None
    for g in range(max(0, gen - tcfg["replay_generations"] + 1), gen + 1):
        _, positions = load_positions(run_dir / "shards" / f"gen{g}.bin", game_offset=g * 1_000_000)
        train, held = split_validation(positions, tcfg["validation_games"])
        parts.append(train)
        if g == gen:
            val = held
    assert val is not None
    joined = concat_positions(parts)
    return Data(joined, device), val, len(joined)


def run(
    config_path: Path, overrides: list[str], out_dir: Path | None, generations: int | None
) -> None:
    cfg = load_config(config_path, overrides)
    tcfg = cfg["train"]
    run_dir = Path(out_dir or cfg["loop"]["out_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    total = generations if generations is not None else cfg["loop"]["generations"]
    g = geometry_of(cfg)
    device = train_device()
    build_binaries()

    steps_log, gen_log = run_dir / "steps.jsonl", run_dir / "log.jsonl"
    (run_dir / "config.toml").write_text(Path(config_path).read_text())

    latest = run_dir / "latest.pt"
    model: gridcnn.GridCNN | None = None
    opt: torch.optim.Optimizer | None = None
    gen, global_step = 0, 0
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model = gridcnn.GridCNN(g).to(device)
        model.load_state_dict(ckpt["model"])
        opt = make_optimizer(model, tcfg["learning_rate"])
        opt.load_state_dict(ckpt["opt"])
        gen, global_step = ckpt["gen"], ckpt["global_step"]
        print(f"resuming at generation {gen} (step {global_step})", flush=True)
    else:
        export_weights(gridcnn.zero_model(g), run_dir / "gen0.bin")

    while True:
        weights = run_dir / f"gen{gen}.bin"
        if model is not None and not weights.exists():
            export_weights(model, weights)
        if gen > 0 and gen not in {r["gen"] for r in read_jsonl(gen_log)}:
            print(f"[gen {gen}] checkpoint without a log line: running its gate", flush=True)
            finish_generation(cfg, run_dir, gen, {"gen": gen, "resumed": True}, time.perf_counter())
        if gen >= total:
            break

        row: dict[str, Any] = {"gen": gen + 1}
        started = time.perf_counter()

        shard = run_dir / "shards" / f"gen{gen}.bin"
        if not shard.exists():
            print(f"[gen {gen}] self-play", flush=True)
            row["selfplay"] = run_selfplay(config_path, weights, shard, seed=1000 + gen)
        else:
            stats = shard.with_name(shard.name + ".stats.json")
            row["selfplay"] = json.loads(stats.read_text()) if stats.exists() else {"reused": True}
        row["selfplay_seconds"] = time.perf_counter() - started

        print(f"[gen {gen}] fit", flush=True)
        window, val, n_window = load_window(run_dir, gen, cfg, device)
        row["window_positions"] = n_window
        if model is None:
            model, opt, global_step, summary, attempts = fit_first_generation(
                g, cfg, window, val, device, gen, steps_log
            )
            row["init_attempts"] = attempts
        else:
            assert opt is not None
            row["val_before"] = evaluate(model, val, device)
            global_step, summary = fit(
                model,
                opt,
                window,
                val,
                tcfg,
                gen=gen,
                global_step=global_step,
                device=device,
                step_log=functools.partial(_log_step, steps_log, {}),
                stall_check=False,
            )
        row["train"] = summary
        row["val_after"] = summary["validation"][-1]
        gen += 1
        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "gen": gen,
            "global_step": global_step,
        }
        atomic_torch_save(ckpt, latest)
        export_weights(model, run_dir / f"gen{gen}.bin")
        del window
        torch.mps.empty_cache()
        finish_generation(cfg, run_dir, gen, row, started)


def finish_generation(
    cfg: dict[str, Any], run_dir: Path, gen: int, row: dict[str, Any], started: float
) -> None:
    """The progress gate for generation ``gen``'s weights, then its log line."""
    gate_started = time.perf_counter()
    print(f"[gen {gen}] progress gate", flush=True)
    gate = run_progress_gate(cfg, run_dir, gen, run_dir / f"gen{gen}.bin")
    row["gate"] = {
        "opponent": cfg["gate"]["opponent"],
        "games": gate["games"],
        "score": gate["score_a"],
        "wilson_lo": gate["wilson_lo"],
        "wilson_hi": gate["wilson_hi"],
        "wins": gate["a_wins"],
        "losses": gate["b_wins"],
        "cnn_ms_per_move": gate["a_ms_per_move"],
        "opponent_ms_per_move": gate["b_ms_per_move"],
    }
    row["gate_seconds"] = time.perf_counter() - gate_started
    row["wall_seconds"] = time.perf_counter() - started
    append_jsonl(run_dir / "log.jsonl", row)
    summary = row.get(
        "train", {"loss_first": float("nan"), "loss_last": float("nan"), "fit_seconds": 0.0}
    )
    print(
        f"[gen {gen}] gate {row['gate']['score']:.3f} [{row['gate']['wilson_lo']:.3f}, "
        f"{row['gate']['wilson_hi']:.3f}]  "
        f"loss {summary['loss_first']:.3f}->{summary['loss_last']:.3f}  "
        f"selfplay {row.get('selfplay_seconds', 0):.0f}s fit {summary['fit_seconds']:.0f}s "
        f"gate {row['gate_seconds']:.0f}s",
        flush=True,
    )


def fit_first_generation(
    g: gridcnn.Geometry,
    cfg: dict[str, Any],
    window: Data,
    val: Positions,
    device: torch.device,
    gen: int,
    steps_log: Path,
) -> tuple[gridcnn.GridCNN, torch.optim.Optimizer, int, dict[str, Any], list[dict[str, Any]]]:
    """Generation 1's fit from a fresh init; some inits are dead (constant output), so a fit whose
    value predictions are still constant at ``stall_check_step`` is abandoned and retried at the
    next seed."""
    tcfg = cfg["train"]
    attempts: list[dict[str, Any]] = []
    for attempt in range(tcfg["max_init_retries"] + 1):
        seed = tcfg["seed"] + attempt
        model = new_model(g, seed, device)
        opt = make_optimizer(model, tcfg["learning_rate"])
        try:
            global_step, summary = fit(
                model,
                opt,
                window,
                val,
                tcfg,
                gen=gen,
                global_step=0,
                device=device,
                step_log=functools.partial(_log_step, steps_log, {"init_seed": seed}),
                stall_check=True,
            )
        except Stalled as e:
            attempts.append({"seed": seed, "stalled": str(e)})
            continue
        attempts.append({"seed": seed, "stalled": None})
        return model, opt, global_step, summary, attempts
    raise RuntimeError(f"every initialization stalled: {attempts}")


def curve_report(rows: list[dict[str, Any]], first: int = 5) -> dict[str, Any]:
    """The learning-curve requirement: from generation ``first`` on, the mean progress-gate score
    of the last quarter of generations must exceed that of the first quarter by more than the
    Wilson half-width of one gate."""
    gates = [(r["gen"], r["gate"]["score"], r["gate"]["games"]) for r in rows if r["gen"] >= first]
    if len(gates) < 8:
        return {"generations": len(gates), "rising": None}
    q = len(gates) // 4
    q1 = float(np.mean([s for _, s, _ in gates[:q]]))
    q4 = float(np.mean([s for _, s, _ in gates[-q:]]))
    n = gates[0][2]
    lo, hi = wilson(n / 2, n)
    half = (hi - lo) / 2
    return {
        "generations": len(gates),
        "first_quartile_mean": q1,
        "last_quartile_mean": q4,
        "wilson_half_width": half,
        "rising": q4 - q1 > half,
    }


def best_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest raw progress-gate score; ties go to the later generation."""
    return max(rows, key=lambda r: (r["gate"]["score"], r["gen"]))


def final_gate_config(
    cfg: dict[str, Any], run_dir: Path, weights: Path, *, name: str, openings: int,
    opening_plies: int, pairs: list[str],
) -> str:  # fmt: skip
    fin, gate, play = cfg["final"], cfg["gate"], cfg["play"]

    def cnn(agent: str, sims: int) -> str:
        return f"""
[[agent]]
name = "{agent}"
kind = "cnn-gumbel"
weights = "{weights}"
iterations = {sims}
considered_actions = {play["considered_actions"]}
value_scale = {play["value_scale"]}
max_visit_init = {play["max_visit_init"]}
chunk_size = {play["chunk_size"]}
"""

    def preset(agent: str, which: str, iterations: int) -> str:
        return f"""
[[agent]]
name = "{agent}"
kind = "preset"
preset = "{which}"
iterations = {iterations}
"""

    return (
        f"""size = {cfg["net"]["size"]}
openings = {openings}
opening_plies = {opening_plies}
max_plies = {gate["max_plies"]}
seed = {fin["seed"]}
workers = {fin["workers"]}
out = "{run_dir / "final-gate" / f"{name}.jsonl"}"
presets = "{gate["presets"]}"
pairs = {json.dumps(pairs)}
"""
        + cnn("cnn-100", 100)
        + cnn("cnn-32", 32)
        + preset("strong-100", "strong", 100)
        + preset("strong-1000", "strong", 1000)
        + preset("strong-10000", "strong", 10000)
        + preset("tuned-100", fin["tuned_preset"], 100)
    )


MAIN_ROWS = ["cnn-100:tuned-100", "cnn-100:strong-100", "cnn-100:strong-1000", "cnn-32:strong-1000"]


def final_gate(
    config_path: Path, run_dir: Path, overrides: list[str] | None = None
) -> dict[str, Any]:
    """The pre-declared final gate (plan: edax5-northstar.md, "FINAL kill line"): the best
    generation by progress-gate score against the MCTS ladder, 200 paired games per row (100 for
    the anchor), seconds per move next to every score."""
    cfg = load_config(config_path, overrides)
    rows = read_jsonl(run_dir / "log.jsonl")
    best = best_generation(rows)
    weights = run_dir / f"gen{best['gen']}.bin"
    out_dir = run_dir / "final-gate"
    out_dir.mkdir(exist_ok=True)
    fin = cfg["final"]
    plans = [
        ("main", fin["openings"], 2, MAIN_ROWS),
        ("anchor", fin["anchor_openings"], 2, ["cnn-100:strong-10000"]),
        ("swap-probe", fin["openings"], 1, ["cnn-100:strong-1000"]),
    ]  # fmt: skip
    table: list[dict[str, Any]] = []
    for name, openings, opening_plies, pairs in plans:
        conf = out_dir / f"{name}.toml"
        conf.write_text(
            final_gate_config(
                cfg,
                run_dir,
                weights,
                name=name,
                openings=openings,
                opening_plies=opening_plies,
                pairs=pairs,
            )
        )
        (out_dir / f"{name}.jsonl").unlink(missing_ok=True)
        print(f"[final gate] {name}: {pairs}", flush=True)
        subprocess.run(
            [str(EXAMPLES / "gonnect_gate"), "--config", str(conf)],
            cwd=ROOT,
            env=rust_env(),
            check=True,
            capture_output=True,
        )
        for r in read_jsonl(out_dir / f"{name}.jsonl"):
            if r.get("type") == "pairing":
                table.append(
                    {
                        "set": name,
                        "row": f"{r['a']} vs {r['b']}",
                        "games": r["games"],
                        "score": r["score_a"],
                        "wilson_lo": r["wilson_lo"],
                        "wilson_hi": r["wilson_hi"],
                        "cnn_ms_per_move": r["a_ms_per_move"],
                        "opponent_ms_per_move": r["b_ms_per_move"],
                        "capped": r["capped"],
                    }
                )
    primary = next(t for t in table if t["row"] == "cnn-100 vs tuned-100")
    strong = next(t for t in table if t["row"] == "cnn-100 vs strong-1000" and t["set"] == "main")
    report = {
        "best_generation": best["gen"],
        "best_progress_gate": best["gate"]["score"],
        "curve": curve_report(rows),
        "rows": table,
        "kill_line_passed": primary["score"] > 0.5 and primary["wilson_lo"] > 0.5,
        "beats_strong_1000_at_100_sims": strong["score"] > 0.5 and strong["wilson_lo"] > 0.5,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def smoke_checks(config_path: Path, run_dir: Path) -> dict[str, Any]:
    """Plumbing checks on a finished run (no strength claim): the Rust/MLX forward equals the torch
    forward on real positions with the newest weights, an agent against itself scores exactly one
    half over both seats, the loss fell, and the checkpoint reloads to the exported weights."""
    cfg = load_config(config_path)
    g = geometry_of(cfg)
    latest = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    gen = latest["gen"]
    model = gridcnn.GridCNN(g)
    model.load_state_dict(latest["model"])
    model.eval()
    report: dict[str, Any] = {"generation": gen}

    _, records = read_shard(run_dir / "shards" / "gen0.bin")
    count = 64
    out = subprocess.run(
        [str(EXAMPLES / "gonnect_check"), "--weights", str(run_dir / f"gen{gen}.bin")]
        + ["--shard", str(run_dir / "shards" / "gen0.bin"), "--count", str(count)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )  # fmt: skip
    rust = json.loads(out.stdout)
    with torch.no_grad():
        value, logits = model(torch.from_numpy(decode_planes(records[:count], g.size)).float())
    report["rust_vs_torch_max_value_diff"] = float(
        np.abs(np.array(rust["values"]) - value.numpy()).max()
    )
    logit_diff = np.abs(np.array(rust["logits"]).reshape(count, -1) - logits.numpy())
    report["rust_vs_torch_max_logit_diff"] = float(logit_diff.max())
    report["value_spread"] = float(value.numpy().std())

    steps = [r for r in read_jsonl(run_dir / "steps.jsonl") if r["gen"] == 0 and "init_seed" in r]
    k = 50
    report["loss_first_50"] = float(np.mean([r["loss"] for r in steps[:k]]))
    report["loss_last_50"] = float(np.mean([r["loss"] for r in steps[-k:]]))

    rebuilt = gridcnn.GridCNN(g)
    rebuilt.load_state_dict(latest["model"])
    export_weights(rebuilt, run_dir / "reload-check.bin")
    report["checkpoint_reload_bit_exact"] = (run_dir / "reload-check.bin").read_bytes() == (
        run_dir / f"gen{gen}.bin"
    ).read_bytes()

    weights = run_dir / f"gen{gen}.bin"
    conf = run_dir / "gate" / "self-match.toml"
    text = gate_config_text(cfg, run_dir, gen, weights).replace(
        f"gen{gen}.jsonl", "self-match.jsonl"
    )
    twin = text[text.index('[[agent]]\nname = "cnn"') :].replace('name = "cnn"', 'name = "cnn-b"')
    conf.write_text(text + "\n" + twin)
    (run_dir / "gate" / "self-match.jsonl").unlink(missing_ok=True)
    subprocess.run(
        [str(EXAMPLES / "gonnect_gate"), "--config", str(conf), "--pair", "cnn:cnn-b"],
        cwd=ROOT,
        env=rust_env(),
        check=True,
        capture_output=True,
    )
    rows = read_jsonl(run_dir / "gate" / "self-match.jsonl")
    row = [r for r in rows if r.get("type") == "pairing"][-1]
    report["self_match"] = {"games": row["games"], "a_wins": row["a_wins"], "b_wins": row["b_wins"]}

    checks = {
        "rust_matches_torch": report["rust_vs_torch_max_value_diff"] < 1e-4
        and report["rust_vs_torch_max_logit_diff"] < 1e-3,
        "self_match_exactly_half": row["a_wins"] == row["b_wins"] and row["draws"] == 0,
        "loss_fell": report["loss_last_50"] < report["loss_first_50"],
        "checkpoint_reloads": report["checkpoint_reload_bit_exact"],
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    (run_dir / "smoke-report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "games/gonnect/cnn/az-7x7.toml"))
    p.add_argument("--set", action="append", default=[], help="section.key=value override")
    p.add_argument("--out-dir")
    p.add_argument("--generations", type=int)
    p.add_argument("--smoke-checks", action="store_true", help="plumbing checks on a finished run")
    p.add_argument("--final-gate", action="store_true", help="the pre-declared final gate")
    args = p.parse_args()
    if args.final_gate:
        run_dir = Path(args.out_dir)
        report = final_gate(
            Path(args.config), run_dir if run_dir.is_absolute() else ROOT / run_dir, args.set
        )
        print(json.dumps(report, indent=2))
        raise SystemExit(0)
    if args.smoke_checks:
        run_dir = Path(args.out_dir)
        report = smoke_checks(
            Path(args.config), run_dir if run_dir.is_absolute() else ROOT / run_dir
        )
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["pass"] else 1)
    run(Path(args.config), args.set, Path(args.out_dir) if args.out_dir else None, args.generations)


if __name__ == "__main__":
    main()
