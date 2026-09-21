# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Trainer and coordinator for the Gonnect CNN track (``games/gonnect/cnn/az-9x9.toml``).

One generation is: Gumbel self-play with the current weights (Rust, MLX) -> a fixed number of
batch-32 Adam steps on a sliding replay window, warm-started from the previous generation (torch,
MPS) -> the progress gate, the new net against the net ``gate.lag`` generations ago and against the
current champion (Rust, MLX). Progress is measured only by nets playing nets (``yardstick.py``);
no reference engine, MCTS ladder or solver takes part, except the informational ``--sanity-gate``
at the end of a run, which nothing reads. Everything a generation produces is written before the
next starts, so a killed run resumes from ``latest.pt``.

Files under the run directory: ``gen<N>.bin`` (exported weights, ``crates/grid-cnn`` format),
``shards/gen<N>.bin`` (self-play), ``latest.pt`` (model + optimizer of the newest generation),
``steps.jsonl`` (one line per optimizer step, written as it happens), ``log.jsonl`` and
``diagnostics.jsonl`` (one line per generation), ``gate/gen<N>.jsonl`` (the progress gate's games),
``ratings/`` (``--round-robin``), ``verdict.json`` (``--verdict``), ``sanity/`` (``--sanity-gate``).
"""

from __future__ import annotations

import argparse
import functools
import json
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

from az_train import gridcnn, yardstick
from az_train.gonnect_diagnostics import diagnostics
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


def _toml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    return repr(v)


def dump_toml(cfg: dict[str, Any], prefix: str = "") -> str:
    """The config as TOML text (tables, arrays of tables, scalars and scalar lists), so the Rust
    binaries read the same effective settings the coordinator runs with, ``--set`` overrides
    included."""
    lines = [f"{k} = {_toml_scalar(v)}" for k, v in cfg.items() if _is_scalar(v)]
    for k, v in cfg.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            lines += ["", f"[{name}]", dump_toml(v, name)]
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            for item in v:
                lines += ["", f"[[{name}]]", dump_toml(item, name)]
    return "\n".join(lines).strip("\n") + "\n"


def _is_scalar(v: Any) -> bool:
    return not isinstance(v, dict) and not (
        isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
    )


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


@torch.no_grad()
def predict(
    model: gridcnn.GridCNN, positions: Positions, device: torch.device, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """``(values (N,), logits (N, actions))`` of the net in eval mode, no augmentation."""
    was_training = model.training
    model.eval()
    values, logits = [], []
    for start in range(0, len(positions), chunk):
        part = Data(positions.slice(start, start + chunk), device)
        v, lg = model(part.planes.float())
        values.append(v.cpu().numpy())
        logits.append(lg.cpu().numpy())
    model.train(was_training)
    return np.concatenate(values), np.concatenate(logits)


def diagnose(
    model: gridcnn.GridCNN,
    shard: Path,
    stats: dict[str, Any],
    held: Positions,
    device: torch.device,
) -> dict[str, Any]:
    """Oracle-free diagnostics of one generation: its self-play shard, and the fitted net's
    predictions on the shard's held-out games (see ``gonnect_diagnostics``)."""
    _, positions = load_positions(shard)
    values, logits = predict(model, held, device)
    return diagnostics(positions, stats, held, values, logits)


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


def agent_stanza(name: str, weights: Path, simulations: int, play: dict[str, Any]) -> str:
    return f"""
[[agent]]
name = "{name}"
kind = "cnn-gumbel"
weights = "{weights}"
iterations = {simulations}
considered_actions = {play["considered_actions"]}
value_scale = {play["value_scale"]}
max_visit_init = {play["max_visit_init"]}
chunk_size = {play["chunk_size"]}
"""


def match_config_text(
    cfg: dict[str, Any],
    *,
    out: Path,
    agents: list[tuple[str, Path, int]],
    pairs: list[tuple[str, str]],
    openings: int,
    opening_plies: int,
    seed: int,
    workers: int,
    max_plies: int,
) -> str:
    """A ``gonnect_gate`` config: net-vs-net pairings of ``(name, weights, simulations)`` agents,
    each opening played from both seats, openings random legal plies from the empty board."""
    head = f"""size = {cfg["net"]["size"]}
openings = {openings}
opening_plies = {opening_plies}
max_plies = {max_plies}
seed = {seed}
workers = {workers}
out = "{out}"
pairs = {json.dumps([f"{a}:{b}" for a, b in pairs])}
"""
    return head + "".join(agent_stanza(n, w, sims, cfg["play"]) for n, w, sims in agents)


def run_gate_binary(
    conf: Path, text: str, out: Path, *, resume: bool = False, extra: list[str] | None = None
) -> list[dict[str, Any]]:
    """Write ``conf``, run ``gonnect_gate`` on it and return the pairing summary rows of ``out``
    (which is cleared first unless ``resume``)."""
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(text)
    if not resume:
        out.unlink(missing_ok=True)
    cmd = [str(EXAMPLES / "gonnect_gate"), "--config", str(conf)] + (extra or [])
    done = subprocess.run(
        cmd + (["--resume"] if resume else []),
        cwd=ROOT,
        env=rust_env(),
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"gonnect_gate failed ({done.returncode}): {done.stderr[-2000:]}")
    return [r for r in read_jsonl(out) if r.get("type") == "pairing"]


def pairing_summary(row: dict[str, Any], opponent_gen: int) -> dict[str, Any]:
    return {
        "opponent_gen": opponent_gen,
        "games": row["games"],
        "score": row["score_a"],
        "wilson_lo": row["wilson_lo"],
        "wilson_hi": row["wilson_hi"],
        "wins": row["a_wins"],
        "losses": row["b_wins"],
        "draws": row["draws"],
        "capped": row["capped"],
        "ms_per_move": row["a_ms_per_move"],
        "opponent_ms_per_move": row["b_ms_per_move"],
    }


def current_champion(rows: list[dict[str, Any]]) -> int:
    """The champion after the last gated generation (generation 0, the zero net, before any)."""
    for r in reversed(rows):
        if "gate" in r:
            return r["gate"]["champion"]
    return 0


def run_progress_gate(
    cfg: dict[str, Any], run_dir: Path, gen: int, champion: int
) -> dict[str, Any]:
    """Generation ``gen`` against the net ``gate.lag`` generations earlier (generation 0 while
    ``gen <= lag``) and against the champion, on one shared opening set. The new net is promoted
    when its score against the champion reaches ``gate.promote_score``. When the two opponents
    are the same net there is one match."""
    gate = cfg["gate"]
    lag_gen = max(0, gen - gate["lag"])
    opponents = [lag_gen] + ([champion] if champion != lag_gen else [])
    names = {g: f"gen{g}" for g in [gen, *opponents]}
    out = run_dir / "gate" / f"gen{gen}.jsonl"
    text = match_config_text(
        cfg,
        out=out,
        agents=[(n, run_dir / f"gen{g}.bin", gate["simulations"]) for g, n in names.items()],
        pairs=[(names[gen], names[o]) for o in opponents],
        openings=gate["games"] // 2,
        opening_plies=gate["opening_plies"],
        seed=gate["seed"],
        workers=gate["workers"],
        max_plies=gate["max_plies"],
    )
    rows = {r["b"]: r for r in run_gate_binary(run_dir / "gate" / f"gen{gen}.toml", text, out)}
    lag = pairing_summary(rows[names[lag_gen]], lag_gen)
    best = pairing_summary(rows[names[champion]], champion)
    promoted = best["score"] >= gate["promote_score"]
    return {
        "lag": lag,
        "best": best,
        "champion_before": champion,
        "promoted": promoted,
        "champion": gen if promoted else champion,
    }


# ------------------------------------------------------------------------------ coordinator


wilson = yardstick.wilson


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
    effective = run_dir / "config.effective.toml"
    effective.write_text(dump_toml(cfg))

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
            row["selfplay"] = run_selfplay(effective, weights, shard, seed=1000 + gen)
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
        row["diagnostics"] = diagnose(model, shard, row["selfplay"], val, device)
        append_jsonl(run_dir / "diagnostics.jsonl", {"gen": gen + 1, **row["diagnostics"]})
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
        if gen == cfg["rules"]["early_gen"] and check_early_kill(cfg, run_dir, total):
            raise SystemExit(3)


def finish_generation(
    cfg: dict[str, Any], run_dir: Path, gen: int, row: dict[str, Any], started: float
) -> None:
    """The progress gate for generation ``gen``'s weights (every ``gate.every`` generations),
    then its log line."""
    gate_started = time.perf_counter()
    if gen % cfg["gate"].get("every", 1) == 0:
        print(f"[gen {gen}] progress gate", flush=True)
        row["gate"] = run_progress_gate(
            cfg, run_dir, gen, current_champion(read_jsonl(run_dir / "log.jsonl"))
        )
    row["gate_seconds"] = time.perf_counter() - gate_started
    row["wall_seconds"] = time.perf_counter() - started
    append_jsonl(run_dir / "log.jsonl", row)
    summary = row.get(
        "train", {"loss_first": float("nan"), "loss_last": float("nan"), "fit_seconds": 0.0}
    )
    gate = row.get("gate")
    verdict = (
        f"lag {gate['lag']['score']:.3f} [{gate['lag']['wilson_lo']:.3f}, "
        f"{gate['lag']['wilson_hi']:.3f}] vs gen{gate['lag']['opponent_gen']}, "
        f"best {gate['best']['score']:.3f} vs gen{gate['best']['opponent_gen']}"
        f"{' PROMOTED' if gate['promoted'] else ''}"
        if gate
        else "no gate"
    )
    print(
        f"[gen {gen}] {verdict}  "
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def evaluate_run(
    cfg: dict[str, Any], run_dir: Path, total: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The pre-registered rules applied to the run so far."""
    rows = read_jsonl(run_dir / "log.jsonl")
    report = run_dir / "ratings" / "report.json"
    ratings = json.loads(report.read_text()) if report.exists() else None
    verdict = yardstick.evaluate_rules(
        cfg["rules"],
        cfg["gate"]["lag"],
        total,
        rows,
        ratings,
        cfg["train"]["stall_check_min_value_std"],
    )
    return verdict, rows


def check_early_kill(cfg: dict[str, Any], run_dir: Path, total: int) -> bool:
    """Rule 1: at ``rules.early_gen`` the mean lag-gate score must be at least
    ``early_min_mean_lag_score``, otherwise the run stops and ``verdict.json`` says KILL_EARLY."""
    verdict, _ = evaluate_run(cfg, run_dir, total)
    early = verdict["clauses"]["early"]
    print(f"[early check] {early}", flush=True)
    if early["status"] == "fail":
        write_json(run_dir / "verdict.json", verdict)
        print("[early check] KILL_EARLY: the run is stopped", flush=True)
        return True
    return False


def checkpoint_generations(cfg: dict[str, Any], last: int) -> list[int]:
    """Round-robin checkpoints: generation 0, every ``round_robin.every``-th generation, and the
    last generation."""
    every = cfg["round_robin"]["every"]
    gens = list(range(0, last + 1, every))
    return gens if gens[-1] == last else [*gens, last]


def round_robin_pairs(gens: list[int], max_gap: int) -> list[tuple[str, str]]:
    """Every pair of checkpoints (later against earlier, so a positive score is progress), or only
    those at most ``max_gap`` checkpoints apart when ``max_gap`` is positive. The last generation
    against half of it is always played: rule 3 needs that pair."""
    half = gens[-1] // 2
    return [
        (f"gen{gens[j]}", f"gen{gens[i]}")
        for i in range(len(gens))
        for j in range(i + 1, len(gens))
        if max_gap <= 0 or j - i <= max_gap or (j == len(gens) - 1 and gens[i] == half)
    ]


def round_robin(
    config_path: Path, run_dir: Path, overrides: list[str], last: int | None = None
) -> dict[str, Any]:
    """The end-of-run rating curve: every pair of checkpoints plays ``round_robin.games`` paired
    games (resumable: finished pairings are skipped), then a Bradley-Terry fit, the pair table,
    intransitivity and the rating plot are written under ``ratings/``."""
    cfg = load_config(config_path, overrides)
    rr = cfg["round_robin"]
    rows = read_jsonl(run_dir / "log.jsonl")
    gens = checkpoint_generations(cfg, last if last is not None else max(r["gen"] for r in rows))
    pairs = round_robin_pairs(gens, rr["max_gap"])
    out = run_dir / "ratings" / "round-robin.jsonl"
    text = match_config_text(
        cfg,
        out=out,
        agents=[(f"gen{g}", run_dir / f"gen{g}.bin", rr["simulations"]) for g in gens],
        pairs=pairs,
        openings=rr["games"] // 2,
        opening_plies=rr["opening_plies"],
        seed=rr["seed"],
        workers=rr["workers"],
        max_plies=rr["max_plies"],
    )
    print(f"[round robin] {len(gens)} checkpoints, {len(pairs)} pairings x {rr['games']} games")
    pair_rows = run_gate_binary(run_dir / "ratings" / "round-robin.toml", text, out, resume=True)
    report = yardstick.rating_report(pair_rows, [f"gen{g}" for g in gens])
    write_json(run_dir / "ratings" / "report.json", report)
    (run_dir / "ratings" / "curve.svg").write_text(rating_plot_svg(report))
    return report


def rating_plot_svg(report: dict[str, Any]) -> str:
    """Rating against generation with 95 percent error bars, one self-contained SVG (light and
    dark)."""
    gens = [int(n[3:]) for n in report["players"]]
    elo, se = report["elo"], report["se"]
    w, h, left, right, top, bottom = 720, 360, 64, 20, 24, 44
    lo = min(e - 1.96 * s for e, s in zip(elo, se, strict=True))
    hi = max(e + 1.96 * s for e, s in zip(elo, se, strict=True))
    pad = 0.06 * max(hi - lo, 1.0)
    lo, hi = lo - pad, hi + pad
    gmax = max(max(gens), 1)

    def x(g: float) -> float:
        return left + (w - left - right) * g / gmax

    def y(e: float) -> float:
        return top + (h - top - bottom) * (1 - (e - lo) / (hi - lo))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="Bradley-Terry rating against generation">',
        "<style>:root{--bg:#fff;--ink:#101418;--muted:#6f7a86;--grid:#e3e7eb;--s1:#2a78d6}"
        "@media(prefers-color-scheme:dark){:root{--bg:#161b21;--ink:#f1f4f7;--muted:#8b96a2;"
        "--grid:#262d35;--s1:#3987e5}}"
        "text{fill:var(--muted);font:11px ui-monospace,Menlo,monospace}"
        ".grid{stroke:var(--grid)}.line{stroke:var(--s1);fill:none;stroke-width:2}"
        ".bar{stroke:var(--s1);stroke-width:1;opacity:.55}.dot{fill:var(--s1)}</style>",
        f'<rect width="{w}" height="{h}" fill="var(--bg)"/>',
    ]
    ticks = 5
    for i in range(ticks + 1):
        e = lo + (hi - lo) * i / ticks
        parts.append(
            f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y(e):.1f}" y2="{y(e):.1f}"/>'
        )
        parts.append(f'<text x="{left - 8}" y="{y(e) + 4:.1f}" text-anchor="end">{e:.0f}</text>')
    for g in gens:
        parts.append(f'<text x="{x(g):.1f}" y="{h - bottom + 16}" text-anchor="middle">{g}</text>')
    mid = (left + w - right) / 2
    parts.append(f'<text x="{mid}" y="{h - 6}" text-anchor="middle">generation</text>')
    parts.append(f'<text x="14" y="{top + 4}">Elo (generation 0 = 0)</text>')
    pts = " ".join(f"{x(g):.1f},{y(e):.1f}" for g, e in zip(gens, elo, strict=True))
    parts.append(f'<polyline class="line" points="{pts}"/>')
    for g, e, s in zip(gens, elo, se, strict=True):
        parts.append(
            f'<line class="bar" x1="{x(g):.1f}" x2="{x(g):.1f}" '
            f'y1="{y(e - 1.96 * s):.1f}" y2="{y(e + 1.96 * s):.1f}"/>'
        )
        parts.append(f'<circle class="dot" cx="{x(g):.1f}" cy="{y(e):.1f}" r="3.5"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def verdict_command(
    config_path: Path, run_dir: Path, overrides: list[str], total: int | None
) -> dict[str, Any]:
    cfg = load_config(config_path, overrides)
    verdict, _ = evaluate_run(cfg, run_dir, total or cfg["loop"]["generations"])
    write_json(run_dir / "verdict.json", verdict)
    return verdict


def sanity_gate(config_path: Path, run_dir: Path, overrides: list[str]) -> dict[str, Any]:
    """INFORMATIONAL ONLY. The champion against MCTS presets at fixed iterations, once, at the end
    of a run. Nothing in the coordinator, the rules or the verdict reads this file: it gates,
    promotes and stops nothing."""
    cfg = load_config(config_path, overrides)
    san = cfg["sanity"]
    champion = current_champion(read_jsonl(run_dir / "log.jsonl"))
    name = f"gen{champion}"
    out = run_dir / "sanity" / "sanity.jsonl"
    text = match_config_text(
        cfg,
        out=out,
        agents=[(name, run_dir / f"gen{champion}.bin", san["simulations"])],
        pairs=[(name, o["name"]) for o in san["opponent"]],
        openings=san["openings"],
        opening_plies=san["opening_plies"],
        seed=san["seed"],
        workers=san["workers"],
        max_plies=san["max_plies"],
    )
    text = text.replace(f'out = "{out}"', f'out = "{out}"\npresets = "{san["presets"]}"')
    for o in san["opponent"]:
        text += f"""
[[agent]]
name = "{o["name"]}"
kind = "preset"
preset = "{o["preset"]}"
iterations = {o["iterations"]}
"""
    rows = run_gate_binary(run_dir / "sanity" / "sanity.toml", text, out)
    report = {
        "informational": True,
        "champion": champion,
        "rows": [
            {"vs": r["b"], "games": r["games"], "score": r["score_a"],
             "wilson_lo": r["wilson_lo"], "wilson_hi": r["wilson_hi"],
             "ms_per_move": r["a_ms_per_move"], "opponent_ms_per_move": r["b_ms_per_move"]}
            for r in rows
        ],
    }  # fmt: skip
    write_json(run_dir / "sanity" / "report.json", report)
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
    gate = cfg["gate"]
    out = run_dir / "gate" / "self-match.jsonl"
    text = match_config_text(
        cfg,
        out=out,
        agents=[("cnn", weights, gate["simulations"]), ("cnn-b", weights, gate["simulations"])],
        pairs=[("cnn", "cnn-b")],
        openings=gate["games"] // 2,
        opening_plies=gate["opening_plies"],
        seed=gate["seed"],
        workers=gate["workers"],
        max_plies=gate["max_plies"],
    )
    row = run_gate_binary(run_dir / "gate" / "self-match.toml", text, out)[-1]
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
    p.add_argument("--config", default=str(ROOT / "games/gonnect/cnn/az-9x9.toml"))
    p.add_argument("--set", action="append", default=[], help="section.key=value override")
    p.add_argument("--out-dir")
    p.add_argument("--generations", type=int)
    p.add_argument("--smoke-checks", action="store_true", help="plumbing checks on a finished run")
    p.add_argument("--round-robin", action="store_true", help="checkpoint round robin + ratings")
    p.add_argument("--verdict", action="store_true", help="apply the pre-registered rules")
    p.add_argument("--sanity-gate", action="store_true", help="informational MCTS-preset check")
    args = p.parse_args()

    def run_dir() -> Path:
        d = Path(args.out_dir)
        return d if d.is_absolute() else ROOT / d

    config = Path(args.config)
    if args.round_robin:
        report = round_robin(config, run_dir(), args.set)
        print(json.dumps({k: report[k] for k in ("players", "elo", "se", "cycles")}, indent=2))
        raise SystemExit(0)
    if args.verdict:
        print(json.dumps(verdict_command(config, run_dir(), args.set, args.generations), indent=2))
        raise SystemExit(0)
    if args.sanity_gate:
        print(json.dumps(sanity_gate(config, run_dir(), args.set), indent=2))
        raise SystemExit(0)
    if args.smoke_checks:
        report = smoke_checks(config, run_dir())
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["pass"] else 1)
    run(config, args.set, Path(args.out_dir) if args.out_dir else None, args.generations)


if __name__ == "__main__":
    main()
