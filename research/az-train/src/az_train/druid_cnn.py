# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Trainer and coordinator for the Druid CNN track (``games/druid/cnn/az-druid-5x5.toml``).

One generation is: Gumbel self-play with the current weights (Rust, MLX) -> a fixed number of
batch-32 Adam steps on a sliding replay window, warm-started from the previous generation (torch,
MPS) -> weights export -> the reference-free progress gate (the new net against the net ``lag``
generations earlier and against the current champion, paired openings, both seats; a promotion
when it beats the champion) -> a log line. Progress is only ever measured net against net.
``--round-robin`` rates checkpoints (Bradley-Terry, cycles), ``--verdict`` applies the
pre-registered ``[rules]``, and ``--smoke-checks`` proves the pieces agree (Rust and torch
forwards, checkpoint reload, a net against itself). The game-agnostic helpers (config, JSONL,
export, optimizer, match-config text, round-robin pairing, the rating plot) come from
``gonnect_cnn``; the network and the rating code are ``gridcnn`` and ``yardstick``, unchanged.

Files under the run directory: ``gen<N>.bin`` (exported weights, ``crates/grid-cnn`` format),
``shards/gen<N>.bin`` (self-play), ``latest.pt`` (model + optimizer of the newest generation),
``steps.jsonl`` (one line per optimizer step) and ``validation.jsonl`` (one line per validation
evaluation), both written as they happen, ``log.jsonl`` (one line per generation, with its gate),
``diagnostics.jsonl`` (one line per generation, see ``druid_diagnostics``), ``gate/gen<N>.jsonl``
(the gate's games), ``ratings/`` (``--round-robin``), ``verdict.json`` (``--verdict``),
``smoke-report.json`` (``--smoke-checks``).

Augmentation: Druid is not D4-invariant (Black joins top-bottom, White left-right), so only the
4 axis-preserving reflections are used, with lintel anchors remapped
(``druid_records.symmetry_sources``).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from az_train import gridcnn, yardstick
from az_train.druid_diagnostics import diagnostics
from az_train.druid_records import (
    IN_PLANES,
    SYMMETRIES,
    Positions,
    decode_planes,
    load_positions,
    num_actions,
    read_shard,
    symmetry_sources,
)
from az_train.gonnect_cnn import (
    EXAMPLES,
    ROOT,
    _pearson,
    append_jsonl,
    atomic_torch_save,
    checkpoint_generations,
    current_champion,
    dump_toml,
    export_weights,
    load_config,
    make_optimizer,
    match_config_text,
    new_model,
    pairing_summary,
    policy_loss,
    rating_plot_svg,
    read_jsonl,
    round_robin_pairs,
    rust_env,
    write_json,
)

CELL_PLANE = 3  # the legal-cell plane, read in the pending phase's own anchor convention


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
        head=n.get("head", gridcnn.HEAD_DENSE),
    )


def train_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("the Druid CNN trains on MPS; there is no CPU fallback")
    return torch.device("mps")


# ------------------------------------------------------------------------------------- training


class Data:
    """A replay window as device tensors. ``legal`` is stored as uint8 (gather is not defined for
    bool on every backend)."""

    def __init__(self, positions: Positions, device: torch.device) -> None:
        self.planes = torch.from_numpy(positions.planes).to(device)
        self.policy = torch.from_numpy(positions.policy).to(device)
        self.legal = torch.from_numpy(positions.legal.astype(np.uint8)).to(device)
        self.value = torch.from_numpy(positions.value).to(device)
        self.kind = torch.from_numpy(positions.kind).to(device)

    def __len__(self) -> int:
        return len(self.value)


def concat_positions(parts: list[Positions]) -> Positions:
    return Positions(*(np.concatenate([getattr(p, f) for p in parts]) for f in (
        "planes", "policy", "legal", "value", "game", "kind",
    )))  # fmt: skip


def split_validation(positions: Positions, games: int) -> tuple[Positions, Positions]:
    """Positions of the first ``games`` distinct game ids of a shard are held out."""
    held = np.isin(positions.game, np.unique(positions.game)[:games])
    return positions.take(~held), positions.take(held)


def augment(
    x: torch.Tensor,
    pi: torch.Tensor,
    legal: torch.Tensor,
    kind: torch.Tensor,
    src: torch.Tensor,
    size: int,
    sym: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One random reflection per sample (or the given ``sym``, for tests), applied consistently to
    the planes, the policy target and the legal mask. Board planes and sarsen placements permute
    as board cells; in a lintel's cell phase the legal-cell plane, the policy and the legal mask
    hold anchors and permute as anchors (``kind``). The four non-cell action entries do not move."""
    b, c = x.shape[0], x.shape[1]
    cells = size * size
    if sym is None:
        sym = torch.randint(0, SYMMETRIES, (b,), device=x.device)
    board = src[sym, 0]
    anchor = src[sym, kind]
    flat = x.reshape(b, c, cells)
    moved = flat.gather(2, board[:, None, :].expand(-1, c, -1))
    moved[:, CELL_PLANE] = flat[:, CELL_PLANE].gather(1, anchor)
    pi = torch.cat([pi[:, :cells].gather(1, anchor), pi[:, cells:]], dim=1)
    legal = torch.cat([legal[:, :cells].gather(1, anchor), legal[:, cells:]], dim=1)
    return moved.reshape(b, c, size, size), pi, legal


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
        v, lg = model(part.planes)
        values.append(v.cpu().numpy())
        logits.append(lg.cpu().numpy())
    model.train(was_training)
    return np.concatenate(values), np.concatenate(logits)


def evaluate(
    model: gridcnn.GridCNN, positions: Positions, device: torch.device
) -> dict[str, float]:
    """Fit metrics on positions with no augmentation, batch-norm in eval mode."""
    pred_v, logits = predict(model, positions, device)
    legal = positions.legal
    masked = np.where(legal, logits, -1e9)
    logp = masked - masked.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
    target = positions.value
    return {
        "value_std": float(pred_v.std()),
        "value_mse": float(np.mean((pred_v - target) ** 2)),
        "value_pearson": _pearson(pred_v, target),
        "value_sign_agreement": float(np.mean(np.sign(pred_v) == np.sign(target))),
        "policy_ce": float((-(positions.policy * logp).sum(axis=1)).mean()),
        "policy_top1": float(np.mean(masked.argmax(axis=1) == positions.policy.argmax(axis=1))),
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
    val_log: Callable[[dict[str, Any]], None],
    stall_check: bool,
) -> tuple[int, dict[str, Any]]:
    """``steps_per_generation`` Adam steps at ``batch_size`` on ``window``; returns the new global
    step and the generation's training summary."""
    size = model.geometry.size
    src = torch.from_numpy(symmetry_sources(size)).to(device)
    weights = model.weight_tensors()
    batch, steps, l2 = tcfg["batch_size"], tcfg["steps_per_generation"], tcfg["l2"]
    model.train()
    losses: list[float] = []
    val_trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = torch.randint(0, len(window), (batch,), device=device)
        x, pi, legal = augment(
            window.planes[idx].float(),
            window.policy[idx],
            window.legal[idx],
            window.kind[idx],
            src,
            size,
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
            metrics = {"step": step, **evaluate(model, val, device)}
            val_trace.append(metrics)
            val_log({"gen": gen, "global_step": global_step, **metrics})
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


def fit_first_generation(
    g: gridcnn.Geometry,
    cfg: dict[str, Any],
    window: Data,
    val: Positions,
    device: torch.device,
    gen: int,
    run_dir: Path,
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
                step_log=functools.partial(_log, run_dir / "steps.jsonl", {"init_seed": seed}),
                val_log=functools.partial(_log, run_dir / "validation.jsonl", {"init_seed": seed}),
                stall_check=True,
            )
        except Stalled as e:
            attempts.append({"seed": seed, "stalled": str(e)})
            continue
        attempts.append({"seed": seed, "stalled": None})
        return model, opt, global_step, summary, attempts
    raise RuntimeError(f"every initialization stalled: {attempts}")


def _log(path: Path, extra: dict[str, Any], row: dict[str, Any]) -> None:
    append_jsonl(path, {**row, **extra})


# ------------------------------------------------------------------------------ Rust binaries


def build_binaries() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "-p", "game-druid"]
        + ["--example", "druid_selfplay", "--example", "druid_check", "--example", "druid_gate"],
        cwd=ROOT,
        env=rust_env(),
        check=True,
    )


def run_selfplay(config: Path, weights: Path, shard: Path, seed: int) -> dict[str, Any]:
    cmd = [str(EXAMPLES / "druid_selfplay"), "--config", str(config), "--weights", str(weights)]
    out = subprocess.run(
        cmd + ["--out", str(shard), "--seed", str(seed)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------ the gate


def run_gate_binary(
    conf: Path, text: str, out: Path, *, resume: bool = False
) -> list[dict[str, Any]]:
    """Write ``conf``, run ``druid_gate`` on it and return the pairing summary rows of ``out``
    (which is cleared first unless ``resume``)."""
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(text)
    if not resume:
        out.unlink(missing_ok=True)
    cmd = [str(EXAMPLES / "druid_gate"), "--config", str(conf)] + (["--resume"] if resume else [])
    done = subprocess.run(cmd, cwd=ROOT, env=rust_env(), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"druid_gate failed ({done.returncode}): {done.stderr[-2000:]}")
    return [r for r in read_jsonl(out) if r.get("type") == "pairing"]


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


def diagnose(
    model: gridcnn.GridCNN,
    shard: Path,
    stats: dict[str, Any],
    held: Positions,
    device: torch.device,
    validation_games: int,
    train_sample: int = 2000,
) -> dict[str, Any]:
    """Oracle-free diagnostics of one generation: its self-play shard, the fitted net's predictions
    on the shard's held-out games, and on an evenly spaced sample of the shard's own training
    positions (see ``druid_diagnostics``)."""
    size, positions = load_positions(shard)
    values, logits = predict(model, held, device)
    train, _ = split_validation(positions, validation_games)
    if len(train) > train_sample:
        train = train.take(np.linspace(0, len(train) - 1, train_sample).astype(np.int64))
    train_values, _ = predict(model, train, device)
    return diagnostics(positions, stats, held, values, logits, size, train_values, train.value)


# ----------------------------------------------------------------------------------- run loop


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
        if tcfg.get("window_dtype", "float32") == "float16":
            # Halves the window's memory (the planes dominate it); every plane value is a 0/1
            # flag or a small fraction, and batches are widened back to float32 before use.
            train = dataclasses.replace(train, planes=train.planes.astype(np.float16))
        parts.append(train)
        if g == gen:
            val = held
    assert val is not None
    joined = concat_positions(parts)
    return Data(joined, device), val, len(joined)


def warm_started_model(
    g: gridcnn.Geometry, init_from: Path, mode: str, device: torch.device
) -> gridcnn.GridCNN:
    """A fresh net for ``g`` with the weights of the ``latest.pt`` at ``init_from`` (a run at any
    board size) copied in per ``gridcnn.warm_start``."""
    ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
    model = new_model(g, 1, device)
    copied = gridcnn.warm_start(model, ckpt["model"], mode)
    print(f"warm start ({mode}) from {init_from}: {len(copied)} entries copied", flush=True)
    return model


def run(
    config_path: Path,
    overrides: list[str],
    out_dir: Path | None,
    generations: int | None,
    init_from: Path | None = None,
    init_mode: str = "trunk",
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
    elif init_from is not None:
        # Generation 0 is the warm-started net (fresh optimizer), so generation 1's self-play
        # already uses it and the lag gate's generation-0 opponent is that same net.
        model = warm_started_model(g, init_from, init_mode, device)
        opt = make_optimizer(model, tcfg["learning_rate"])
        (run_dir / "warm-start.json").write_text(
            json.dumps({"init_from": str(init_from), "mode": init_mode})
        )
    else:
        export_weights(gridcnn.zero_model(g), run_dir / "gen0.bin")

    gen_log = run_dir / "log.jsonl"
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
                g, cfg, window, val, device, gen, run_dir
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
                step_log=functools.partial(_log, run_dir / "steps.jsonl", {}),
                val_log=functools.partial(_log, run_dir / "validation.jsonl", {}),
                stall_check=False,
            )
        row["train"] = summary
        row["val_after"] = summary["validation"][-1]
        row["diagnostics"] = diagnose(
            model, shard, row["selfplay"], val, device, tcfg["validation_games"]
        )
        append_jsonl(run_dir / "diagnostics.jsonl", {"gen": gen + 1, **row["diagnostics"]})
        gen += 1
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "gen": gen,
                "global_step": global_step,
            },
            run_dir / "latest.pt",
        )
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
    val = row.get("val_after", {"value_mse": float("nan"), "policy_ce": float("nan")})
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
        f"val value_mse {val['value_mse']:.3f} policy_ce {val['policy_ce']:.3f}  "
        f"selfplay {row.get('selfplay_seconds', 0):.0f}s fit {summary['fit_seconds']:.0f}s "
        f"gate {row['gate_seconds']:.0f}s",
        flush=True,
    )


# ------------------------------------------------------------------ round robin and the verdict


def druid_warnings(rows: list[dict[str, Any]]) -> list[str]:
    """Health warnings ``yardstick`` does not know about (it is game-agnostic and unchanged). They
    never gate: they are appended to the verdict's health clause for the reader."""
    recent = [r for r in rows[-10:] if "diagnostics" in r]
    warnings: list[str] = []
    if not recent:
        return warnings
    draw = float(np.mean([r["diagnostics"]["selfplay"]["draw_rate"] for r in recent]))
    if draw > 0.10:
        warnings.append(f"self-play draw rate {draw:.2f} above 0.10 over the last {len(recent)}")
    pear = [r["diagnostics"]["value"]["held"]["pearson"] for r in recent]
    pear = [p for p in pear if p is not None]
    if pear and float(np.mean(pear)) < 0.0:
        warnings.append(
            f"held-out value Pearson {np.mean(pear):.2f} below 0 over the last {len(recent)}: "
            "the value head is anti-correlated with outcomes on unseen games"
        )
    over = [r["diagnostics"]["value"]["held"]["mse_vs_constant"] for r in recent]
    over = [m for m in over if m is not None]
    if over and float(np.mean(over)) > 1.0:
        warnings.append(
            f"held-out value MSE {np.mean(over):.2f}x a constant predictor "
            f"over the last {len(recent)}"
        )
    return warnings


def evaluate_run(
    cfg: dict[str, Any], run_dir: Path, total: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The pre-registered rules applied to the run so far, plus Druid's non-gating warnings."""
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
    verdict["clauses"]["health"]["warnings"] += druid_warnings(rows)
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


def round_robin(
    config_path: Path, run_dir: Path, overrides: list[str], last: int | None = None
) -> dict[str, Any]:
    """The end-of-run rating curve: every pair of checkpoints (within ``round_robin.max_gap``)
    plays ``round_robin.games`` paired games (resumable: finished pairings are skipped), then a
    Bradley-Terry fit, the pair table, intransitivity and the rating plot are written under
    ``ratings/``."""
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


def verdict_command(
    config_path: Path, run_dir: Path, overrides: list[str], total: int | None
) -> dict[str, Any]:
    cfg = load_config(config_path, overrides)
    verdict, _ = evaluate_run(cfg, run_dir, total or cfg["loop"]["generations"])
    write_json(run_dir / "verdict.json", verdict)
    return verdict


# --------------------------------------------------------------------------------- smoke checks


def smoke_checks(config_path: Path, run_dir: Path, overrides: list[str]) -> dict[str, Any]:
    """Plumbing checks on a finished run (no strength claim): the Rust/MLX forward equals the torch
    forward on real positions with the newest weights, an agent against itself scores exactly one
    half over both seats, the loss fell, and the checkpoint reloads to the exported weights."""
    cfg = load_config(config_path, overrides)
    g = geometry_of(cfg)
    latest = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    gen = latest["gen"]
    model = gridcnn.GridCNN(g)
    model.load_state_dict(latest["model"])
    model.eval()
    report: dict[str, Any] = {"generation": gen}
    weights = run_dir / f"gen{gen}.bin"
    shard = run_dir / "shards" / "gen0.bin"

    _, records = read_shard(shard)
    count = 64
    out = subprocess.run(
        [str(EXAMPLES / "druid_check"), "forward", "--weights", str(weights)]
        + ["--shard", str(shard), "--count", str(count)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )  # fmt: skip
    rust = json.loads(out.stdout)
    with torch.no_grad():
        value, logits = model(torch.from_numpy(decode_planes(records[:count], g.size)))
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
        weights.read_bytes()
    )

    smoke = cfg["smoke"]
    out = subprocess.run(
        [str(EXAMPLES / "druid_check"), "self-match", "--config", str(config_path)]
        + ["--weights", str(weights), "--openings", str(smoke["openings"])]
        + ["--opening-plies", str(smoke["opening_plies"]), "--seed", str(smoke["seed"])]
        + ["--max-plies", str(smoke["max_plies"])],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )  # fmt: skip
    row = json.loads(out.stdout.strip().splitlines()[-1])
    report["self_match"] = row

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
    p.add_argument("--config", default=str(ROOT / "games/druid/cnn/az-druid-5x5.toml"))
    p.add_argument("--set", action="append", default=[], help="section.key=value override")
    p.add_argument("--out-dir")
    p.add_argument("--generations", type=int)
    p.add_argument("--init-from", help="latest.pt of a run (any board size) to warm-start from")
    p.add_argument("--init-mode", choices=("trunk", "full"), default="trunk")
    p.add_argument("--smoke-checks", action="store_true", help="plumbing checks on a finished run")
    p.add_argument("--round-robin", action="store_true", help="checkpoint round robin + ratings")
    p.add_argument("--verdict", action="store_true", help="apply the pre-registered rules")
    args = p.parse_args()
    config = Path(args.config)

    def run_dir() -> Path:
        d = Path(args.out_dir)
        return d if d.is_absolute() else ROOT / d

    if args.round_robin:
        report = round_robin(config, run_dir(), args.set)
        print(json.dumps({k: report[k] for k in ("players", "elo", "se", "cycles")}, indent=2))
        raise SystemExit(0)
    if args.verdict:
        print(json.dumps(verdict_command(config, run_dir(), args.set, args.generations), indent=2))
        raise SystemExit(0)
    if args.smoke_checks:
        report = smoke_checks(config, run_dir(), args.set)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["pass"] else 1)
    init_from = Path(args.init_from) if args.init_from else None
    out_dir = Path(args.out_dir) if args.out_dir else None
    run(config, args.set, out_dir, args.generations, init_from, args.init_mode)


if __name__ == "__main__":
    main()
