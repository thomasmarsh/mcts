# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Trainer and coordinator for the Druid CNN track (``games/druid/cnn/az-druid-5x5.toml``).

One generation is: Gumbel self-play with the current weights (Rust, MLX) -> a fixed number of
batch-32 Adam steps on a sliding replay window, warm-started from the previous generation (torch,
MPS) -> weights export and a log line. There is no gate yet: this is the plumbing, and
``--smoke-checks`` proves the pieces agree (Rust and torch forwards, checkpoint reload, a net
against itself). The game-agnostic helpers (config, JSONL, export, optimizer) come from
``gonnect_cnn``; the network and the rating code are ``gridcnn`` and ``yardstick``, unchanged.

Files under the run directory: ``gen<N>.bin`` (exported weights, ``crates/grid-cnn`` format),
``shards/gen<N>.bin`` (self-play), ``latest.pt`` (model + optimizer of the newest generation),
``steps.jsonl`` (one line per optimizer step) and ``validation.jsonl`` (one line per validation
evaluation), both written as they happen, ``log.jsonl`` (one line per generation),
``smoke-report.json`` (``--smoke-checks``).

Augmentation: Druid is not D4-invariant (Black joins top-bottom, White left-right), so only the
4 axis-preserving reflections are used, with lintel anchors remapped
(``druid_records.symmetry_sources``).
"""

from __future__ import annotations

import argparse
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

from az_train import gridcnn
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
    dump_toml,
    export_weights,
    load_config,
    make_optimizer,
    new_model,
    policy_loss,
    read_jsonl,
    rust_env,
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
            window.planes[idx], window.policy[idx], window.legal[idx], window.kind[idx], src, size
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
        + ["--example", "druid_selfplay", "--example", "druid_check"],
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

    while gen < total:
        weights = run_dir / f"gen{gen}.bin"
        if model is not None and not weights.exists():
            export_weights(model, weights)
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
        row["wall_seconds"] = time.perf_counter() - started
        append_jsonl(run_dir / "log.jsonl", row)
        print(
            f"[gen {gen}] loss {summary['loss_first']:.3f}->{summary['loss_last']:.3f}  "
            f"val value_mse {row['val_after']['value_mse']:.3f} "
            f"policy_ce {row['val_after']['policy_ce']:.3f}  "
            f"selfplay {row['selfplay_seconds']:.0f}s fit {summary['fit_seconds']:.0f}s",
            flush=True,
        )


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
    p.add_argument("--smoke-checks", action="store_true", help="plumbing checks on a finished run")
    args = p.parse_args()
    config = Path(args.config)
    if args.smoke_checks:
        d = Path(args.out_dir)
        report = smoke_checks(config, d if d.is_absolute() else ROOT / d, args.set)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["pass"] else 1)
    run(config, args.set, Path(args.out_dir) if args.out_dir else None, args.generations)


if __name__ == "__main__":
    main()
