"""The PPO run loop: one long process, incremental JSONL metrics, full-state resume checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ppo_train.config import DEFAULT_CONFIG, Config, load_config, parse_override
from ppo_train.env import OthelloEnv
from ppo_train.match import evaluate_vs_random
from ppo_train.net import build_net
from ppo_train.ppo import (
    Env,
    collect_rollout,
    compute_gae,
    explained_variance,
    opponent_slot,
    update,
)

Evaluator = Callable[[torch.nn.Module, np.random.Generator], dict[str, float]]
StateDict = dict[str, torch.Tensor]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _snapshot(net: torch.nn.Module) -> StateDict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


class Trainer:
    def __init__(
        self,
        cfg: Config,
        env: Env,
        device: torch.device,
        out_dir: Path | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.cfg, self.env, self.device, self.out_dir, self.evaluator = (
            cfg, env, device, out_dir, evaluator,
        )
        torch.manual_seed(cfg.run.seed)
        self.rng = np.random.default_rng(cfg.run.seed)
        self.net = build_net(cfg.net).to(device)
        self.opp_net = build_net(cfg.net).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo.lr, eps=cfg.ppo.adam_eps)
        first = _snapshot(self.net)
        self.ring: list[StateDict] = [first] * cfg.opponent.ring_size
        self.push_count = 0
        self.iteration = 0
        pool = env.reset(cfg.reset.pool, cfg.reset.max_depth, int(self.rng.integers(2**62)))
        self.states = pool[self.rng.integers(0, len(pool), cfg.rollout.num_envs)].copy()

    def run_iteration(self) -> dict[str, Any]:
        cfg, rng = self.cfg, self.rng
        n_env = cfg.rollout.num_envs
        t0 = time.perf_counter()
        pool = self.env.reset(cfg.reset.pool, cfg.reset.max_depth, int(rng.integers(2**62)))
        self.opp_net.load_state_dict(self.ring[opponent_slot(self.push_count, cfg.opponent, rng)])
        learner_seat = rng.integers(0, 2, n_env)
        is_pool_env = rng.random(n_env) < cfg.opponent.pool_frac

        roll, info = collect_rollout(
            self.net, self.opp_net, self.env, self.states, cfg.rollout, pool, learner_seat,
            is_pool_env, rng, self.device,
        )
        _sync(self.device)
        t1 = time.perf_counter()
        adv, target = compute_gae(
            roll.value, roll.reward, roll.done, roll.last_value, cfg.ppo.gamma, cfg.ppo.gae_lambda
        )
        learner = roll.weight > 0
        ev = {
            "explained_var_learner": explained_variance(roll.value[learner], target[learner]),
            "explained_var_opponent": explained_variance(roll.value[~learner], target[~learner]),
        }
        stats = update(self.net, self.opt, roll, adv, target, cfg.ppo, rng, self.device)
        _sync(self.device)
        t2 = time.perf_counter()

        self.iteration += 1
        if self.iteration % cfg.opponent.push_every == 0:
            self.ring[self.push_count % cfg.opponent.ring_size] = _snapshot(self.net)
            self.push_count += 1
        steps = n_env * cfg.rollout.rollout_len
        return {
            "iter": self.iteration,
            "env_steps": self.iteration * steps,
            **info,
            **stats,
            **ev,
            "sps": steps / (t2 - t0),
            "t_rollout": t1 - t0,
            "t_update": t2 - t1,
        }

    def save(self, path: Path) -> None:
        state = {
            "cfg": dataclasses.asdict(self.cfg),
            "net": _snapshot(self.net),
            "opt": self.opt.state_dict(),
            "ring": self.ring,
            "push_count": self.push_count,
            "iteration": self.iteration,
            "rng": self.rng.bit_generator.state,
            "states": self.states,
        }
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved = state["cfg"]
        for section in ("net", "rollout"):
            if saved[section] != dataclasses.asdict(getattr(self.cfg, section)):
                raise ValueError(f"resume checkpoint disagrees with config on [{section}]")
        self.net.load_state_dict(state["net"])
        self.opt.load_state_dict(state["opt"])
        self.ring = state["ring"]
        self.push_count = state["push_count"]
        self.iteration = state["iteration"]
        self.rng.bit_generator.state = state["rng"]
        self.states = state["states"]

    def run(self, resume: bool = False) -> None:
        assert self.out_dir is not None
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(self.cfg), indent=2))
        last = out / "last.pt"
        if resume and last.exists():
            self.load(last)
            print(f"resumed at iteration {self.iteration}")
        cfg = self.cfg
        t_start = time.time()
        with (out / "metrics.jsonl").open("a") as log:
            while self.iteration < cfg.run.iters:
                metrics = self.run_iteration()
                it = metrics["iter"]
                if self.evaluator and (
                    it == 1 or it % cfg.eval.every == 0 or it == cfg.run.iters
                ):
                    metrics.update(self.evaluator(self.net, self.rng))
                metrics["wall"] = time.time() - t_start
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                _print_line(metrics)
                if it % cfg.run.ckpt_every == 0 or it == cfg.run.iters:
                    self.save(last)
                    torch.save(_snapshot(self.net), out / f"net_{it:06d}.pt")


def _print_line(m: dict[str, Any]) -> None:
    vs = f" | vs_random {m['vs_random_score']:.3f}" if "vs_random_score" in m else ""
    print(
        f"it {m['iter']:5d} | steps {m['env_steps']:.2e} | sps {m['sps']:7.0f}"
        f" | ent {m['entropy']:.3f} | kl {m['approx_kl']:.4f} | clip {m['clip_frac']:.3f}"
        f" | ev {m['explained_var_learner']:.3f}{vs}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="section.key=value")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config, dict(parse_override(s) for s in args.set))
    device = pick_device(cfg.run.device)
    env = OthelloEnv()

    def evaluator(net: torch.nn.Module, rng: np.random.Generator) -> dict[str, float]:
        return evaluate_vs_random(env, net, device, cfg.eval.games, cfg.eval.opening_plies, rng)

    print(f"device={device} arch={cfg.net.arch} {cfg.net.channels}x{cfg.net.blocks}")
    Trainer(cfg, env, device, args.out, evaluator).run(args.resume)


if __name__ == "__main__":
    main()
