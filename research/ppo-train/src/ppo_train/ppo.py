"""PPO core: rollout, negamax GAE, weighted clipped loss and the micro-batched update.

Semantics port `../selfplay/pgx4/train.py`. Value is always from the mover's perspective, so the
bootstrap enters with a flipped sign; frozen-opponent plies keep their place in the GAE chain but
carry loss weight 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from ppo_train.config import OpponentCfg, PpoCfg, RolloutCfg
from ppo_train.net import NEG_INF, policy_value


class Env(Protocol):
    def reset(self, n: int, max_depth: int, seed: int) -> np.ndarray: ...
    def observe(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def step(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def mover(self, states: np.ndarray) -> np.ndarray: ...


@dataclass
class Rollout:
    """Time-major `(T, E, ...)` arrays."""

    obs: np.ndarray  # uint8 (T, E, 2, 8, 8)
    mask: np.ndarray  # bool (T, E, 65)
    action: np.ndarray  # int64 (T, E)
    log_prob: np.ndarray  # float32, under the current policy
    value: np.ndarray  # float32, mover's perspective
    reward: np.ndarray  # float32, for the player who just moved
    done: np.ndarray  # float32 0/1
    weight: np.ndarray  # float32, 1 learner ply / 0 frozen-opponent ply
    last_value: np.ndarray  # float32 (E,), value of the state after the final step


@torch.no_grad()
def infer(
    net: torch.nn.Module, obs: np.ndarray, mask: np.ndarray, chunk: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """No-grad masked logits `(n, 65)` and values `(n,)` as numpy; the net must be in eval mode."""
    logits, values = [], []
    for s in range(0, len(obs), chunk):
        o = torch.from_numpy(np.ascontiguousarray(obs[s : s + chunk])).to(device, torch.float32)
        m = torch.from_numpy(np.ascontiguousarray(mask[s : s + chunk])).to(device)
        lg, v = policy_value(net, o, m)  # pyright: ignore[reportArgumentType]
        logits.append(lg.cpu().numpy())
        values.append(v.cpu().numpy())
    return np.concatenate(logits), np.concatenate(values)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def sample(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Categorical sample per row via Gumbel-max; masked entries (-1e9) are never drawn."""
    noise = rng.gumbel(size=logits.shape).astype(np.float32)
    return (logits + noise).argmax(axis=-1)


def collect_rollout(
    net: torch.nn.Module,
    opp_net: torch.nn.Module,
    env: Env,
    states: np.ndarray,
    cfg: RolloutCfg,
    reset_pool: np.ndarray,
    learner_seat: np.ndarray,
    is_pool_env: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[Rollout, dict[str, float]]:
    """Act `rollout_len` steps in every env, mutating `states` in place.

    Envs flagged `is_pool_env` play their non-learner seat with the frozen `opp_net`; finished
    envs are replaced by a random draw from `reset_pool`.
    """
    net.eval()
    opp_net.eval()
    t_len, n_env = cfg.rollout_len, len(states)
    obs_buf = np.zeros((t_len, n_env, 2, 8, 8), dtype=np.uint8)
    mask_buf = np.zeros((t_len, n_env, 65), dtype=np.bool_)
    action_buf = np.zeros((t_len, n_env), dtype=np.int64)
    logp_buf = np.zeros((t_len, n_env), dtype=np.float32)
    value_buf = np.zeros((t_len, n_env), dtype=np.float32)
    reward_buf = np.zeros((t_len, n_env), dtype=np.float32)
    done_buf = np.zeros((t_len, n_env), dtype=np.float32)
    weight_buf = np.zeros((t_len, n_env), dtype=np.float32)
    episodes = decisive = 0

    for t in range(t_len):
        obs, mask = env.observe(states)
        logits, value = infer(net, obs, mask, cfg.infer_chunk, device)
        action = sample(logits, rng)
        logp = log_softmax(logits)[np.arange(n_env), action]

        use_opp = is_pool_env & (env.mover(states) != learner_seat)
        opp_idx = np.flatnonzero(use_opp)
        if opp_idx.size:
            opp_logits, _ = infer(opp_net, obs[opp_idx], mask[opp_idx], cfg.infer_chunk, device)
            action[opp_idx] = sample(opp_logits, rng)

        reward, done = env.step(states, action.astype(np.uint8))
        finished = np.flatnonzero(done)
        if finished.size:
            states[finished] = reset_pool[rng.integers(0, len(reset_pool), finished.size)]
            episodes += finished.size
            decisive += int((reward[finished] != 0).sum())

        obs_buf[t] = obs.astype(np.uint8)
        mask_buf[t] = mask
        action_buf[t] = action
        logp_buf[t] = logp
        value_buf[t] = value
        reward_buf[t] = reward
        done_buf[t] = done
        weight_buf[t] = np.where(use_opp, 0.0, 1.0)

    obs, mask = env.observe(states)
    _, last_value = infer(net, obs, mask, cfg.infer_chunk, device)
    roll = Rollout(
        obs_buf, mask_buf, action_buf, logp_buf, value_buf, reward_buf, done_buf, weight_buf,
        last_value.astype(np.float32),
    )
    info = {
        "episodes": float(episodes),
        "decisive_rate": decisive / max(episodes, 1),
        "draw_rate": (episodes - decisive) / max(episodes, 1),
    }
    return roll, info


def compute_gae(
    value: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Negamax GAE over `(T, E)` arrays: the next state belongs to the opponent, so its value and
    the advantage chain enter with a flipped sign; a terminal cuts both."""
    adv = np.zeros_like(value)
    gae = np.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(len(value))):
        not_done = 1.0 - done[t]
        delta = reward[t] + gamma * (-next_value) * not_done - value[t]
        gae = delta + gamma * lam * (-1.0) * not_done * gae
        adv[t] = gae
        next_value = value[t]
    return adv, adv + value


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    if target.size < 2 or target.var() < 1e-12:
        return float("nan")
    return float(1.0 - (target - pred).var() / target.var())


@dataclass
class MinibatchRows:
    obs: torch.Tensor  # float (n, 2, 8, 8)
    mask: torch.Tensor  # bool (n, 65)
    action: torch.Tensor  # int64 (n,)
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    adv: torch.Tensor  # already normalised over the full minibatch
    target: torch.Tensor
    weight: torch.Tensor


def ppo_loss(
    net: torch.nn.Module, rows: MinibatchRows, cfg: PpoCfg, weight_sum: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted clipped PPO loss. Every term is divided by `weight_sum`, the weight total of the
    full minibatch, so summing micro-batch losses reproduces the full-minibatch loss exactly."""
    logits, value = policy_value(net, rows.obs, rows.mask)
    log_probs = torch.log_softmax(logits, dim=-1)
    log_prob = log_probs.gather(1, rows.action[:, None])[:, 0]
    log_ratio = log_prob - rows.old_log_prob
    ratio = log_ratio.exp()
    w = rows.weight
    pg1 = ratio * rows.adv
    pg2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * rows.adv
    pg_loss = -(torch.minimum(pg1, pg2) * w).sum() / weight_sum
    v_clipped = rows.old_value + (value - rows.old_value).clamp(-cfg.clip_eps, cfg.clip_eps)
    v_loss = (
        0.5
        * (torch.maximum((value - rows.target) ** 2, (v_clipped - rows.target) ** 2) * w).sum()
        / weight_sum
    )
    entropy = -((log_probs.exp() * log_probs.clamp_min(NEG_INF)).sum(-1) * w).sum() / weight_sum
    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
    with torch.no_grad():
        stats = {
            "pg_loss": pg_loss.detach(),
            "v_loss": v_loss.detach(),
            "entropy": entropy.detach(),
            "approx_kl": (((ratio - 1) - log_ratio) * w).sum() / weight_sum,
            "clip_frac": (((ratio - 1).abs() > cfg.clip_eps).float() * w).sum() / weight_sum,
            "abs_log_ratio_max": (log_ratio.abs() * (w > 0)).max(),
        }
    return loss, stats


def update(
    net: torch.nn.Module,
    opt: torch.optim.Optimizer,
    roll: Rollout,
    adv: np.ndarray,
    target: np.ndarray,
    cfg: PpoCfg,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    """`epochs` passes over shuffled minibatches, each accumulated over `micro_batch` chunks.

    The net is in train mode, so BatchNorm uses micro-batch statistics and updates its running
    averages. `first_mb_*` are measured on the very first minibatch, before any update.
    """
    n = roll.action.size
    minibatch = n // cfg.minibatches
    flat = {
        "obs": roll.obs.reshape(n, 2, 8, 8),
        "mask": roll.mask.reshape(n, 65),
        "action": roll.action.reshape(n),
        "log_prob": roll.log_prob.reshape(n),
        "value": roll.value.reshape(n),
        "weight": roll.weight.reshape(n),
        "adv": adv.reshape(n),
        "target": target.reshape(n),
    }
    net.train()
    names = ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "grad_norm")
    totals = dict.fromkeys(names, 0.0)
    first: dict[str, float] = {}
    n_mb = 0
    for _epoch in range(cfg.epochs):
        perm = rng.permutation(n)
        for m in range(cfg.minibatches):
            idx = perm[m * minibatch : (m + 1) * minibatch]
            adv_mb = flat["adv"][idx]
            adv_n = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
            weight_sum = max(float(flat["weight"][idx].sum()), 1.0)
            acc: dict[str, torch.Tensor] = {}
            opt.zero_grad(set_to_none=True)
            for s in range(0, minibatch, cfg.micro_batch):
                sel = idx[s : s + cfg.micro_batch]
                rows = MinibatchRows(
                    obs=torch.from_numpy(flat["obs"][sel]).to(device, torch.float32),
                    mask=torch.from_numpy(flat["mask"][sel]).to(device),
                    action=torch.from_numpy(flat["action"][sel]).to(device),
                    old_log_prob=torch.from_numpy(flat["log_prob"][sel]).to(device),
                    old_value=torch.from_numpy(flat["value"][sel]).to(device),
                    adv=torch.from_numpy(adv_n[s : s + cfg.micro_batch]).to(device),
                    target=torch.from_numpy(flat["target"][sel]).to(device),
                    weight=torch.from_numpy(flat["weight"][sel]).to(device),
                )
                loss, stats = ppo_loss(net, rows, cfg, weight_sum)
                loss.backward()
                for k, v in stats.items():
                    if k == "abs_log_ratio_max":
                        acc[k] = torch.maximum(acc[k], v) if k in acc else v
                    else:
                        acc[k] = acc[k] + v if k in acc else v
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            done_stats = {k: float(v) for k, v in acc.items()}
            for k in totals:
                totals[k] += float(grad_norm) if k == "grad_norm" else done_stats[k]
            if not first:
                first = {
                    "first_mb_kl": done_stats["approx_kl"],
                    "first_mb_abs_log_ratio_max": done_stats["abs_log_ratio_max"],
                }
            n_mb += 1
    net.eval()
    return {**{k: v / n_mb for k, v in totals.items()}, **first}


def opponent_slot(push_count: int, cfg: OpponentCfg, rng: np.random.Generator) -> int:
    """Ring slot of the frozen opponent for this iteration (uniform over slots pushed so far)."""
    return int(rng.integers(0, max(min(push_count, cfg.ring_size), 1)))
