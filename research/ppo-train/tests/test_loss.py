"""Loss invariants from the reference audit: ratio == 1 at the start, weight-0 rows are inert."""

import numpy as np
import torch
from conftest import tiny_cfg

from ppo_train.net import build_net, policy_value
from ppo_train.ppo import MinibatchRows, collect_rollout, ppo_loss
from ppo_train.train import Trainer


def _rows(env, net, n: int, seed: int) -> MinibatchRows:
    """Real Othello positions with a sampled legal action and a perturbed old log-prob."""
    rng = np.random.default_rng(seed)
    states = env.reset(n, 40, seed)
    obs, mask = env.observe(states)
    o, m = torch.from_numpy(obs), torch.from_numpy(mask)
    with torch.no_grad():
        logits, value = policy_value(net, o, m)
    action = torch.from_numpy((rng.random(mask.shape) * mask).argmax(-1))
    logp = torch.log_softmax(logits, -1).gather(1, action[:, None])[:, 0]
    return MinibatchRows(
        obs=o, mask=m, action=action,
        old_log_prob=logp + 0.1 * torch.randn(n),
        old_value=value, adv=torch.randn(n), target=torch.randn(n).clamp(-1, 1),
        weight=torch.ones(n),
    )


def _slice(rows: MinibatchRows, sl: slice) -> MinibatchRows:
    return MinibatchRows(*(getattr(rows, f)[sl] for f in MinibatchRows.__dataclass_fields__))


def _grads(net, rows, cfg, weight_sum):
    net.zero_grad()
    ppo_loss(net, rows, cfg, weight_sum)[0].backward()
    return [p.grad.clone() for p in net.parameters() if p.grad is not None]


def test_ratio_is_one_on_the_first_minibatch(othello):
    for arch in ("none", "bn"):
        cfg = tiny_cfg(net__arch=arch)
        tr = Trainer(cfg, othello, torch.device("cpu"))
        pool = othello.reset(cfg.reset.pool, cfg.reset.max_depth, 1)
        roll, _ = collect_rollout(
            tr.net, tr.opp_net, othello, tr.states, cfg.rollout, pool,
            np.zeros(64, np.int64), np.zeros(64, bool), tr.rng, tr.device,
        )
        n = 64
        rows = MinibatchRows(
            obs=torch.from_numpy(roll.obs[0].astype(np.float32)),
            mask=torch.from_numpy(roll.mask[0]),
            action=torch.from_numpy(roll.action[0]),
            old_log_prob=torch.from_numpy(roll.log_prob[0]),
            old_value=torch.from_numpy(roll.value[0]),
            adv=torch.randn(n), target=torch.zeros(n), weight=torch.ones(n),
        )
        tr.net.eval()  # BN uses running stats, exactly as during acting
        _, stats = ppo_loss(tr.net, rows, cfg.ppo, float(n))
        assert float(stats["abs_log_ratio_max"]) < 1e-4, arch
        assert float(stats["approx_kl"]) < 1e-8, arch
        assert float(stats["clip_frac"]) == 0.0, arch


def test_opponent_plies_contribute_zero_gradient(othello):
    cfg = tiny_cfg(net__arch="none")
    net = build_net(cfg.net)
    full = _rows(othello, net, 48, seed=3)
    keep = 32
    weights = torch.cat([torch.ones(keep), torch.zeros(48 - keep)])
    full.weight = weights
    only_learner = _slice(full, slice(0, keep))

    g_full = _grads(net, full, cfg.ppo, float(keep))
    g_learner = _grads(net, only_learner, cfg.ppo, float(keep))
    for a, b in zip(g_full, g_learner, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    full.weight = torch.zeros(48)
    for g in _grads(net, full, cfg.ppo, 1.0):
        assert float(g.abs().max()) == 0.0


def test_micro_batch_accumulation_equals_one_big_batch(othello):
    cfg = tiny_cfg(net__arch="none")
    net = build_net(cfg.net)
    rows = _rows(othello, net, 64, seed=5)
    whole = _grads(net, rows, cfg.ppo, 64.0)
    net.zero_grad()
    for s in range(0, 64, 16):
        part = _slice(rows, slice(s, s + 16))
        ppo_loss(net, part, cfg.ppo, 64.0)[0].backward()
    for a, p in zip(whole, [q for q in net.parameters() if q.grad is not None], strict=True):
        torch.testing.assert_close(a, p.grad, atol=1e-6, rtol=1e-5)
