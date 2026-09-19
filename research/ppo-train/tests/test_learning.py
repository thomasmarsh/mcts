"""End-to-end plumbing: the full trainer must learn a scripted win-in-one game."""

import json

import numpy as np
import pytest
import torch
from conftest import WinInOneEnv, tiny_cfg

from ppo_train.ppo import policy_value
from ppo_train.train import Trainer


def _accuracy(tr: Trainer, env: WinInOneEnv) -> float:
    states = env.reset(512, 0, 999)
    obs, mask = env.observe(states)
    tr.net.eval()
    with torch.no_grad():
        logits, _ = policy_value(tr.net, torch.from_numpy(obs), torch.from_numpy(mask))
    return float((logits.argmax(-1).numpy() == env.target(states)).mean())


@pytest.mark.parametrize("arch", ["none", "bn"])
def test_learns_an_always_win_in_one_position(arch):
    env = WinInOneEnv()
    cfg = tiny_cfg(
        net__arch=arch, ppo__lr=3e-3, rollout__num_envs=256, rollout__rollout_len=2,
        ppo__micro_batch=256, opponent__pool_frac=0.0,
    )
    tr = Trainer(cfg, env, torch.device("cpu"))
    assert _accuracy(tr, env) < 0.5
    for _ in range(15):
        m = tr.run_iteration()
    assert _accuracy(tr, env) > 0.9
    assert m["decisive_rate"] == 1.0


def test_resume_restores_state_and_metrics_are_incremental(tmp_path):
    env = WinInOneEnv()
    cfg = tiny_cfg(run__iters=3, run__ckpt_every=1, opponent__pool_frac=0.5)
    a = Trainer(cfg, env, torch.device("cpu"), tmp_path)
    a.run()
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["iter"] for x in lines] == [1, 2, 3]

    b = Trainer(cfg, env, torch.device("cpu"), tmp_path)
    b.load(tmp_path / "last.pt")
    assert b.iteration == 3 and b.push_count == a.push_count
    for k, v in a.net.state_dict().items():
        torch.testing.assert_close(b.net.state_dict()[k], v)
    np.testing.assert_array_equal(b.states, a.states)
    assert a.rng.integers(2**62) == b.rng.integers(2**62)
