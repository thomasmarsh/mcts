import pytest

from ppo_train.config import DEFAULT_CONFIG, load_config, parse_override


def test_reference_config_mirrors_run_sweep():
    cfg = load_config(DEFAULT_CONFIG)
    assert (cfg.net.channels, cfg.net.blocks) == (128, 6)
    assert (cfg.rollout.num_envs, cfg.rollout.rollout_len) == (4096, 32)
    assert (cfg.ppo.epochs, cfg.ppo.minibatches) == (3, 8)
    assert (cfg.reset.max_depth, cfg.reset.pool) == (50, 8192)
    opp = cfg.opponent
    assert (opp.pool_frac, opp.ring_size, opp.push_every) == (0.5, 12, 100)


def test_overrides_must_name_existing_keys():
    with pytest.raises(ValueError):
        load_config(DEFAULT_CONFIG, {"ppo.no_such_key": 1})
    assert parse_override("ppo.lr=1e-4") == ("ppo.lr", 1e-4)


def test_micro_batch_must_divide_minibatch():
    with pytest.raises(ValueError):
        load_config(DEFAULT_CONFIG, {"ppo.micro_batch": 1000})
