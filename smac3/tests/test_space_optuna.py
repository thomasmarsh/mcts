"""`space_optuna.suggest_config` must only sample parameters whose conditions
are actually satisfied, and must sample every root parameter unconditionally.
"""

from __future__ import annotations

import optuna

from smac3_cli.config import SearchConfig
from smac3_cli.space_optuna import default_config, suggest_config

_SPACE_YAML = """
parameters:
  family:
    type: categorical
    choices: [rave, ucb1_pn]
    default: rave
  final_action:
    type: categorical
    choices: [max_avg, secure_child, robust_child]
    default: robust_child
  a:
    type: float
    bounds: [0, 10]
    default: 4.0
  schedule:
    type: categorical
    choices: [hand_selected, threshold]
    default: threshold
  k:
    type: int
    bounds: [0, 2000]
    default: 1000
  rave:
    type: int
    bounds: [0, 2000]
    default: 700
  contempt:
    type: categorical
    choices: ["off", "on"]
    default: "off"
  contempt_factor:
    type: float
    bounds: [-1, 1]
    default: 0.0
  fixed:
    type: constant
    value: always

conditions:
  - if: { family: rave }
    then: [final_action, schedule]
  - if: { final_action: secure_child }
    then: [a]
  - if: { schedule: hand_selected }
    then: [k]
  - if: { schedule: threshold }
    then: [rave]
  - if: { contempt: "on" }
    then: [contempt_factor]
"""


def _cfg() -> SearchConfig:
    import yaml

    raw = yaml.safe_load(_SPACE_YAML)
    return SearchConfig._from_dict(raw)


def test_only_satisfied_conditions_sample_their_children():
    cfg = _cfg()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    seen_families = set()
    for _ in range(50):
        trial = study.ask()
        params = suggest_config(trial, cfg)
        seen_families.add(params["family"])

        # `contempt` is a root (never a `then` target), always sampled.
        assert "contempt" in params
        assert ("contempt_factor" in params) == (params["contempt"] == "on")

        if params["family"] == "rave":
            assert "final_action" in params and "schedule" in params
            assert ("a" in params) == (params["final_action"] == "secure_child")
            assert ("k" in params) == (params["schedule"] == "hand_selected")
            assert ("rave" in params) == (params["schedule"] == "threshold")
        else:
            # ucb1_pn: none of `family`'s dependents should be sampled at all.
            for name in ("final_action", "schedule", "a", "k", "rave"):
                assert name not in params

        study.tell(trial, 0.0)

    assert seen_families == {"rave", "ucb1_pn"}


def test_default_config_takes_defaults_and_honors_conditions():
    cfg = _cfg()
    params = default_config(cfg)

    assert params["family"] == "rave"
    assert params["contempt"] == "off"
    assert params["fixed"] == "always"
    assert "contempt_factor" not in params  # gated by contempt == "on"

    # family defaults to "rave", so its dependents are active and defaulted.
    assert params["final_action"] == "robust_child"
    assert params["schedule"] == "threshold"
    assert "a" not in params  # gated by final_action == "secure_child"
    assert "k" not in params  # gated by schedule == "hand_selected"
    assert params["rave"] == 700  # gated by schedule == "threshold"
