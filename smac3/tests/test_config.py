"""`SearchConfig.parameters_from_binary` must reproduce the search space that
used to live hand-maintained in `smac3/config/default.yaml`'s `parameters:`/
`conditions:` blocks (removed once the binary's own `tune describe` became
the single source of truth -- see `config.py`'s docstring for why).

This is a frozen snapshot of that removed YAML, not a spec: if `mcts-tune`'s
catalog changes, this snapshot goes stale by design and this test starts
failing -- exactly the drift-detection this test exists for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from ConfigSpace import Configuration, ConfigurationSpace

from smac3_cli.config import CondDef, OptimizerConfig, ParamDef, SearchConfig, TargetConfig
from smac3_cli.space import build_space

# The `parameters:`/`conditions:` blocks that lived in `default.yaml` prior
# to this session, verbatim (minus comments).
_LEGACY_YAML_SPACE = """
parameters:
  family:
    type: categorical
    choices:
      [ucb1, ucb1_dm, ucb1_mast, ucb1_nst, ucb1_progressive_history,
       ucb1_max_robust, amaf, amaf_mast, ucb1_tuned, ucb1_tuned_mast,
       ucb1_tuned_dm, ucb1_tuned_dm_mast, meta_mcts, rave]
    default: rave
  q_init:
    type: categorical
    choices: [Draw, Infinity, Loss, Parent, Win]
    default: Infinity
  final_action:
    type: categorical
    choices: [max_avg, secure_child, robust_child]
    default: robust_child
  a:
    type: float
    bounds: [0, 10]
    default: 4.0
  c:
    type: float
    bounds: [0, 3]
    default: 1.4142135623730951
  epsilon:
    type: float
    bounds: [0, 1]
    default: 0.1
  amaf_alpha:
    type: float
    bounds: [0, 1]
    default: 1.0
  ph_weight:
    type: float
    bounds: [0, 5]
    default: 1.0
  nst_backoff_threshold:
    type: int
    bounds: [0, 100]
    default: 5
  bias:
    type: float
    bounds: [0, 10]
    default: 0.00001
  k:
    type: int
    bounds: [0, 2000]
    default: 1000
  rave:
    type: int
    bounds: [0, 2000]
    default: 700
  schedule:
    type: categorical
    choices: [hand_selected, min_mse, threshold]
    default: threshold
  threshold:
    type: int
    bounds: [0, 2000]
    default: 700
  rave_ucb:
    type: categorical
    choices: [none, ucb1, tuned]
    default: tuned

conditions:
  - if:
      family: [ucb1, ucb1_dm, ucb1_mast, ucb1_nst, ucb1_progressive_history,
                amaf, amaf_mast, ucb1_tuned, ucb1_tuned_mast, ucb1_tuned_dm,
                ucb1_tuned_dm_mast, rave]
    then: [final_action]
  - if: { final_action: secure_child }
    then: [a]
  - if:
      family: [ucb1, ucb1_dm, ucb1_mast, ucb1_nst, ucb1_progressive_history,
                amaf, amaf_mast, ucb1_tuned, ucb1_tuned_mast, ucb1_tuned_dm,
                ucb1_tuned_dm_mast, ucb1_max_robust, meta_mcts]
    then: [c]
  - if:
      family: [ucb1_mast, ucb1_nst, amaf_mast, ucb1_tuned_dm_mast, rave]
    then: [epsilon]
  - if: { family: [amaf, amaf_mast] }
    then: [amaf_alpha]
  - if: { family: ucb1_progressive_history }
    then: [ph_weight]
  - if: { family: ucb1_nst }
    then: [nst_backoff_threshold]
  - if: { family: rave }
    then: [threshold, schedule, rave_ucb]
  - if: { schedule: hand_selected }
    then: [k]
  - if: { schedule: min_mse }
    then: [bias]
  - if: { schedule: threshold }
    then: [rave]
  - if: { rave_ucb: [ucb1, tuned] }
    then: [c]
"""


def _cfg(parameters: list[ParamDef], conditions: list[CondDef]) -> SearchConfig:
    return SearchConfig(
        optimizer=OptimizerConfig(seed=1234),
        target=TargetConfig(),
        parameters=parameters,
        conditions=conditions,
    )


@pytest.fixture(scope="module")
def legacy_space() -> ConfigurationSpace:
    raw = yaml.safe_load(_LEGACY_YAML_SPACE)
    cfg = SearchConfig._from_dict({"optimizer": {"seed": 1234}, **raw})
    return build_space(cfg)


@pytest.fixture(scope="module")
def binary_space(game_nim_binary: Path) -> ConfigurationSpace:
    parameters, conditions, _baselines = SearchConfig.parameters_from_binary(game_nim_binary)
    return build_space(_cfg(parameters, conditions))


def test_binary_sourced_space_matches_legacy_yaml_space(legacy_space, binary_space):
    assert binary_space == legacy_space


def test_parameters_from_binary_reports_baselines(game_nim_binary: Path):
    """`baselines` rides along with `parameters`/`conditions` in the same
    `tune describe` call -- nim has a single "strong" baseline preset, same
    as every tunable game except druid (which lists a second, "master")."""
    _parameters, _conditions, baselines = SearchConfig.parameters_from_binary(game_nim_binary)
    assert baselines == ["strong"]


def test_binary_sourced_space_round_trips_through_legacy_space(legacy_space, binary_space):
    """Sample many configs from the binary-sourced space and confirm every
    one's active-parameter set is also valid in the other space, rather than
    relying on `==` alone.
    """
    for config in binary_space.sample_configuration(200):
        active = dict(config)
        # Re-parsing an active-parameter dict as a Configuration in the
        # legacy space raises if any name/value/activation disagrees.
        Configuration(legacy_space, values=active)
