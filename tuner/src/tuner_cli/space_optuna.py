"""Sample one trial's parameter dict from the YAML/binary-driven search config.

Optuna has no `ConfigurationSpace`/`Condition` object -- a trial samples
parameters imperatively, one `trial.suggest_*` call at a time, and a param
conditioned on another simply isn't sampled unless its parent's active value
warrants it. This is the Optuna analogue of `space.py`'s `build_space`,
walking `cfg.parameters`/`cfg.conditions` (already parsed from YAML or
`tune describe`, not re-parsed here) in dependency order instead of building
a `ConfigurationSpace` graph.
"""

from __future__ import annotations

import optuna

from .config import CondDef, ParamDef, SearchConfig


def _suggest_one(trial: optuna.Trial, p: ParamDef):
    if p.type == "constant":
        return p.value
    if p.type == "float":
        lo, hi = p.bounds
        return trial.suggest_float(p.name, float(lo), float(hi))
    if p.type == "int":
        lo, hi = p.bounds
        return trial.suggest_int(p.name, int(lo), int(hi))
    if p.type == "categorical":
        return trial.suggest_categorical(p.name, p.choices)
    if p.type == "bool":
        return trial.suggest_categorical(p.name, [False, True])
    raise ValueError(f"Unknown parameter type '{p.type}' for '{p.name}'")


def suggest_config(trial: optuna.Trial, cfg: SearchConfig) -> dict:
    """Sample a full parameter dict for one trial, honoring `cfg.conditions`.

    A child listed in more than one `CondDef` (e.g. `c` gated both by a
    plain UCB-flavored `family` and by RAVE's own `rave_ucb`) is active if
    *any* of its conditions is satisfied -- same OR semantics as `space.py`'s
    `per_child` merge. Only active parameters are sampled; skipping the rest
    (rather than sampling and discarding) keeps Optuna's own per-parameter
    search stats meaningful.
    """
    by_name: dict[str, ParamDef] = {p.name: p for p in cfg.parameters}
    children_of: dict[str, list[CondDef]] = {}
    for cd in cfg.conditions:
        for child in cd.children:
            children_of.setdefault(child, []).append(cd)

    all_children = set(children_of)
    roots = [p.name for p in cfg.parameters if p.name not in all_children]

    active: dict[str, object] = {}
    for name in roots:
        active[name] = _suggest_one(trial, by_name[name])

    # Conditions form a shallow DAG (parent -> child, never a cycle) -- repeat
    # passes until no new parameter activates, rather than assuming one pass
    # covers every depth.
    changed = True
    while changed:
        changed = False
        for name, conds in children_of.items():
            if name in active:
                continue
            satisfied = any(cd.parent in active and active[cd.parent] in cd.values for cd in conds)
            if satisfied:
                active[name] = _suggest_one(trial, by_name[name])
                changed = True

    return active


def default_config(cfg: SearchConfig) -> dict:
    """The all-defaults parameter dict, honoring `cfg.conditions` like `suggest_config`.

    Same parent/child activation walk as `suggest_config`, but taking each
    active `ParamDef`'s `.default` instead of sampling -- used to seed the
    opponent pool's `"default"` anchor with a reasonable, deterministic
    opponent rather than a random point in the search space.
    """
    by_name: dict[str, ParamDef] = {p.name: p for p in cfg.parameters}

    def value_for(parameter: ParamDef):
        return parameter.value if parameter.type == "constant" else parameter.default

    children_of: dict[str, list[CondDef]] = {}
    for cd in cfg.conditions:
        for child in cd.children:
            children_of.setdefault(child, []).append(cd)

    all_children = set(children_of)
    roots = [p.name for p in cfg.parameters if p.name not in all_children]

    active: dict[str, object] = {}
    for name in roots:
        active[name] = value_for(by_name[name])

    changed = True
    while changed:
        changed = False
        for name, conds in children_of.items():
            if name in active:
                continue
            satisfied = any(cd.parent in active and active[cd.parent] in cd.values for cd in conds)
            if satisfied:
                active[name] = value_for(by_name[name])
                changed = True

    return active
