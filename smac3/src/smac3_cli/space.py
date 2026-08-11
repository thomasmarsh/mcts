"""Build a ConfigSpace ConfigurationSpace from the YAML-driven search config."""

from __future__ import annotations

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    Float,
    Integer,
    OrConjunction,
)

from .config import CondDef, ParamDef, SearchConfig


def build_space(cfg: SearchConfig) -> ConfigurationSpace:
    """Construct a ``ConfigurationSpace`` from the parameter definitions.

    * ``constant``-type parameters are added as active ``Constant`` values.
    * Conditional dependencies follow the ``conditions`` section of the config.
    """
    cs = ConfigurationSpace(seed=cfg.optimizer.seed)

    # -- hyperparameters ---------------------------------------------------
    hyperparams = []
    for p in cfg.parameters:
        if p.type == "constant":
            hp = Constant(p.name, p.value)
        elif p.type == "float":
            hp = Float(p.name, bounds := p.bounds, default=p.default or bounds[0])
        elif p.type == "int":
            lo, hi = p.bounds
            hp = Integer(p.name, (int(lo), int(hi)), default=int(p.default or lo))
        elif p.type == "categorical":
            hp = Categorical(
                p.name,
                p.choices,
                default=p.default or p.choices[0],
            )
        else:
            raise ValueError(f"Unknown parameter type '{p.type}' for '{p.name}'")
        hyperparams.append(hp)

    cs.add(hyperparams)

    # -- conditions --------------------------------------------------------
    # A child can be named by more than one `conditions` entry (e.g. "c" is
    # active both for a plain UCB-flavored family and for RAVE's own
    # rave_ucb-gated schedule) -- unrelated `if` blocks, not alternate values
    # of one parent. ConfigSpace only accepts a single Condition/Conjunction
    # object per child, so every individual (parent == value) requirement
    # across every entry naming a given child is collected first and then
    # OR'd together into one object, rather than calling `cs.add` once per
    # entry (which would raise on the second `add` for the same child).
    per_child: dict[str, list] = {}
    for cd in cfg.conditions:
        for child in cd.children:
            per_child.setdefault(child, []).extend(
                EqualsCondition(cs[child], cs[cd.parent], v) for v in cd.values
            )

    for child, atoms in per_child.items():
        cs.add(atoms[0] if len(atoms) == 1 else OrConjunction(*atoms))

    return cs