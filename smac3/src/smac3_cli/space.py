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
    for cd in cfg.conditions:
        if len(cd.values) == 1:
            for child in cd.children:
                cs.add(EqualsCondition(cs[child], cs[cd.parent], cd.values[0]))
        else:
            # Build one OrConjunction per child: param.active if parent IN {values}
            for child in cd.children:
                cs.add(
                    OrConjunction(
                        *(EqualsCondition(cs[child], cs[cd.parent], v) for v in cd.values)
                    )
                )

    return cs