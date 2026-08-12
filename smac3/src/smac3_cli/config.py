"""Hyperparameter search configuration — loaded from YAML, overridable at the CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    n_trials: int = 1000
    deterministic: bool = False
    n_workers: int | None = None  # None → cpu_count // 2
    seed: int = 42


@dataclass
class TargetConfig:
    binary: Path = Path("target/release/game-traffic-lights")
    rounds: int = 20
    # Baseline instance ids to evaluate each trial config against (SMAC3's
    # `Scenario(instances=...)`). Sourced from the binary's own `tune
    # describe` at launch time, same as `parameters`/`conditions` --  see
    # `SearchConfig.parameters_from_binary`'s docstring for why the binary,
    # not this dataclass's default, is the source of truth.
    baselines: list[str] = field(default_factory=list)


@dataclass
class ParamDef:
    """Definition of one hyperparameter in the search space."""

    name: str
    type: str  # "float" | "int" | "categorical" | "constant"
    bounds: tuple[float, float] | None = None  # float/int only
    choices: list[str] | None = None  # categorical only
    default: Any = None
    value: Any = None  # constant only


@dataclass
class CondDef:
    parent: str
    values: list[str]
    children: list[str]


@dataclass
class SearchConfig:
    """Top-level configuration."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    parameters: list[ParamDef] = field(default_factory=list)
    conditions: list[CondDef] = field(default_factory=list)

    # Path this config was loaded from (for resolving relative binary paths)
    _source: Path | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Load / merge
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> SearchConfig:
        """Load from a YAML file."""
        path = Path(path).expanduser().resolve(strict=True)
        with open(path) as f:
            raw: dict = yaml.safe_load(f)
        return cls._from_dict(raw).with_source(path)

    @classmethod
    def defaults(cls) -> SearchConfig:
        """Load the packaged default config."""
        pkg_root = Path(__file__).resolve().parent.parent.parent
        return cls.load(pkg_root / "config" / "default.yaml")

    def resolve_binary(self) -> Path:
        """Return the absolute path to the game binary.

        * Relative paths are resolved from the **current working directory**
          (not the config file), so the user can run from the project root.
        * Absolute paths are used as-is.
        """
        p = self.target.binary
        return p if p.is_absolute() else (Path.cwd() / p).resolve()

    @classmethod
    def parameters_from_binary(
        cls, binary: Path
    ) -> tuple[list[ParamDef], list[CondDef], list[str]]:
        """Query ``<binary> tune describe`` for its search-space schema.

        The binary is the single source of truth for the search space (what
        `mcts-tune`'s `strategy_tuner_info` actually builds), not the YAML
        config -- the two drifted apart once already (a family missing from
        a hand-maintained YAML list). `tune describe`'s JSON reports the
        same `type`/`bounds`/`choices`/`default`/`value` shape as the YAML
        `parameters:`/`conditions:` blocks, just as an array of
        ``{"name": ..., ...}`` objects instead of a name-keyed mapping, so
        it's reshaped into that mapping and run back through `_from_dict`
        rather than duplicating its field-extraction logic. ``baselines``
        (the list of opponent-instance ids, e.g. ``["strong", "master"]``)
        is reported alongside `parameters`/`conditions` for the same reason
        -- it's part of the binary's tuner metadata, not something to
        hand-maintain here.
        """
        result = subprocess.run(
            [str(binary), "tune", "describe"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        info = json.loads(result.stdout)
        raw = {
            "parameters": {
                p["name"]: {k: v for k, v in p.items() if k != "name"}
                for p in info["parameters"]
            },
            "conditions": info["conditions"],
        }
        parsed = cls._from_dict(raw)
        return parsed.parameters, parsed.conditions, list(info["baselines"])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @classmethod
    def _from_dict(cls, raw: dict) -> SearchConfig:
        opt = raw.get("optimizer", {})
        tgt = raw.get("target", {})

        params: list[ParamDef] = []
        for name, pd in raw.get("parameters", {}).items():
            typ = pd["type"]
            p = ParamDef(name=name, type=typ, default=pd.get("default"))
            if typ == "float":
                p.bounds = tuple(pd["bounds"])
            elif typ == "int":
                p.bounds = tuple(pd["bounds"])
            elif typ == "categorical":
                p.choices = list(pd["choices"])
                p.default = pd.get("default", p.choices[0])
            elif typ == "constant":
                p.value = pd["value"]
            params.append(p)

        conds: list[CondDef] = []
        for c in raw.get("conditions", []):
            for parent, vals in c["if"].items():
                if isinstance(vals, str):
                    vals = [vals]
                conds.append(CondDef(parent=parent, values=vals, children=c["then"]))

        return SearchConfig(
            optimizer=OptimizerConfig(
                n_trials=opt.get("n_trials", 1000),
                deterministic=opt.get("deterministic", False),
                n_workers=opt.get("n_workers"),
                seed=opt.get("seed", 42),
            ),
            target=TargetConfig(
                binary=Path(tgt.get("binary", "target/release/game-traffic-lights")),
                rounds=tgt.get("rounds", 20),
                baselines=list(tgt.get("baselines", [])),
            ),
            parameters=params,
            conditions=conds,
        )

    def with_source(self, path: Path) -> SearchConfig:
        self._source = path
        return self

    @staticmethod
    def _to_snake(name: str) -> str:
        """Convert kebab-case or lower-case to snake_case."""
        return name.replace("-", "_")