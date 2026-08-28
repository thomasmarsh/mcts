"""Hyperparameter search configuration — loaded from YAML, overridable at the CLI."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def json_default(value: Any) -> Any:
    """Convert scalar-like values to built-in JSON types when needed."""
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if isinstance(scalar, (bool, int, float, str)) or scalar is None:
            return scalar
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_dumps(value: Any) -> str:
    """Serialize protocol values accepted by the game binary."""
    return json.dumps(value, default=json_default)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ResourcePolicy:
    min_pairs: int = 5
    max_pairs: int = 15


@dataclass
class RatingPolicy:
    sigma_stop: float | None = 2.0
    conservative_k: float = 3.0


@dataclass
class PruningPolicy:
    enabled: bool = False
    kind: str = "hyperband"
    reduction_factor: int = 3
    startup_trials: int = 100


@dataclass
class SamplerPolicy:
    kind: str = "tpe"
    startup_trials: int = 10


@dataclass
class OptimizerConfig:
    n_trials: int = 1000
    deterministic: bool = False
    n_workers: int | None = None  # None → cpu_count // 2
    seed: int = 42
    resource: ResourcePolicy = field(default_factory=ResourcePolicy)
    rating: RatingPolicy = field(default_factory=RatingPolicy)
    pruning: PruningPolicy = field(default_factory=PruningPolicy)
    sampler: SamplerPolicy = field(default_factory=SamplerPolicy)


@dataclass
class TargetConfig:
    binary: Path = Path("target/release/game-traffic-lights")
    rounds: int = 20
    # Baseline instance ids to evaluate each trial config against (the tuner's
    # `Scenario(instances=...)`). Must be provided explicitly for every run;
    # the binary's `tune describe` baselines are available choices, not a
    # default selection.
    baselines: list[str] = field(default_factory=list)
    # Extra instances backed by a raw discovered config rather than a named
    # preset -- id -> the exact JSON `tune eval --baseline-config` expects
    # (see `game_host::GameAdapter::tune_eval`'s `baseline_config`
    # parameter). Populated from repeated `--baseline-config <id>=<json>`
    # CLI flags; `train()` forwards a matching instance id as
    # `--baseline-config <json>` instead of `--baseline <id>`. Empty unless
    # a caller opts in.
    baseline_configs: dict[str, dict] = field(default_factory=dict)
    # Game-setup config (e.g. Druid's board size) pinning every trial in
    # this run to a non-default `GameAdapter::default_config()` -- an axis
    # tuner never searches over, unlike `parameters`/`conditions` (the
    # strategy search space). `None` means "use the game binary's own
    # default"; `train()` only forwards `--game-config` when this is set.
    # Settable via `--game-config <json>` or this same key in YAML.
    game_config: dict | None = None
    # Per-*run* search-effort ceiling forwarded as `--max-iterations` to
    # every `tune eval` call this run makes -- `mcts_tune::SearchBudget`'s
    # `max_iterations` on the Rust side. Deliberately a launch-time setting,
    # not a `parameters:` entry: it's how *much* compute a trial's candidate
    # (and, for a `baseline_config`-backed opponent, that opponent too) gets
    # to spend, not a hyperparameter the tuner should search over -- letting
    # the tuner tune this would just reward whichever trial got the biggest
    # budget, not the best hyperparameters at a fixed budget. `None` (the
    # default) means "use the game binary's own historical default"
    # (`mcts-tune`'s `MAX_ITER` constant). Settable via `--max-iterations
    # <n>` or this same key in YAML.
    max_iterations: int | None = None
    # Per-*run* search-effort ceiling forwarded as `--max-time-ms` instead of
    # `--max-iterations` -- the Rust side's `SearchBudget` accepts either a
    # fixed iteration count or a wall-clock budget per move (never both:
    # `game-host::run_tune_eval` rejects a command line that sets both
    # flags). Same launch-time-not-searched rationale as `max_iterations`
    # above. `None` (the default) means "use `max_iterations`, or the game
    # binary's own historical default if that's also unset." Settable via
    # `--override target.max_time_ms=<n>` or this same key in YAML.
    max_time_ms: int | None = None


@dataclass
class ParamDef:
    """Definition of one hyperparameter in the search space."""

    name: str
    type: str  # "float" | "int" | "categorical" | "bool" | "constant"
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
        (the available named opponent ids, e.g. ``["strong", "master"]``)
        is reported alongside `parameters`/`conditions` for the same reason.
        The caller must still explicitly provide ``target.baselines`` for a
        run.
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
                p["name"]: {k: v for k, v in p.items() if k != "name"} for p in info["parameters"]
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
        if not isinstance(opt, dict):
            raise ValueError("optimizer must be a mapping")
        if not isinstance(tgt, dict):
            raise ValueError("target must be a mapping")

        resource = cls._policy_mapping(opt, "resource", {"min_pairs", "max_pairs"})
        rating = cls._policy_mapping(opt, "rating", {"sigma_stop", "conservative_k"})
        pruning = cls._policy_mapping(
            opt,
            "pruning",
            {"enabled", "kind", "reduction_factor", "startup_trials"},
        )
        sampler = cls._policy_mapping(opt, "sampler", {"kind", "startup_trials"})
        if "eta" in opt:
            if "reduction_factor" in pruning:
                raise ValueError("optimizer.eta conflicts with optimizer.pruning.reduction_factor")
            pruning["reduction_factor"] = opt["eta"]

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
            elif typ == "bool":
                p.default = pd.get("default", False)
            elif typ == "constant":
                p.value = pd["value"]
            params.append(p)

        conds: list[CondDef] = []
        for c in raw.get("conditions", []):
            for parent, vals in c["if"].items():
                if isinstance(vals, bool):
                    vals = [str(vals).lower()]
                elif isinstance(vals, str):
                    vals = [vals]
                elif not isinstance(vals, list):
                    vals = [str(vals)]
                conds.append(CondDef(parent=parent, values=vals, children=c["then"]))

        cfg = SearchConfig(
            optimizer=OptimizerConfig(
                n_trials=opt.get("n_trials", 1000),
                deterministic=opt.get("deterministic", False),
                n_workers=opt.get("n_workers"),
                seed=opt.get("seed", 42),
                resource=ResourcePolicy(
                    min_pairs=resource.get("min_pairs", 5),
                    max_pairs=resource.get("max_pairs", 15),
                ),
                rating=RatingPolicy(
                    sigma_stop=rating.get("sigma_stop", 2.0),
                    conservative_k=rating.get("conservative_k", 3.0),
                ),
                pruning=PruningPolicy(
                    enabled=pruning.get("enabled", False),
                    kind=pruning.get("kind", "hyperband"),
                    reduction_factor=pruning.get("reduction_factor", 3),
                    startup_trials=pruning.get("startup_trials", 100),
                ),
                sampler=SamplerPolicy(
                    kind=sampler.get("kind", "tpe"),
                    startup_trials=sampler.get("startup_trials", 10),
                ),
            ),
            target=TargetConfig(
                binary=Path(tgt.get("binary", "target/release/game-traffic-lights")),
                rounds=tgt.get("rounds", 20),
                baselines=list(tgt.get("baselines", [])),
                game_config=tgt.get("game_config"),
                max_iterations=tgt.get("max_iterations"),
                max_time_ms=tgt.get("max_time_ms"),
            ),
            parameters=params,
            conditions=conds,
        )
        cfg.validate()
        return cfg

    @staticmethod
    def _policy_mapping(parent: dict, name: str, allowed: set[str]) -> dict:
        value = parent.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"optimizer.{name} must be a mapping")
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"optimizer.{name} has unsupported keys: {', '.join(sorted(unknown))}")
        return dict(value)

    def validate(self) -> None:
        """Reject invalid explicit policy values before a tuning run starts."""
        opt = self.optimizer
        resource = opt.resource
        rating = opt.rating
        pruning = opt.pruning
        sampler = opt.sampler

        self._positive_int(opt.n_trials, "optimizer.n_trials")
        if opt.n_workers is not None:
            self._positive_int(opt.n_workers, "optimizer.n_workers")
        self._positive_int(resource.min_pairs, "optimizer.resource.min_pairs")
        self._positive_int(resource.max_pairs, "optimizer.resource.max_pairs")
        if resource.min_pairs > resource.max_pairs:
            raise ValueError("optimizer.resource.min_pairs must not exceed max_pairs")
        if rating.sigma_stop is not None:
            self._positive_finite(rating.sigma_stop, "optimizer.rating.sigma_stop")
        self._positive_finite(rating.conservative_k, "optimizer.rating.conservative_k")
        if not isinstance(pruning.enabled, bool):
            raise ValueError("optimizer.pruning.enabled must be a boolean")
        if pruning.kind != "hyperband":
            raise ValueError("optimizer.pruning.kind must be 'hyperband'")
        self._integer_at_least(
            pruning.reduction_factor,
            2,
            "optimizer.pruning.reduction_factor",
        )
        self._nonnegative_int(
            pruning.startup_trials,
            "optimizer.pruning.startup_trials",
        )
        if sampler.kind != "tpe":
            raise ValueError("optimizer.sampler.kind must be 'tpe'")
        self._nonnegative_int(sampler.startup_trials, "optimizer.sampler.startup_trials")

    @staticmethod
    def _positive_int(value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    @staticmethod
    def _integer_at_least(value: Any, minimum: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")

    @staticmethod
    def _positive_finite(value: Any, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number")

    def with_source(self, path: Path) -> SearchConfig:
        self._source = path
        return self

    @staticmethod
    def _to_snake(name: str) -> str:
        """Convert kebab-case or lower-case to snake_case."""
        return name.replace("-", "_")
