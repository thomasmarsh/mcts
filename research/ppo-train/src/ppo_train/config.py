"""Run configuration: every hyperparameter lives in a TOML file, none are defaulted in code."""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "othello-ref.toml"


@dataclass(frozen=True)
class RunCfg:
    seed: int
    iters: int
    device: str
    ckpt_every: int


@dataclass(frozen=True)
class NetCfg:
    arch: str
    blocks: int
    channels: int
    value_hidden: int


@dataclass(frozen=True)
class RolloutCfg:
    num_envs: int
    rollout_len: int
    infer_chunk: int


@dataclass(frozen=True)
class PpoCfg:
    epochs: int
    minibatches: int
    micro_batch: int
    lr: float
    adam_eps: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float


@dataclass(frozen=True)
class ResetCfg:
    max_depth: int
    pool: int


@dataclass(frozen=True)
class OpponentCfg:
    pool_frac: float
    ring_size: int
    push_every: int


@dataclass(frozen=True)
class EvalCfg:
    every: int
    games: int
    opening_plies: int


@dataclass(frozen=True)
class Config:
    run: RunCfg
    net: NetCfg
    rollout: RolloutCfg
    ppo: PpoCfg
    reset: ResetCfg
    opponent: OpponentCfg
    eval: EvalCfg

    def validate(self) -> None:
        if self.net.arch not in ("bn", "none"):
            raise ValueError(f"net.arch must be 'bn' or 'none', got {self.net.arch!r}")
        rows = self.rollout.num_envs * self.rollout.rollout_len
        if rows % self.ppo.minibatches:
            raise ValueError(f"{rows} rows do not split into {self.ppo.minibatches} minibatches")
        if (rows // self.ppo.minibatches) % self.ppo.micro_batch:
            raise ValueError("micro_batch must divide the minibatch size")
        if self.opponent.ring_size < 1 or self.opponent.push_every < 1:
            raise ValueError("opponent.ring_size and opponent.push_every must be >= 1")


_SECTIONS: dict[str, type] = {f.name: t for f, t in zip(dataclasses.fields(Config), (
    RunCfg, NetCfg, RolloutCfg, PpoCfg, ResetCfg, OpponentCfg, EvalCfg), strict=True)}


def _build(raw: Mapping[str, Any]) -> Config:
    extra = set(raw) - set(_SECTIONS)
    if extra:
        raise ValueError(f"unknown config sections: {sorted(extra)}")
    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        body = dict(raw[name])
        unknown = set(body) - {g.name for g in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown keys in [{name}]: {sorted(unknown)}")
        sections[name] = cls(**body)
    cfg = Config(**sections)
    cfg.validate()
    return cfg


def load_config(
    path: Path | str = DEFAULT_CONFIG, overrides: Mapping[str, Any] | None = None
) -> Config:
    """Load a TOML config; `overrides` maps 'section.key' to a replacement value."""
    with open(path, "rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if section not in raw or key not in raw[section]:
            raise ValueError(f"override {dotted!r} does not name an existing config key")
        raw[section][key] = value
    return _build(raw)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a 'section.key=value' CLI override; the value is a TOML literal."""
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"override must look like section.key=value, got {text!r}")
    return key.strip(), tomllib.loads(f"v = {value}")["v"]
