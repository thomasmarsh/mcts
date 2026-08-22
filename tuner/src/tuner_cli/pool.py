"""The dynamic opponent pool -- OpenSkill-rated anchors matchmaking plays against.

Unlike the tuner's fixed named-baseline instances, this pool starts with just two
frozen anchors (`"default"`, `"random"`) and grows over the course of a run:
a finished trial's config becomes a new anchor whenever it's either a new
champion (higher `mu` than every existing anchor) or fills a new skill band
(more than `_NEW_BAND_DELTA_MU` away from the nearest anchor's `mu`). Anchors
never mutate after insertion -- a stable, replayable rung on the ladder every
future trial can be matched against, not a rating that drifts under later
play.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import SearchConfig, json_default
from .space_optuna import default_config
from .target import FLOOR_BASELINES

# A candidate more than this far (in `mu`) from every current anchor opens a
# new skill band, rather than being folded into its nearest neighbor's band.
_NEW_BAND_DELTA_MU = 1.5

# Frozen anchors never accumulate more games, so their `sigma` is fixed at a
# small nonzero value (not 0.0 -- `_MODEL.rate` divides by variance terms
# derived from both players' sigma) rather than ever being updated.
_ANCHOR_SIGMA = 0.5

_DEFAULT_ANCHOR_MU = 25.0
_RANDOM_ANCHOR_MU = 0.0


@dataclass
class Anchor:
    id: str
    config: dict
    mu: float
    sigma: float


@dataclass
class OpponentPool:
    anchors: list[Anchor] = field(default_factory=list)

    @classmethod
    def bootstrap(cls, cfg: SearchConfig) -> OpponentPool:
        """Seed the pool with the two floor anchors every run needs.

        `"default"` is the game binary's own `default_config`-sampled
        strategy (a reasonable, not maximal, opponent); `"random"` is the
        weakest possible opponent (`target.FLOOR_BASELINES["random"]`). Both
        are frozen from the start -- their `mu` values are fixed reference
        points, not something matchmaking should ever revise.
        """
        return cls(
            anchors=[
                Anchor(
                    id="default",
                    config=default_config(cfg),
                    mu=_DEFAULT_ANCHOR_MU,
                    sigma=_ANCHOR_SIGMA,
                ),
                Anchor(
                    id="random",
                    config=dict(FLOOR_BASELINES["random"]),
                    mu=_RANDOM_ANCHOR_MU,
                    sigma=_ANCHOR_SIGMA,
                ),
            ]
        )

    def closest(self, mu: float) -> Anchor:
        """Return the anchor whose `mu` is nearest a candidate's current rating."""
        return min(self.anchors, key=lambda a: abs(a.mu - mu))

    def maybe_insert(self, config: dict, mu: float, sigma: float) -> Anchor | None:
        """Freeze `config` as a new anchor if it's a new champion or skill band.

        Returns the inserted `Anchor`, or `None` if neither rule fired (the
        pool is unchanged). Uses the candidate's own final `sigma`, unlike
        the fixed `_ANCHOR_SIGMA` bootstrap anchors get, so the confidence a
        real matchmaking run produced is preserved.
        """
        best_mu = max(a.mu for a in self.anchors)
        nearest_delta = min(abs(a.mu - mu) for a in self.anchors)

        is_new_champion = mu > best_mu
        is_new_band = nearest_delta > _NEW_BAND_DELTA_MU
        if not (is_new_champion or is_new_band):
            return None

        anchor = Anchor(id=f"trial-{len(self.anchors)}", config=deepcopy(config), mu=mu, sigma=sigma)
        self.anchors.append(anchor)
        return anchor

    def save(self, path: str | Path) -> None:
        data = {"anchors": [asdict(a) for a in self.anchors]}
        Path(path).write_text(json.dumps(data, default=json_default, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> OpponentPool:
        data = json.loads(Path(path).read_text())
        return cls(anchors=[Anchor(**a) for a in data["anchors"]])
