"""Custom SMAC callback that logs every new incumbent."""

from __future__ import annotations

import logging

from smac import Callback
from smac.main.smbo import SMBO
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.utils.configspace import get_config_hash

logger = logging.getLogger(__name__)


class IncumbentTracker(Callback):
    """Print each new incumbent configuration and its cost as it's found."""

    def __init__(self) -> None:
        self._last_hash: str | None = None

    def on_next_configurations_end(self, config_selector, config) -> None:
        """Log every configuration the config selector proposes."""
        logger.info("Proposed config: %s", dict(config))

    def on_tell_end(
        self,
        smbo: SMBO,
        info: TrialInfo,
        value: TrialValue,
    ) -> bool | None:
        """Check whether the incumbent changed and print it if so."""
        incumbent = smbo.intensifier.get_incumbent()
        if incumbent is None:
            return None

        h = get_config_hash(incumbent)
        if h != self._last_hash:
            cost = smbo.runhistory.get_cost(incumbent)
            logger.info(
                "New incumbent  hash=%s  cost=%s  config=%s",
                h,
                cost,
                dict(incumbent),
            )
            self._last_hash = h
        return None