"""The narrow Optuna boundary for Hyperband pruning decisions.

The coordinator owns trial termination.  This module only creates trials,
captures their startup eligibility, and asks Optuna for a decision after a
caller has reported an evaluation pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import optuna
from .config import PruningPolicy, ResourcePolicy


_COMPLETED_RUNG_PREFIX = "completed_rung_"


@dataclass
class HyperbandTrial:
    """One Optuna trial together with its immutable startup eligibility."""

    trial: optuna.Trial
    pruning_exempt: bool
    _observed_rungs: int = 0


@dataclass(frozen=True)
class HyperbandDecision:
    """Optuna's decision and only the pruning metadata it actually observed."""

    should_prune: bool
    pruning_exempt: bool
    bracket_id: str | None
    rung_resource: int | None


class OptunaHyperbandAdapter:
    """Adapt explicit pair resources to the pinned Optuna Hyperband pruner."""

    def __init__(self, resource: ResourcePolicy, pruning: PruningPolicy) -> None:
        self._min_resource = _integer_resource(resource.min_pairs, "min_pairs")
        self._max_resource = _integer_resource(resource.max_pairs, "max_pairs")
        self.pruner = optuna.pruners.HyperbandPruner(
            min_resource=self._min_resource,
            max_resource=self._max_resource,
            reduction_factor=pruning.reduction_factor,
        )

    def create_trial(
        self, study: optuna.Study, pruning_exempt: bool = False
    ) -> HyperbandTrial:
        """Ask for a trial with coordinator-assigned immutable eligibility."""
        return HyperbandTrial(
            trial=study.ask(),
            pruning_exempt=pruning_exempt,
        )

    def observe_after_report(
        self, hyperband_trial: HyperbandTrial
    ) -> HyperbandDecision:
        """Return Optuna's decision after the caller has reported one pair.

        A startup-exempt trial is deliberately not passed to the pruner, so it
        cannot contribute a rung observation before it becomes eligible.
        """
        if hyperband_trial.pruning_exempt:
            return HyperbandDecision(False, True, None, None)

        trial = hyperband_trial.trial
        if trial.study.pruner is not self.pruner:
            raise ValueError("study must use this adapter's Hyperband pruner")

        should_prune = trial.should_prune()
        frozen_trial = _frozen_trial(trial)
        bracket_id = str(self._bracket_id(trial.study, frozen_trial))
        completed_rungs = _completed_rungs(frozen_trial)
        rung_resource = None
        if len(completed_rungs) > hyperband_trial._observed_rungs:
            rung_resource = frozen_trial.last_step
            hyperband_trial._observed_rungs = len(completed_rungs)

        return HyperbandDecision(
            should_prune=should_prune,
            pruning_exempt=False,
            bracket_id=bracket_id,
            rung_resource=rung_resource,
        )

    def bracket_id_for(self, study: optuna.Study, trial: optuna.Trial) -> str:
        """Return the pinned Optuna bracket identity after pruner initialization."""
        return str(self._bracket_id(study, _frozen_trial(trial)))

    def _bracket_id(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> int:
        # Optuna 4.9 intentionally derives this from study_name and trial.number.
        # Keeping its internal implementation here avoids a second bracket formula.
        return self.pruner._get_bracket_id(study, trial)


def _integer_resource(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer resource")
    return value


def _frozen_trial(trial: optuna.Trial) -> optuna.trial.FrozenTrial:
    for frozen_trial in trial.study.get_trials(deepcopy=False):
        if frozen_trial.number == trial.number:
            return frozen_trial
    raise RuntimeError(f"Optuna trial {trial.number} was not found in its study")


def _completed_rungs(trial: optuna.trial.FrozenTrial) -> list[int]:
    """Return only rungs Optuna has recorded, never a projected schedule."""
    return sorted(
        int(key.removeprefix(_COMPLETED_RUNG_PREFIX))
        for key in trial.system_attrs
        if key.startswith(_COMPLETED_RUNG_PREFIX)
    )
