"""Manual resume support.

SMAC3's own continue path (`SMBO._initialize_state`) only auto-resumes when
the new `Scenario` is byte-identical to the one saved on disk -- bumping
`n_trials` (the actual "run more trials" use case) trips its mismatch
handling, which falls back to an interactive `input()` prompt that would
hang a background-launched run. So resume is implemented by hand instead:
build a fresh `Scenario`/facade as normal, then seed its runhistory from a
prior run's saved `RunHistory` before calling `optimize()`.

This is squarely within SMAC3's own intended usage, not a workaround --
`Intensifier.__iter__` explicitly re-queues every config already present in
the runhistory handed to it ("supports user-inputs"), so merging prior
trials into a fresh runhistory before the first `ask()` is exactly the
mechanism it expects, not internal state this module has to fight.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ConfigSpace import ConfigurationSpace
from smac.runhistory.runhistory import RunHistory

logger = logging.getLogger(__name__)

# Matches `Scenario`'s own default `output_directory` in `__main__.py`.
SMAC3_OUTPUT_ROOT = Path("smac3_output")


def find_prior_output_directory(resume_id: str) -> Path:
    """Locate a prior run's SMAC3 output directory by its pinned ``Scenario.name``.

    Layout is ``smac3_output/<name>/<seed>/`` -- `Scenario` appends the seed
    subdirectory itself, so a resuming caller (which may use a different
    seed than the original run) can't predict it. In practice each name has
    exactly one seed subdirectory (one launch uses one seed), so it's
    resolved by globbing rather than needing the original seed as an input.
    """
    parent = SMAC3_OUTPUT_ROOT / resume_id
    if not parent.is_dir():
        raise FileNotFoundError(f"no prior SMAC3 run found at {parent}")
    candidates = [p for p in parent.iterdir() if p.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one seed directory under {parent}, found "
            f"{len(candidates)}: {candidates}"
        )
    return candidates[0]


def load_prior_runhistory(resume_id: str, configspace: ConfigurationSpace) -> RunHistory:
    """Load a prior run's saved ``RunHistory`` from disk.

    ``configspace`` must be the *new* run's `ConfigurationSpace` (not the old
    run's) -- `RunHistory.load` reconstructs every `Configuration` against
    it, which is what lets `RunHistory.update` merge the result into the new
    run's own runhistory by value equality rather than object identity.
    """
    output_directory = find_prior_output_directory(resume_id)
    runhistory_path = output_directory / "runhistory.json"
    rh = RunHistory()
    rh.load(runhistory_path, configspace=configspace)
    return rh
