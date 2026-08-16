"""Manual resume (`smac3_cli.resume`, `build_optimizer(..., resume=...)`):
run a tiny optimize, "stop" it, resume with a bigger `n_trials`, and confirm
the already-evaluated configs aren't re-evaluated -- see `resume.py`'s
docstring for why SMAC3's own continue path can't be used here (it hangs a
background process on any scenario mismatch, and bumping `n_trials` always
is one).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smac3_cli.__main__ import build_optimizer
from smac3_cli.config import OptimizerConfig, SearchConfig, TargetConfig


def _cfg(binary: Path, n_trials: int) -> SearchConfig:
    return SearchConfig(
        optimizer=OptimizerConfig(n_trials=n_trials, n_workers=1, deterministic=True, seed=7),
        target=TargetConfig(binary=binary, rounds=2, baselines=["strong"]),
    )


@pytest.fixture
def run_id(tmp_path, monkeypatch) -> str:
    """A fresh, isolated `smac3_output/` per test (cwd-relative, like the
    real CLI), so tests can't collide with each other or a real run."""
    monkeypatch.chdir(tmp_path)
    return "resume-test-run"


def _value_key(config) -> frozenset:
    """A hashable projection of a Configuration's *values*, independent of
    which `ConfigurationSpace` object it belongs to.

    `first` and `second` each build their own `ConfigurationSpace` instance
    (`build_optimizer` owns that construction internally), so `Configuration.
    __hash__`/`__eq__` -- which fold in `self.config_space` -- can't be
    trusted to recognize the same logical config across the two runs. Values
    round-trip exactly through `RunHistory.save`/`.load` (JSON floats are
    read back bit-for-bit in Python), so comparing on values alone is both
    sufficient and exactly what "was this config re-evaluated" means here.
    """
    return frozenset(dict(config).items())


def test_resume_seeds_prior_trials_and_skips_reevaluation(game_nim_binary: Path, run_id: str):
    first = build_optimizer(_cfg(game_nim_binary, n_trials=2), run_id=run_id)
    first.optimize()
    assert first.runhistory.submitted == 2
    first_trial_counts = {
        _value_key(config): len(first.runhistory.get_trials(config))
        for config in first.runhistory.get_configs()
    }
    assert len(first_trial_counts) == 2

    second = build_optimizer(_cfg(game_nim_binary, n_trials=4), resume=run_id)
    second.optimize()

    assert second.runhistory.submitted == 4

    # Every config the first run evaluated must still be present in the
    # resumed run's runhistory, with the *same* trial count -- i.e. the
    # resumed run's extra budget went entirely to new configs, not
    # re-evaluating carried-over ones.
    second_trial_counts: dict[frozenset, int] = {}
    for config in second.runhistory.get_configs():
        second_trial_counts[_value_key(config)] = len(second.runhistory.get_trials(config))

    for key, n in first_trial_counts.items():
        assert key in second_trial_counts, f"prior config {dict(key)} missing after resume"
        assert second_trial_counts[key] == n, (
            f"config {dict(key)} was re-evaluated after resume "
            f"(had {n} trial(s) before, {second_trial_counts[key]} after)"
        )

    # And the resumed run's extra budget (4 - 2 = 2 trials) actually went to
    # configs the first run never saw.
    new_configs = set(second_trial_counts) - set(first_trial_counts)
    assert sum(second_trial_counts[key] for key in new_configs) == 2


def test_resume_without_prior_run_raises(game_nim_binary: Path, run_id: str):
    with pytest.raises(FileNotFoundError):
        build_optimizer(_cfg(game_nim_binary, n_trials=2), resume="no-such-run")


def test_resume_with_changed_instance_uses_remaining_budget_without_old_costs(
    game_nim_binary: Path, run_id: str
):
    first = build_optimizer(_cfg(game_nim_binary, n_trials=2), run_id=run_id)
    first.optimize()

    changed = _cfg(game_nim_binary, n_trials=4)
    changed.target.baselines = ["weak"]
    second = build_optimizer(changed, resume=run_id)

    assert second.scenario.n_trials == 2
    assert len(second.runhistory) == 0

    second.optimize()
    assert second.runhistory.finished == 2
    assert {key.instance for key in second.runhistory.keys()} == {"weak"}
