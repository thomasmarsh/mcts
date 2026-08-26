"""Compatibility checks for the Optuna Hyperband boundary."""

from __future__ import annotations

import optuna
from optuna.trial import TrialState

from tuner_cli import hyperband
from tuner_cli.config import PruningPolicy, ResourcePolicy
from tuner_cli.hyperband import OptunaHyperbandAdapter


def _adapter(
    *, min_pairs: int = 1, max_pairs: int = 3, startup_trials: int = 0
) -> OptunaHyperbandAdapter:
    return OptunaHyperbandAdapter(
        ResourcePolicy(min_pairs=min_pairs, max_pairs=max_pairs),
        PruningPolicy(
            reduction_factor=3,
            startup_trials=startup_trials,
        ),
    )


def _study(
    adapter: OptunaHyperbandAdapter, name: str = "hyperband-test"
) -> optuna.Study:
    return optuna.create_study(
        direction="maximize", study_name=name, pruner=adapter.pruner
    )


def test_constructor_passes_explicit_resources_without_bootstrap_count(monkeypatch):
    arguments: dict[str, object] = {}

    def hyperband_pruner(**kwargs):
        arguments.update(kwargs)
        return object()

    monkeypatch.setattr(hyperband.optuna.pruners, "HyperbandPruner", hyperband_pruner)

    _adapter(min_pairs=2, max_pairs=7, startup_trials=11)

    assert arguments == {
        "min_resource": 2,
        "max_resource": 7,
        "reduction_factor": 3,
    }


def test_startup_policy_does_not_set_optuna_bootstrap_count():
    adapter = _adapter(min_pairs=2, max_pairs=7, startup_trials=11)

    assert adapter.pruner._min_resource == 2
    assert adapter.pruner._max_resource == 7
    assert adapter.pruner._reduction_factor == 3
    assert adapter.pruner._bootstrap_count == 0


def test_bracket_identity_is_stable_for_a_fixed_study_name_and_trial_number():
    def bracket_id() -> str:
        adapter = _adapter()
        study = _study(adapter, "fixed-study")
        hyperband_trial = adapter.create_trial(study)
        hyperband_trial.trial.report(1.0, 1)
        decision = adapter.observe_after_report(hyperband_trial)
        assert decision.bracket_id == adapter.bracket_id_for(
            study, hyperband_trial.trial
        )
        return decision.bracket_id

    assert bracket_id() == bracket_id()


def test_minimum_resource_only_observes_a_rung_when_reached():
    adapter = _adapter(min_pairs=2, max_pairs=5)
    hyperband_trial = adapter.create_trial(_study(adapter))

    hyperband_trial.trial.report(1.0, 1)
    below_minimum = adapter.observe_after_report(hyperband_trial)
    hyperband_trial.trial.report(1.0, 2)
    at_minimum = adapter.observe_after_report(hyperband_trial)

    assert not below_minimum.should_prune
    assert below_minimum.rung_resource is None
    assert not at_minimum.should_prune
    assert at_minimum.rung_resource == 2


def test_startup_exemption_is_the_coordinator_provided_immutable_value():
    adapter = _adapter(startup_trials=3)
    study = _study(adapter)

    first = adapter.create_trial(study, True)
    assert first.pruning_exempt
    assert not adapter.create_trial(study, False).pruning_exempt


def test_observation_delegates_keep_and_prune_results_without_telling():
    adapter = _adapter(min_pairs=1, max_pairs=1)
    study = _study(adapter)

    kept = adapter.create_trial(study)
    kept.trial.report(1.0, 1)
    keep_decision = adapter.observe_after_report(kept)
    assert not keep_decision.should_prune
    assert keep_decision.bracket_id == "0"
    assert keep_decision.rung_resource == 1
    assert study.trials[0].state is TrialState.RUNNING
    study.tell(kept.trial, 1.0)

    pruned = adapter.create_trial(study)
    pruned.trial.report(-1.0, 1)
    prune_decision = adapter.observe_after_report(pruned)

    assert prune_decision.should_prune
    assert prune_decision.bracket_id == "0"
    assert prune_decision.rung_resource == 1
    assert study.trials[1].state is TrialState.RUNNING
