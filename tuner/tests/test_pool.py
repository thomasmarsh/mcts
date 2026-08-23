"""`pool.OpponentPool`: bootstrap anchors, insertion rules, frozen anchors, JSON roundtrip."""

from __future__ import annotations

import yaml

from tuner_cli.config import SearchConfig
from tuner_cli.pool import Anchor, OpponentPool

_SPACE_YAML = """
parameters:
  family:
    type: categorical
    choices: [rave, ucb1_pn]
    default: rave
  c:
    type: float
    bounds: [0, 2]
    default: 1.4
"""


def _cfg() -> SearchConfig:
    return SearchConfig._from_dict(yaml.safe_load(_SPACE_YAML))


def test_bootstrap_seeds_default_and_random_anchors():
    pool = OpponentPool.bootstrap(_cfg())

    ids = {a.id for a in pool.anchors}
    assert ids == {"default", "random"}

    default = next(a for a in pool.anchors if a.id == "default")
    random_ = next(a for a in pool.anchors if a.id == "random")
    assert default.mu > random_.mu
    assert default.config == {"family": "rave", "c": 1.4}
    assert random_.config["family"] == "random"
    assert (default.provenance, default.insertion_reason) == (
        "bootstrap_default",
        "bootstrap",
    )
    assert (random_.provenance, random_.insertion_reason) == (
        "bootstrap_random",
        "bootstrap",
    )


def test_configured_anchor_has_distinct_provenance():
    pool = OpponentPool.bootstrap(_cfg())

    anchor = pool.add_configured_anchor("handicap", {"family": "ucb1_pn"})

    assert (anchor.provenance, anchor.insertion_reason, anchor.source_trial_id) == (
        "configured",
        "configured",
        None,
    )


def test_closest_picks_nearest_anchor_by_mu():
    pool = OpponentPool.bootstrap(_cfg())  # anchors at mu=25.0 ("default"), mu=0.0 ("random")

    assert pool.closest(24.0).id == "default"
    assert pool.closest(1.0).id == "random"
    assert pool.closest(12.5).id in {"default", "random"}  # tie, either is fine


def test_maybe_insert_new_champion():
    pool = OpponentPool.bootstrap(_cfg())  # best mu so far is 25.0

    inserted = pool.maybe_insert(
        {"family": "ucb1_pn"}, mu=30.0, sigma=3.0, source_trial_id="trial-3"
    )

    assert inserted is not None
    assert inserted.mu == 30.0
    assert (inserted.insertion_reason, inserted.source_trial_id) == ("champion", "trial-3")
    assert len(pool.anchors) == 3


def test_maybe_insert_new_skill_band():
    pool = OpponentPool.bootstrap(_cfg())  # nearest anchors at mu=0.0, mu=25.0

    inserted = pool.maybe_insert(
        {"family": "ucb1_pn"}, mu=12.0, sigma=3.0, source_trial_id="trial-4"
    )  # >1.5 from both

    assert inserted is not None
    assert inserted.insertion_reason == "skill_band"
    assert len(pool.anchors) == 3


def test_maybe_insert_rejects_config_too_close_to_an_existing_anchor():
    pool = OpponentPool.bootstrap(_cfg())  # "default" anchor at mu=25.0

    inserted = pool.maybe_insert(
        {"family": "ucb1_pn"}, mu=24.5, sigma=3.0, source_trial_id="trial-5"
    )  # within 1.5, not champion

    assert inserted is None
    assert len(pool.anchors) == 2


def test_anchors_never_mutate_once_inserted():
    pool = OpponentPool.bootstrap(_cfg())
    config = {"family": "ucb1_pn", "nested": {"value": 1}}
    pool.maybe_insert(config, mu=30.0, sigma=3.0, source_trial_id="trial-6")
    config["nested"]["value"] = 2
    assert pool.anchors[-1].config["nested"]["value"] == 1
    snapshot = [Anchor(**vars(a)) for a in pool.anchors]

    # A later trial that would otherwise be a new champion over the first
    # inserted anchor must add a fourth anchor, not revise the third.
    pool.maybe_insert(
        {"family": "ucb1_pn"}, mu=35.0, sigma=2.0, source_trial_id="trial-7"
    )

    assert len(pool.anchors) == 4
    for before, after in zip(snapshot, pool.anchors[:3]):
        assert before == after


def test_save_load_roundtrip(tmp_path):
    pool = OpponentPool.bootstrap(_cfg())
    pool.maybe_insert(
        {"family": "ucb1_pn"}, mu=30.0, sigma=3.0, source_trial_id="trial-8"
    )

    path = tmp_path / "pool.json"
    pool.save(path)
    loaded = OpponentPool.load(path)

    assert loaded == pool


def test_loads_legacy_four_field_anchor_as_unknown(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(
        '{"anchors": [{"id": "old", "config": {"family": "rave"}, "mu": 4.0, "sigma": 1.0}]}'
    )

    anchor = OpponentPool.load(path).anchors[0]

    assert (anchor.provenance, anchor.insertion_reason, anchor.source_trial_id) == (
        "legacy_unknown",
        "legacy_unknown",
        None,
    )
