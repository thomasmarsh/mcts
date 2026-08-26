"""`pool.OpponentPool`: bootstrap anchors, insertion rules, frozen anchors, JSON roundtrip."""

from __future__ import annotations

import yaml

from tuner_cli.config import SearchConfig
from tuner_cli.lifecycle import AttemptId, LifecycleWriter, SessionId, trial_id_for
from tuner_cli.pool import Anchor, OpponentPool, load_checkpoint, recover_pool

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
    assert (inserted.insertion_reason, inserted.source_trial_id) == (
        "champion",
        "trial-3",
    )
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
    pool.maybe_insert({"family": "ucb1_pn"}, mu=35.0, sigma=2.0, source_trial_id="trial-7")

    assert len(pool.anchors) == 4
    for before, after in zip(snapshot, pool.anchors[:3], strict=True):
        assert before == after


def test_save_load_roundtrip(tmp_path):
    pool = OpponentPool.bootstrap(_cfg())
    pool.maybe_insert({"family": "ucb1_pn"}, mu=30.0, sigma=3.0, source_trial_id="trial-8")

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


def test_recovery_records_one_missing_decision_then_becomes_a_no_op(tmp_path):
    cfg = _cfg()
    session = SessionId("pool-recovery")
    trial_id = trial_id_for(session, 0)
    journal = tmp_path / "lifecycle.jsonl"
    pool_path = tmp_path / "pool.json"

    with LifecycleWriter(journal, session, AttemptId("prior")) as writer:
        writer.emit("session_started", {"manifest": {}, "manifest_fingerprint": "manifest"})
        writer.emit("attempt_started", {})
        writer.emit(
            "trial_created",
            {"trial_id": trial_id, "trial_number": 0, "config": {"family": "ucb1_pn"}},
        )
        writer.emit("trial_started", {"trial_id": trial_id, "trial_number": 0})
        writer.emit_trial_terminal(
            "trial_completed",
            trial_id,
            {
                "trial_number": 0,
                "config": {"family": "ucb1_pn"},
                "mu": 30.0,
                "sigma": 2.0,
            },
        )

    study = type("Study", (), {"trials": [object()]})()
    with LifecycleWriter(journal, session, AttemptId("recovery")) as writer:
        writer.emit("attempt_started", {})
        pool = recover_pool(cfg, pool_path, "manifest", writer, study)
        assert [anchor.id for anchor in pool.anchors][-1] == "trial-2"
        writer.emit("attempt_completed", {})

    with LifecycleWriter(journal, session, AttemptId("again")) as writer:
        writer.emit("attempt_started", {})
        recovered = recover_pool(cfg, pool_path, "manifest", writer, study)
        assert recovered == pool

    records = [yaml.safe_load(line) for line in journal.read_text().splitlines()]
    assert [record["event_type"] for record in records].count("pool_anchor_decided") == 1
    loaded, decision, legacy = load_checkpoint(pool_path, "manifest")
    assert loaded == pool
    assert decision is not None and decision.action == "inserted"
    assert not legacy


def test_recovery_applies_a_logged_decision_without_a_checkpoint(tmp_path):
    cfg = _cfg()
    session = SessionId("decision-before-save")
    trial_id = trial_id_for(session, 0)
    journal = tmp_path / "lifecycle.jsonl"
    pool_path = tmp_path / "pool.json"
    decision = OpponentPool.bootstrap(cfg).decide_insertion(
        {"family": "ucb1_pn"}, 30.0, 2.0, trial_id
    )

    with LifecycleWriter(journal, session, AttemptId("prior")) as writer:
        writer.emit("session_started", {"manifest": {}, "manifest_fingerprint": "manifest"})
        writer.emit("attempt_started", {})
        writer.emit(
            "trial_created",
            {"trial_id": trial_id, "trial_number": 0, "config": {"family": "ucb1_pn"}},
        )
        writer.emit("trial_started", {"trial_id": trial_id, "trial_number": 0})
        writer.emit_trial_terminal(
            "trial_completed",
            trial_id,
            {
                "trial_number": 0,
                "config": {"family": "ucb1_pn"},
                "mu": 30.0,
                "sigma": 2.0,
            },
        )
        writer.emit("pool_anchor_decided", decision.payload())
        writer.emit("attempt_completed", {})

    study = type("Study", (), {"trials": [object()]})()
    with LifecycleWriter(journal, session, AttemptId("recovery")) as writer:
        writer.emit("attempt_started", {})
        pool = recover_pool(cfg, pool_path, "manifest", writer, study)

    assert pool.anchors[-1].source_trial_id == trial_id
    _, saved, legacy = load_checkpoint(pool_path, "manifest")
    assert saved == decision
    assert not legacy
