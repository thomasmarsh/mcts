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
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .config import SearchConfig, json_default
from .lifecycle import pool_snapshot_fingerprint
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

AnchorProvenance = Literal[
    "bootstrap_default",
    "bootstrap_random",
    "configured",
    "trial",
    "legacy_unknown",
]
AnchorInsertionReason = Literal[
    "bootstrap",
    "configured",
    "champion",
    "skill_band",
    "legacy_unknown",
]
PoolDecisionAction = Literal["inserted", "rejected"]
PoolDecisionReason = Literal["champion", "skill_band", "covered"]
POOL_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class Anchor:
    id: str
    config: dict
    mu: float
    sigma: float
    provenance: AnchorProvenance = "legacy_unknown"
    insertion_reason: AnchorInsertionReason = "legacy_unknown"
    source_trial_id: str | None = None

    def revision_snapshot(self) -> dict:
        """Return the complete immutable anchor evidence for a pool revision."""
        return {
            "anchor_id": self.id,
            "config": self.config,
            "mu": self.mu,
            "sigma": self.sigma,
            "provenance": self.provenance,
            "insertion_reason": self.insertion_reason,
            "source_trial_id": self.source_trial_id,
        }


@dataclass(frozen=True)
class PoolDecision:
    """The immutable result of considering one completed trial for the pool."""

    trial_id: str
    before_pool_snapshot_fingerprint: str
    action: PoolDecisionAction
    reason: PoolDecisionReason
    anchor: dict[str, Any] | None
    after_pool_snapshot_fingerprint: str

    def payload(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "before_pool_snapshot_fingerprint": self.before_pool_snapshot_fingerprint,
            "action": self.action,
            "reason": self.reason,
            "anchor": self.anchor,
            "after_pool_snapshot_fingerprint": self.after_pool_snapshot_fingerprint,
        }


@dataclass(frozen=True)
class PoolCheckpoint:
    """The durable state needed to resume an immutable opponent pool."""

    manifest_fingerprint: str
    pool_snapshot_fingerprint: str
    anchors: list[Anchor]
    last_applied_decision: dict[str, Any] | None


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
                    provenance="bootstrap_default",
                    insertion_reason="bootstrap",
                ),
                Anchor(
                    id="random",
                    config=dict(FLOOR_BASELINES["random"]),
                    mu=_RANDOM_ANCHOR_MU,
                    sigma=_ANCHOR_SIGMA,
                    provenance="bootstrap_random",
                    insertion_reason="bootstrap",
                ),
            ]
        )

    def closest(self, mu: float) -> Anchor:
        """Return the anchor whose `mu` is nearest a candidate's current rating."""
        return min(self.anchors, key=lambda a: abs(a.mu - mu))

    def decide_insertion(
        self, config: dict, mu: float, sigma: float, source_trial_id: str
    ) -> PoolDecision:
        """Return the deterministic insertion decision without mutating the pool."""
        before = pool_snapshot_fingerprint(self.anchors)
        best_mu = max(a.mu for a in self.anchors)
        nearest_delta = min(abs(a.mu - mu) for a in self.anchors)

        is_new_champion = mu > best_mu
        is_new_band = nearest_delta > _NEW_BAND_DELTA_MU
        if not (is_new_champion or is_new_band):
            return PoolDecision(
                source_trial_id, before, "rejected", "covered", None, before
            )

        anchor = Anchor(
            id=f"trial-{len(self.anchors)}",
            config=deepcopy(config),
            mu=mu,
            sigma=sigma,
            provenance="trial",
            insertion_reason="champion" if is_new_champion else "skill_band",
            source_trial_id=source_trial_id,
        )
        after = pool_snapshot_fingerprint([*self.anchors, anchor])
        return PoolDecision(
            source_trial_id,
            before,
            "inserted",
            anchor.insertion_reason,
            anchor.revision_snapshot(),
            after,
        )

    def apply_decision(self, decision: PoolDecision) -> None:
        """Apply one validated immutable decision exactly at its fingerprint boundary."""
        if (
            pool_snapshot_fingerprint(self.anchors)
            != decision.before_pool_snapshot_fingerprint
        ):
            raise ValueError("pool decision does not follow the current checkpoint")
        if decision.action == "rejected":
            if decision.reason != "covered" or decision.anchor is not None:
                raise ValueError("rejected pool decision has invalid evidence")
        elif decision.action == "inserted":
            if (
                decision.reason not in {"champion", "skill_band"}
                or decision.anchor is None
            ):
                raise ValueError("inserted pool decision has invalid evidence")
            anchor = Anchor(
                id=decision.anchor["anchor_id"],
                config=deepcopy(decision.anchor["config"]),
                mu=decision.anchor["mu"],
                sigma=decision.anchor["sigma"],
                provenance=decision.anchor["provenance"],
                insertion_reason=decision.anchor["insertion_reason"],
                source_trial_id=decision.anchor["source_trial_id"],
            )
            if (
                anchor.id in {existing.id for existing in self.anchors}
                or anchor.provenance != "trial"
                or anchor.insertion_reason != decision.reason
                or anchor.source_trial_id != decision.trial_id
            ):
                raise ValueError("pool decision anchor identity is invalid")
            self.anchors.append(anchor)
        else:
            raise ValueError("unknown pool decision action")
        if (
            pool_snapshot_fingerprint(self.anchors)
            != decision.after_pool_snapshot_fingerprint
        ):
            raise ValueError(
                "pool decision fingerprint does not match its anchor snapshot"
            )

    def maybe_insert(
        self, config: dict, mu: float, sigma: float, source_trial_id: str
    ) -> Anchor | None:
        """Compatibility helper that applies a freshly computed decision."""
        decision = self.decide_insertion(config, mu, sigma, source_trial_id)
        self.apply_decision(decision)
        return self.anchors[-1] if decision.action == "inserted" else None

    def add_configured_anchor(self, anchor_id: str, config: dict) -> Anchor:
        """Freeze a user-configured baseline when it is absent from the pool."""
        anchor = Anchor(
            anchor_id,
            deepcopy(config),
            mu=_DEFAULT_ANCHOR_MU,
            sigma=_ANCHOR_SIGMA,
            provenance="configured",
            insertion_reason="configured",
        )
        self.anchors.append(anchor)
        return anchor

    def revision_payload(self) -> dict:
        """Return the ordered full snapshot attached to a pool-revised event."""
        return {
            "pool_snapshot_fingerprint": pool_snapshot_fingerprint(self.anchors),
            "anchors": [anchor.revision_snapshot() for anchor in self.anchors],
        }

    def save(
        self,
        path: str | Path,
        manifest_fingerprint: str | None = None,
        last_applied_decision: PoolDecision | None = None,
    ) -> None:
        """Atomically replace the versioned checkpoint after a durable decision."""
        if manifest_fingerprint is None:
            # Retain this narrow compatibility path for callers reading/writing a
            # standalone pool fixture; production always supplies a manifest.
            Path(path).write_text(
                json.dumps(
                    {"anchors": [asdict(a) for a in self.anchors]},
                    default=json_default,
                    indent=2,
                )
            )
            return
        data = {
            "schema_version": POOL_CHECKPOINT_SCHEMA_VERSION,
            "manifest_fingerprint": manifest_fingerprint,
            "pool_snapshot_fingerprint": pool_snapshot_fingerprint(self.anchors),
            "anchors": [asdict(a) for a in self.anchors],
            "last_applied_decision": (
                last_applied_decision.payload()
                if last_applied_decision is not None
                else None
            ),
        }
        _replace_checkpoint_atomically(
            Path(path), json.dumps(data, default=json_default, indent=2) + "\n"
        )

    @classmethod
    def load(cls, path: str | Path) -> OpponentPool:
        data = json.loads(Path(path).read_text())
        anchors = []
        for raw_anchor in data["anchors"]:
            anchor = dict(raw_anchor)
            anchor.setdefault("provenance", "legacy_unknown")
            anchor.setdefault("insertion_reason", "legacy_unknown")
            anchor.setdefault("source_trial_id", None)
            anchors.append(Anchor(**anchor))
        return cls(anchors=anchors)


def load_checkpoint(
    path: str | Path, manifest_fingerprint: str
) -> tuple[OpponentPool, PoolDecision | None, bool]:
    """Load a versioned checkpoint, accepting a legacy pool for one adoption."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "schema_version" not in data:
        return OpponentPool.load(path), None, True
    if data.get("schema_version") != POOL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("pool checkpoint has an unsupported schema version")
    if data.get("manifest_fingerprint") != manifest_fingerprint:
        raise ValueError("pool checkpoint manifest fingerprint does not match")
    pool = OpponentPool.load(path)
    fingerprint = pool_snapshot_fingerprint(pool.anchors)
    if data.get("pool_snapshot_fingerprint") != fingerprint:
        raise ValueError("pool checkpoint fingerprint does not match its anchors")
    raw_decision = data.get("last_applied_decision")
    decision = (
        _decision_from_payload(raw_decision) if raw_decision is not None else None
    )
    if decision is not None and decision.after_pool_snapshot_fingerprint != fingerprint:
        raise ValueError("pool checkpoint decision does not match its anchors")
    return pool, decision, False


def decision_from_payload(payload: dict[str, Any]) -> PoolDecision:
    """Decode persisted lifecycle evidence before applying its invariants."""
    return _decision_from_payload(payload)


def _decision_from_payload(payload: dict[str, Any]) -> PoolDecision:
    if not isinstance(payload, dict):
        raise ValueError("pool decision payload is not an object")
    required = {
        "trial_id",
        "before_pool_snapshot_fingerprint",
        "action",
        "reason",
        "anchor",
        "after_pool_snapshot_fingerprint",
    }
    if set(payload) != required or not all(
        isinstance(payload[key], str) and payload[key] for key in required - {"anchor"}
    ):
        raise ValueError("pool decision payload is malformed")
    if payload["action"] not in {"inserted", "rejected"} or payload["reason"] not in {
        "champion",
        "skill_band",
        "covered",
    }:
        raise ValueError("pool decision payload has invalid action or reason")
    if payload["anchor"] is not None and not isinstance(payload["anchor"], dict):
        raise ValueError("pool decision anchor is malformed")
    return PoolDecision(**payload)


def _replace_checkpoint_atomically(destination: Path, contents: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as temporary:
        temporary.write(contents)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def recover_pool(
    cfg: SearchConfig,
    pool_path: Path,
    manifest_fingerprint: str,
    lifecycle: Any,
    study: Any,
) -> OpponentPool:
    """Reconcile immutable decisions and the checkpoint before scheduling work.

    The journal is read completely and validated before this function emits an
    event or replaces the checkpoint.  That makes a malformed historical record
    a startup failure rather than a partly-repaired session.
    """
    terminals, decisions, revisions = _pool_evidence(
        lifecycle.path, lifecycle.session_id
    )
    base = _bootstrap_with_configured_anchors(cfg)
    checkpoint_exists = pool_path.exists()
    checkpoint: OpponentPool | None = None
    checkpoint_last: PoolDecision | None = None
    legacy = False
    if checkpoint_exists:
        checkpoint, checkpoint_last, legacy = load_checkpoint(
            pool_path, manifest_fingerprint
        )
        _add_missing_configured_anchors(checkpoint, cfg)

    _replay_decisions(deepcopy(base), terminals, decisions)
    applied_count = 0
    if checkpoint is None:
        pool = base
        last = None
    elif legacy:
        # A legacy pool predates the decision log.  It can only be adopted when
        # it already agrees with all retained decisions.
        replayed = _replay_decisions(base, terminals, decisions)
        if pool_snapshot_fingerprint(checkpoint.anchors) != pool_snapshot_fingerprint(
            replayed.anchors
        ):
            raise ValueError("legacy pool conflicts with lifecycle decisions")
        pool = checkpoint
        last = decisions[-1] if decisions else None
        applied_count = len(decisions)
        pool.save(pool_path, manifest_fingerprint, last)
    else:
        pool = checkpoint
        last = checkpoint_last
        if checkpoint_last is not None:
            try:
                applied_count = decisions.index(checkpoint_last) + 1
            except ValueError as error:
                raise ValueError(
                    "pool checkpoint last decision conflicts with lifecycle evidence"
                ) from error
        expected = _replay_decisions(
            deepcopy(base), terminals[:applied_count], decisions[:applied_count]
        )
        if pool_snapshot_fingerprint(pool.anchors) != pool_snapshot_fingerprint(
            expected.anchors
        ):
            raise ValueError("pool checkpoint conflicts with lifecycle decisions")
    if checkpoint is None and not terminals and len(study.trials) != 0:
        raise ValueError(
            "cannot reconstruct a nonempty study without a pool checkpoint or decisions"
        )

    for decision in decisions[applied_count:]:
        pool.apply_decision(decision)
        pool.save(pool_path, manifest_fingerprint, decision)
        last = decision

    planned = deepcopy(pool)
    missing: list[PoolDecision] = []
    for terminal in terminals[len(decisions) :]:
        decision = planned.decide_insertion(
            terminal["config"], terminal["mu"], terminal["sigma"], terminal["trial_id"]
        )
        planned.apply_decision(decision)
        missing.append(decision)

    if checkpoint is None and not decisions and not missing:
        pool.save(pool_path, manifest_fingerprint, None)

    for decision in missing:
        lifecycle.emit("pool_anchor_decided", decision.payload())
        pool.apply_decision(decision)
        pool.save(pool_path, manifest_fingerprint, decision)
        last = decision
        if decision.action == "inserted":
            lifecycle.emit("pool_revised", pool.revision_payload())

    # A checkpoint may have made an inserted revision durable just before the
    # process died; publish that snapshot if its corresponding event is absent.
    if (
        last is not None
        and last.action == "inserted"
        and last.after_pool_snapshot_fingerprint not in revisions
    ):
        lifecycle.emit("pool_revised", pool.revision_payload())
    return pool


def _bootstrap_with_configured_anchors(cfg: SearchConfig) -> OpponentPool:
    pool = OpponentPool.bootstrap(cfg)
    _add_missing_configured_anchors(pool, cfg)
    return pool


def _add_missing_configured_anchors(pool: OpponentPool, cfg: SearchConfig) -> None:
    for anchor_id, config in cfg.target.baseline_configs.items():
        if not any(anchor.id == anchor_id for anchor in pool.anchors):
            pool.add_configured_anchor(anchor_id, config)


def _pool_evidence(
    path: Path, session_id: str
) -> tuple[list[dict[str, Any]], list[PoolDecision], set[str]]:
    terminals: list[dict[str, Any]] = []
    decisions: list[PoolDecision] = []
    terminal_ids: set[str] = set()
    revisions: set[str] = set()
    if not path.exists():
        return terminals, decisions, revisions
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("session_id") != session_id:
            raise ValueError("lifecycle journal belongs to a different session")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("lifecycle journal has an invalid event payload")
        event_type = record.get("event_type")
        if event_type == "trial_completed":
            trial_id = payload.get("trial_id")
            config, mu, sigma = (
                payload.get("config"),
                payload.get("mu"),
                payload.get("sigma"),
            )
            if (
                not isinstance(trial_id, str)
                or trial_id in terminal_ids
                or not isinstance(config, dict)
                or not _finite_number(mu)
                or not _finite_number(sigma)
            ):
                raise ValueError(
                    "completed trial has insufficient pool recovery evidence"
                )
            terminal_ids.add(trial_id)
            terminals.append(
                {"trial_id": trial_id, "config": config, "mu": mu, "sigma": sigma}
            )
        elif event_type == "pool_anchor_decided":
            decision = decision_from_payload(payload)
            if decision.trial_id not in terminal_ids or any(
                item.trial_id == decision.trial_id for item in decisions
            ):
                raise ValueError(
                    "pool decision does not have one completed trial source"
                )
            decisions.append(decision)
        elif event_type == "pool_revised":
            fingerprint = payload.get("pool_snapshot_fingerprint")
            if isinstance(fingerprint, str):
                revisions.add(fingerprint)
    if [decision.trial_id for decision in decisions] != [
        item["trial_id"] for item in terminals[: len(decisions)]
    ]:
        raise ValueError("pool decisions are not in completed-trial order")
    return terminals, decisions, revisions


def _replay_decisions(
    pool: OpponentPool, terminals: list[dict[str, Any]], decisions: list[PoolDecision]
) -> OpponentPool:
    # terminals may outnumber decisions (a completed trial whose pool decision
    # wasn't yet recorded); only the common prefix is replayed here.
    for terminal, decision in zip(terminals, decisions, strict=False):
        expected = pool.decide_insertion(
            terminal["config"], terminal["mu"], terminal["sigma"], terminal["trial_id"]
        )
        if decision != expected:
            raise ValueError("pool decision conflicts with completed-trial evidence")
        pool.apply_decision(decision)
    return pool


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
