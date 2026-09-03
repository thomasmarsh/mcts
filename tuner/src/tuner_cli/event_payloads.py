"""Closed tagged union of immutable versioned evidence event payloads.

Each event type owns one frozen payload value with a focused decoder and an
encoder that reproduces exactly the fields the producers wrote before this
module existed. ``evidence.py`` keeps the public envelope, writer, canonical
projection, and file API; it delegates payload narrowing and construction here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

from .codec import (
    JsonObject,
    elements,
    integer,
    json_object,
    literal,
    object_fields,
    optional_integer,
    optional_raw_number,
    optional_string,
    raw_number,
    string,
    strings,
)
from .domain import (
    ApplyElimination,
    BeginValidation,
    CandidateEliminationAction,
    DeepenCohortAllocation,
    DiagnosticPairTask,
    EliminationDecisionMargin,
    EvaluateDiagnosticPair,
    IntroduceCandidate,
    PairedBootstrapEvidence,
    PairedProbabilityMargin,
    RefillCandidate,
    ResourceAllocation,
    RetainElites,
    SearchEffort,
    ShadowCandidateDecision,
    ShadowRaceDecision,
    SuccessiveHalvingEvidence,
    SuccessiveHalvingRankMargin,
    SuspendActiveElimination,
)
from .effort import decode_effort, encode_effort

EventType = Literal[
    "proposal_created",
    "proposal_accepted",
    "proposal_rejected",
    "cohort_completed",
    "pair_started",
    "pair_completed",
    "pair_failed",
    "diagnostic_pair_started",
    "diagnostic_pair_completed",
    "diagnostic_pair_failed",
    "run_interrupted",
    "run_failed",
    "observation_completed",
    "finalists_selected",
    "run_completed",
    "allocation_decided",
    "shadow_race_decided",
    "candidate_failed",
    "budget_extended",
]

SCIENTIFIC: frozenset[EventType] = frozenset(
    {
        "budget_extended",
        "proposal_created",
        "proposal_accepted",
        "proposal_rejected",
        "cohort_completed",
        "pair_completed",
        "diagnostic_pair_completed",
        "observation_completed",
        "finalists_selected",
        "run_completed",
        "allocation_decided",
        "shadow_race_decided",
        "candidate_failed",
    }
)

_PROPOSAL_SOURCES: tuple[
    Literal["schema_default", "bootstrap_random", "smac_model", "random_reserve"], ...
] = (
    "schema_default",
    "bootstrap_random",
    "smac_model",
    "random_reserve",
)
_PHASES: tuple[Literal["tuning", "validation"], ...] = ("tuning", "validation")

ProposalSource = Literal[
    "schema_default",
    "bootstrap_random",
    "smac_model",
    "random_reserve",
    "random_search",
    "qmc_search",
    "irace_model",
]
Phase = Literal["tuning", "validation"]
RejectionReason = Literal["duplicate", "semantic_validation"]


def _decode_elimination_margin(value: object) -> EliminationDecisionMargin:
    raw = json_object(value, "elimination decision margin")
    kind = string(raw.get("kind"), "elimination margin kind")
    if kind == "paired_probability":
        item = object_fields(
            raw,
            {
                "kind",
                "elimination_probability_threshold",
                "favorable_probability",
                "threshold_minus_probability",
            },
            "paired probability margin",
        )
        return PairedProbabilityMargin(
            raw_number(
                item["elimination_probability_threshold"], "elimination probability threshold"
            ),
            raw_number(item["favorable_probability"], "favorable probability"),
            raw_number(item["threshold_minus_probability"], "threshold minus probability"),
        )
    if kind == "successive_halving_rank":
        item = object_fields(
            raw,
            {"kind", "rank", "target_survivor_count", "ranks_below_cutoff", "spared_count"},
            "successive halving rank margin",
        )
        return SuccessiveHalvingRankMargin(
            integer(item["rank"], "elimination rank", positive=True),
            integer(item["target_survivor_count"], "target survivor count", positive=True),
            integer(item["ranks_below_cutoff"], "ranks below cutoff", positive=True),
            integer(item["spared_count"], "spared count"),
        )
    raise ValueError("unknown elimination decision margin kind")


def _decode_elimination_action(value: object) -> CandidateEliminationAction:
    item = object_fields(value, {"candidate_id", "action", "margin"}, "elimination action")
    return CandidateEliminationAction(
        string(item["candidate_id"], "elimination candidate id"),
        literal(item["action"], ("prune", "audit_continue"), "elimination action"),
        _decode_elimination_margin(item["margin"]),
    )


def _encode_elimination_margin(margin: EliminationDecisionMargin) -> JsonObject:
    if isinstance(margin, PairedProbabilityMargin):
        return {
            "kind": "paired_probability",
            "elimination_probability_threshold": margin.elimination_probability_threshold,
            "favorable_probability": margin.favorable_probability,
            "threshold_minus_probability": margin.threshold_minus_probability,
        }
    return {
        "kind": "successive_halving_rank",
        "rank": margin.rank,
        "target_survivor_count": margin.target_survivor_count,
        "ranks_below_cutoff": margin.ranks_below_cutoff,
        "spared_count": margin.spared_count,
    }


def _encode_elimination_action(action: CandidateEliminationAction) -> JsonObject:
    return {
        "candidate_id": action.candidate_id,
        "action": action.action,
        "margin": _encode_elimination_margin(action.margin),
    }


def _decode_resource_allocation(value: object) -> ResourceAllocation:
    raw = json_object(value, "resource allocation")
    kind = string(raw.get("kind"), "resource allocation kind")
    if kind == "introduce_candidate":
        item = object_fields(raw, {"kind", "cohort_slot", "source"}, "introduce allocation")
        return IntroduceCandidate(
            integer(item["cohort_slot"], "allocation cohort slot"),
            literal(item["source"], _PROPOSAL_SOURCES, "allocation proposal source"),
        )
    if kind == "refill_candidate":
        item = object_fields(
            raw, {"kind", "cohort_slot", "source", "failed_candidate_id"}, "refill allocation"
        )
        return RefillCandidate(
            integer(item["cohort_slot"], "allocation cohort slot"),
            literal(item["source"], _PROPOSAL_SOURCES, "allocation proposal source"),
            string(item["failed_candidate_id"], "failed candidate id"),
        )
    if kind == "deepen_cohort":
        item = object_fields(raw, {"kind", "block_index", "prefix_id"}, "deepen allocation")
        return DeepenCohortAllocation(
            integer(item["block_index"], "allocation block index"),
            string(item["prefix_id"], "allocation prefix id"),
        )
    if kind == "begin_validation":
        item = object_fields(raw, {"kind", "tuning_prefix_id"}, "validation allocation")
        return BeginValidation(string(item["tuning_prefix_id"], "allocation tuning prefix id"))
    if kind == "evaluate_diagnostic_pair":
        item = object_fields(
            raw, {"kind", "cohort_index", "reason", "task"}, "diagnostic allocation"
        )
        task = _decode_diagnostic_task(item["task"])
        return EvaluateDiagnosticPair(
            integer(item["cohort_index"], "diagnostic cohort index"),
            literal(
                item["reason"],
                (
                    "graph_connectivity",
                    "potential_cycle_closure",
                    "ranking_boundary",
                    "unresolved_edge",
                ),
                "diagnostic reason",
            ),
            task,
        )
    if kind == "retain_elites":
        item = object_fields(
            raw,
            {"kind", "cohort_index", "candidate_ids", "prefix_id"},
            "elite retention allocation",
        )
        return RetainElites(
            integer(item["cohort_index"], "retained cohort index"),
            strings(item["candidate_ids"], "retained candidate ids"),
            string(item["prefix_id"], "retained prefix id"),
        )
    if kind == "apply_elimination":
        item = object_fields(
            raw, {"kind", "cohort_index", "prefix_id", "actions"}, "elimination allocation"
        )
        return ApplyElimination(
            integer(item["cohort_index"], "elimination cohort index"),
            string(item["prefix_id"], "elimination prefix id"),
            tuple(
                _decode_elimination_action(value)
                for value in elements(item["actions"], "elimination actions")
            ),
        )
    if kind == "suspend_active_elimination":
        item = object_fields(
            raw,
            {
                "kind",
                "after_cohort_index",
                "triggering_candidate_ids",
                "triggering_prefix_ids",
                "safety_rule_version",
            },
            "active elimination suspension",
        )
        return SuspendActiveElimination(
            integer(item["after_cohort_index"], "suspension cohort index"),
            strings(item["triggering_candidate_ids"], "suspension candidate ids"),
            strings(item["triggering_prefix_ids"], "suspension prefix ids"),
            string(item["safety_rule_version"], "safety rule version"),
        )
    raise ValueError("unknown resource allocation kind")


def _encode_resource_allocation(value: ResourceAllocation) -> JsonObject:
    match value:
        case IntroduceCandidate(cohort_slot, source):
            return {"kind": "introduce_candidate", "cohort_slot": cohort_slot, "source": source}
        case RefillCandidate(cohort_slot, source, failed_candidate_id):
            return {
                "kind": "refill_candidate",
                "cohort_slot": cohort_slot,
                "source": source,
                "failed_candidate_id": failed_candidate_id,
            }
        case DeepenCohortAllocation(block_index, prefix_id):
            return {"kind": "deepen_cohort", "block_index": block_index, "prefix_id": prefix_id}
        case BeginValidation(tuning_prefix_id):
            return {"kind": "begin_validation", "tuning_prefix_id": tuning_prefix_id}
        case EvaluateDiagnosticPair(cohort_index, reason, task):
            return {
                "kind": "evaluate_diagnostic_pair",
                "cohort_index": cohort_index,
                "reason": reason,
                "task": _encode_diagnostic_task(task),
            }
        case RetainElites(cohort_index, candidate_ids, prefix_id):
            return {
                "kind": "retain_elites",
                "cohort_index": cohort_index,
                "candidate_ids": list(candidate_ids),
                "prefix_id": prefix_id,
            }
        case ApplyElimination(cohort_index, prefix_id, actions):
            return {
                "kind": "apply_elimination",
                "cohort_index": cohort_index,
                "prefix_id": prefix_id,
                "actions": [_encode_elimination_action(action) for action in actions],
            }
        case SuspendActiveElimination(
            after_cohort_index, triggering_candidate_ids, triggering_prefix_ids, safety_rule_version
        ):
            return {
                "kind": "suspend_active_elimination",
                "after_cohort_index": after_cohort_index,
                "triggering_candidate_ids": list(triggering_candidate_ids),
                "triggering_prefix_ids": list(triggering_prefix_ids),
                "safety_rule_version": safety_rule_version,
            }


@dataclass(frozen=True, slots=True)
class AllocationDecidedPayload:
    event_type: ClassVar[EventType] = "allocation_decided"
    allocation: ResourceAllocation
    policy_version: str

    @staticmethod
    def decode(value: object) -> AllocationDecidedPayload:
        item = object_fields(value, {"allocation", "policy_version"}, "allocation decision")
        return AllocationDecidedPayload(
            _decode_resource_allocation(item["allocation"]),
            string(item["policy_version"], "allocation policy version"),
        )

    def encode(self) -> JsonObject:
        return {
            "allocation": _encode_resource_allocation(self.allocation),
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class ShadowRaceDecidedPayload:
    event_type: ClassVar[EventType] = "shadow_race_decided"
    decision: ShadowRaceDecision

    @staticmethod
    def decode(value: object) -> ShadowRaceDecidedPayload:
        item = object_fields(
            value,
            {
                "cohort_index",
                "prefix_id",
                "observation_ids",
                "boundary_candidate_id",
                "decisions",
                "policy_kind",
                "policy_version",
            },
            "shadow race",
        )
        decisions: list[ShadowCandidateDecision] = []
        for raw in elements(item["decisions"], "shadow decisions"):
            candidate = object_fields(
                raw, {"candidate_id", "disposition", "evidence"}, "shadow candidate decision"
            )
            evidence_raw = candidate["evidence"]
            from .codec import json_object

            evidence_object = json_object(evidence_raw, "shadow evidence")
            evidence_kind = string(evidence_object.get("kind"), "shadow evidence kind")
            if evidence_kind == "paired_bootstrap":
                encoded_evidence = object_fields(
                    evidence_object,
                    {"kind", "favorable_resamples", "total_resamples"},
                    "paired shadow evidence",
                )
                evidence = PairedBootstrapEvidence(
                    integer(encoded_evidence["favorable_resamples"], "favorable resamples"),
                    integer(encoded_evidence["total_resamples"], "total resamples", positive=True),
                )
            elif evidence_kind == "successive_halving":
                encoded_evidence = object_fields(
                    evidence_object,
                    {
                        "kind",
                        "rank",
                        "prior_survivor_count",
                        "target_survivor_count",
                        "newly_eliminated",
                    },
                    "successive halving evidence",
                )
                rank = encoded_evidence["rank"]
                newly_eliminated = encoded_evidence["newly_eliminated"]
                if type(newly_eliminated) is not bool:
                    raise ValueError("newly eliminated must be Boolean")
                evidence = SuccessiveHalvingEvidence(
                    None if rank is None else integer(rank, "shadow rank", positive=True),
                    integer(
                        encoded_evidence["prior_survivor_count"],
                        "prior survivor count",
                        positive=True,
                    ),
                    integer(
                        encoded_evidence["target_survivor_count"],
                        "target survivor count",
                        positive=True,
                    ),
                    newly_eliminated,
                )
            else:
                raise ValueError("unsupported shadow evidence")
            decisions.append(
                ShadowCandidateDecision(
                    string(candidate["candidate_id"], "shadow candidate id"),
                    literal(
                        candidate["disposition"],
                        ("continue", "eliminate", "protected"),
                        "shadow disposition",
                    ),
                    evidence,
                )
            )
        return ShadowRaceDecidedPayload(
            ShadowRaceDecision(
                integer(item["cohort_index"], "shadow cohort index"),
                string(item["prefix_id"], "shadow prefix id"),
                strings(item["observation_ids"], "shadow observation ids"),
                string(item["boundary_candidate_id"], "shadow boundary candidate id"),
                tuple(decisions),
                literal(
                    item["policy_kind"],
                    ("paired_bootstrap", "successive_halving"),
                    "shadow policy kind",
                ),
                literal(
                    item["policy_version"],
                    (
                        "stratified-paired-bootstrap-v1",
                        "stratified-paired-bootstrap-all-strata-v2",
                        "successive-halving-common-prefix-eta2-v1",
                        "successive-halving-spare-near-tie-v1",
                    ),
                    "shadow policy version",
                ),
            )
        )

    def encode(self) -> JsonObject:
        value = self.decision
        return {
            "cohort_index": value.cohort_index,
            "prefix_id": value.prefix_id,
            "observation_ids": list(value.observation_ids),
            "boundary_candidate_id": value.boundary_candidate_id,
            "decisions": [
                {
                    "candidate_id": item.candidate_id,
                    "evidence": (
                        {
                            "kind": "paired_bootstrap",
                            "favorable_resamples": item.evidence.favorable_resamples,
                            "total_resamples": item.evidence.total_resamples,
                        }
                        if isinstance(item.evidence, PairedBootstrapEvidence)
                        else {
                            "kind": "successive_halving",
                            "rank": item.evidence.rank,
                            "prior_survivor_count": item.evidence.prior_survivor_count,
                            "target_survivor_count": item.evidence.target_survivor_count,
                            "newly_eliminated": item.evidence.newly_eliminated,
                        }
                    ),
                    "disposition": item.disposition,
                }
                for item in value.decisions
            ],
            "policy_kind": value.policy_kind,
            "policy_version": value.policy_version,
        }


def _numbers(value: object, label: str) -> tuple[int | float, ...]:
    return tuple(raw_number(item, label) for item in elements(value, label))


@dataclass(frozen=True, slots=True)
class ProposalIdentity:
    proposal_index: int
    cohort_index: int
    cohort_slot: int
    source: ProposalSource
    source_attempt: int
    candidate_id: str
    fingerprint: str
    canonical_config: str

    @staticmethod
    def decode(item: JsonObject) -> ProposalIdentity:
        return ProposalIdentity(
            integer(item["proposal_index"], "proposal index"),
            integer(item["cohort_index"], "cohort index"),
            integer(item["cohort_slot"], "cohort slot"),
            literal(item["source"], _PROPOSAL_SOURCES, "proposal source"),
            integer(item["source_attempt"], "source attempt", positive=True),
            string(item["candidate_id"], "candidate id"),
            string(item["fingerprint"], "candidate fingerprint"),
            string(item["canonical_config"], "canonical config"),
        )

    def encode(self) -> JsonObject:
        return {
            "proposal_index": self.proposal_index,
            "cohort_index": self.cohort_index,
            "cohort_slot": self.cohort_slot,
            "source": self.source,
            "source_attempt": self.source_attempt,
            "candidate_id": self.candidate_id,
            "fingerprint": self.fingerprint,
            "canonical_config": self.canonical_config,
        }


_IDENTITY_FIELDS = {
    "proposal_index",
    "cohort_index",
    "cohort_slot",
    "source",
    "source_attempt",
    "candidate_id",
    "fingerprint",
    "canonical_config",
}


@dataclass(frozen=True, slots=True)
class ProposalCreatedPayload:
    event_type: ClassVar[EventType] = "proposal_created"
    identity: ProposalIdentity
    frontier_id: str
    frontier_observation_ids: tuple[str, ...]
    proposer_version: str
    origin: str | None
    acquisition: int | float | None
    prediction: int | float | None
    uncertainty: int | float | None
    parent_candidate_id: str | None

    @staticmethod
    def decode(value: object) -> ProposalCreatedPayload:
        item = object_fields(
            value,
            _IDENTITY_FIELDS
            | {
                "frontier_id",
                "frontier_observation_ids",
                "proposer_version",
                "origin",
                "acquisition",
                "prediction",
                "uncertainty",
                "parent_candidate_id",
            },
            "proposal payload",
        )
        return ProposalCreatedPayload(
            ProposalIdentity.decode(item),
            string(item["frontier_id"], "frontier id"),
            strings(item["frontier_observation_ids"], "proposal frontier"),
            string(item["proposer_version"], "proposer version"),
            optional_string(item["origin"], "proposal origin"),
            optional_raw_number(item["acquisition"], "proposal acquisition"),
            optional_raw_number(item["prediction"], "proposal prediction"),
            optional_raw_number(item["uncertainty"], "proposal uncertainty"),
            optional_string(item["parent_candidate_id"], "proposal parent candidate id"),
        )

    def encode(self) -> JsonObject:
        return {
            **self.identity.encode(),
            "frontier_id": self.frontier_id,
            "frontier_observation_ids": list(self.frontier_observation_ids),
            "proposer_version": self.proposer_version,
            "origin": self.origin,
            "acquisition": self.acquisition,
            "prediction": self.prediction,
            "uncertainty": self.uncertainty,
            "parent_candidate_id": self.parent_candidate_id,
        }


@dataclass(frozen=True, slots=True)
class ProposalAcceptedPayload:
    event_type: ClassVar[EventType] = "proposal_accepted"
    identity: ProposalIdentity
    panel_response_fingerprints: tuple[str, ...]

    @staticmethod
    def decode(value: object) -> ProposalAcceptedPayload:
        item = object_fields(
            value, _IDENTITY_FIELDS | {"panel_response_fingerprints"}, "proposal acceptance"
        )
        return ProposalAcceptedPayload(
            ProposalIdentity.decode(item),
            strings(item["panel_response_fingerprints"], "panel response fingerprints"),
        )

    def encode(self) -> JsonObject:
        return {
            **self.identity.encode(),
            "panel_response_fingerprints": list(self.panel_response_fingerprints),
        }


@dataclass(frozen=True, slots=True)
class PanelFieldError:
    field: str
    message: str
    candidate_index: int | None

    @staticmethod
    def decode(value: object) -> PanelFieldError:
        item = object_fields(value, {"field", "message", "candidate_index"}, "panel field error")
        return PanelFieldError(
            string(item["field"], "panel field error field"),
            string(item["message"], "panel field error message"),
            optional_integer(item["candidate_index"], "panel field error candidate index"),
        )

    def encode(self) -> JsonObject:
        return {
            "field": self.field,
            "message": self.message,
            "candidate_index": self.candidate_index,
        }


@dataclass(frozen=True, slots=True)
class PanelRejection:
    opponent_id: str
    errors: tuple[PanelFieldError, ...]

    @staticmethod
    def decode(value: object) -> PanelRejection:
        item = object_fields(value, {"opponent_id", "errors"}, "panel rejection")
        return PanelRejection(
            string(item["opponent_id"], "rejected opponent id"),
            tuple(
                PanelFieldError.decode(entry)
                for entry in elements(item["errors"], "panel rejection errors")
            ),
        )

    def encode(self) -> JsonObject:
        return {
            "opponent_id": self.opponent_id,
            "errors": [entry.encode() for entry in self.errors],
        }


@dataclass(frozen=True, slots=True)
class ProposalRejectedPayload:
    event_type: ClassVar[EventType] = "proposal_rejected"
    identity: ProposalIdentity
    reason: RejectionReason
    errors: tuple[PanelRejection, ...]

    @staticmethod
    def decode(value: object) -> ProposalRejectedPayload:
        item = object_fields(value, _IDENTITY_FIELDS | {"reason", "errors"}, "proposal rejection")
        raw_errors = item["errors"]
        if not isinstance(raw_errors, list):
            raise ValueError("proposal rejection errors must be an array")
        return ProposalRejectedPayload(
            ProposalIdentity.decode(item),
            literal(item["reason"], ("duplicate", "semantic_validation"), "rejection reason"),
            tuple(PanelRejection.decode(entry) for entry in raw_errors),
        )

    def encode(self) -> JsonObject:
        return {
            **self.identity.encode(),
            "reason": self.reason,
            "errors": [entry.encode() for entry in self.errors],
        }


@dataclass(frozen=True, slots=True)
class CohortCompletedPayload:
    event_type: ClassVar[EventType] = "cohort_completed"
    cohort_index: int
    candidate_ids: tuple[str, ...]
    retained_candidate_ids: tuple[str, ...]
    proposal_sources: tuple[str, ...]
    schedule_version: str
    final_frontier_id: str

    @staticmethod
    def decode(value: object) -> CohortCompletedPayload:
        item = object_fields(
            value,
            {
                "cohort_index",
                "candidate_ids",
                "retained_candidate_ids",
                "proposal_sources",
                "schedule_version",
                "final_frontier_id",
            },
            "cohort completion",
        )
        return CohortCompletedPayload(
            integer(item["cohort_index"], "cohort index"),
            strings(item["candidate_ids"], "cohort candidate ids"),
            strings(item["retained_candidate_ids"], "cohort retained candidate ids"),
            strings(item["proposal_sources"], "cohort proposal sources"),
            string(item["schedule_version"], "cohort schedule version"),
            string(item["final_frontier_id"], "cohort frontier id"),
        )

    def encode(self) -> JsonObject:
        return {
            "cohort_index": self.cohort_index,
            "candidate_ids": list(self.candidate_ids),
            "retained_candidate_ids": list(self.retained_candidate_ids),
            "proposal_sources": list(self.proposal_sources),
            "schedule_version": self.schedule_version,
            "final_frontier_id": self.final_frontier_id,
        }


@dataclass(frozen=True, slots=True)
class PairIdentity:
    phase: Phase
    candidate_id: str
    task_id: str
    pair_id: str
    opponent_id: str
    search_effort: SearchEffort

    @staticmethod
    def decode(item: JsonObject) -> PairIdentity:
        return PairIdentity(
            literal(item["phase"], _PHASES, "pair phase"),
            string(item["candidate_id"], "candidate id"),
            string(item["task_id"], "task id"),
            string(item["pair_id"], "pair id"),
            string(item["opponent_id"], "opponent id"),
            decode_effort(item["search_effort"], "pair search effort"),
        )

    def encode(self) -> JsonObject:
        return {
            "phase": self.phase,
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "pair_id": self.pair_id,
            "opponent_id": self.opponent_id,
            "search_effort": encode_effort(self.search_effort),
        }


_PAIR_IDENTITY_FIELDS = {
    "phase",
    "candidate_id",
    "task_id",
    "pair_id",
    "opponent_id",
    "search_effort",
}


def _decode_diagnostic_task(value: object) -> DiagnosticPairTask:
    item = object_fields(
        value,
        {
            "pair_id",
            "edge_id",
            "ordinal",
            "left_candidate_id",
            "right_candidate_id",
            "seed",
            "search_effort",
        },
        "diagnostic task",
    )
    return DiagnosticPairTask(
        string(item["pair_id"], "diagnostic pair id"),
        string(item["edge_id"], "diagnostic edge id"),
        integer(item["ordinal"], "diagnostic ordinal"),
        string(item["left_candidate_id"], "diagnostic left candidate id"),
        string(item["right_candidate_id"], "diagnostic right candidate id"),
        integer(item["seed"], "diagnostic seed"),
        decode_effort(item["search_effort"], "diagnostic search effort"),
    )


def _encode_diagnostic_task(task: DiagnosticPairTask) -> JsonObject:
    return {
        "pair_id": task.pair_id,
        "edge_id": task.edge_id,
        "ordinal": task.ordinal,
        "left_candidate_id": task.left_candidate_id,
        "right_candidate_id": task.right_candidate_id,
        "seed": task.seed,
        "search_effort": encode_effort(task.search_effort),
    }


@dataclass(frozen=True, slots=True)
class DiagnosticPairStartedPayload:
    event_type: ClassVar[EventType] = "diagnostic_pair_started"
    task: DiagnosticPairTask

    @staticmethod
    def decode(value: object) -> DiagnosticPairStartedPayload:
        return DiagnosticPairStartedPayload(_decode_diagnostic_task(value))

    def encode(self) -> JsonObject:
        return _encode_diagnostic_task(self.task)


@dataclass(frozen=True, slots=True)
class DiagnosticPairCompletedPayload:
    event_type: ClassVar[EventType] = "diagnostic_pair_completed"
    task: DiagnosticPairTask
    games: tuple[JsonObject, JsonObject]

    @staticmethod
    def decode(value: object) -> DiagnosticPairCompletedPayload:
        item = object_fields(value, {"task", "games"}, "diagnostic completion")
        games = item["games"]
        if not isinstance(games, list) or len(games) != 2:
            raise ValueError("diagnostic completion needs two games")
        return DiagnosticPairCompletedPayload(
            _decode_diagnostic_task(item["task"]),
            (json_object(games[0], "diagnostic game"), json_object(games[1], "diagnostic game")),
        )

    def encode(self) -> JsonObject:
        return {
            "task": _encode_diagnostic_task(self.task),
            "games": [dict(self.games[0]), dict(self.games[1])],
        }


@dataclass(frozen=True, slots=True)
class DiagnosticPairFailedPayload:
    event_type: ClassVar[EventType] = "diagnostic_pair_failed"
    task: DiagnosticPairTask
    kind: str
    message: str

    @staticmethod
    def decode(value: object) -> DiagnosticPairFailedPayload:
        item = object_fields(value, {"task", "kind", "message"}, "diagnostic failure")
        return DiagnosticPairFailedPayload(
            _decode_diagnostic_task(item["task"]),
            string(item["kind"], "diagnostic failure kind"),
            string(item["message"], "diagnostic failure message"),
        )

    def encode(self) -> JsonObject:
        return {
            "task": _encode_diagnostic_task(self.task),
            "kind": self.kind,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class PairStartedPayload:
    event_type: ClassVar[EventType] = "pair_started"
    identity: PairIdentity
    task_seed: int

    @staticmethod
    def decode(value: object) -> PairStartedPayload:
        item = object_fields(value, _PAIR_IDENTITY_FIELDS | {"task_seed"}, "pair start")
        return PairStartedPayload(
            PairIdentity.decode(item), integer(item["task_seed"], "task seed")
        )

    def encode(self) -> JsonObject:
        return {**self.identity.encode(), "task_seed": self.task_seed}


@dataclass(frozen=True, slots=True)
class PairCompletedPayload:
    event_type: ClassVar[EventType] = "pair_completed"
    identity: PairIdentity
    games: tuple[JsonObject, JsonObject]
    pair_utility: int | float

    @staticmethod
    def decode(value: object) -> PairCompletedPayload:
        item = object_fields(
            value, _PAIR_IDENTITY_FIELDS | {"games", "pair_utility"}, "pair completion"
        )
        games = item["games"]
        if not isinstance(games, list) or len(games) != 2:
            raise ValueError("pair completion must record exactly two games")
        first, second = (
            json_object(games[0], "completed game"),
            json_object(games[1], "completed game"),
        )
        return PairCompletedPayload(
            PairIdentity.decode(item),
            (first, second),
            raw_number(item["pair_utility"], "pair utility"),
        )

    def encode(self) -> JsonObject:
        return {
            **self.identity.encode(),
            "games": [dict(self.games[0]), dict(self.games[1])],
            "pair_utility": self.pair_utility,
        }


@dataclass(frozen=True, slots=True)
class PairFailedPayload:
    event_type: ClassVar[EventType] = "pair_failed"
    identity: PairIdentity
    kind: str
    command: tuple[str, ...]
    returncode: int | None
    stderr: str
    stdout: str
    partial_output: tuple[str, ...]

    @staticmethod
    def decode(value: object) -> PairFailedPayload:
        item = object_fields(
            value,
            _PAIR_IDENTITY_FIELDS
            | {"kind", "command", "returncode", "stderr", "stdout", "partial_output"},
            "pair failure",
        )
        return PairFailedPayload(
            PairIdentity.decode(item),
            string(item["kind"], "pair failure kind"),
            strings(item["command"], "pair failure command"),
            optional_integer(item["returncode"], "pair failure return code"),
            string(item["stderr"], "pair failure stderr"),
            string(item["stdout"], "pair failure stdout"),
            strings(item["partial_output"], "pair failure partial output"),
        )

    def encode(self) -> JsonObject:
        return {
            **self.identity.encode(),
            "kind": self.kind,
            "command": list(self.command),
            "returncode": self.returncode,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "partial_output": list(self.partial_output),
        }


@dataclass(frozen=True, slots=True)
class CandidateFailedPayload:
    event_type: ClassVar[EventType] = "candidate_failed"
    policy_version: str
    reason: str
    cohort_index: int
    candidate_id: str
    triggering_pair: PairIdentity
    started_attempts: int
    failed_attempts: int
    censored_attempts: int
    completed_tuning_pair_ids: tuple[str, ...]

    @staticmethod
    def decode(value: object) -> CandidateFailedPayload:
        item = object_fields(
            value,
            {
                "policy_version",
                "reason",
                "cohort_index",
                "candidate_id",
                "triggering_pair",
                "started_attempts",
                "failed_attempts",
                "censored_attempts",
                "completed_tuning_pair_ids",
            },
            "candidate failure",
        )
        return CandidateFailedPayload(
            string(item["policy_version"], "candidate failure policy version"),
            string(item["reason"], "candidate failure reason"),
            integer(item["cohort_index"], "candidate failure cohort index"),
            string(item["candidate_id"], "failed candidate id"),
            PairIdentity.decode(
                object_fields(item["triggering_pair"], _PAIR_IDENTITY_FIELDS, "triggering pair")
            ),
            integer(item["started_attempts"], "started attempts", positive=True),
            integer(item["failed_attempts"], "failed attempts"),
            integer(item["censored_attempts"], "censored attempts"),
            strings(item["completed_tuning_pair_ids"], "completed tuning pair ids"),
        )

    def encode(self) -> JsonObject:
        return {
            "policy_version": self.policy_version,
            "reason": self.reason,
            "cohort_index": self.cohort_index,
            "candidate_id": self.candidate_id,
            "triggering_pair": self.triggering_pair.encode(),
            "started_attempts": self.started_attempts,
            "failed_attempts": self.failed_attempts,
            "censored_attempts": self.censored_attempts,
            "completed_tuning_pair_ids": list(self.completed_tuning_pair_ids),
        }


@dataclass(frozen=True, slots=True)
class RunInterruptedPayload:
    event_type: ClassVar[EventType] = "run_interrupted"
    stage: str
    pair_id: str | None

    @staticmethod
    def decode(value: object) -> RunInterruptedPayload:
        item = object_fields(value, {"stage", "pair_id"}, "run interruption")
        return RunInterruptedPayload(
            string(item["stage"], "interruption stage"),
            optional_string(item["pair_id"], "interruption pair id"),
        )

    def encode(self) -> JsonObject:
        return {"stage": self.stage, "pair_id": self.pair_id}


@dataclass(frozen=True, slots=True)
class RunFailedPayload:
    event_type: ClassVar[EventType] = "run_failed"
    kind: Literal["configuration"]
    message: str

    @staticmethod
    def decode(value: object) -> RunFailedPayload:
        item = object_fields(value, {"kind", "message"}, "run failure")
        return RunFailedPayload(
            literal(item["kind"], ("configuration",), "run failure kind"),
            string(item["message"], "run failure message"),
        )

    def encode(self) -> JsonObject:
        return {"kind": self.kind, "message": self.message}


@dataclass(frozen=True, slots=True)
class ObservationCompletedPayload:
    event_type: ClassVar[EventType] = "observation_completed"
    observation_id: str
    candidate_id: str
    phase: Phase
    objective_epoch_id: str
    corpus_id: str
    prefix_id: str
    prefix_task_ids: tuple[str, ...]
    prefix_length: int
    search_effort: SearchEffort
    pair_utilities: tuple[int | float, ...]
    estimate: JsonObject
    counts: JsonObject

    @staticmethod
    def decode(value: object) -> ObservationCompletedPayload:
        item = object_fields(
            value,
            {
                "observation_id",
                "candidate_id",
                "phase",
                "objective_epoch_id",
                "corpus_id",
                "prefix_id",
                "prefix_task_ids",
                "prefix_length",
                "search_effort",
                "pair_utilities",
                "estimate",
                "counts",
            },
            "observation",
        )
        return ObservationCompletedPayload(
            string(item["observation_id"], "observation id"),
            string(item["candidate_id"], "candidate id"),
            literal(item["phase"], _PHASES, "observation phase"),
            string(item["objective_epoch_id"], "objective epoch id"),
            string(item["corpus_id"], "corpus id"),
            string(item["prefix_id"], "prefix id"),
            strings(item["prefix_task_ids"], "prefix task ids"),
            integer(item["prefix_length"], "prefix length", positive=True),
            decode_effort(item["search_effort"], "observation search effort"),
            _numbers(item["pair_utilities"], "observation pair utilities"),
            json_object(item["estimate"], "observation estimate"),
            json_object(item["counts"], "observation counts"),
        )

    def encode(self) -> JsonObject:
        return {
            "observation_id": self.observation_id,
            "candidate_id": self.candidate_id,
            "phase": self.phase,
            "objective_epoch_id": self.objective_epoch_id,
            "corpus_id": self.corpus_id,
            "prefix_id": self.prefix_id,
            "prefix_task_ids": list(self.prefix_task_ids),
            "prefix_length": self.prefix_length,
            "search_effort": encode_effort(self.search_effort),
            "pair_utilities": list(self.pair_utilities),
            "estimate": dict(self.estimate),
            "counts": dict(self.counts),
        }


@dataclass(frozen=True, slots=True)
class FinalistsSelectedPayload:
    event_type: ClassVar[EventType] = "finalists_selected"
    finalist_ids: tuple[str, ...]
    tuning_estimates: JsonObject
    objective_epoch_id: str
    corpus_id: str
    prefix_id: str
    prefix_task_ids: tuple[str, ...]
    search_effort: SearchEffort
    selection_rule_version: str

    @staticmethod
    def decode(value: object) -> FinalistsSelectedPayload:
        item = object_fields(
            value,
            {
                "finalist_ids",
                "tuning_estimates",
                "objective_epoch_id",
                "corpus_id",
                "prefix_id",
                "prefix_task_ids",
                "search_effort",
                "selection_rule_version",
            },
            "finalists",
        )
        return FinalistsSelectedPayload(
            strings(item["finalist_ids"], "finalist ids"),
            json_object(item["tuning_estimates"], "tuning estimates"),
            string(item["objective_epoch_id"], "objective epoch id"),
            string(item["corpus_id"], "corpus id"),
            string(item["prefix_id"], "prefix id"),
            strings(item["prefix_task_ids"], "prefix task ids"),
            decode_effort(item["search_effort"], "selection search effort"),
            string(item["selection_rule_version"], "selection rule version"),
        )

    def encode(self) -> JsonObject:
        return {
            "finalist_ids": list(self.finalist_ids),
            "tuning_estimates": dict(self.tuning_estimates),
            "objective_epoch_id": self.objective_epoch_id,
            "corpus_id": self.corpus_id,
            "prefix_id": self.prefix_id,
            "prefix_task_ids": list(self.prefix_task_ids),
            "search_effort": encode_effort(self.search_effort),
            "selection_rule_version": self.selection_rule_version,
        }


@dataclass(frozen=True, slots=True)
class RunCompletedPayload:
    event_type: ClassVar[EventType] = "run_completed"
    manifest_fingerprint: str
    accepted_ids: tuple[str, ...]
    finalist_ids: tuple[str, ...]
    evidence_counts: JsonObject
    validation_claim: str
    objective_epoch_id: str
    validation_prefix_id: str
    validation_search_effort: SearchEffort
    missing_production_axes: tuple[str, ...]
    cohort_frontier_id: str

    @staticmethod
    def decode(value: object) -> RunCompletedPayload:
        item = object_fields(
            value,
            {
                "manifest_fingerprint",
                "accepted_ids",
                "finalist_ids",
                "evidence_counts",
                "validation_claim",
                "objective_epoch_id",
                "validation_prefix_id",
                "validation_search_effort",
                "missing_production_axes",
                "cohort_frontier_id",
            },
            "run completion",
        )
        return RunCompletedPayload(
            string(item["manifest_fingerprint"], "manifest fingerprint"),
            strings(item["accepted_ids"], "accepted ids"),
            strings(item["finalist_ids"], "finalist ids"),
            json_object(item["evidence_counts"], "evidence counts"),
            string(item["validation_claim"], "validation claim"),
            string(item["objective_epoch_id"], "objective epoch id"),
            string(item["validation_prefix_id"], "validation prefix id"),
            decode_effort(item["validation_search_effort"], "validation search effort"),
            strings(item["missing_production_axes"], "missing production axes"),
            string(item["cohort_frontier_id"], "cohort frontier id"),
        )

    def encode(self) -> JsonObject:
        return {
            "manifest_fingerprint": self.manifest_fingerprint,
            "accepted_ids": list(self.accepted_ids),
            "finalist_ids": list(self.finalist_ids),
            "evidence_counts": dict(self.evidence_counts),
            "validation_claim": self.validation_claim,
            "objective_epoch_id": self.objective_epoch_id,
            "validation_prefix_id": self.validation_prefix_id,
            "validation_search_effort": encode_effort(self.validation_search_effort),
            "missing_production_axes": list(self.missing_production_axes),
            "cohort_frontier_id": self.cohort_frontier_id,
        }


@dataclass(frozen=True, slots=True)
class BudgetExtendedPayload:
    """An operator-recorded increase to one or more frozen pair budgets.

    Deltas are non-negative and at least one is strictly positive. The event is
    append-only evidence: it never edits ``manifest.compute_budget``. Replay
    folds the ordered deltas into ``ReplayState.effective_budget`` and, when the
    run had already completed, re-opens it at the next cohort boundary.
    """

    event_type: ClassVar[EventType] = "budget_extended"
    tuning_pair_attempts_delta: int
    validation_pair_attempts_delta: int
    diagnostic_pair_attempts_delta: int
    reason: str
    requested_at: str

    @staticmethod
    def decode(value: object) -> BudgetExtendedPayload:
        item = object_fields(
            value,
            {
                "tuning_pair_attempts_delta",
                "validation_pair_attempts_delta",
                "diagnostic_pair_attempts_delta",
                "reason",
                "requested_at",
            },
            "budget extension",
        )
        deltas = tuple(
            integer(item[name], label)
            for name, label in (
                ("tuning_pair_attempts_delta", "tuning pair attempts delta"),
                ("validation_pair_attempts_delta", "validation pair attempts delta"),
                ("diagnostic_pair_attempts_delta", "diagnostic pair attempts delta"),
            )
        )
        if any(delta < 0 for delta in deltas):
            raise ValueError("budget extension deltas must be non-negative")
        if not any(deltas):
            raise ValueError("budget extension must raise at least one budget")
        return BudgetExtendedPayload(
            deltas[0],
            deltas[1],
            deltas[2],
            string(item["reason"], "budget extension reason"),
            string(item["requested_at"], "budget extension request time"),
        )

    def encode(self) -> JsonObject:
        return {
            "tuning_pair_attempts_delta": self.tuning_pair_attempts_delta,
            "validation_pair_attempts_delta": self.validation_pair_attempts_delta,
            "diagnostic_pair_attempts_delta": self.diagnostic_pair_attempts_delta,
            "reason": self.reason,
            "requested_at": self.requested_at,
        }


EventPayload = (
    AllocationDecidedPayload
    | BudgetExtendedPayload
    | ShadowRaceDecidedPayload
    | ProposalCreatedPayload
    | ProposalAcceptedPayload
    | ProposalRejectedPayload
    | CohortCompletedPayload
    | PairStartedPayload
    | PairCompletedPayload
    | PairFailedPayload
    | DiagnosticPairStartedPayload
    | DiagnosticPairCompletedPayload
    | DiagnosticPairFailedPayload
    | CandidateFailedPayload
    | RunInterruptedPayload
    | RunFailedPayload
    | ObservationCompletedPayload
    | FinalistsSelectedPayload
    | RunCompletedPayload
)

_DECODERS: dict[EventType, Callable[[object], EventPayload]] = {
    "allocation_decided": AllocationDecidedPayload.decode,
    "budget_extended": BudgetExtendedPayload.decode,
    "shadow_race_decided": ShadowRaceDecidedPayload.decode,
    "proposal_created": ProposalCreatedPayload.decode,
    "proposal_accepted": ProposalAcceptedPayload.decode,
    "proposal_rejected": ProposalRejectedPayload.decode,
    "cohort_completed": CohortCompletedPayload.decode,
    "pair_started": PairStartedPayload.decode,
    "pair_completed": PairCompletedPayload.decode,
    "pair_failed": PairFailedPayload.decode,
    "diagnostic_pair_started": DiagnosticPairStartedPayload.decode,
    "diagnostic_pair_completed": DiagnosticPairCompletedPayload.decode,
    "diagnostic_pair_failed": DiagnosticPairFailedPayload.decode,
    "candidate_failed": CandidateFailedPayload.decode,
    "run_interrupted": RunInterruptedPayload.decode,
    "run_failed": RunFailedPayload.decode,
    "observation_completed": ObservationCompletedPayload.decode,
    "finalists_selected": FinalistsSelectedPayload.decode,
    "run_completed": RunCompletedPayload.decode,
}


def decode_payload(event_type: EventType, value: object) -> EventPayload:
    return _DECODERS[event_type](value)


def narrow_event_type(value: object) -> EventType:
    for name in _DECODERS:
        if value == name:
            return name
    raise ValueError(f"unknown evidence event type {value!r}")
