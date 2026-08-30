"""Closed tagged union of immutable version-4 evidence event payloads.

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

EventType = Literal[
    "proposal_created",
    "proposal_accepted",
    "proposal_rejected",
    "cohort_completed",
    "pair_started",
    "pair_completed",
    "pair_failed",
    "run_interrupted",
    "run_failed",
    "observation_completed",
    "finalists_selected",
    "run_completed",
]

SCIENTIFIC: frozenset[EventType] = frozenset(
    {
        "proposal_created",
        "proposal_accepted",
        "proposal_rejected",
        "cohort_completed",
        "pair_completed",
        "observation_completed",
        "finalists_selected",
        "run_completed",
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

ProposalSource = Literal["schema_default", "bootstrap_random", "smac_model", "random_reserve"]
Phase = Literal["tuning", "validation"]
RejectionReason = Literal["duplicate", "semantic_validation"]


def _numbers(value: object, label: str) -> tuple[int | float, ...]:
    return tuple(raw_number(item, label) for item in elements(value, label))


@dataclass(frozen=True, slots=True)
class ProposalIdentity:
    proposal_index: int
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
            "cohort_slot": self.cohort_slot,
            "source": self.source,
            "source_attempt": self.source_attempt,
            "candidate_id": self.candidate_id,
            "fingerprint": self.fingerprint,
            "canonical_config": self.canonical_config,
        }


_IDENTITY_FIELDS = {
    "proposal_index",
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
    candidate_ids: tuple[str, ...]
    sources: tuple[str, ...]
    schedule_version: str
    final_frontier_id: str

    @staticmethod
    def decode(value: object) -> CohortCompletedPayload:
        item = object_fields(
            value,
            {"candidate_ids", "sources", "schedule_version", "final_frontier_id"},
            "cohort completion",
        )
        return CohortCompletedPayload(
            strings(item["candidate_ids"], "cohort candidate ids"),
            strings(item["sources"], "cohort sources"),
            string(item["schedule_version"], "cohort schedule version"),
            string(item["final_frontier_id"], "cohort frontier id"),
        )

    def encode(self) -> JsonObject:
        return {
            "candidate_ids": list(self.candidate_ids),
            "sources": list(self.sources),
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
    budget: int

    @staticmethod
    def decode(item: JsonObject) -> PairIdentity:
        return PairIdentity(
            literal(item["phase"], _PHASES, "pair phase"),
            string(item["candidate_id"], "candidate id"),
            string(item["task_id"], "task id"),
            string(item["pair_id"], "pair id"),
            string(item["opponent_id"], "opponent id"),
            integer(item["budget"], "pair budget", positive=True),
        )

    def encode(self) -> JsonObject:
        return {
            "phase": self.phase,
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "pair_id": self.pair_id,
            "opponent_id": self.opponent_id,
            "budget": self.budget,
        }


_PAIR_IDENTITY_FIELDS = {"phase", "candidate_id", "task_id", "pair_id", "opponent_id", "budget"}


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
    search_effort: int
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
            integer(item["search_effort"], "observation search effort", positive=True),
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
            "search_effort": self.search_effort,
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
    search_effort: int
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
            integer(item["search_effort"], "selection search effort"),
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
            "search_effort": self.search_effort,
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
    validation_search_effort: int
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
            integer(item["validation_search_effort"], "validation search effort"),
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
            "validation_search_effort": self.validation_search_effort,
            "missing_production_axes": list(self.missing_production_axes),
            "cohort_frontier_id": self.cohort_frontier_id,
        }


EventPayload = (
    ProposalCreatedPayload
    | ProposalAcceptedPayload
    | ProposalRejectedPayload
    | CohortCompletedPayload
    | PairStartedPayload
    | PairCompletedPayload
    | PairFailedPayload
    | RunInterruptedPayload
    | RunFailedPayload
    | ObservationCompletedPayload
    | FinalistsSelectedPayload
    | RunCompletedPayload
)

_DECODERS: dict[EventType, Callable[[object], EventPayload]] = {
    "proposal_created": ProposalCreatedPayload.decode,
    "proposal_accepted": ProposalAcceptedPayload.decode,
    "proposal_rejected": ProposalRejectedPayload.decode,
    "cohort_completed": CohortCompletedPayload.decode,
    "pair_started": PairStartedPayload.decode,
    "pair_completed": PairCompletedPayload.decode,
    "pair_failed": PairFailedPayload.decode,
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
