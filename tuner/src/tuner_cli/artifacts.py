"""Versioned immutable manifest encoding and strict artifact decoding."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, TypeAlias

from .codec import (
    JsonObject,
    integer,
    json_object,
    literal,
    number,
    object_fields,
    strict_json,
    string,
)
from .constraints import (
    CONSTRAINT_POLICY_VERSION,
    Constraints,
    decode_constraints,
    encode_constraints,
    validate_constraints,
)
from .domain import (
    ComputeBudget,
    ObjectiveEpoch,
    Opponent,
    OpponentPanel,
    OpponentRole,
    Phase,
    ProposalSource,
    SearchEffort,
    ShadowMethodVersion,
    TaskCase,
    TaskCorpus,
    TaskPrefix,
)
from .effort import decode_effort, encode_effort, exceeds_same_kind
from .identity import (
    canonical_json,
    fingerprint,
    objective_epoch,
    opponent_panel,
    task_prefix,
)
from .objective import ResolvedObjective
from .proposer import (
    COST_POLICY_VERSION,
    POLICIES,
    POLICY_VERSION,
    SAMPLER_VERSION,
    ProposerPolicy,
    challenger_source_schedule,
    derived_seed,
    source_schedule,
)
from .schema import GameSpec, decode_game_spec
from .smac_proposer import ADAPTER_VERSION
from .tasks import (
    build_corpus,
    selected_prefix,
    tuning_blocks,
    validate_cycle_endpoint,
    verify_weighted_corpus,
)

SCHEMA_VERSION = 5
CANDIDATE_FAILURE_POLICY_VERSION = "terminal-candidate-refill-v1"
MINIMUM_ELIGIBLE_PREFIX_PAIRS = 12
_OPPONENT_ROLES: tuple[OpponentRole, OpponentRole] = ("default", "historical_reference")
_OPPONENT_SOURCES: tuple[
    Literal["schema_default", "inline"], Literal["schema_default", "inline"]
] = (
    "schema_default",
    "inline",
)


def configspace_version() -> str:
    return version("ConfigSpace")


def runtime_versions() -> dict[str, str]:
    return {
        "smac": version("smac"),
        "configspace": configspace_version(),
        "scikit_learn": version("scikit-learn"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }


@dataclass(frozen=True, slots=True)
class ProposerSpecification:
    policy: ProposerPolicy
    proposal_seed: int
    task_seed: int
    cohort_size: int
    finalists: int
    bootstrap_candidates: int
    random_reserve_candidates: int
    source_schedule: tuple[ProposalSource, ...]
    challenger_source_schedule: tuple[ProposalSource, ...]
    bootstrap_seed: int
    reserve_seed: int
    runtime_versions: tuple[tuple[str, str], ...]
    constraints: Constraints

    @property
    def model_candidates(self) -> int:
        return sum(
            source not in {"schema_default", "bootstrap_random", "random_reserve"}
            for source in self.source_schedule
        )

    @property
    def attempt_cap(self) -> int:
        return max(100, self.cohort_size * 100)

    def encoded(self) -> JsonObject:
        return {
            "policy_version": POLICY_VERSION,
            "policy": self.policy,
            "guided_source": {
                "smac_mixed": "smac_model",
                "random": "random_search",
                "qmc": "qmc_search",
                "irace_generational": "irace_model",
            }[self.policy],
            "constraint_policy_version": CONSTRAINT_POLICY_VERSION,
            "constraints": encode_constraints(self.constraints),
            "proposal_seed": self.proposal_seed,
            "task_seed": self.task_seed,
            "cohort_size": self.cohort_size,
            "finalists": self.finalists,
            "bootstrap_candidates": self.bootstrap_candidates,
            "model_candidates": self.model_candidates,
            "random_reserve_candidates": self.random_reserve_candidates,
            "source_schedule": list(self.source_schedule),
            "challenger_source_schedule": list(self.challenger_source_schedule),
            "attempt_cap": self.attempt_cap,
            "seed_derivation_version": "proposal-seed-v1",
            "cost_policy_version": COST_POLICY_VERSION,
            "bootstrap_sampler_version": SAMPLER_VERSION,
            "reserve_sampler_version": SAMPLER_VERSION,
            "bootstrap_seed": self.bootstrap_seed,
            "reserve_seed": self.reserve_seed,
            "guided_adapter_version": {
                "smac_mixed": ADAPTER_VERSION,
                "random": SAMPLER_VERSION,
                "qmc": "scipy-sobol-scrambled-v1",
                "irace_generational": "irace-elite-generational-v1",
            }[self.policy],
            "runtime_versions": dict(self.runtime_versions),
        }


@dataclass(frozen=True, slots=True)
class PairedBootstrapPolicySpecification:
    kind: Literal["paired_bootstrap"]
    practical_effect_margin: float
    elimination_probability_threshold: float
    resamples: int
    method_version: ShadowMethodVersion
    minimum_eligible_prefix_pairs: int = MINIMUM_ELIGIBLE_PREFIX_PAIRS

    def encoded(self) -> JsonObject:
        return {
            "kind": self.kind,
            "practical_effect_margin": self.practical_effect_margin,
            "elimination_probability_threshold": self.elimination_probability_threshold,
            "resamples": self.resamples,
            "method_version": self.method_version,
            "minimum_eligible_prefix_pairs": self.minimum_eligible_prefix_pairs,
        }


@dataclass(frozen=True, slots=True)
class SuccessiveHalvingPolicySpecification:
    kind: Literal["successive_halving"]
    method_version: Literal[
        "successive-halving-common-prefix-eta2-v1",
        "successive-halving-spare-near-tie-v1",
    ]
    reduction_factor: Literal[2]
    practical_effect_margin: float
    minimum_eligible_prefix_pairs: int
    survivor_floor: int
    ranking_rule: Literal["tuning-point-estimate-fingerprint-v1"]
    spare_margin: float = 0.0

    def encoded(self) -> JsonObject:
        return {
            "kind": self.kind,
            "method_version": self.method_version,
            "reduction_factor": self.reduction_factor,
            "practical_effect_margin": self.practical_effect_margin,
            "minimum_eligible_prefix_pairs": self.minimum_eligible_prefix_pairs,
            "survivor_floor": self.survivor_floor,
            "ranking_rule": self.ranking_rule,
            "spare_margin": self.spare_margin,
        }


ShadowPolicySpecification: TypeAlias = (
    PairedBootstrapPolicySpecification | SuccessiveHalvingPolicySpecification
)


@dataclass(frozen=True, slots=True)
class CandidateFailurePolicySpecification:
    max_pair_attempts: int = 2

    def encoded(self) -> JsonObject:
        return {
            "policy_version": CANDIDATE_FAILURE_POLICY_VERSION,
            "phase": "tuning",
            "max_pair_attempts": self.max_pair_attempts,
            "exhaustion_basis": "started_attempts",
            "overflow_source": "random_reserve",
        }


@dataclass(frozen=True, slots=True)
class ActiveEliminationSpecification:
    audit_probability: float
    shadow_policy_kind: Literal["paired_bootstrap", "successive_halving"] = "paired_bootstrap"
    shadow_method_version: ShadowMethodVersion = "stratified-paired-bootstrap-all-strata-v2"
    shadow_spare_margin: float = 0.0
    sampler_version: Literal["stage-stratified-sha256-v1"] = "stage-stratified-sha256-v1"
    safety_rule_version: Literal["suspend-after-first-audited-boundary-reversal-v1"] = (
        "suspend-after-first-audited-boundary-reversal-v1"
    )

    def encoded(self) -> JsonObject:
        return {
            "audit_probability": self.audit_probability,
            "shadow_policy_kind": self.shadow_policy_kind,
            "shadow_method_version": self.shadow_method_version,
            "shadow_spare_margin": self.shadow_spare_margin,
            "sampler_version": self.sampler_version,
            "safety_rule_version": self.safety_rule_version,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticPolicySpecification:
    maximum_reserve_slots: int = 1

    def encoded(self) -> JsonObject:
        return {
            "edge_policy_version": "connected-cycle-boundary-uncertainty-v1",
            "seed_policy_version": "diagnostic-allocation-seed-v1",
            "graph_rule_version": "direct-hoeffding-cycle-components-v1",
            "shortlist_rule_version": "objective-top-with-one-cycle-reserve-v1",
            "maximum_reserve_slots": self.maximum_reserve_slots,
        }


def proposer_specification(
    proposal_seed: int,
    task_seed: int,
    cohort_size: int,
    finalists: int,
    bootstrap_candidates: int,
    random_reserve_candidates: int,
    constraints: Constraints = (),
    versions: dict[str, str] | None = None,
    policy: ProposerPolicy = "smac_mixed",
) -> ProposerSpecification:
    schedule = source_schedule(cohort_size, bootstrap_candidates, random_reserve_candidates, policy)
    if finalists >= cohort_size:
        raise ValueError("finalists must be smaller than cohort size")
    return ProposerSpecification(
        policy,
        proposal_seed,
        task_seed,
        cohort_size,
        finalists,
        bootstrap_candidates,
        random_reserve_candidates,
        schedule,
        challenger_source_schedule(cohort_size, finalists, random_reserve_candidates, policy),
        derived_seed(proposal_seed, "bootstrap"),
        derived_seed(proposal_seed, "reserve"),
        tuple(sorted((runtime_versions() if versions is None else versions).items())),
        constraints,
    )


@dataclass(frozen=True, slots=True)
class Manifest:
    fingerprint: str
    spec: GameSpec
    objective_source_path: Path
    objective_id: str
    objective_fingerprint: str
    panel: OpponentPanel
    tuning_corpus: TaskCorpus
    production_validation_corpus: TaskCorpus
    tuning_prefix: TaskPrefix
    tuning_blocks: tuple[TaskPrefix, ...]
    validation_prefix: TaskPrefix
    epoch: ObjectiveEpoch
    proposer_spec: ProposerSpecification
    run_id: str
    game_config: str
    game_config_fingerprint: str
    effort_values: tuple[SearchEffort, SearchEffort, SearchEffort]
    compute_budget: ComputeBudget
    shadow_policy: ShadowPolicySpecification
    candidate_failure_policy: CandidateFailurePolicySpecification
    active_elimination: ActiveEliminationSpecification | None
    diagnostic_policy: DiagnosticPolicySpecification
    constraints: Constraints

    @property
    def seed(self) -> int:
        return self.proposer_spec.proposal_seed

    @property
    def task_seed(self) -> int:
        return self.proposer_spec.task_seed

    @property
    def proposer(self) -> ProposerSpecification:
        return self.proposer_spec

    @property
    def cohort_size(self) -> int:
        return self.proposer_spec.cohort_size

    @property
    def finalists(self) -> int:
        return self.proposer_spec.finalists

    @property
    def bootstrap_candidates(self) -> int:
        return self.proposer_spec.bootstrap_candidates

    @property
    def random_reserve_candidates(self) -> int:
        return self.proposer_spec.random_reserve_candidates

    @property
    def source_schedule(self) -> tuple[ProposalSource, ...]:
        return self.proposer_spec.source_schedule

    @property
    def challenger_source_schedule(self) -> tuple[ProposalSource, ...]:
        return self.proposer_spec.challenger_source_schedule

    @property
    def efforts(self) -> dict[str, SearchEffort]:
        tuning, validation, production = self.effort_values
        return {"tuning": tuning, "validation": validation, "production": production}

    @property
    def opponent(self) -> Opponent:
        return next(item for item in self.panel.opponents if item.role == "default")

    @property
    def tuning(self) -> TaskCorpus:
        return self.tuning_corpus

    @property
    def validation(self) -> TaskCorpus:
        return self.production_validation_corpus

    def prefix_cases(self, phase: Phase) -> tuple[TaskCase, ...]:
        corpus, prefix = (
            (self.tuning_corpus, self.tuning_prefix)
            if phase == "tuning"
            else (self.production_validation_corpus, self.validation_prefix)
        )
        return corpus.cases[: prefix.length]


def _opponent_dict(item: Opponent) -> JsonObject:
    return {
        "id": item.opponent_id,
        "source": item.source_id,
        "label": item.label,
        "role": item.role,
        "weight": item.weight,
        "canonical_config": item.canonical_config,
        "configuration_fingerprint": item.configuration_fingerprint,
    }


def _panel_dict(panel: OpponentPanel) -> JsonObject:
    return {
        "panel_id": panel.panel_id,
        "fingerprint": panel.fingerprint,
        "total_weight": panel.total_weight,
        "opponents": [_opponent_dict(item) for item in panel.opponents],
    }


def _case_dict(case: TaskCase) -> JsonObject:
    return {
        "task_id": case.task_id,
        "phase": case.phase,
        "ordinal": case.ordinal,
        "seed": case.seed,
        "stratum_id": case.stratum_id,
        "opponent_id": case.opponent_id,
        "opponent_fingerprint": case.opponent_fingerprint,
        "panel_fingerprint": case.panel_fingerprint,
        "game_config_fingerprint": case.game_config_fingerprint,
        "start": case.start,
    }


def _corpus_dict(corpus: TaskCorpus) -> JsonObject:
    return {
        "corpus_id": corpus.corpus_id,
        "fingerprint": corpus.fingerprint,
        "phase": corpus.phase,
        "task_policy_version": corpus.task_policy_version,
        "cases": [_case_dict(case) for case in corpus.cases],
    }


def _prefix_dict(prefix: TaskPrefix) -> JsonObject:
    return {
        "prefix_id": prefix.prefix_id,
        "corpus_id": prefix.corpus_id,
        "length": prefix.length,
        "task_ids": list(prefix.task_ids),
    }


def production_claim(
    validation_prefix: TaskPrefix,
    production_corpus: TaskCorpus,
    validation_effort: SearchEffort,
    production_effort: SearchEffort,
) -> tuple[str, tuple[str, ...]]:
    missing: list[str] = []
    if validation_prefix.length != len(
        production_corpus.cases
    ) or validation_prefix.task_ids != tuple(case.task_id for case in production_corpus.cases):
        missing.append("task_count")
    if validation_effort != production_effort:
        missing.append("search_effort")
    return ("production", ()) if not missing else ("mechanics_smoke", tuple(missing))


def _epoch_payload(
    spec: GameSpec,
    objective_id: str,
    objective_fingerprint: str,
    panel_fingerprint: str,
    start_distribution_fingerprint: str,
    tuning: TaskCorpus,
    validation: TaskCorpus,
    production_effort: SearchEffort,
    game_config_fingerprint: str,
    constraints: Constraints,
) -> JsonObject:
    payload: JsonObject = {
        "version": "objective-epoch-v1",
        "objective_id": objective_id,
        "objective_fingerprint": objective_fingerprint,
        "game_kind": spec.kind,
        "engine_fingerprint": spec.engine_fingerprint,
        "schema_fingerprint": spec.schema_fingerprint,
        "game_config_fingerprint": game_config_fingerprint,
        "panel_fingerprint": panel_fingerprint,
        "start_distribution_fingerprint": start_distribution_fingerprint,
        "tuning_corpus_fingerprint": tuning.fingerprint,
        "production_validation_corpus_fingerprint": validation.fingerprint,
        "production_validation_pairs": len(validation.cases),
        "production_search_effort": encode_effort(production_effort),
        "task_policy_version": "weighted-fair-prefix-v1",
        "utility_formula_version": "pair_mean_v1",
        "interval_method": "hoeffding_pair_bound_v1",
        "confidence_level": 0.95,
        "tie_rule_version": "paired_hoeffding_v1",
        "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
    }
    # Unconstrained runs keep their historical epoch identity; a run-scoped
    # constraint makes the run a distinct objective epoch.
    if constraints:
        payload["constraints"] = encode_constraints(constraints)
    return payload


def build_manifest(
    run_id: str,
    spec: GameSpec,
    objective: ResolvedObjective,
    seed: int,
    task_seed: int,
    cohort_size: int,
    finalists: int,
    bootstrap_candidates: int,
    random_reserve_candidates: int,
    tuning_pairs: int,
    tuning_pair_budget: int,
    validation_pair_budget: int,
    production_validation_pairs: int,
    tuning_effort: SearchEffort,
    validation_effort: SearchEffort,
    production_effort: SearchEffort,
    shadow_practical_margin: float = 0.0,
    shadow_elimination_threshold: float = 0.05,
    shadow_policy_kind: Literal["paired_bootstrap", "successive_halving"] = "paired_bootstrap",
    shadow_halving_spare_margin: float = 0.0,
    active_elimination_audit_probability: float | None = None,
    diagnostic_pair_budget: int = 0,
    proposer_policy: ProposerPolicy = "smac_mixed",
    constraints: Constraints = (),
) -> Manifest:
    validate_constraints(spec.tuning, constraints)
    proposer = proposer_specification(
        seed,
        task_seed,
        cohort_size,
        finalists,
        bootstrap_candidates,
        random_reserve_candidates,
        constraints,
        policy=proposer_policy,
    )
    if (
        isinstance(tuning_pair_budget, bool)
        or isinstance(validation_pair_budget, bool)
        or tuning_pair_budget <= 0
        or validation_pair_budget <= 0
        or isinstance(diagnostic_pair_budget, bool)
        or diagnostic_pair_budget < 0
    ):
        raise ValueError("compute budgets must be positive integers")
    if validation_pair_budget % finalists:
        raise ValueError("validation pair budget must divide finalists")
    validation_pairs = validation_pair_budget // finalists
    validate_cycle_endpoint(objective.panel, validation_pairs, "validation pairs")
    if validation_pairs > production_validation_pairs:
        raise ValueError("validation pairs cannot exceed production validation pairs")
    if tuning_pair_budget < cohort_size * tuning_pairs:
        raise ValueError("tuning pair budget cannot fund initial cohort")
    _validate_manifest_inputs(
        objective.panel,
        (tuning_pairs, production_validation_pairs),
        (tuning_effort, validation_effort, production_effort),
    )
    shadow_policy = _shadow_policy(
        shadow_practical_margin,
        shadow_elimination_threshold,
        shadow_policy_kind,
        finalists,
        shadow_halving_spare_margin,
    )
    candidate_failure_policy = CandidateFailurePolicySpecification()
    active_elimination = _active_elimination(active_elimination_audit_probability, shadow_policy)
    game_config_fingerprint = fingerprint(strict_json(objective.game_config, "game configuration"))
    tuning = build_corpus(
        "tuning", tuning_pairs, task_seed, objective.panel, game_config_fingerprint
    )
    validation = build_corpus(
        "validation",
        production_validation_pairs,
        task_seed,
        objective.panel,
        game_config_fingerprint,
    )
    efforts = (tuning_effort, validation_effort, production_effort)
    epoch = objective_epoch(
        _epoch_payload(
            spec,
            objective.objective_id,
            objective.fingerprint,
            objective.panel.fingerprint,
            objective.start_distribution_fingerprint,
            tuning,
            validation,
            efforts[2],
            game_config_fingerprint,
            constraints,
        )
    )
    raw = _encode_manifest_object(
        run_id,
        spec,
        objective.source_path,
        objective.objective_id,
        objective.fingerprint,
        objective.game_config,
        game_config_fingerprint,
        proposer,
        objective.panel,
        tuning,
        validation,
        selected_prefix(tuning, tuning_pairs),
        tuning_blocks(tuning, objective.panel),
        selected_prefix(validation, validation_pairs),
        selected_prefix(validation, production_validation_pairs),
        epoch,
        efforts,
        ComputeBudget(tuning_pair_budget, validation_pair_budget, diagnostic_pair_budget),
        shadow_policy,
        candidate_failure_policy,
        active_elimination,
        DiagnosticPolicySpecification(),
        constraints,
    )
    return decode_manifest_object({**raw, "fingerprint": fingerprint(raw)})


def _validate_manifest_inputs(
    panel: OpponentPanel,
    pairs: tuple[int, int],
    efforts: tuple[SearchEffort, SearchEffort, SearchEffort],
) -> None:
    tuning_pairs, production_validation_pairs = pairs
    for count, label in (
        (tuning_pairs, "tuning pairs"),
        (production_validation_pairs, "production validation pairs"),
    ):
        validate_cycle_endpoint(panel, count, label)
    tuning_effort, validation_effort, production_effort = efforts
    if exceeds_same_kind(tuning_effort, production_effort) or exceeds_same_kind(
        validation_effort, production_effort
    ):
        raise ValueError("observed search effort cannot exceed production effort")


def _shadow_policy(
    margin: object,
    threshold: object,
    kind: object = "paired_bootstrap",
    finalists: object = 2,
    spare_margin: object = 0.0,
) -> ShadowPolicySpecification:
    practical_margin = number(margin, "shadow practical margin")
    elimination_threshold = number(threshold, "shadow elimination threshold")
    spare = number(spare_margin, "shadow halving spare margin")
    if not 0.0 <= practical_margin <= 1.0:
        raise ValueError("shadow practical margin must be in [0.0, 1.0]")
    if not 0.0 < elimination_threshold < 0.5:
        raise ValueError("shadow elimination threshold must be in (0.0, 0.5)")
    if not 0.0 <= spare <= 1.0:
        raise ValueError("shadow halving spare margin must be in [0.0, 1.0]")
    if kind == "paired_bootstrap":
        return PairedBootstrapPolicySpecification(
            "paired_bootstrap",
            practical_margin,
            elimination_threshold,
            4096,
            "stratified-paired-bootstrap-all-strata-v2",
        )
    if kind == "successive_halving":
        if elimination_threshold != 0.05:
            raise ValueError("successive halving does not accept a non-default shadow threshold")
        method_version: Literal[
            "successive-halving-common-prefix-eta2-v1",
            "successive-halving-spare-near-tie-v1",
        ] = (
            "successive-halving-spare-near-tie-v1"
            if spare > 0.0
            else "successive-halving-common-prefix-eta2-v1"
        )
        return SuccessiveHalvingPolicySpecification(
            "successive_halving",
            method_version,
            2,
            practical_margin,
            MINIMUM_ELIGIBLE_PREFIX_PAIRS,
            integer(finalists, "finalists", positive=True),
            "tuning-point-estimate-fingerprint-v1",
            spare,
        )
    raise ValueError("unsupported shadow policy")


def _active_elimination(
    value: object | None, shadow_policy: ShadowPolicySpecification
) -> ActiveEliminationSpecification | None:
    if value is None:
        return None
    probability = number(value, "active elimination audit probability")
    if not 0.0 < probability < 1.0:
        raise ValueError("active elimination audit probability must be in (0.0, 1.0)")
    if isinstance(shadow_policy, SuccessiveHalvingPolicySpecification):
        if shadow_policy.method_version != "successive-halving-spare-near-tie-v1":
            raise ValueError(
                "active elimination with successive halving requires the gate-approved "
                "spare-near-tie policy (a positive spare margin)"
            )
        spare_margin = shadow_policy.spare_margin
    else:
        spare_margin = 0.0
    return ActiveEliminationSpecification(
        probability,
        shadow_policy.kind,
        shadow_policy.method_version,
        spare_margin,
    )


_STATISTICAL_POLICY: JsonObject = {
    "utility_formula_version": "pair_mean_v1",
    "selection_rule_version": "tuning_point_estimate_fingerprint_v1",
    "interval_method": "hoeffding_pair_bound_v1",
    "confidence_level": 0.95,
    "tie_rule_version": "paired_hoeffding_v1",
}

_LIMITATIONS: tuple[str, ...] = (
    "default-only start distribution",
    "sequential execution",
    "fixed search effort",
    "explicit resume",
)


def _game_identity_section(
    spec: GameSpec, game_config: str, game_config_fingerprint: str
) -> JsonObject:
    return {
        "binary": {"path": str(spec.binary_path), "sha256": spec.binary_sha256},
        "engine_fingerprint": spec.engine_fingerprint,
        "description": spec.raw_description,
        "description_fingerprint": spec.description_fingerprint,
        "kind": spec.kind,
        "label": spec.label,
        "game_description": spec.description,
        "tuning_schema_fingerprint": spec.schema_fingerprint,
        "game_config": game_config,
        "game_config_fingerprint": game_config_fingerprint,
    }


def _fidelity_section(
    tuning_prefix: TaskPrefix,
    validation_prefix: TaskPrefix,
    production_prefix: TaskPrefix,
    efforts: tuple[SearchEffort, SearchEffort, SearchEffort],
) -> JsonObject:
    names = ("tuning", "validation", "production")
    prefixes = (tuning_prefix, validation_prefix, production_prefix)
    section: JsonObject = {}
    for name, prefix, effort in zip(names, prefixes, efforts, strict=True):
        section[name] = {
            "task_prefix_id": prefix.prefix_id,
            "search_effort": encode_effort(effort),
        }
    return section


def _encode_manifest_object(
    run_id: str,
    spec: GameSpec,
    objective_source_path: Path,
    objective_id: str,
    objective_fingerprint: str,
    game_config: str,
    game_config_fingerprint: str,
    proposer: ProposerSpecification,
    panel: OpponentPanel,
    tuning: TaskCorpus,
    validation: TaskCorpus,
    tuning_prefix: TaskPrefix,
    blocks: tuple[TaskPrefix, ...],
    validation_prefix: TaskPrefix,
    production_prefix: TaskPrefix,
    epoch: ObjectiveEpoch,
    efforts: tuple[SearchEffort, SearchEffort, SearchEffort],
    compute_budget: ComputeBudget,
    shadow_policy: ShadowPolicySpecification,
    candidate_failure_policy: CandidateFailurePolicySpecification,
    active_elimination: ActiveEliminationSpecification | None,
    diagnostic_policy: DiagnosticPolicySpecification,
    constraints: Constraints,
) -> JsonObject:
    encoded: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "command_policy_version": POLICY_VERSION,
        "compute_budget": {
            "policy_version": "safe-boundary-pair-attempts-v1",
            "tuning_pair_attempts": compute_budget.tuning_pair_attempts,
            "validation_pair_attempts": compute_budget.validation_pair_attempts,
            "diagnostic_pair_attempts": compute_budget.diagnostic_pair_attempts,
        },
        "shadow_policy": shadow_policy.encoded(),
        "candidate_failure_policy": candidate_failure_policy.encoded(),
        "active_elimination": None if active_elimination is None else active_elimination.encoded(),
        "diagnostic_policy": diagnostic_policy.encoded(),
        **_game_identity_section(spec, game_config, game_config_fingerprint),
        "proposer": proposer.encoded(),
        "objective": {
            "source_path": str(objective_source_path),
            "objective_id": objective_id,
            "fingerprint": objective_fingerprint,
        },
        "opponent_panel": _panel_dict(panel),
        "start_distribution": {
            "kind": "default_only",
            "fingerprint": fingerprint({"kind": "default_only"}),
        },
        "corpora": {
            "tuning": _corpus_dict(tuning),
            "production_validation": _corpus_dict(validation),
        },
        "prefixes": {
            "tuning": _prefix_dict(tuning_prefix),
            "validation": _prefix_dict(validation_prefix),
        },
        "tuning_blocks": [
            {"ordinal": ordinal, "prefix": _prefix_dict(prefix)}
            for ordinal, prefix in enumerate(blocks)
        ],
        "fidelity": _fidelity_section(tuning_prefix, validation_prefix, production_prefix, efforts),
        "epoch": {"epoch_id": epoch.epoch_id, "fingerprint": epoch.fingerprint},
        **_STATISTICAL_POLICY,
        "limitations": list(_LIMITATIONS),
    }
    # Emitted only when the run actually constrains the space, so an
    # unconstrained run's manifest carries no constraint block.
    if constraints:
        encoded["constraints"] = encode_constraints(constraints)
    return encoded


_FIELDS = {
    "schema_version",
    "run_id",
    "command_policy_version",
    "compute_budget",
    "shadow_policy",
    "candidate_failure_policy",
    "active_elimination",
    "diagnostic_policy",
    "binary",
    "engine_fingerprint",
    "description",
    "description_fingerprint",
    "kind",
    "label",
    "game_description",
    "tuning_schema_fingerprint",
    "game_config",
    "game_config_fingerprint",
    "proposer",
    "objective",
    "opponent_panel",
    "start_distribution",
    "corpora",
    "prefixes",
    "tuning_blocks",
    "fidelity",
    "epoch",
    "utility_formula_version",
    "selection_rule_version",
    "interval_method",
    "confidence_level",
    "tie_rule_version",
    "limitations",
    "fingerprint",
}


def _decode_panel(value: object) -> OpponentPanel:
    raw = object_fields(
        value, {"panel_id", "fingerprint", "total_weight", "opponents"}, "opponent panel"
    )
    if not isinstance(raw["opponents"], list) or len(raw["opponents"]) < 2:
        raise ValueError("opponent panel needs at least two entries")
    opponents: list[Opponent] = []
    for item in raw["opponents"]:
        entry = object_fields(
            item,
            {
                "id",
                "source",
                "label",
                "role",
                "weight",
                "canonical_config",
                "configuration_fingerprint",
            },
            "panel opponent",
        )
        role_value = string(entry["role"], "panel role")
        source_value = string(entry["source"], "panel source")
        if role_value not in _OPPONENT_ROLES or source_value not in _OPPONENT_SOURCES:
            raise ValueError("invalid panel opponent role or source")
        role: OpponentRole = "default" if role_value == "default" else "historical_reference"
        source: Literal["schema_default", "inline"] = (
            "schema_default" if source_value == "schema_default" else "inline"
        )
        canonical = string(entry["canonical_config"], "panel configuration")
        if (
            canonical_json(strict_json(canonical, "panel configuration")) != canonical
            or fingerprint(strict_json(canonical)) != entry["configuration_fingerprint"]
        ):
            raise ValueError("panel configuration identity is invalid")
        opponents.append(
            Opponent(
                string(entry["id"], "opponent id", nonempty=True),
                source,
                string(entry["label"], "opponent label", nonempty=True),
                role,
                integer(entry["weight"], "opponent weight", positive=True),
                canonical,
                string(entry["configuration_fingerprint"], "configuration fingerprint"),
            )
        )
    panel = opponent_panel(tuple(opponents))
    if _panel_dict(panel) != raw:
        raise ValueError("opponent panel identity is inconsistent")
    return panel


def _decode_corpus(
    value: object, phase: Phase, panel: OpponentPanel, task_seed: int, game_config_fingerprint: str
) -> TaskCorpus:
    raw = object_fields(
        value,
        {"corpus_id", "fingerprint", "phase", "task_policy_version", "cases"},
        f"{phase} corpus",
    )
    if (
        raw["phase"] != phase
        or raw["task_policy_version"] != "weighted-fair-prefix-v1"
        or not isinstance(raw["cases"], list)
    ):
        raise ValueError(f"invalid {phase} corpus")
    expected = build_corpus(phase, len(raw["cases"]), task_seed, panel, game_config_fingerprint)
    if _corpus_dict(expected) != raw:
        raise ValueError(f"{phase} corpus identities do not match frozen inputs")
    verify_weighted_corpus(expected, panel)
    return expected


def _decode_prefix(value: object, corpus: TaskCorpus, label: str) -> TaskPrefix:
    raw = object_fields(value, {"prefix_id", "corpus_id", "length", "task_ids"}, f"{label} prefix")
    length = integer(raw["length"], f"{label} prefix length", positive=True)
    expected = task_prefix(corpus, length)
    if _prefix_dict(expected) != raw:
        raise ValueError(f"{label} prefix identity is inconsistent")
    return expected


def _decode_proposer(value: object) -> ProposerSpecification:
    fields = {
        "policy_version",
        "policy",
        "guided_source",
        "constraint_policy_version",
        "constraints",
        "proposal_seed",
        "task_seed",
        "cohort_size",
        "finalists",
        "bootstrap_candidates",
        "model_candidates",
        "random_reserve_candidates",
        "source_schedule",
        "challenger_source_schedule",
        "attempt_cap",
        "seed_derivation_version",
        "cost_policy_version",
        "bootstrap_sampler_version",
        "reserve_sampler_version",
        "bootstrap_seed",
        "reserve_seed",
        "guided_adapter_version",
        "runtime_versions",
    }
    raw = object_fields(value, fields, "proposer")
    versions = object_fields(
        raw["runtime_versions"],
        {"smac", "configspace", "scikit_learn", "numpy", "scipy"},
        "runtime versions",
    )
    if not all(isinstance(item, str) and item for item in versions.values()):
        raise ValueError("runtime versions must be nonempty strings")
    if raw["constraint_policy_version"] != CONSTRAINT_POLICY_VERSION:
        raise ValueError("run predates the taxonomy cutover -- replay with the v5 CLI")
    constraints = decode_constraints(raw["constraints"])
    policy: ProposerPolicy = literal(raw["policy"], POLICIES, "proposer policy")
    specification = proposer_specification(
        integer(raw["proposal_seed"], "proposal seed", positive=True),
        integer(raw["task_seed"], "task seed", positive=True),
        integer(raw["cohort_size"], "cohort size", positive=True),
        integer(raw["finalists"], "finalists", positive=True),
        integer(raw["bootstrap_candidates"], "bootstrap candidates", positive=True),
        integer(raw["random_reserve_candidates"], "random reserve candidates", positive=True),
        constraints,
        {
            key: string(item, f"runtime version {key}", nonempty=True)
            for key, item in versions.items()
        },
        policy,
    )
    if raw != specification.encoded():
        raise ValueError("proposer specification is inconsistent")
    return specification


def _decode_game_identity(raw: JsonObject) -> tuple[GameSpec, str, str]:
    binary = object_fields(raw["binary"], {"path", "sha256"}, "binary")
    spec = decode_game_spec(
        strict_json(string(raw["description"], "description"), "description"),
        Path(string(binary["path"], "binary path")),
        string(binary["sha256"], "binary hash"),
    )
    duplicated = {
        "engine_fingerprint": spec.engine_fingerprint,
        "description_fingerprint": spec.description_fingerprint,
        "kind": spec.kind,
        "label": spec.label,
        "game_description": spec.description,
        "tuning_schema_fingerprint": spec.schema_fingerprint,
    }
    if any(
        canonical_json(raw[key]) != canonical_json(expected) for key, expected in duplicated.items()
    ):
        raise ValueError("manifest disagrees with game description")
    game_config = string(raw["game_config"], "game configuration")
    game_config_fingerprint = string(
        raw["game_config_fingerprint"], "game configuration fingerprint"
    )
    if fingerprint(strict_json(game_config)) != game_config_fingerprint:
        raise ValueError("game configuration fingerprint is invalid")
    if not spec.game_config_schema.is_empty:
        errors = spec.game_config_schema.validate_config(strict_json(game_config))
        if errors:
            raise ValueError(f"manifest game_config is invalid: {'; '.join(errors)}")
    return spec, game_config, game_config_fingerprint


def _decode_objective_ref(raw: JsonObject) -> tuple[Path, str, str]:
    objective = object_fields(
        raw["objective"], {"source_path", "objective_id", "fingerprint"}, "objective"
    )
    return (
        Path(string(objective["source_path"], "objective source path", nonempty=True)),
        string(objective["objective_id"], "objective id", nonempty=True),
        string(objective["fingerprint"], "objective fingerprint"),
    )


def _decode_start_distribution(raw: JsonObject) -> str:
    start = object_fields(raw["start_distribution"], {"kind", "fingerprint"}, "start distribution")
    expected = fingerprint({"kind": "default_only"})
    if start["kind"] != "default_only" or start["fingerprint"] != expected:
        raise ValueError("invalid start distribution")
    return expected


def _decode_corpora_and_prefixes(
    raw: JsonObject, panel: OpponentPanel, task_seed: int, game_config_fingerprint: str
) -> tuple[TaskCorpus, TaskCorpus, TaskPrefix, tuple[TaskPrefix, ...], TaskPrefix]:
    corpora = object_fields(raw["corpora"], {"tuning", "production_validation"}, "corpora")
    tuning = _decode_corpus(corpora["tuning"], "tuning", panel, task_seed, game_config_fingerprint)
    validation = _decode_corpus(
        corpora["production_validation"], "validation", panel, task_seed, game_config_fingerprint
    )
    if set(case.seed for case in tuning.cases) & set(case.seed for case in validation.cases):
        raise ValueError("task corpora have colliding seeds")
    prefixes = object_fields(raw["prefixes"], {"tuning", "validation"}, "prefixes")
    tuning_prefix = _decode_prefix(prefixes["tuning"], tuning, "tuning")
    validation_prefix = _decode_prefix(prefixes["validation"], validation, "validation")
    raw_blocks = raw["tuning_blocks"]
    if not isinstance(raw_blocks, list):
        raise ValueError("tuning blocks must be an array")
    expected_blocks = tuning_blocks(tuning, panel)
    blocks = tuple(
        _decode_prefix(
            object_fields(item, {"ordinal", "prefix"}, "tuning block")["prefix"],
            tuning,
            "tuning block",
        )
        for item in raw_blocks
    )
    if (
        [
            object_fields(item, {"ordinal", "prefix"}, "tuning block")["ordinal"]
            for item in raw_blocks
        ]
        != list(range(len(blocks)))
        or blocks != expected_blocks
        or blocks[-1] != tuning_prefix
    ):
        raise ValueError("tuning blocks are inconsistent")
    return tuning, validation, tuning_prefix, blocks, validation_prefix


def _decode_fidelity(
    raw: JsonObject,
    validation: TaskCorpus,
    tuning_prefix: TaskPrefix,
    validation_prefix: TaskPrefix,
) -> tuple[SearchEffort, SearchEffort, SearchEffort]:
    fidelity = object_fields(raw["fidelity"], {"tuning", "validation", "production"}, "fidelity")
    expected: dict[str, TaskPrefix] = {
        "tuning": tuning_prefix,
        "validation": validation_prefix,
        "production": task_prefix(validation, len(validation.cases)),
    }
    efforts: dict[str, SearchEffort] = {}
    for name, prefix in expected.items():
        item = object_fields(
            fidelity[name], {"task_prefix_id", "search_effort"}, f"{name} fidelity"
        )
        if item["task_prefix_id"] != prefix.prefix_id:
            raise ValueError(f"{name} fidelity prefix is inconsistent")
        efforts[name] = decode_effort(item["search_effort"], f"{name} search effort")
    if exceeds_same_kind(efforts["tuning"], efforts["production"]) or exceeds_same_kind(
        efforts["validation"], efforts["production"]
    ):
        raise ValueError("observed search effort exceeds production effort")
    return efforts["tuning"], efforts["validation"], efforts["production"]


def _check_epoch(
    raw: JsonObject,
    spec: GameSpec,
    objective_id: str,
    objective_fingerprint: str,
    panel_fingerprint: str,
    start_fingerprint: str,
    game_config_fingerprint: str,
    tuning: TaskCorpus,
    validation: TaskCorpus,
    production_effort: SearchEffort,
    constraints: Constraints,
) -> ObjectiveEpoch:
    epoch = objective_epoch(
        _epoch_payload(
            spec,
            objective_id,
            objective_fingerprint,
            panel_fingerprint,
            start_fingerprint,
            tuning,
            validation,
            production_effort,
            game_config_fingerprint,
            constraints,
        )
    )
    if raw["epoch"] != {"epoch_id": epoch.epoch_id, "fingerprint": epoch.fingerprint}:
        raise ValueError("objective epoch is inconsistent")
    return epoch


def _check_statistical_policy(raw: JsonObject) -> None:
    if any(raw[key] != value for key, value in _STATISTICAL_POLICY.items()):
        raise ValueError("unsupported statistical policy")
    if not isinstance(raw["limitations"], list) or not all(
        isinstance(item, str) for item in raw["limitations"]
    ):
        raise ValueError("limitations must be strings")


def _decode_candidate_failure_policy(value: object) -> CandidateFailurePolicySpecification:
    raw = object_fields(
        value,
        {"policy_version", "phase", "max_pair_attempts", "exhaustion_basis", "overflow_source"},
        "candidate failure policy",
    )
    policy = CandidateFailurePolicySpecification()
    if raw != policy.encoded():
        raise ValueError("unsupported candidate failure policy")
    return policy


def _decode_active_elimination(
    value: object, shadow_policy: ShadowPolicySpecification
) -> ActiveEliminationSpecification | None:
    if value is None:
        return None
    raw = object_fields(
        value,
        {
            "audit_probability",
            "shadow_policy_kind",
            "shadow_method_version",
            "shadow_spare_margin",
            "sampler_version",
            "safety_rule_version",
        },
        "active elimination",
    )
    policy = _active_elimination(raw["audit_probability"], shadow_policy)
    if policy is None or raw != policy.encoded():
        raise ValueError("unsupported active elimination policy")
    return policy


def _decode_diagnostic_policy(value: object) -> DiagnosticPolicySpecification:
    raw = object_fields(
        value,
        {
            "edge_policy_version",
            "seed_policy_version",
            "graph_rule_version",
            "shortlist_rule_version",
            "maximum_reserve_slots",
        },
        "diagnostic policy",
    )
    policy = DiagnosticPolicySpecification()
    if raw != policy.encoded():
        raise ValueError("unsupported diagnostic policy")
    return policy


def _decode_paired_shadow_policy(shadow_raw: JsonObject) -> PairedBootstrapPolicySpecification:
    policy_raw = object_fields(
        shadow_raw,
        {
            "kind",
            "practical_effect_margin",
            "elimination_probability_threshold",
            "resamples",
            "method_version",
            "minimum_eligible_prefix_pairs",
        },
        "paired bootstrap shadow policy",
    )
    validated = _shadow_policy(
        policy_raw["practical_effect_margin"], policy_raw["elimination_probability_threshold"]
    )
    if not isinstance(validated, PairedBootstrapPolicySpecification):
        raise ValueError("unsupported shadow policy")
    if policy_raw["method_version"] != "stratified-paired-bootstrap-all-strata-v2":
        raise ValueError("unsupported shadow policy")
    return PairedBootstrapPolicySpecification(
        "paired_bootstrap",
        validated.practical_effect_margin,
        validated.elimination_probability_threshold,
        integer(policy_raw["resamples"], "shadow resamples", positive=True),
        "stratified-paired-bootstrap-all-strata-v2",
        integer(
            policy_raw["minimum_eligible_prefix_pairs"],
            "minimum eligible prefix pairs",
            positive=True,
        ),
    )


def _decode_successive_halving_policy(
    shadow_raw: JsonObject,
) -> SuccessiveHalvingPolicySpecification:
    policy_raw = object_fields(
        shadow_raw,
        {
            "kind",
            "method_version",
            "reduction_factor",
            "practical_effect_margin",
            "minimum_eligible_prefix_pairs",
            "survivor_floor",
            "ranking_rule",
            "spare_margin",
        },
        "successive halving shadow policy",
    )
    policy = _shadow_policy(
        policy_raw["practical_effect_margin"],
        0.05,
        "successive_halving",
        policy_raw["survivor_floor"],
        policy_raw["spare_margin"],
    )
    if not isinstance(policy, SuccessiveHalvingPolicySpecification) or (
        policy_raw["method_version"] != policy.method_version
        or policy_raw["reduction_factor"] != 2
        or policy_raw["minimum_eligible_prefix_pairs"] != MINIMUM_ELIGIBLE_PREFIX_PAIRS
        or policy_raw["ranking_rule"] != policy.ranking_rule
    ):
        raise ValueError("unsupported shadow policy")
    return policy


def _decode_shadow_policy(value: object) -> ShadowPolicySpecification:
    shadow_raw = json_object(value, "shadow policy")
    if "kind" not in shadow_raw:
        raise ValueError("shadow policy is missing kind")
    kind = string(shadow_raw["kind"], "shadow policy kind")
    if kind == "paired_bootstrap":
        shadow_policy: ShadowPolicySpecification = _decode_paired_shadow_policy(shadow_raw)
    elif kind == "successive_halving":
        shadow_policy = _decode_successive_halving_policy(shadow_raw)
    else:
        raise ValueError("unsupported shadow policy")
    paired = isinstance(shadow_policy, PairedBootstrapPolicySpecification)
    if shadow_policy.minimum_eligible_prefix_pairs != MINIMUM_ELIGIBLE_PREFIX_PAIRS or (
        paired
        and (
            shadow_policy.resamples != 4096
            or shadow_policy.method_version != "stratified-paired-bootstrap-all-strata-v2"
        )
    ):
        raise ValueError("unsupported shadow policy")
    return shadow_policy


def _decode_constraints_block(
    raw: JsonObject, spec: GameSpec, proposer: ProposerSpecification
) -> Constraints:
    constraints = decode_constraints(raw.get("constraints"))
    validate_constraints(spec.tuning, constraints)
    if not constraints and "constraints" in raw:
        raise ValueError("constraints block is present but empty")
    if constraints != proposer.constraints:
        raise ValueError("manifest constraints disagree with the proposer block")
    return constraints


def decode_manifest_object(value: object) -> Manifest:
    full = json_object(value, "manifest")
    # The `constraints` block is present only for a run that constrains the
    # space, so an unconstrained run's manifest carries the base field set.
    allowed: set[str] = set(_FIELDS)
    if "constraints" in full:
        allowed.add("constraints")
    raw = object_fields(full, allowed, "manifest")
    if raw["schema_version"] != SCHEMA_VERSION or raw["command_policy_version"] != POLICY_VERSION:
        raise ValueError("unsupported manifest schema version or command policy")
    stored = string(raw["fingerprint"], "manifest fingerprint")
    if fingerprint({key: item for key, item in raw.items() if key != "fingerprint"}) != stored:
        raise ValueError("manifest fingerprint does not match content")
    spec, game_config, game_config_fingerprint = _decode_game_identity(raw)
    proposer = _decode_proposer(raw["proposer"])
    validate_constraints(spec.tuning, proposer.constraints)
    budget_raw = object_fields(
        raw["compute_budget"],
        {
            "policy_version",
            "tuning_pair_attempts",
            "validation_pair_attempts",
            "diagnostic_pair_attempts",
        },
        "compute budget",
    )
    if budget_raw["policy_version"] != "safe-boundary-pair-attempts-v1":
        raise ValueError("unsupported compute budget policy")
    compute_budget = ComputeBudget(
        integer(budget_raw["tuning_pair_attempts"], "tuning pair attempts", positive=True),
        integer(budget_raw["validation_pair_attempts"], "validation pair attempts", positive=True),
        integer(budget_raw["diagnostic_pair_attempts"], "diagnostic pair attempts"),
    )
    shadow_policy = _decode_shadow_policy(raw["shadow_policy"])
    candidate_failure_policy = _decode_candidate_failure_policy(raw["candidate_failure_policy"])
    active_elimination = _decode_active_elimination(raw["active_elimination"], shadow_policy)
    diagnostic_policy = _decode_diagnostic_policy(raw["diagnostic_policy"])
    panel = _decode_panel(raw["opponent_panel"])
    source_path, objective_id, objective_fingerprint = _decode_objective_ref(raw)
    start_fingerprint = _decode_start_distribution(raw)
    tuning, validation, tuning_prefix, blocks, validation_prefix = _decode_corpora_and_prefixes(
        raw, panel, proposer.task_seed, game_config_fingerprint
    )
    if compute_budget.validation_pair_attempts % proposer.finalists:
        raise ValueError("validation pair budget must divide finalists")
    if validation_prefix.length != compute_budget.validation_pair_attempts // proposer.finalists:
        raise ValueError("validation prefix does not match compute budget")
    validate_cycle_endpoint(panel, validation_prefix.length, "validation pairs")
    if compute_budget.tuning_pair_attempts < proposer.cohort_size * tuning_prefix.length:
        raise ValueError("tuning pair budget cannot fund initial cohort")
    efforts = _decode_fidelity(raw, validation, tuning_prefix, validation_prefix)
    constraints = _decode_constraints_block(raw, spec, proposer)
    epoch = _check_epoch(
        raw,
        spec,
        objective_id,
        objective_fingerprint,
        panel.fingerprint,
        start_fingerprint,
        game_config_fingerprint,
        tuning,
        validation,
        efforts[2],
        constraints,
    )
    _check_statistical_policy(raw)
    return Manifest(
        stored,
        spec,
        source_path,
        objective_id,
        objective_fingerprint,
        panel,
        tuning,
        validation,
        tuning_prefix,
        blocks,
        validation_prefix,
        epoch,
        proposer,
        string(raw["run_id"], "run id", nonempty=True),
        game_config,
        game_config_fingerprint,
        efforts,
        compute_budget,
        shadow_policy,
        candidate_failure_policy,
        active_elimination,
        diagnostic_policy,
        constraints,
    )


def read_manifest(path: Path) -> Manifest:
    return decode_manifest_object(strict_json(path.read_text(encoding="utf-8"), "manifest"))


def manifest_json(manifest: Manifest) -> JsonObject:
    """Return the frozen transport representation at the publishing boundary."""
    validation = manifest.production_validation_corpus
    encoded = _encode_manifest_object(
        manifest.run_id,
        manifest.spec,
        manifest.objective_source_path,
        manifest.objective_id,
        manifest.objective_fingerprint,
        manifest.game_config,
        manifest.game_config_fingerprint,
        manifest.proposer_spec,
        manifest.panel,
        manifest.tuning_corpus,
        validation,
        manifest.tuning_prefix,
        manifest.tuning_blocks,
        manifest.validation_prefix,
        task_prefix(validation, len(validation.cases)),
        manifest.epoch,
        manifest.effort_values,
        manifest.compute_budget,
        manifest.shadow_policy,
        manifest.candidate_failure_policy,
        manifest.active_elimination,
        manifest.diagnostic_policy,
        manifest.constraints,
    )
    return {**encoded, "fingerprint": manifest.fingerprint}
