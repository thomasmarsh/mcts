use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;

use super::{
    EventValidationError, TuningAttemptId, TuningEventType, TuningGameId, TuningPairId,
    TuningTrialId,
};

#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateSide {
    First,
    Second,
}

impl CandidateSide {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Second => "second",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuningGameOutcome {
    CandidateWin,
    BaselineWin,
    Draw,
}

impl TuningGameOutcome {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CandidateWin => "candidate_win",
            Self::BaselineWin => "baseline_win",
            Self::Draw => "draw",
        }
    }
}

#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct StrategyMetrics {
    pub iterations_total: u64,
    pub iterations_first_half: u64,
    pub move_time_ms: u64,
}

#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct Rating {
    pub mu: f64,
    pub sigma: f64,
}

#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct OpponentSnapshot {
    pub anchor_id: String,
    pub config: Value,
    pub mu: f64,
    pub sigma: f64,
    pub label: Option<String>,
    pub provenance: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PoolAnchorProvenance {
    BootstrapDefault,
    BootstrapRandom,
    Configured,
    Trial,
    LegacyUnknown,
}

impl PoolAnchorProvenance {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BootstrapDefault => "bootstrap_default",
            Self::BootstrapRandom => "bootstrap_random",
            Self::Configured => "configured",
            Self::Trial => "trial",
            Self::LegacyUnknown => "legacy_unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PoolAnchorInsertionReason {
    Bootstrap,
    Configured,
    Champion,
    SkillBand,
    LegacyUnknown,
}

impl PoolAnchorInsertionReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::Configured => "configured",
            Self::Champion => "champion",
            Self::SkillBand => "skill_band",
            Self::LegacyUnknown => "legacy_unknown",
        }
    }
}

#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq)]
pub struct PoolAnchorSnapshot {
    pub anchor_id: String,
    pub config: Value,
    pub mu: f64,
    pub sigma: f64,
    pub provenance: PoolAnchorProvenance,
    pub insertion_reason: PoolAnchorInsertionReason,
    pub source_trial_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PoolRevisedPayload {
    pub pool_snapshot_fingerprint: String,
    pub anchors: Vec<PoolAnchorSnapshot>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SessionStartedPayload {
    pub manifest: Value,
    pub manifest_fingerprint: String,
    pub optimizer_id: Option<String>,
    pub lifecycle_path: Option<String>,
    pub target_trial_count: Option<i64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AttemptStartedPayload {
    pub optimizer_id: Option<String>,
    pub bench_run_id: Option<String>,
    /// Legacy physical benchmark identity emitted before `bench_run_id`.
    pub run_id: Option<String>,
    pub study_name: Option<String>,
    pub storage: Option<String>,
    pub target_trial_count: Option<i64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryTrialReason {
    AbruptAttemptRecovery,
    RecoveryEvidenceGap,
}

impl RecoveryTrialReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AbruptAttemptRecovery => "abrupt_attempt_recovery",
            Self::RecoveryEvidenceGap => "recovery_evidence_gap",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct RecoveredTrialPayload {
    pub trial_id: TuningTrialId,
    pub trial_number: i64,
    pub reason: RecoveryTrialReason,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AttemptRecoveredPayload {
    pub prior_attempt_id: TuningAttemptId,
    pub prior_bench_run_id: Option<String>,
    pub trials: Vec<RecoveredTrialPayload>,
    pub pair_ids: Vec<TuningPairId>,
    pub reason: RecoveryTrialReason,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrialCreatedPayload {
    pub trial_id: TuningTrialId,
    pub trial_number: i64,
    pub config: Value,
    pub seed: Option<i64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrialStartedPayload {
    pub trial_id: TuningTrialId,
    pub trial_number: i64,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrialReportOutcome {
    Continue,
    Complete,
    Prune,
}

impl TrialReportOutcome {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::Complete => "complete",
            Self::Prune => "prune",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrialReportReason {
    BelowMinPairs,
    PruningDisabled,
    StartupExempt,
    HyperbandKeep,
    Confidence,
    MaxPairs,
    HyperbandPrune,
}

impl TrialReportReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BelowMinPairs => "below_min_pairs",
            Self::PruningDisabled => "pruning_disabled",
            Self::StartupExempt => "startup_exempt",
            Self::HyperbandKeep => "hyperband_keep",
            Self::Confidence => "confidence",
            Self::MaxPairs => "max_pairs",
            Self::HyperbandPrune => "hyperband_prune",
        }
    }

    #[must_use]
    pub const fn is_valid_for(self, outcome: TrialReportOutcome) -> bool {
        matches!(
            (outcome, self),
            (
                TrialReportOutcome::Continue,
                Self::BelowMinPairs
                    | Self::PruningDisabled
                    | Self::StartupExempt
                    | Self::HyperbandKeep
            ) | (
                TrialReportOutcome::Complete,
                Self::Confidence | Self::MaxPairs
            ) | (TrialReportOutcome::Prune, Self::HyperbandPrune)
        )
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrialReportedPayload {
    pub trial_id: TuningTrialId,
    pub trial_number: i64,
    pub completed_pairs: u64,
    pub mu: f64,
    pub sigma: f64,
    pub score: f64,
    pub score_formula_version: u32,
    pub conservative_k: f64,
    pub outcome: TrialReportOutcome,
    pub reason: TrialReportReason,
    pub pruning_exempt: bool,
    pub bracket_id: Option<String>,
    pub rung_resource: Option<u64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrialTerminalPayload {
    pub trial_id: TuningTrialId,
    pub trial_number: Option<i64>,
    pub config: Option<Value>,
    pub seed: Option<i64>,
    pub completed_pairs: Option<u64>,
    pub mu: Option<f64>,
    pub sigma: Option<f64>,
    pub score: Option<f64>,
    pub score_formula_version: Option<u32>,
    pub conservative_k: Option<f64>,
    pub pruning_exempt: Option<bool>,
    pub bracket_id: Option<String>,
    pub rung_resource: Option<u64>,
    #[serde(alias = "reason")]
    pub stop_reason: Option<TrialReportReason>,
    pub error: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AttemptTerminalPayload {
    pub target_trial_count: Option<i64>,
    pub error: Option<String>,
    pub reason: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PairStartedPayload {
    pub trial_id: TuningTrialId,
    pub pair_id: TuningPairId,
    pub pair_index: u32,
    pub seed: u64,
    pub round: u32,
    pub opponent: OpponentSnapshot,
    pub pool_snapshot_fingerprint: String,
    pub rating_before: Rating,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GameFinishedPayload {
    pub trial_id: TuningTrialId,
    pub pair_id: TuningPairId,
    pub game_id: TuningGameId,
    pub candidate_side: CandidateSide,
    pub outcome: TuningGameOutcome,
    pub seed: u64,
    pub round: u32,
    pub trace_game_seq: Option<u64>,
    pub plies: u32,
    pub elapsed_ms: u64,
    pub candidate: StrategyMetrics,
    pub baseline: StrategyMetrics,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PairFinishedPayload {
    pub trial_id: TuningTrialId,
    pub pair_id: TuningPairId,
    pub pair_index: u32,
    pub rating_before: Rating,
    pub rating_after: Rating,
    pub score: f64,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PairFailedPayload {
    pub trial_id: TuningTrialId,
    pub pair_id: TuningPairId,
    pub pair_index: u32,
    pub error: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug)]
pub enum TuningPayload {
    SessionStarted(SessionStartedPayload),
    AttemptStarted(AttemptStartedPayload),
    AttemptRecovered(AttemptRecoveredPayload),
    PoolRevised(PoolRevisedPayload),
    TrialCreated(TrialCreatedPayload),
    TrialStarted(TrialStartedPayload),
    TrialReported(TrialReportedPayload),
    TrialCompleted(TrialTerminalPayload),
    TrialPruned(TrialTerminalPayload),
    TrialFailed(TrialTerminalPayload),
    TrialCancelled(TrialTerminalPayload),
    AttemptCompleted(AttemptTerminalPayload),
    AttemptFailed(AttemptTerminalPayload),
    AttemptStopped(AttemptTerminalPayload),
    PairStarted(PairStartedPayload),
    GameFinished(GameFinishedPayload),
    PairFinished(PairFinishedPayload),
    PairFailed(PairFailedPayload),
}

fn parse<T: DeserializeOwned>(value: &Value) -> Result<T, EventValidationError> {
    serde_json::from_value(value.clone())
        .map_err(|error| EventValidationError::PayloadType(error.to_string()))
}

pub(super) fn parse_typed(
    event_type: TuningEventType,
    value: &Value,
) -> Result<TuningPayload, EventValidationError> {
    if !value.is_object() {
        return Err(EventValidationError::PayloadMustBeObject);
    }
    match event_type {
        TuningEventType::SessionStarted => parse(value).map(TuningPayload::SessionStarted),
        TuningEventType::AttemptStarted => parse(value).map(TuningPayload::AttemptStarted),
        TuningEventType::AttemptRecovered => parse(value).map(TuningPayload::AttemptRecovered),
        TuningEventType::PoolRevised => parse(value).map(TuningPayload::PoolRevised),
        TuningEventType::TrialCreated => parse(value).map(TuningPayload::TrialCreated),
        TuningEventType::TrialStarted => parse(value).map(TuningPayload::TrialStarted),
        TuningEventType::TrialReported => parse(value).map(TuningPayload::TrialReported),
        TuningEventType::TrialCompleted => parse(value).map(TuningPayload::TrialCompleted),
        TuningEventType::TrialPruned => parse(value).map(TuningPayload::TrialPruned),
        TuningEventType::TrialFailed => parse(value).map(TuningPayload::TrialFailed),
        TuningEventType::TrialCancelled => parse(value).map(TuningPayload::TrialCancelled),
        TuningEventType::AttemptCompleted => parse(value).map(TuningPayload::AttemptCompleted),
        TuningEventType::AttemptFailed => parse(value).map(TuningPayload::AttemptFailed),
        TuningEventType::AttemptStopped => parse(value).map(TuningPayload::AttemptStopped),
        TuningEventType::PairStarted => parse(value).map(TuningPayload::PairStarted),
        TuningEventType::GameFinished => parse(value).map(TuningPayload::GameFinished),
        TuningEventType::PairFinished => parse(value).map(TuningPayload::PairFinished),
        TuningEventType::PairFailed => parse(value).map(TuningPayload::PairFailed),
    }
}
