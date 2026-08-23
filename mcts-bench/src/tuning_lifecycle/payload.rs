use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;

use super::{EventValidationError, TuningEventType, TuningGameId, TuningPairId, TuningTrialId};

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

#[derive(Clone, Debug, Deserialize)]
pub struct SessionStartedPayload {
    pub manifest: Value,
    pub manifest_fingerprint: String,
    pub target_trial_count: Option<i64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AttemptStartedPayload {
    pub run_id: Option<String>,
    pub study_name: Option<String>,
    pub storage: Option<String>,
    pub target_trial_count: Option<i64>,
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

#[derive(Clone, Debug, Deserialize)]
pub struct TrialTerminalPayload {
    pub trial_id: TuningTrialId,
    pub trial_number: Option<i64>,
    pub config: Option<Value>,
    pub seed: Option<i64>,
    pub mu: Option<f64>,
    pub sigma: Option<f64>,
    pub score: Option<f64>,
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
    TrialCreated(TrialCreatedPayload),
    TrialStarted(TrialStartedPayload),
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
        TuningEventType::TrialCreated => parse(value).map(TuningPayload::TrialCreated),
        TuningEventType::TrialStarted => parse(value).map(TuningPayload::TrialStarted),
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
