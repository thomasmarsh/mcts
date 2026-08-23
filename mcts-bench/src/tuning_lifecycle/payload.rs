use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;

use super::{EventValidationError, TuningEventType, TuningTrialId};

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
    }
}
