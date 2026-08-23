//! Versioned evidence emitted by the Python tuner coordinator.

mod payload;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use payload::{
    AttemptStartedPayload, AttemptTerminalPayload, CandidateSide, GameFinishedPayload,
    OpponentSnapshot, PairFailedPayload, PairFinishedPayload, PairStartedPayload,
    PoolAnchorInsertionReason, PoolAnchorProvenance, PoolAnchorSnapshot, PoolRevisedPayload,
    Rating, SessionStartedPayload, StrategyMetrics, TrialCreatedPayload, TrialReportOutcome,
    TrialReportReason, TrialReportedPayload, TrialStartedPayload, TrialTerminalPayload,
    TuningGameOutcome, TuningPayload,
};

pub const TUNING_LIFECYCLE_SCHEMA_VERSION: u32 = 1;

macro_rules! opaque_id {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

opaque_id!(TuningSessionId);
opaque_id!(TuningAttemptId);
opaque_id!(TuningEventId);
opaque_id!(TuningTrialId);
opaque_id!(TuningPairId);
opaque_id!(TuningGameId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TuningEventType {
    SessionStarted,
    AttemptStarted,
    PoolRevised,
    TrialCreated,
    TrialStarted,
    TrialReported,
    TrialCompleted,
    TrialPruned,
    TrialFailed,
    TrialCancelled,
    PairStarted,
    GameFinished,
    PairFinished,
    PairFailed,
    AttemptCompleted,
    AttemptFailed,
    AttemptStopped,
}

impl TuningEventType {
    #[must_use]
    pub const fn is_trial(&self) -> bool {
        matches!(
            self,
            Self::TrialCreated
                | Self::TrialStarted
                | Self::TrialReported
                | Self::TrialCompleted
                | Self::TrialPruned
                | Self::TrialFailed
                | Self::TrialCancelled
        )
    }

    #[must_use]
    pub const fn is_pair(&self) -> bool {
        matches!(
            self,
            Self::PairStarted | Self::GameFinished | Self::PairFinished | Self::PairFailed
        )
    }

    #[must_use]
    pub const fn is_attempt_terminal(&self) -> bool {
        matches!(
            self,
            Self::AttemptCompleted | Self::AttemptFailed | Self::AttemptStopped
        )
    }

    #[must_use]
    pub const fn is_trial_terminal(&self) -> bool {
        matches!(
            self,
            Self::TrialCompleted | Self::TrialPruned | Self::TrialFailed | Self::TrialCancelled
        )
    }

    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SessionStarted => "session_started",
            Self::AttemptStarted => "attempt_started",
            Self::PoolRevised => "pool_revised",
            Self::TrialCreated => "trial_created",
            Self::TrialStarted => "trial_started",
            Self::TrialReported => "trial_reported",
            Self::TrialCompleted => "trial_completed",
            Self::TrialPruned => "trial_pruned",
            Self::TrialFailed => "trial_failed",
            Self::TrialCancelled => "trial_cancelled",
            Self::PairStarted => "pair_started",
            Self::GameFinished => "game_finished",
            Self::PairFinished => "pair_finished",
            Self::PairFailed => "pair_failed",
            Self::AttemptCompleted => "attempt_completed",
            Self::AttemptFailed => "attempt_failed",
            Self::AttemptStopped => "attempt_stopped",
        }
    }
}

/// A deliberately small envelope. Payload is retained verbatim so a later
/// projection can add pair or game fields without rewriting stored evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuningLifecycleEvent {
    pub schema_version: u32,
    pub event_id: TuningEventId,
    pub session_id: TuningSessionId,
    pub attempt_id: TuningAttemptId,
    pub session_sequence: u64,
    pub timestamp: String,
    pub event_type: TuningEventType,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventValidationError {
    UnsupportedSchemaVersion(u32),
    Empty(&'static str),
    PayloadMustBeObject,
    MissingTrialId,
    PayloadType(String),
}

impl std::fmt::Display for EventValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported tuning lifecycle schema version {version}")
            }
            Self::Empty(field) => write!(f, "empty {field}"),
            Self::PayloadMustBeObject => f.write_str("payload must be an object"),
            Self::MissingTrialId => f.write_str("trial lifecycle payload is missing trial_id"),
            Self::PayloadType(error) => write!(f, "invalid typed payload: {error}"),
        }
    }
}

impl std::error::Error for EventValidationError {}

impl TuningLifecycleEvent {
    pub fn validate_shape(&self) -> Result<(), EventValidationError> {
        if self.schema_version != TUNING_LIFECYCLE_SCHEMA_VERSION {
            return Err(EventValidationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        for (value, field) in [
            (self.event_id.as_str(), "event_id"),
            (self.session_id.as_str(), "session_id"),
            (self.attempt_id.as_str(), "attempt_id"),
            (self.timestamp.as_str(), "timestamp"),
        ] {
            if value.is_empty() {
                return Err(EventValidationError::Empty(field));
            }
        }
        self.typed_payload()?;
        Ok(())
    }

    pub fn typed_payload(&self) -> Result<TuningPayload, EventValidationError> {
        payload::parse_typed(self.event_type, &self.payload)
    }

    pub fn trial_id(&self) -> Result<Option<TuningTrialId>, EventValidationError> {
        let payload = self.typed_payload()?;
        Ok(match payload {
            TuningPayload::TrialCreated(value) => Some(value.trial_id),
            TuningPayload::TrialStarted(value) => Some(value.trial_id),
            TuningPayload::TrialReported(value) => Some(value.trial_id),
            TuningPayload::TrialCompleted(value)
            | TuningPayload::TrialPruned(value)
            | TuningPayload::TrialFailed(value)
            | TuningPayload::TrialCancelled(value) => Some(value.trial_id),
            TuningPayload::PairStarted(value) => Some(value.trial_id),
            TuningPayload::GameFinished(value) => Some(value.trial_id),
            TuningPayload::PairFinished(value) => Some(value.trial_id),
            TuningPayload::PairFailed(value) => Some(value.trial_id),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_the_v1_envelope_without_extra_fields() {
        let event = TuningLifecycleEvent {
            schema_version: 1,
            event_id: "event-1".to_owned().into(),
            session_id: "session-1".to_owned().into(),
            attempt_id: "attempt-1".to_owned().into(),
            session_sequence: 0,
            timestamp: "2026-08-23T00:00:00Z".into(),
            event_type: TuningEventType::SessionStarted,
            payload: serde_json::json!({"manifest": {}, "manifest_fingerprint": "fingerprint"}),
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 8);
        assert_eq!(value["event_type"], "session_started");
        event.validate_shape().unwrap();
    }
}
