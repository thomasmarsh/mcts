//! Logical reads for tuning-session analysis.
//!
//! The analysis route consumes these application records without depending on
//! a particular database driver.

use serde_json::Value;

use crate::tuning_lifecycle::{
    PoolAnchorInsertionReason, PoolAnchorProvenance, TrialReportOutcome, TrialReportReason,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningAnalysisRepositoryError {
    Storage(String),
    InvalidData(String),
}

impl std::fmt::Display for TuningAnalysisRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(message) => write!(f, "tuning analysis storage failure: {message}"),
            Self::InvalidData(message) => {
                write!(f, "invalid persisted tuning analysis data: {message}")
            }
        }
    }
}

impl std::error::Error for TuningAnalysisRepositoryError {}

#[derive(Debug)]
pub struct TuningAnalysisSession {
    pub manifest: String,
    pub last_sequence: i64,
}

#[derive(Debug)]
pub struct TuningAnalysisTrialCounts {
    pub total: i64,
    pub queued: i64,
    pub running: i64,
    pub terminal: i64,
    pub completed: i64,
    pub failed: i64,
    pub pruned: i64,
    pub cancelled: i64,
}

#[derive(Debug)]
pub struct TuningAnalysisReport {
    pub trial_id: String,
    pub trial_number: i64,
    pub trial_status: String,
    pub resource: u64,
    pub mu: f64,
    pub sigma: f64,
    pub score: f64,
    pub outcome: TrialReportOutcome,
    pub reason: TrialReportReason,
    pub pruning_exempt: bool,
    pub bracket_id: Option<String>,
    pub rung_resource: Option<u64>,
}

#[derive(Debug)]
pub struct TuningAnalysisPairCoverage {
    pub total: i64,
    pub running: i64,
    pub complete: i64,
    pub failed: i64,
    pub unmatched_pool_revisions: i64,
}

#[derive(Debug)]
pub struct TuningAnalysisBest {
    pub score: f64,
    pub trial_ids: Vec<String>,
}

#[derive(Debug)]
pub struct TuningAnalysisPoolAnchor {
    pub anchor_ordinal: u32,
    pub anchor_id: String,
    pub config: Value,
    pub mu: f64,
    pub sigma: f64,
    pub provenance: PoolAnchorProvenance,
    pub insertion_reason: PoolAnchorInsertionReason,
    pub source_trial_id: Option<String>,
}

#[derive(Debug)]
pub struct TuningAnalysisPoolRevision {
    pub pool_snapshot_fingerprint: String,
    pub display_ordinal: u32,
    pub observed_at: String,
    pub pair_count: i64,
    pub anchors: Vec<TuningAnalysisPoolAnchor>,
}

#[derive(Debug)]
pub struct TuningAnalysisData {
    pub session: TuningAnalysisSession,
    pub trial_counts: TuningAnalysisTrialCounts,
    pub reports: Vec<TuningAnalysisReport>,
    pub pair_coverage: TuningAnalysisPairCoverage,
    pub best: Option<TuningAnalysisBest>,
    pub pool_revisions: Vec<TuningAnalysisPoolRevision>,
}

/// Logical storage operations needed to render a tuning-session analysis.
pub trait TuningAnalysisRepository {
    fn load_analysis(
        &self,
        session_id: &str,
    ) -> Result<Option<TuningAnalysisData>, TuningAnalysisRepositoryError>;
    fn load_trial_pool_revisions(
        &self,
        session_id: &str,
        trial_id: &str,
    ) -> Result<Vec<TuningAnalysisPoolRevision>, TuningAnalysisRepositoryError>;
}
