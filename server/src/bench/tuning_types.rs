use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
pub(crate) struct TuningSessionCommandBody {
    pub(crate) command_id: String,
    pub(crate) expected_version: u64,
}

#[derive(Deserialize)]
pub(crate) struct TuningSessionBudgetBody {
    pub(crate) command_id: String,
    pub(crate) expected_version: u64,
    pub(crate) delta: u64,
    pub(crate) start: bool,
    pub(crate) n_workers: Option<u64>,
}

#[derive(Serialize)]
pub(crate) struct TuningSessionCommandResponse {
    pub(crate) schema_version: u32,
    pub(crate) command_id: String,
    pub(crate) replay: bool,
    pub(crate) status: &'static str,
    pub(crate) attempt_id: Option<String>,
    pub(crate) bench_run_id: Option<String>,
    pub(crate) signal: Option<TuningStopSignal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) budget: Option<TuningBudgetResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) launch_error: Option<String>,
    pub(crate) control: TuningSessionControl,
}

#[derive(Serialize)]
pub(crate) struct TuningBudgetResult {
    pub(crate) previous_target_trial_count: u64,
    pub(crate) delta: u64,
    pub(crate) target_trial_count: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuningStopSignal {
    Sent,
    NotFound,
}

/// Stable control information is deliberately separate from lifecycle
/// progress: ordinary reports do not invalidate an operator command form.
#[derive(Serialize)]
pub(crate) struct TuningSessionControl {
    pub(crate) version: u64,
    pub(crate) continuation: TuningContinuation,
    pub(crate) allowed_commands: Vec<mcts_bench::tuning_command_store::AllowedCommand>,
}

#[derive(Serialize)]
pub(crate) struct TuningContinuation {
    pub(crate) target_trial_count: Option<u64>,
    pub(crate) consumed_trial_count: u64,
    pub(crate) remaining_trial_count: Option<u64>,
    pub(crate) active_attempt_id: Option<String>,
    pub(crate) launch_reservation: Option<mcts_bench::tuning_command_store::LaunchReservation>,
    pub(crate) stop_attempt_id: Option<String>,
    pub(crate) recovery_required: bool,
}

impl From<mcts_bench::tuning_command_store::SessionControl> for TuningSessionControl {
    fn from(control: mcts_bench::tuning_command_store::SessionControl) -> Self {
        let remaining_trial_count = control
            .target_trial_count
            .map(|target| target.saturating_sub(control.consumed_trial_count));
        Self {
            version: control.control_version,
            continuation: TuningContinuation {
                target_trial_count: control.target_trial_count,
                consumed_trial_count: control.consumed_trial_count,
                remaining_trial_count,
                active_attempt_id: control.active_attempt_id,
                launch_reservation: control.launch_reservation,
                stop_attempt_id: control.stop_attempt_id,
                recovery_required: control.recovery_required,
            },
            allowed_commands: control.allowed_commands,
        }
    }
}

#[derive(Serialize)]
pub(crate) struct TuningSessionList {
    pub(crate) schema_version: u32,
    pub(crate) sessions: Vec<TuningSessionListItem>,
}

#[derive(Serialize)]
pub(crate) struct TuningSessionListItem {
    pub(crate) session_id: String,
    pub(crate) game: Option<String>,
    pub(crate) label: Option<String>,
    pub(crate) status: String,
    pub(crate) target_trial_count: Option<i64>,
    pub(crate) counts: TuningTrialCounts,
    pub(crate) created_at: String,
    pub(crate) last_activity_at: String,
    pub(crate) attempts: Vec<TuningAttemptSummary>,
    pub(crate) capabilities: TuningCapabilities,
    pub(crate) control: TuningSessionControl,
}

#[derive(Serialize)]
pub(crate) struct TuningAttemptSummary {
    pub(crate) attempt_id: String,
    pub(crate) bench_run_id: Option<String>,
    pub(crate) status: String,
    pub(crate) started_at: String,
    pub(crate) ended_at: Option<String>,
    pub(crate) failure: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TuningSessionSummary {
    pub(crate) session_id: String,
    pub(crate) status: String,
    pub(crate) target_trial_count: Option<i64>,
    pub(crate) counts: TuningTrialCounts,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialCounts {
    pub(crate) total: i64,
    pub(crate) queued: i64,
    pub(crate) running: i64,
    pub(crate) terminal: i64,
    pub(crate) completed: i64,
    pub(crate) failed: i64,
    pub(crate) pruned: i64,
    pub(crate) cancelled: i64,
}

#[derive(Serialize)]
pub(crate) struct TuningAttemptView {
    pub(crate) attempt_id: String,
    pub(crate) bench_run_id: Option<String>,
    pub(crate) status: String,
    pub(crate) started_at: String,
    pub(crate) ended_at: Option<String>,
    pub(crate) failure: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialView {
    pub(crate) trial_id: String,
    pub(crate) trial_number: i64,
    pub(crate) attempt_id: String,
    pub(crate) status: String,
    pub(crate) config: Option<Value>,
    pub(crate) score: Option<f64>,
    pub(crate) mu: Option<f64>,
    pub(crate) sigma: Option<f64>,
    pub(crate) stop_reason: Option<mcts_bench::tuning_lifecycle::TrialReportReason>,
    pub(crate) failure: Option<String>,
    pub(crate) pairs: Vec<TuningPairView>,
    pub(crate) reports: Vec<TuningTrialReportView>,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialReportView {
    pub(crate) completed_pairs: u64,
    pub(crate) rating: TuningRatingView,
    pub(crate) score: f64,
    pub(crate) score_formula_version: u32,
    pub(crate) conservative_k: f64,
    pub(crate) decision: TuningTrialReportDecisionView,
    pub(crate) reported_at: String,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialReportDecisionView {
    pub(crate) outcome: mcts_bench::tuning_lifecycle::TrialReportOutcome,
    pub(crate) reason: mcts_bench::tuning_lifecycle::TrialReportReason,
    pub(crate) pruning_exempt: bool,
    pub(crate) bracket_id: Option<String>,
    pub(crate) rung_resource: Option<u64>,
}

#[derive(Serialize)]
pub(crate) struct TuningOpponentView {
    pub(crate) anchor_id: String,
    pub(crate) config: Value,
    pub(crate) mu: f64,
    pub(crate) sigma: f64,
    pub(crate) label: Option<String>,
    pub(crate) provenance: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TuningRatingView {
    pub(crate) mu: f64,
    pub(crate) sigma: f64,
}

#[derive(Serialize)]
pub(crate) struct TuningStrategyMetricsView {
    pub(crate) iterations_total: u64,
    pub(crate) iterations_first_half: u64,
    pub(crate) move_time_ms: u64,
}

#[derive(Serialize)]
pub(crate) struct TuningGameView {
    pub(crate) game_id: String,
    pub(crate) candidate_side: String,
    pub(crate) outcome: String,
    pub(crate) seed: u64,
    pub(crate) round: u32,
    pub(crate) trace_game_seq: Option<u64>,
    pub(crate) plies: u32,
    pub(crate) elapsed_ms: u64,
    pub(crate) candidate: TuningStrategyMetricsView,
    pub(crate) baseline: TuningStrategyMetricsView,
}

#[derive(Serialize)]
pub(crate) struct TuningPairView {
    pub(crate) pair_id: String,
    pub(crate) pair_index: u32,
    pub(crate) status: String,
    pub(crate) seed: u64,
    pub(crate) round: u32,
    pub(crate) opponent: TuningOpponentView,
    pub(crate) pool_snapshot_fingerprint: String,
    pub(crate) rating_before: TuningRatingView,
    pub(crate) rating_after: Option<TuningRatingView>,
    pub(crate) score: Option<f64>,
    pub(crate) failure: Option<String>,
    pub(crate) games: Vec<TuningGameView>,
}

#[derive(Serialize)]
pub(crate) struct TuningCapabilities {
    pub(crate) has_lifecycle: bool,
    pub(crate) has_pairs: bool,
    pub(crate) has_renderer_trace: bool,
    pub(crate) has_search_reports: bool,
    pub(crate) has_trial_reports: bool,
}

#[derive(Serialize)]
pub(crate) struct TuningCursorBoundary {
    pub(crate) session_sequence: i64,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialPage {
    pub(crate) schema_version: u32,
    pub(crate) trials: Vec<TuningTrialSummaryView>,
    pub(crate) total_count: i64,
    pub(crate) limit: u16,
    pub(crate) next_cursor: Option<String>,
    pub(crate) cursor: TuningCursorBoundary,
}

/// A compact row deliberately omitting the candidate configuration and all
/// child evidence. Those are loaded only from the one-trial endpoint.
#[derive(Serialize)]
pub(crate) struct TuningTrialSummaryView {
    pub(crate) trial_id: String,
    pub(crate) trial_number: i64,
    pub(crate) attempt_id: String,
    pub(crate) state: String,
    pub(crate) reason: Option<mcts_bench::tuning_lifecycle::TrialReportReason>,
    pub(crate) rating: Option<TuningRatingView>,
    pub(crate) score: Option<f64>,
    pub(crate) family: Option<String>,
    pub(crate) config_summary: Option<String>,
    pub(crate) bracket_id: Option<String>,
    pub(crate) resource: Option<u64>,
    pub(crate) pair_count: u64,
    pub(crate) wins: u64,
    pub(crate) losses: u64,
    pub(crate) draws: u64,
    pub(crate) elapsed_ms: u64,
    pub(crate) search_iterations_total: u64,
    pub(crate) search_move_time_ms: u64,
    pub(crate) has_detail: bool,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialDetail {
    pub(crate) schema_version: u32,
    pub(crate) trial: TuningTrialDetailView,
    pub(crate) cursor: TuningCursorBoundary,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialDetailView {
    pub(crate) trial_id: String,
    pub(crate) trial_number: i64,
    pub(crate) attempt_id: String,
    pub(crate) state: String,
    pub(crate) config: Option<Value>,
    pub(crate) score: Option<f64>,
    pub(crate) rating: Option<TuningRatingView>,
    pub(crate) reason: Option<mcts_bench::tuning_lifecycle::TrialReportReason>,
    pub(crate) failure: Option<String>,
    pub(crate) reports: Vec<TuningTrialReportView>,
    pub(crate) pairs: Vec<TuningTrialDetailPairView>,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialDetailPairView {
    pub(crate) pair_id: String,
    pub(crate) pair_index: u32,
    pub(crate) state: String,
    pub(crate) seed: u64,
    pub(crate) round: u32,
    pub(crate) opponent: TuningOpponentView,
    pub(crate) pool_snapshot_fingerprint: String,
    pub(crate) pool_revision: Option<TuningPoolRevisionView>,
    pub(crate) rating_before: TuningRatingView,
    pub(crate) rating_after: Option<TuningRatingView>,
    pub(crate) score: Option<f64>,
    pub(crate) failure: Option<String>,
    pub(crate) games: Vec<TuningTrialDetailGameView>,
}

#[derive(Serialize)]
pub(crate) struct TuningTrialDetailGameView {
    pub(crate) game_id: String,
    pub(crate) candidate_side: String,
    pub(crate) outcome: String,
    pub(crate) seed: u64,
    pub(crate) round: u32,
    pub(crate) plies: u32,
    pub(crate) elapsed_ms: u64,
    pub(crate) candidate: TuningStrategyMetricsView,
    pub(crate) baseline: TuningStrategyMetricsView,
    pub(crate) replay: Option<TuningReplayReference>,
}

#[derive(Serialize)]
pub(crate) struct TuningReplayReference {
    pub(crate) run_id: String,
    pub(crate) game_seq: u64,
    pub(crate) has_renderer_trace: bool,
    pub(crate) has_search_reports: bool,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct TuningResourcePolicyView {
    pub(crate) min_pairs: u64,
    pub(crate) max_pairs: u64,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct TuningRatingPolicyView {
    pub(crate) model: String,
    pub(crate) score: String,
    pub(crate) sigma_stop: Option<f64>,
    pub(crate) conservative_k: f64,
}

#[derive(Serialize)]
pub(crate) struct TuningSamplerPolicyView {
    pub(crate) kind: String,
    pub(crate) seed: u64,
    pub(crate) deterministic: bool,
    pub(crate) startup_trials: u64,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct TuningPruningPolicyView {
    pub(crate) enabled: bool,
    pub(crate) kind: String,
    pub(crate) reduction_factor: f64,
    pub(crate) startup_trials: u64,
}

#[derive(Serialize)]
pub(crate) struct TuningPolicyView {
    pub(crate) resource: TuningResourcePolicyView,
    pub(crate) rating: TuningRatingPolicyView,
    pub(crate) sampler: TuningSamplerPolicyView,
    pub(crate) pruning: TuningPruningPolicyView,
}

#[derive(Serialize)]
pub(crate) struct TuningSessionDetail {
    pub(crate) schema_version: u32,
    pub(crate) summary: TuningSessionSummary,
    pub(crate) attempts: Vec<TuningAttemptView>,
    pub(crate) trials: Vec<TuningTrialView>,
    pub(crate) policy: Option<TuningPolicyView>,
    pub(crate) manifest: Value,
    pub(crate) fingerprint: Option<String>,
    pub(crate) capabilities: TuningCapabilities,
    pub(crate) control: TuningSessionControl,
    pub(crate) cursor: TuningCursorBoundary,
}

/// Compact, bounded evidence used by the tuning analysis views.  Detailed
/// trial, pair, and game evidence is intentionally served by separate routes.
#[derive(Serialize)]
pub(crate) struct TuningAnalysisOverview {
    pub(crate) schema_version: u32,
    pub(crate) policy: Option<TuningPolicyView>,
    pub(crate) objective: TuningAnalysisObjective,
    pub(crate) cursor: TuningCursorBoundary,
    pub(crate) coverage: TuningAnalysisCoverage,
    pub(crate) bracket_resources: Vec<TuningBracketResourceAggregate>,
    pub(crate) decision_groups: Vec<TuningDecisionAggregate>,
    pub(crate) points: Vec<TuningAnalysisPoint>,
    pub(crate) best: Option<TuningAnalysisBest>,
    pub(crate) pool_revisions: Vec<TuningPoolRevisionView>,
    pub(crate) control: TuningSessionControl,
}

#[derive(Serialize)]
pub(crate) struct TuningAnalysisObjective {
    pub(crate) metric: &'static str,
    pub(crate) direction: &'static str,
    pub(crate) complete_trials_only: bool,
}

#[derive(Serialize)]
pub(crate) struct TuningAnalysisCoverage {
    pub(crate) trials: TuningTrialCounts,
    pub(crate) reports: i64,
    pub(crate) pairs: TuningAnalysisPairCoverage,
    pub(crate) points: TuningAnalysisPointCoverage,
}

#[derive(Serialize)]
pub(crate) struct TuningAnalysisPairCoverage {
    pub(crate) total: i64,
    pub(crate) running: i64,
    pub(crate) complete: i64,
    pub(crate) failed: i64,
    pub(crate) unmatched_pool_revisions: i64,
}

#[derive(Serialize)]
pub(crate) struct TuningAnalysisPointCoverage {
    pub(crate) total: i64,
    pub(crate) returned: i64,
    pub(crate) sampled: bool,
}

#[derive(Serialize)]
pub(crate) struct TuningBracketResourceAggregate {
    pub(crate) bracket_id: Option<String>,
    /// The resource consumed by a report, always its completed pair count.
    pub(crate) resource: u64,
    /// The optional rung reported by the tuner is retained as evidence and is
    /// never substituted for `resource`.
    pub(crate) rung_resource: Option<u64>,
    pub(crate) reports: i64,
    pub(crate) trials: i64,
}

#[derive(Serialize)]
pub(crate) struct TuningDecisionAggregate {
    pub(crate) outcome: mcts_bench::tuning_lifecycle::TrialReportOutcome,
    pub(crate) reason: mcts_bench::tuning_lifecycle::TrialReportReason,
    pub(crate) pruning_exempt: bool,
    pub(crate) reports: i64,
}

#[derive(Serialize)]
pub(crate) struct TuningAnalysisPoint {
    pub(crate) trial_id: String,
    pub(crate) trial_number: i64,
    pub(crate) trial_status: String,
    pub(crate) resource: u64,
    pub(crate) rating: TuningRatingView,
    pub(crate) score: f64,
    pub(crate) outcome: mcts_bench::tuning_lifecycle::TrialReportOutcome,
    pub(crate) reason: mcts_bench::tuning_lifecycle::TrialReportReason,
    pub(crate) pruning_exempt: bool,
    pub(crate) bracket_id: Option<String>,
    pub(crate) rung_resource: Option<u64>,
}

#[derive(Serialize)]
pub(crate) struct TuningAnalysisBest {
    pub(crate) score: f64,
    pub(crate) trial_ids: Vec<String>,
}

#[derive(Serialize)]
pub(crate) struct TuningPoolRevisionView {
    pub(crate) pool_snapshot_fingerprint: String,
    pub(crate) display_ordinal: u32,
    pub(crate) observed_at: String,
    pub(crate) pair_count: i64,
    pub(crate) anchors: Vec<TuningPoolAnchorView>,
}

#[derive(Serialize)]
pub(crate) struct TuningPoolAnchorView {
    pub(crate) anchor_ordinal: u32,
    pub(crate) anchor_id: String,
    pub(crate) config: Value,
    pub(crate) rating: TuningRatingView,
    pub(crate) provenance: mcts_bench::tuning_lifecycle::PoolAnchorProvenance,
    pub(crate) insertion_reason: mcts_bench::tuning_lifecycle::PoolAnchorInsertionReason,
    pub(crate) source_trial_id: Option<String>,
}
