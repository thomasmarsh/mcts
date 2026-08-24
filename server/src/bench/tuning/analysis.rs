use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use axum::{
    extract::{Path as AxumPath, State as AxumState},
    response::Json,
};
use mcts_bench::tuning_analysis_repository::{
    TuningAnalysisData, TuningAnalysisPoolAnchor, TuningAnalysisPoolRevision, TuningAnalysisReport,
    TuningAnalysisRepositoryError, TuningAnalysisTrialCounts,
};
use mcts_bench::tuning_lifecycle::{TrialReportOutcome, TrialReportReason};

use super::super::{
    BenchError, BenchState, TuningAnalysisBest, TuningAnalysisCoverage, TuningAnalysisObjective,
    TuningAnalysisOverview, TuningAnalysisPairCoverage, TuningAnalysisPoint,
    TuningAnalysisPointCoverage, TuningBracketResourceAggregate, TuningCursorBoundary,
    TuningDecisionAggregate, TuningPoolAnchorView, TuningPoolRevisionView, TuningSessionControl,
};
use super::commands::session_control;
use super::sessions::{decode_manifest, decode_manifest_policy, rating_view};

const ANALYSIS_POINT_LIMIT: usize = 2_000;

pub(crate) async fn get_tuning_analysis_overview(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Result<Json<TuningAnalysisOverview>, BenchError> {
    let control = {
        let db = state.db.lock().unwrap();
        session_control(&db, &session_id)?
    };
    let analysis = state
        .tuning_analysis_repository
        .load_analysis(&session_id)
        .map_err(tuning_analysis_repository_error)?;
    let overview = analysis
        .map(|analysis| load_tuning_analysis_overview(analysis, control))
        .transpose()?
        .ok_or_else(|| BenchError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: format!("tuning session '{session_id}' not found"),
        })?;
    Ok(Json(overview))
}

fn load_tuning_analysis_overview(
    analysis: TuningAnalysisData,
    control: TuningSessionControl,
) -> Result<TuningAnalysisOverview, BenchError> {
    let manifest = decode_manifest(&analysis.session.manifest)?;
    let reports = analysis
        .reports
        .into_iter()
        .map(AnalysisReportRow::from)
        .collect::<Vec<_>>();
    let counts = trial_counts(analysis.trial_counts);
    let pairs = pair_coverage(analysis.pair_coverage);
    let pool_revisions = analysis
        .pool_revisions
        .into_iter()
        .map(|revision| pool_revision(&revision))
        .collect();
    let best = analysis.best.map(|best| TuningAnalysisBest {
        score: best.score,
        trial_ids: best.trial_ids,
    });
    let (bracket_resources, decision_groups) = aggregate_analysis_reports(&reports);
    let points = sample_analysis_points(&reports);
    let returned = points.len() as i64;
    let total = reports.len() as i64;

    Ok(TuningAnalysisOverview {
        schema_version: 1,
        policy: decode_manifest_policy(&manifest)?,
        objective: TuningAnalysisObjective {
            metric: "score",
            direction: "maximize",
            complete_trials_only: true,
        },
        cursor: TuningCursorBoundary {
            session_sequence: analysis.session.last_sequence,
        },
        coverage: TuningAnalysisCoverage {
            trials: counts,
            reports: total,
            pairs,
            points: TuningAnalysisPointCoverage {
                total,
                returned,
                sampled: total > returned,
            },
        },
        bracket_resources,
        decision_groups,
        points,
        best,
        pool_revisions,
        control,
    })
}

struct AnalysisReportRow {
    trial_id: String,
    trial_number: i64,
    trial_status: String,
    resource: u64,
    mu: f64,
    sigma: f64,
    score: f64,
    outcome: TrialReportOutcome,
    reason: TrialReportReason,
    pruning_exempt: bool,
    bracket_id: Option<String>,
    rung_resource: Option<u64>,
}

impl From<TuningAnalysisReport> for AnalysisReportRow {
    fn from(report: TuningAnalysisReport) -> Self {
        Self {
            trial_id: report.trial_id,
            trial_number: report.trial_number,
            trial_status: report.trial_status,
            resource: report.resource,
            mu: report.mu,
            sigma: report.sigma,
            score: report.score,
            outcome: report.outcome,
            reason: report.reason,
            pruning_exempt: report.pruning_exempt,
            bracket_id: report.bracket_id,
            rung_resource: report.rung_resource,
        }
    }
}

type AnalysisResourceKey = (Option<String>, u64, Option<u64>);
type AnalysisResourceAggregate = (i64, BTreeSet<String>);
type AnalysisDecisionKey = (u8, u8, bool);
type AnalysisDecisionAggregate = (TrialReportOutcome, TrialReportReason, i64);

fn trial_counts(counts: TuningAnalysisTrialCounts) -> super::super::TuningTrialCounts {
    super::super::TuningTrialCounts {
        total: counts.total,
        queued: counts.queued,
        running: counts.running,
        terminal: counts.terminal,
        completed: counts.completed,
        failed: counts.failed,
        pruned: counts.pruned,
        cancelled: counts.cancelled,
    }
}

fn pair_coverage(
    coverage: mcts_bench::tuning_analysis_repository::TuningAnalysisPairCoverage,
) -> TuningAnalysisPairCoverage {
    TuningAnalysisPairCoverage {
        total: coverage.total,
        running: coverage.running,
        complete: coverage.complete,
        failed: coverage.failed,
        unmatched_pool_revisions: coverage.unmatched_pool_revisions,
    }
}

fn aggregate_analysis_reports(
    reports: &[AnalysisReportRow],
) -> (
    Vec<TuningBracketResourceAggregate>,
    Vec<TuningDecisionAggregate>,
) {
    let mut resources: BTreeMap<AnalysisResourceKey, AnalysisResourceAggregate> = BTreeMap::new();
    let mut decisions: BTreeMap<AnalysisDecisionKey, AnalysisDecisionAggregate> = BTreeMap::new();
    for report in reports {
        let resource = resources
            .entry((
                report.bracket_id.clone(),
                report.resource,
                report.rung_resource,
            ))
            .or_insert_with(|| (0, BTreeSet::new()));
        resource.0 += 1;
        resource.1.insert(report.trial_id.clone());

        let decision = decisions
            .entry((
                report_outcome_rank(report.outcome),
                report_reason_rank(report.reason),
                report.pruning_exempt,
            ))
            .or_insert((report.outcome, report.reason, 0));
        decision.2 += 1;
    }
    (
        resources
            .into_iter()
            .map(
                |((bracket_id, resource, rung_resource), (reports, trials))| {
                    TuningBracketResourceAggregate {
                        bracket_id,
                        resource,
                        rung_resource,
                        reports,
                        trials: trials.len() as i64,
                    }
                },
            )
            .collect(),
        decisions
            .into_iter()
            .map(
                |((_, _, pruning_exempt), (outcome, reason, reports))| TuningDecisionAggregate {
                    outcome,
                    reason,
                    pruning_exempt,
                    reports,
                },
            )
            .collect(),
    )
}

fn sample_analysis_points(reports: &[AnalysisReportRow]) -> Vec<TuningAnalysisPoint> {
    let selected = if reports.len() <= ANALYSIS_POINT_LIMIT {
        vec![true; reports.len()]
    } else {
        let mut strata: BTreeMap<(Option<String>, u64, u8), Vec<usize>> = BTreeMap::new();
        for (index, report) in reports.iter().enumerate() {
            strata
                .entry((
                    report.bracket_id.clone(),
                    report.resource,
                    report_outcome_rank(report.outcome),
                ))
                .or_default()
                .push(index);
        }
        let mut selected = vec![false; reports.len()];
        let mut returned = 0;
        let mut strata_by_coverage: Vec<&Vec<usize>> = strata.values().collect();
        strata_by_coverage.sort_by_key(|indices| indices.len());
        for indices in &strata_by_coverage {
            if returned == ANALYSIS_POINT_LIMIT {
                break;
            }
            selected[indices[0]] = true;
            returned += 1;
        }
        let mut offset = 1;
        while returned < ANALYSIS_POINT_LIMIT {
            let mut added = false;
            for indices in strata.values() {
                if returned == ANALYSIS_POINT_LIMIT {
                    break;
                }
                if let Some(&index) = indices.get(offset) {
                    selected[index] = true;
                    returned += 1;
                    added = true;
                }
            }
            if !added {
                break;
            }
            offset += 1;
        }
        selected
    };
    reports
        .iter()
        .zip(selected)
        .filter(|(_, selected)| *selected)
        .map(|(report, _)| analysis_point(report))
        .collect()
}

fn analysis_point(report: &AnalysisReportRow) -> TuningAnalysisPoint {
    TuningAnalysisPoint {
        trial_id: report.trial_id.clone(),
        trial_number: report.trial_number,
        trial_status: report.trial_status.clone(),
        resource: report.resource,
        rating: rating_view(report.mu, report.sigma),
        score: report.score,
        outcome: report.outcome,
        reason: report.reason,
        pruning_exempt: report.pruning_exempt,
        bracket_id: report.bracket_id.clone(),
        rung_resource: report.rung_resource,
    }
}

pub(super) fn pool_revision(revision: &TuningAnalysisPoolRevision) -> TuningPoolRevisionView {
    TuningPoolRevisionView {
        pool_snapshot_fingerprint: revision.pool_snapshot_fingerprint.clone(),
        display_ordinal: revision.display_ordinal,
        observed_at: revision.observed_at.clone(),
        pair_count: revision.pair_count,
        anchors: revision.anchors.iter().map(pool_anchor).collect(),
    }
}

fn pool_anchor(anchor: &TuningAnalysisPoolAnchor) -> TuningPoolAnchorView {
    TuningPoolAnchorView {
        anchor_ordinal: anchor.anchor_ordinal,
        anchor_id: anchor.anchor_id.clone(),
        config: anchor.config.clone(),
        rating: rating_view(anchor.mu, anchor.sigma),
        provenance: anchor.provenance.clone(),
        insertion_reason: anchor.insertion_reason.clone(),
        source_trial_id: anchor.source_trial_id.clone(),
    }
}

pub(super) fn tuning_analysis_repository_error(error: TuningAnalysisRepositoryError) -> BenchError {
    BenchError {
        status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("tuning analysis storage error: {error}"),
    }
}

fn report_outcome_rank(outcome: TrialReportOutcome) -> u8 {
    match outcome {
        TrialReportOutcome::Continue => 0,
        TrialReportOutcome::Complete => 1,
        TrialReportOutcome::Prune => 2,
    }
}

fn report_reason_rank(reason: TrialReportReason) -> u8 {
    match reason {
        TrialReportReason::BelowMinPairs => 0,
        TrialReportReason::PruningDisabled => 1,
        TrialReportReason::StartupExempt => 2,
        TrialReportReason::HyperbandKeep => 3,
        TrialReportReason::Confidence => 4,
        TrialReportReason::MaxPairs => 5,
        TrialReportReason::HyperbandPrune => 6,
    }
}
