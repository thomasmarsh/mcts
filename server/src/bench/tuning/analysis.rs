use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use axum::{
    extract::{Path as AxumPath, State as AxumState},
    response::Json,
};
use duckdb::Connection;
use mcts_bench::tuning_lifecycle::{TrialReportOutcome, TrialReportReason};

use super::super::{
    BenchError, BenchState, TuningAnalysisBest, TuningAnalysisCoverage, TuningAnalysisObjective,
    TuningAnalysisOverview, TuningAnalysisPairCoverage, TuningAnalysisPoint,
    TuningAnalysisPointCoverage, TuningBracketResourceAggregate, TuningCursorBoundary,
    TuningDecisionAggregate, TuningPoolAnchorView, TuningPoolRevisionView, TuningSessionControl,
};
use super::commands::session_control;
use super::sessions::{
    decode_json, decode_manifest, decode_manifest_policy, decode_report_enum, load_session,
    load_trial_counts, rating_view,
};

const ANALYSIS_POINT_LIMIT: usize = 2_000;

pub(crate) async fn get_tuning_analysis_overview(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Result<Json<TuningAnalysisOverview>, BenchError> {
    let db = state.db.lock().unwrap();
    let control = session_control(&db, &session_id)?;
    let overview =
        load_tuning_analysis_overview(&db, &session_id, control)?.ok_or_else(|| BenchError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: format!("tuning session '{session_id}' not found"),
        })?;
    Ok(Json(overview))
}

fn load_tuning_analysis_overview(
    db: &Connection,
    session_id: &str,
    control: TuningSessionControl,
) -> Result<Option<TuningAnalysisOverview>, BenchError> {
    let Some(session) = load_session(db, session_id)? else {
        return Ok(None);
    };
    let manifest = decode_manifest(&session.manifest)?;
    let reports = load_analysis_reports(db, session_id)?;
    let counts = load_trial_counts(db, session_id)?;
    let pairs = load_analysis_pair_coverage(db, session_id)?;
    let pool_revisions = load_pool_revisions(db, session_id)?;
    let best = load_analysis_best(db, session_id)?;
    let (bracket_resources, decision_groups) = aggregate_analysis_reports(&reports);
    let points = sample_analysis_points(&reports);
    let returned = points.len() as i64;
    let total = reports.len() as i64;

    Ok(Some(TuningAnalysisOverview {
        schema_version: 1,
        policy: decode_manifest_policy(&manifest)?,
        objective: TuningAnalysisObjective {
            metric: "score",
            direction: "maximize",
            complete_trials_only: true,
        },
        cursor: TuningCursorBoundary {
            session_sequence: session.last_sequence,
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
    }))
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

type AnalysisResourceKey = (Option<String>, u64, Option<u64>);
type AnalysisResourceAggregate = (i64, BTreeSet<String>);
type AnalysisDecisionKey = (u8, u8, bool);
type AnalysisDecisionAggregate = (TrialReportOutcome, TrialReportReason, i64);

fn load_analysis_reports(
    db: &Connection,
    session_id: &str,
) -> Result<Vec<AnalysisReportRow>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT reports.trial_id, reports.trial_number, trials.status, reports.completed_pairs, \
                reports.mu, reports.sigma, reports.score, reports.outcome, reports.reason, \
                reports.pruning_exempt, reports.bracket_id, reports.rung_resource \
         FROM tuning_trial_reports reports \
         JOIN tuning_trials trials USING (session_id, trial_id) \
         WHERE reports.session_id = ?1 \
         ORDER BY reports.bracket_id ASC NULLS FIRST, reports.completed_pairs ASC, \
                  reports.outcome ASC, reports.trial_number ASC, reports.event_id ASC",
    )?;
    query
        .query_map(duckdb::params![session_id], |row| {
            let outcome: String = row.get(7)?;
            let reason: String = row.get(8)?;
            Ok(AnalysisReportRow {
                trial_id: row.get(0)?,
                trial_number: row.get(1)?,
                trial_status: row.get(2)?,
                resource: row.get(3)?,
                mu: row.get(4)?,
                sigma: row.get(5)?,
                score: row.get(6)?,
                outcome: decode_report_enum(&outcome, 7)?,
                reason: decode_report_enum(&reason, 8)?,
                pruning_exempt: row.get(9)?,
                bracket_id: row.get(10)?,
                rung_resource: row.get(11)?,
            })
        })?
        .collect()
}

fn load_analysis_pair_coverage(
    db: &Connection,
    session_id: &str,
) -> Result<TuningAnalysisPairCoverage, duckdb::Error> {
    db.query_row(
        "SELECT COUNT(*), \
                COALESCE(SUM(CASE WHEN pairs.status = 'running' THEN 1 ELSE 0 END), 0), \
                COALESCE(SUM(CASE WHEN pairs.status = 'complete' THEN 1 ELSE 0 END), 0), \
                COALESCE(SUM(CASE WHEN pairs.status = 'failed' THEN 1 ELSE 0 END), 0), \
                COALESCE(SUM(CASE WHEN revisions.pool_snapshot_fingerprint IS NULL THEN 1 ELSE 0 END), 0) \
         FROM tuning_evaluation_pairs pairs \
         LEFT JOIN tuning_pool_revisions revisions \
           ON revisions.session_id = pairs.session_id \
          AND revisions.pool_snapshot_fingerprint = pairs.pool_snapshot_fingerprint \
         WHERE pairs.session_id = ?1",
        duckdb::params![session_id],
        |row| {
            Ok(TuningAnalysisPairCoverage {
                total: row.get(0)?,
                running: row.get(1)?,
                complete: row.get(2)?,
                failed: row.get(3)?,
                unmatched_pool_revisions: row.get(4)?,
            })
        },
    )
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

fn load_analysis_best(
    db: &Connection,
    session_id: &str,
) -> Result<Option<TuningAnalysisBest>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT trial_id, trial_number, score \
         FROM tuning_trials \
         WHERE session_id = ?1 AND status = 'complete' AND score IS NOT NULL \
         ORDER BY score DESC, trial_number ASC, trial_id ASC",
    )?;
    let trials: Vec<(String, i64, f64)> = query
        .query_map(duckdb::params![session_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .collect::<Result<_, _>>()?;
    let Some((_, _, score)) = trials.first() else {
        return Ok(None);
    };
    let score = *score;
    Ok(Some(TuningAnalysisBest {
        score,
        trial_ids: trials
            .into_iter()
            .take_while(|(_, _, trial_score)| trial_score.total_cmp(&score).is_eq())
            .map(|(trial_id, _, _)| trial_id)
            .collect(),
    }))
}

fn load_pool_revisions(
    db: &Connection,
    session_id: &str,
) -> Result<Vec<TuningPoolRevisionView>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT revisions.pool_snapshot_fingerprint, revisions.display_ordinal, \
                CAST(revisions.observed_at AS TEXT), COUNT(pairs.pair_id) \
         FROM tuning_pool_revisions revisions \
         LEFT JOIN tuning_evaluation_pairs pairs \
           ON pairs.session_id = revisions.session_id \
          AND pairs.pool_snapshot_fingerprint = revisions.pool_snapshot_fingerprint \
         WHERE revisions.session_id = ?1 \
         GROUP BY revisions.pool_snapshot_fingerprint, revisions.display_ordinal, revisions.observed_at \
         ORDER BY revisions.display_ordinal ASC, revisions.pool_snapshot_fingerprint ASC",
    )?;
    query
        .query_map(duckdb::params![session_id], |row| {
            let fingerprint: String = row.get(0)?;
            Ok(TuningPoolRevisionView {
                pool_snapshot_fingerprint: fingerprint.clone(),
                display_ordinal: row.get(1)?,
                observed_at: row.get(2)?,
                pair_count: row.get(3)?,
                anchors: load_pool_anchors(db, session_id, &fingerprint)?,
            })
        })?
        .collect()
}

pub(super) fn load_pool_anchors(
    db: &Connection,
    session_id: &str,
    fingerprint: &str,
) -> Result<Vec<TuningPoolAnchorView>, duckdb::Error> {
    let mut query = db.prepare(
        "SELECT anchor_ordinal, anchor_id, CAST(config AS TEXT), mu, sigma, provenance, \
                insertion_reason, source_trial_id \
         FROM tuning_pool_anchors \
         WHERE session_id = ?1 AND pool_snapshot_fingerprint = ?2 \
         ORDER BY anchor_ordinal ASC, anchor_id ASC",
    )?;
    query
        .query_map(duckdb::params![session_id, fingerprint], |row| {
            let config: String = row.get(2)?;
            let provenance: String = row.get(5)?;
            let insertion_reason: String = row.get(6)?;
            Ok(TuningPoolAnchorView {
                anchor_ordinal: row.get(0)?,
                anchor_id: row.get(1)?,
                config: decode_json(&config, 2)?,
                rating: rating_view(row.get(3)?, row.get(4)?),
                provenance: decode_report_enum(&provenance, 5)?,
                insertion_reason: decode_report_enum(&insertion_reason, 6)?,
                source_trial_id: row.get(7)?,
            })
        })?
        .collect()
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
