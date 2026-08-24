//! DuckDB implementation of [`crate::tuning_analysis_repository::TuningAnalysisRepository`].

use std::sync::{Arc, Mutex};

use duckdb::{params, Connection};

use crate::tuning_analysis_repository::{
    TuningAnalysisBest, TuningAnalysisData, TuningAnalysisPairCoverage, TuningAnalysisPoolAnchor,
    TuningAnalysisPoolRevision, TuningAnalysisReport, TuningAnalysisRepository,
    TuningAnalysisRepositoryError, TuningAnalysisSession, TuningAnalysisTrialCounts,
};

/// A tuning analysis repository backed by a shared DuckDB connection.
#[derive(Clone)]
pub struct SharedDuckDbTuningAnalysisRepository {
    connection: Arc<Mutex<Connection>>,
}

impl SharedDuckDbTuningAnalysisRepository {
    pub fn new(connection: Arc<Mutex<Connection>>) -> Self {
        Self { connection }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, TuningAnalysisRepositoryError> {
        self.connection.lock().map_err(|_| {
            TuningAnalysisRepositoryError::Storage("benchmark database mutex poisoned".into())
        })
    }
}

impl TuningAnalysisRepository for SharedDuckDbTuningAnalysisRepository {
    fn load_analysis(
        &self,
        session_id: &str,
    ) -> Result<Option<TuningAnalysisData>, TuningAnalysisRepositoryError> {
        let connection = self.lock()?;
        load_analysis(&connection, session_id)
    }

    fn load_trial_pool_revisions(
        &self,
        session_id: &str,
        trial_id: &str,
    ) -> Result<Vec<TuningAnalysisPoolRevision>, TuningAnalysisRepositoryError> {
        let connection = self.lock()?;
        load_trial_pool_revisions(&connection, session_id, trial_id)
    }
}

fn load_analysis(
    connection: &Connection,
    session_id: &str,
) -> Result<Option<TuningAnalysisData>, TuningAnalysisRepositoryError> {
    let Some(session) = load_session(connection, session_id)? else {
        return Ok(None);
    };
    Ok(Some(TuningAnalysisData {
        session,
        trial_counts: load_trial_counts(connection, session_id)?,
        reports: load_reports(connection, session_id)?,
        pair_coverage: load_pair_coverage(connection, session_id)?,
        best: load_best(connection, session_id)?,
        pool_revisions: load_pool_revisions(connection, session_id)?,
    }))
}

fn load_session(
    connection: &Connection,
    session_id: &str,
) -> Result<Option<TuningAnalysisSession>, TuningAnalysisRepositoryError> {
    match connection.query_row(
        "SELECT CAST(manifest AS TEXT), last_sequence FROM tuning_sessions WHERE session_id = ?1",
        params![session_id],
        |row| {
            Ok(TuningAnalysisSession {
                manifest: row.get(0)?,
                last_sequence: row.get(1)?,
            })
        },
    ) {
        Ok(session) => Ok(Some(session)),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(storage(error)),
    }
}

fn load_trial_counts(
    connection: &Connection,
    session_id: &str,
) -> Result<TuningAnalysisTrialCounts, TuningAnalysisRepositoryError> {
    connection
        .query_row(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END), 0), \
                    COALESCE(SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END), 0), \
                    COALESCE(SUM(CASE WHEN status IN ('complete', 'failed', 'pruned', 'cancelled') THEN 1 ELSE 0 END), 0), \
                    COALESCE(SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END), 0), \
                    COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0), \
                    COALESCE(SUM(CASE WHEN status = 'pruned' THEN 1 ELSE 0 END), 0), \
                    COALESCE(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END), 0) \
             FROM tuning_trials WHERE session_id = ?1",
            params![session_id],
            |row| {
                Ok(TuningAnalysisTrialCounts {
                    total: row.get(0)?,
                    queued: row.get(1)?,
                    running: row.get(2)?,
                    terminal: row.get(3)?,
                    completed: row.get(4)?,
                    failed: row.get(5)?,
                    pruned: row.get(6)?,
                    cancelled: row.get(7)?,
                })
            },
        )
        .map_err(storage)
}

fn load_reports(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<TuningAnalysisReport>, TuningAnalysisRepositoryError> {
    let mut statement = connection
        .prepare(
            "SELECT reports.trial_id, reports.trial_number, trials.status, reports.completed_pairs, \
                    reports.mu, reports.sigma, reports.score, reports.outcome, reports.reason, \
                    reports.pruning_exempt, reports.bracket_id, reports.rung_resource \
             FROM tuning_trial_reports reports \
             JOIN tuning_trials trials USING (session_id, trial_id) \
             WHERE reports.session_id = ?1 \
             ORDER BY reports.bracket_id ASC NULLS FIRST, reports.completed_pairs ASC, \
                      reports.outcome ASC, reports.trial_number ASC, reports.event_id ASC",
        )
        .map_err(storage)?;
    statement
        .query_map(params![session_id], |row| {
            let outcome: String = row.get(7)?;
            let reason: String = row.get(8)?;
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                outcome,
                reason,
                row.get(9)?,
                row.get(10)?,
                row.get(11)?,
            ))
        })
        .map_err(storage)?
        .map(|row| {
            let (
                trial_id,
                trial_number,
                trial_status,
                resource,
                mu,
                sigma,
                score,
                outcome,
                reason,
                pruning_exempt,
                bracket_id,
                rung_resource,
            ) = row.map_err(storage)?;
            Ok(TuningAnalysisReport {
                trial_id,
                trial_number,
                trial_status,
                resource,
                mu,
                sigma,
                score,
                outcome: decode_enum(&outcome)?,
                reason: decode_enum(&reason)?,
                pruning_exempt,
                bracket_id,
                rung_resource,
            })
        })
        .collect()
}

fn load_pair_coverage(
    connection: &Connection,
    session_id: &str,
) -> Result<TuningAnalysisPairCoverage, TuningAnalysisRepositoryError> {
    connection
        .query_row(
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
            params![session_id],
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
        .map_err(storage)
}

fn load_best(
    connection: &Connection,
    session_id: &str,
) -> Result<Option<TuningAnalysisBest>, TuningAnalysisRepositoryError> {
    let mut statement = connection
        .prepare(
            "SELECT trial_id, trial_number, score \
             FROM tuning_trials \
             WHERE session_id = ?1 AND status = 'complete' AND score IS NOT NULL \
             ORDER BY score DESC, trial_number ASC, trial_id ASC",
        )
        .map_err(storage)?;
    let trials: Vec<(String, i64, f64)> = statement
        .query_map(params![session_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)?;
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
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<TuningAnalysisPoolRevision>, TuningAnalysisRepositoryError> {
    let mut statement = connection
        .prepare(
            "SELECT revisions.pool_snapshot_fingerprint, revisions.display_ordinal, \
                    CAST(revisions.observed_at AS TEXT), COUNT(pairs.pair_id) \
             FROM tuning_pool_revisions revisions \
             LEFT JOIN tuning_evaluation_pairs pairs \
               ON pairs.session_id = revisions.session_id \
              AND pairs.pool_snapshot_fingerprint = revisions.pool_snapshot_fingerprint \
             WHERE revisions.session_id = ?1 \
             GROUP BY revisions.pool_snapshot_fingerprint, revisions.display_ordinal, revisions.observed_at \
             ORDER BY revisions.display_ordinal ASC, revisions.pool_snapshot_fingerprint ASC",
        )
        .map_err(storage)?;
    statement
        .query_map(params![session_id], |row| {
            Ok::<(String, u32, String, i64), duckdb::Error>((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
            ))
        })
        .map_err(storage)?
        .map(|row| {
            let (pool_snapshot_fingerprint, display_ordinal, observed_at, pair_count) =
                row.map_err(storage)?;
            Ok(TuningAnalysisPoolRevision {
                anchors: load_pool_anchors(connection, session_id, &pool_snapshot_fingerprint)?,
                pool_snapshot_fingerprint,
                display_ordinal,
                observed_at,
                pair_count,
            })
        })
        .collect()
}

fn load_trial_pool_revisions(
    connection: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Vec<TuningAnalysisPoolRevision>, TuningAnalysisRepositoryError> {
    let mut statement = connection
        .prepare(
            "SELECT revisions.pool_snapshot_fingerprint, revisions.display_ordinal, \
                    CAST(revisions.observed_at AS TEXT), COUNT(pairs.pair_id) \
         FROM tuning_pool_revisions revisions \
         LEFT JOIN tuning_evaluation_pairs pairs \
           ON pairs.session_id = revisions.session_id \
          AND pairs.pool_snapshot_fingerprint = revisions.pool_snapshot_fingerprint \
         WHERE revisions.session_id = ?1 AND EXISTS ( \
             SELECT 1 FROM tuning_evaluation_pairs selected \
             WHERE selected.session_id = revisions.session_id \
               AND selected.trial_id = ?2 \
               AND selected.pool_snapshot_fingerprint = revisions.pool_snapshot_fingerprint \
         ) \
         GROUP BY revisions.pool_snapshot_fingerprint, revisions.display_ordinal, revisions.observed_at \
         ORDER BY revisions.display_ordinal ASC, revisions.pool_snapshot_fingerprint ASC",
        )
        .map_err(storage)?;
    statement
        .query_map(params![session_id, trial_id], |row| {
            Ok::<(String, u32, String, i64), duckdb::Error>((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
            ))
        })
        .map_err(storage)?
        .map(|row| {
            let (pool_snapshot_fingerprint, display_ordinal, observed_at, pair_count) =
                row.map_err(storage)?;
            Ok(TuningAnalysisPoolRevision {
                anchors: load_pool_anchors(connection, session_id, &pool_snapshot_fingerprint)?,
                pool_snapshot_fingerprint,
                display_ordinal,
                observed_at,
                pair_count,
            })
        })
        .collect()
}

fn load_pool_anchors(
    connection: &Connection,
    session_id: &str,
    fingerprint: &str,
) -> Result<Vec<TuningAnalysisPoolAnchor>, TuningAnalysisRepositoryError> {
    let mut statement = connection
        .prepare(
            "SELECT anchor_ordinal, anchor_id, CAST(config AS TEXT), mu, sigma, provenance, \
                    insertion_reason, source_trial_id \
             FROM tuning_pool_anchors \
             WHERE session_id = ?1 AND pool_snapshot_fingerprint = ?2 \
             ORDER BY anchor_ordinal ASC, anchor_id ASC",
        )
        .map_err(storage)?;
    statement
        .query_map(params![session_id, fingerprint], |row| {
            Ok::<
                (
                    u32,
                    String,
                    String,
                    f64,
                    f64,
                    String,
                    String,
                    Option<String>,
                ),
                duckdb::Error,
            >((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
            ))
        })
        .map_err(storage)?
        .map(|row| {
            let (
                anchor_ordinal,
                anchor_id,
                config,
                mu,
                sigma,
                provenance,
                insertion_reason,
                source_trial_id,
            ) = row.map_err(storage)?;
            Ok(TuningAnalysisPoolAnchor {
                anchor_ordinal,
                anchor_id,
                config: serde_json::from_str(&config)
                    .map_err(|error| TuningAnalysisRepositoryError::Storage(error.to_string()))?,
                mu,
                sigma,
                provenance: decode_enum(&provenance)?,
                insertion_reason: decode_enum(&insertion_reason)?,
                source_trial_id,
            })
        })
        .collect()
}

fn decode_enum<T: serde::de::DeserializeOwned>(
    value: &str,
) -> Result<T, TuningAnalysisRepositoryError> {
    serde_json::from_value(serde_json::Value::String(value.into()))
        .map_err(|error| TuningAnalysisRepositoryError::Storage(error.to_string()))
}

fn storage(error: duckdb::Error) -> TuningAnalysisRepositoryError {
    TuningAnalysisRepositoryError::Storage(error.to_string())
}
