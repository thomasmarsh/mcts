//! DuckDB implementation of [`crate::tuning_session_repository::TuningSessionRepository`].

use std::sync::{Arc, Mutex};

use duckdb::{params, Connection};

use crate::tuning_session_repository::*;

/// A tuning-session repository backed by a shared DuckDB connection.
#[derive(Clone)]
pub struct SharedDuckDbTuningSessionRepository {
    connection: Arc<Mutex<Connection>>,
}

impl SharedDuckDbTuningSessionRepository {
    pub fn new(connection: Arc<Mutex<Connection>>) -> Self {
        Self { connection }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, TuningSessionRepositoryError> {
        self.connection.lock().map_err(|_| {
            TuningSessionRepositoryError::Storage("benchmark database mutex poisoned".into())
        })
    }
}

impl TuningSessionRepository for SharedDuckDbTuningSessionRepository {
    fn load_session_list(&self) -> Result<TuningSessionListData, TuningSessionRepositoryError> {
        let connection = self.lock()?;
        let mut sessions = load_session_list_rows(&connection)?;
        for session in &mut sessions {
            session.control =
                crate::tuning_command_store::reconcile(&connection, &session.session_id)
                    .map_err(|error| TuningSessionRepositoryError::Storage(error.to_string()))?;
        }
        let data = TuningSessionListData {
            sessions,
            attempts: load_session_list_attempts(&connection)?,
        };
        validate_list_data(&data)?;
        Ok(data)
    }

    fn load_session_detail(
        &self,
        session_id: &str,
    ) -> Result<Option<TuningSessionDetailData>, TuningSessionRepositoryError> {
        let connection = self.lock()?;
        let Some(session) = load_session(&connection, session_id)? else {
            return Ok(None);
        };
        let data = TuningSessionDetailData {
            trial_counts: load_trial_counts(&connection, session_id)?,
            attempts: load_attempts(&connection, session_id)?,
            trials: load_trials(&connection, session_id)?,
            reports: load_trial_reports(&connection, session_id)?,
            pairs: load_pairs(&connection, session_id)?,
            games: load_games(&connection, session_id)?,
            capabilities: load_capabilities(&connection, session_id)?,
            control: crate::tuning_command_store::reconcile(&connection, session_id)
                .map_err(|error| TuningSessionRepositoryError::Storage(error.to_string()))?,
            session,
        };
        validate_detail_data(&data)?;
        Ok(Some(data))
    }

    fn load_session_control(
        &self,
        session_id: &str,
    ) -> Result<Option<crate::tuning_command_store::SessionControl>, TuningSessionRepositoryError>
    {
        let connection = self.lock()?;
        if load_session(&connection, session_id)?.is_none() {
            return Ok(None);
        }
        crate::tuning_command_store::reconcile(&connection, session_id)
            .map(Some)
            .map_err(|error| TuningSessionRepositoryError::Storage(error.to_string()))
    }
}

fn load_session_list_rows(
    connection: &Connection,
) -> Result<Vec<TuningSessionListRow>, TuningSessionRepositoryError> {
    let mut query = connection.prepare(
        "WITH trial_counts AS ( \
             SELECT session_id, COUNT(*) AS total, \
                    SUM(status = 'queued') AS queued, SUM(status = 'running') AS running, \
                    SUM(status IN ('complete', 'failed', 'pruned', 'cancelled')) AS terminal, \
                    SUM(status = 'complete') AS completed, SUM(status = 'failed') AS failed, \
                    SUM(status = 'pruned') AS pruned, SUM(status = 'cancelled') AS cancelled \
             FROM tuning_trials GROUP BY session_id \
         ), pair_capabilities AS ( \
             SELECT pairs.session_id, COUNT(*) AS pair_count, \
                    COUNT(DISTINCT renderer_moves.run_id || ':' || renderer_moves.game_seq) AS renderer_trace_count, \
                    COUNT(DISTINCT report_moves.run_id || ':' || report_moves.game_seq) AS search_report_count \
             FROM tuning_evaluation_pairs pairs \
             LEFT JOIN tuning_games games USING (session_id, pair_id) \
             LEFT JOIN tuning_attempts attempts ON attempts.attempt_id = pairs.attempt_id \
             LEFT JOIN (SELECT DISTINCT run_id, game_seq FROM game_moves WHERE trace_schema_version = 1) renderer_moves \
                    ON renderer_moves.run_id = attempts.bench_run_id AND renderer_moves.game_seq = games.trace_game_seq \
             LEFT JOIN (SELECT DISTINCT run_id, game_seq FROM game_moves WHERE search_report IS NOT NULL) report_moves \
                    ON report_moves.run_id = attempts.bench_run_id AND report_moves.game_seq = games.trace_game_seq \
             GROUP BY pairs.session_id \
         ), trial_report_capabilities AS ( \
             SELECT session_id, COUNT(*) AS trial_report_count FROM tuning_trial_reports GROUP BY session_id \
         ), activity AS ( \
             SELECT session_id, created_at AS occurred_at FROM tuning_sessions \
             UNION ALL SELECT session_id, started_at FROM tuning_attempts \
             UNION ALL SELECT session_id, ended_at FROM tuning_attempts WHERE ended_at IS NOT NULL \
             UNION ALL SELECT session_id, created_at FROM tuning_trials \
             UNION ALL SELECT session_id, started_at FROM tuning_trials WHERE started_at IS NOT NULL \
             UNION ALL SELECT session_id, ended_at FROM tuning_trials WHERE ended_at IS NOT NULL \
             UNION ALL SELECT session_id, started_at FROM tuning_evaluation_pairs \
             UNION ALL SELECT session_id, ended_at FROM tuning_evaluation_pairs WHERE ended_at IS NOT NULL \
             UNION ALL SELECT session_id, finished_at FROM tuning_games \
         ) \
         SELECT sessions.session_id, sessions.status, sessions.target_trial_count, \
                CAST(sessions.manifest AS TEXT), CAST(sessions.created_at AS TEXT), \
                CAST(MAX(activity.occurred_at) AS TEXT), \
                COALESCE(trial_counts.total, 0), COALESCE(trial_counts.queued, 0), \
                COALESCE(trial_counts.running, 0), COALESCE(trial_counts.terminal, 0), \
                COALESCE(trial_counts.completed, 0), COALESCE(trial_counts.failed, 0), \
                COALESCE(trial_counts.pruned, 0), COALESCE(trial_counts.cancelled, 0), \
                COALESCE(pair_capabilities.pair_count, 0), COALESCE(pair_capabilities.renderer_trace_count, 0), \
                COALESCE(pair_capabilities.search_report_count, 0), COALESCE(trial_report_capabilities.trial_report_count, 0) \
         FROM tuning_sessions sessions \
         LEFT JOIN trial_counts USING (session_id) \
         LEFT JOIN pair_capabilities USING (session_id) \
         LEFT JOIN trial_report_capabilities USING (session_id) \
         JOIN activity USING (session_id) \
         GROUP BY sessions.session_id, sessions.status, sessions.target_trial_count, sessions.manifest, \
                  sessions.created_at, trial_counts.total, trial_counts.queued, trial_counts.running, \
                  trial_counts.terminal, trial_counts.completed, trial_counts.failed, trial_counts.pruned, \
                  trial_counts.cancelled, pair_capabilities.pair_count, pair_capabilities.renderer_trace_count, \
                  pair_capabilities.search_report_count, trial_report_capabilities.trial_report_count \
         ORDER BY MAX(activity.occurred_at) DESC, sessions.session_id DESC",
    ).map_err(storage)?;
    query
        .query_map([], |row| {
            Ok(TuningSessionListRow {
                session_id: row.get(0)?,
                status: row.get(1)?,
                target_trial_count: row.get(2)?,
                manifest: row.get(3)?,
                created_at: row.get(4)?,
                last_activity_at: row.get(5)?,
                trial_counts: TuningTrialCountsRow {
                    total: row.get(6)?,
                    queued: row.get(7)?,
                    running: row.get(8)?,
                    terminal: row.get(9)?,
                    completed: row.get(10)?,
                    failed: row.get(11)?,
                    pruned: row.get(12)?,
                    cancelled: row.get(13)?,
                },
                pair_count: row.get(14)?,
                renderer_trace_count: row.get(15)?,
                search_report_count: row.get(16)?,
                trial_report_count: row.get(17)?,
                control: crate::tuning_command_store::SessionControl {
                    control_version: 0,
                    target_trial_count: None,
                    consumed_trial_count: 0,
                    active_attempt_id: None,
                    launch_reservation: None,
                    stop_attempt_id: None,
                    recovery_required: false,
                    allowed_commands: Vec::new(),
                },
            })
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn load_session_list_attempts(
    connection: &Connection,
) -> Result<Vec<TuningSessionAttemptRow>, TuningSessionRepositoryError> {
    let mut query = connection.prepare(
        "SELECT session_id, attempt_id, bench_run_id, status, CAST(started_at AS TEXT), CAST(ended_at AS TEXT), failure FROM tuning_attempts ORDER BY session_id, started_at ASC, attempt_id ASC",
    ).map_err(storage)?;
    query
        .query_map([], attempt_from_row)
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn load_session(
    connection: &Connection,
    session_id: &str,
) -> Result<Option<TuningSessionRow>, TuningSessionRepositoryError> {
    match connection.query_row(
        "SELECT session_id, status, target_trial_count, CAST(manifest AS TEXT), manifest_fingerprint, last_sequence FROM tuning_sessions WHERE session_id = ?1",
        params![session_id],
        |row| Ok(TuningSessionRow { session_id: row.get(0)?, status: row.get(1)?, target_trial_count: row.get(2)?, manifest: row.get(3)?, fingerprint: row.get(4)?, last_sequence: row.get(5)? }),
    ) {
        Ok(row) => Ok(Some(row)), Err(duckdb::Error::QueryReturnedNoRows) => Ok(None), Err(error) => Err(storage(error)),
    }
}

fn load_trial_counts(
    connection: &Connection,
    session_id: &str,
) -> Result<TuningTrialCountsRow, TuningSessionRepositoryError> {
    connection.query_row(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status IN ('complete', 'failed', 'pruned', 'cancelled') THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'pruned' THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END), 0) FROM tuning_trials WHERE session_id = ?1",
        params![session_id],
        |row| Ok(TuningTrialCountsRow { total: row.get(0)?, queued: row.get(1)?, running: row.get(2)?, terminal: row.get(3)?, completed: row.get(4)?, failed: row.get(5)?, pruned: row.get(6)?, cancelled: row.get(7)? }),
    ).map_err(storage)
}

fn load_attempts(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<TuningSessionAttemptRow>, TuningSessionRepositoryError> {
    let mut query = connection.prepare("SELECT session_id, attempt_id, bench_run_id, status, CAST(started_at AS TEXT), CAST(ended_at AS TEXT), failure FROM tuning_attempts WHERE session_id = ?1 ORDER BY started_at").map_err(storage)?;
    query
        .query_map(params![session_id], attempt_from_row)
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn attempt_from_row(row: &duckdb::Row<'_>) -> Result<TuningSessionAttemptRow, duckdb::Error> {
    Ok(TuningSessionAttemptRow {
        session_id: row.get(0)?,
        attempt_id: row.get(1)?,
        bench_run_id: row.get(2)?,
        status: row.get(3)?,
        started_at: row.get(4)?,
        ended_at: row.get(5)?,
        failure: row.get(6)?,
    })
}

fn load_trials(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<TuningSessionTrialRow>, TuningSessionRepositoryError> {
    let mut query = connection.prepare("SELECT trial_id, trial_number, attempt_id, status, CAST(config AS TEXT), score, mu, sigma, stop_reason, failure FROM tuning_trials WHERE session_id = ?1 ORDER BY trial_number").map_err(storage)?;
    query
        .query_map(params![session_id], |row| {
            Ok(TuningSessionTrialRow {
                trial_id: row.get(0)?,
                trial_number: row.get(1)?,
                attempt_id: row.get(2)?,
                status: row.get(3)?,
                config: row.get(4)?,
                score: row.get(5)?,
                mu: row.get(6)?,
                sigma: row.get(7)?,
                stop_reason: row.get(8)?,
                failure: row.get(9)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn load_trial_reports(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<TuningSessionTrialReportRow>, TuningSessionRepositoryError> {
    let mut query = connection.prepare("SELECT trial_id, completed_pairs, CAST(reported_at AS TEXT), mu, sigma, score, score_formula_version, conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource FROM tuning_trial_reports WHERE session_id = ?1 ORDER BY trial_number, completed_pairs, event_id").map_err(storage)?;
    query
        .query_map(params![session_id], |row| {
            Ok(TuningSessionTrialReportRow {
                trial_id: row.get(0)?,
                completed_pairs: row.get(1)?,
                reported_at: row.get(2)?,
                mu: row.get(3)?,
                sigma: row.get(4)?,
                score: row.get(5)?,
                score_formula_version: row.get(6)?,
                conservative_k: row.get(7)?,
                outcome: row.get(8)?,
                reason: row.get(9)?,
                pruning_exempt: row.get(10)?,
                bracket_id: row.get(11)?,
                rung_resource: row.get(12)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn load_pairs(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<TuningSessionPairRow>, TuningSessionRepositoryError> {
    let mut query = connection.prepare("SELECT trial_id, pair_id, pair_index, status, seed, round, CAST(opponent AS TEXT), pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, rating_after_mu, rating_after_sigma, score, failure FROM tuning_evaluation_pairs WHERE session_id = ?1 ORDER BY trial_id, pair_index").map_err(storage)?;
    query
        .query_map(params![session_id], |row| {
            Ok(TuningSessionPairRow {
                trial_id: row.get(0)?,
                pair_id: row.get(1)?,
                pair_index: row.get(2)?,
                status: row.get(3)?,
                seed: row.get(4)?,
                round: row.get(5)?,
                opponent: row.get(6)?,
                pool_snapshot_fingerprint: row.get(7)?,
                rating_before_mu: row.get(8)?,
                rating_before_sigma: row.get(9)?,
                rating_after_mu: row.get(10)?,
                rating_after_sigma: row.get(11)?,
                score: row.get(12)?,
                failure: row.get(13)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn load_games(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<TuningSessionGameRow>, TuningSessionRepositoryError> {
    let mut query = connection.prepare("SELECT pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, CAST(candidate_metrics AS TEXT), CAST(baseline_metrics AS TEXT) FROM tuning_games WHERE session_id = ?1 ORDER BY pair_id, candidate_side").map_err(storage)?;
    query
        .query_map(params![session_id], |row| {
            Ok(TuningSessionGameRow {
                pair_id: row.get(0)?,
                game_id: row.get(1)?,
                candidate_side: row.get(2)?,
                outcome: row.get(3)?,
                seed: row.get(4)?,
                round: row.get(5)?,
                trace_game_seq: row.get(6)?,
                plies: row.get(7)?,
                elapsed_ms: row.get(8)?,
                candidate_metrics: row.get(9)?,
                baseline_metrics: row.get(10)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn load_capabilities(
    connection: &Connection,
    session_id: &str,
) -> Result<TuningSessionCapabilities, TuningSessionRepositoryError> {
    connection.query_row(
        "WITH joined_games AS ( SELECT pairs.session_id, pairs.attempt_id, games.trace_game_seq FROM tuning_evaluation_pairs pairs LEFT JOIN tuning_games games USING (session_id, pair_id) WHERE pairs.session_id = ?1 ), renderer_moves AS ( SELECT DISTINCT run_id, game_seq FROM game_moves WHERE trace_schema_version = 1 ), report_moves AS ( SELECT DISTINCT run_id, game_seq FROM game_moves WHERE search_report IS NOT NULL ) SELECT COUNT(*), COUNT(DISTINCT renderer_moves.run_id || ':' || renderer_moves.game_seq), COUNT(DISTINCT report_moves.run_id || ':' || report_moves.game_seq), (SELECT COUNT(*) FROM tuning_trial_reports WHERE session_id = ?1) FROM joined_games LEFT JOIN tuning_attempts attempts ON attempts.attempt_id = joined_games.attempt_id LEFT JOIN renderer_moves ON renderer_moves.run_id = attempts.bench_run_id AND renderer_moves.game_seq = joined_games.trace_game_seq LEFT JOIN report_moves ON report_moves.run_id = attempts.bench_run_id AND report_moves.game_seq = joined_games.trace_game_seq",
        params![session_id],
        |row| Ok(TuningSessionCapabilities { pair_count: row.get(0)?, renderer_trace_count: row.get(1)?, search_report_count: row.get(2)?, trial_report_count: row.get(3)? }),
    ).map_err(storage)
}

fn storage(error: duckdb::Error) -> TuningSessionRepositoryError {
    TuningSessionRepositoryError::Storage(error.to_string())
}

fn invalid_data(error: impl std::fmt::Display) -> TuningSessionRepositoryError {
    TuningSessionRepositoryError::InvalidData(error.to_string())
}

fn validate_list_data(data: &TuningSessionListData) -> Result<(), TuningSessionRepositoryError> {
    for session in &data.sessions {
        serde_json::from_str::<serde_json::Value>(&session.manifest).map_err(invalid_data)?;
    }
    Ok(())
}

fn validate_detail_data(
    data: &TuningSessionDetailData,
) -> Result<(), TuningSessionRepositoryError> {
    serde_json::from_str::<serde_json::Value>(&data.session.manifest).map_err(invalid_data)?;
    for trial in &data.trials {
        trial
            .config
            .as_deref()
            .map(serde_json::from_str::<serde_json::Value>)
            .transpose()
            .map_err(invalid_data)?;
        trial
            .stop_reason
            .as_deref()
            .map(validate_reason)
            .transpose()?;
    }
    for report in &data.reports {
        validate_outcome(&report.outcome)?;
        validate_reason(&report.reason)?;
    }
    for pair in &data.pairs {
        serde_json::from_str::<crate::tuning_lifecycle::OpponentSnapshot>(&pair.opponent)
            .map_err(invalid_data)?;
    }
    for game in &data.games {
        serde_json::from_str::<crate::tuning_lifecycle::StrategyMetrics>(&game.candidate_metrics)
            .map_err(invalid_data)?;
        serde_json::from_str::<crate::tuning_lifecycle::StrategyMetrics>(&game.baseline_metrics)
            .map_err(invalid_data)?;
    }
    Ok(())
}

fn validate_outcome(value: &str) -> Result<(), TuningSessionRepositoryError> {
    serde_json::from_value::<crate::tuning_lifecycle::TrialReportOutcome>(
        serde_json::Value::String(value.into()),
    )
    .map(|_| ())
    .map_err(invalid_data)
}

fn validate_reason(value: &str) -> Result<(), TuningSessionRepositoryError> {
    serde_json::from_value::<crate::tuning_lifecycle::TrialReportReason>(serde_json::Value::String(
        value.into(),
    ))
    .map(|_| ())
    .map_err(invalid_data)
}
