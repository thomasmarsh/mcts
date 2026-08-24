//! DuckDB implementation of [`crate::tuning_trial_repository::TuningTrialRepository`].

use std::sync::{Arc, Mutex};

use duckdb::{params, Connection};

use crate::tuning_trial_repository::{
    TuningTrialDetailData, TuningTrialDetailRow, TuningTrialGameRow, TuningTrialPageData,
    TuningTrialPageRow, TuningTrialPairRow, TuningTrialReportRow, TuningTrialRepository,
    TuningTrialRepositoryError,
};

/// A tuning-trial repository backed by a shared DuckDB connection.
#[derive(Clone)]
pub struct SharedDuckDbTuningTrialRepository {
    connection: Arc<Mutex<Connection>>,
}

impl SharedDuckDbTuningTrialRepository {
    pub fn new(connection: Arc<Mutex<Connection>>) -> Self {
        Self { connection }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, TuningTrialRepositoryError> {
        self.connection.lock().map_err(|_| {
            TuningTrialRepositoryError::Storage("benchmark database mutex poisoned".into())
        })
    }
}

impl TuningTrialRepository for SharedDuckDbTuningTrialRepository {
    fn load_trial_page(
        &self,
        session_id: &str,
    ) -> Result<Option<TuningTrialPageData>, TuningTrialRepositoryError> {
        let connection = self.lock()?;
        let Some(session_sequence) = load_session_sequence(&connection, session_id)? else {
            return Ok(None);
        };
        Ok(Some(TuningTrialPageData {
            session_sequence,
            trials: load_trial_page_rows(&connection, session_id)?,
        }))
    }

    fn load_trial_detail(
        &self,
        session_id: &str,
        trial_id: &str,
    ) -> Result<Option<TuningTrialDetailData>, TuningTrialRepositoryError> {
        let connection = self.lock()?;
        let Some(session_sequence) = load_session_sequence(&connection, session_id)? else {
            return Ok(None);
        };
        let Some(trial) = load_trial_detail_row(&connection, session_id, trial_id)? else {
            return Ok(None);
        };
        Ok(Some(TuningTrialDetailData {
            session_sequence,
            reports: load_trial_reports(&connection, session_id, trial_id)?,
            pairs: load_trial_pairs(&connection, session_id, trial_id)?,
            trial,
        }))
    }
}

fn load_session_sequence(
    connection: &Connection,
    session_id: &str,
) -> Result<Option<i64>, TuningTrialRepositoryError> {
    match connection.query_row(
        "SELECT last_sequence FROM tuning_sessions WHERE session_id = ?1",
        params![session_id],
        |row| row.get(0),
    ) {
        Ok(sequence) => Ok(Some(sequence)),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(storage(error)),
    }
}

fn load_trial_page_rows(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<TuningTrialPageRow>, TuningTrialRepositoryError> {
    let mut query = connection.prepare(
        "WITH ranked_reports AS ( \
             SELECT session_id, trial_id, completed_pairs, mu, sigma, score, reason, bracket_id, event_id, \
                    ROW_NUMBER() OVER (PARTITION BY session_id, trial_id ORDER BY completed_pairs DESC, event_id DESC) AS rank \
             FROM tuning_trial_reports \
         ), last_reports AS ( \
             SELECT session_id, trial_id, completed_pairs, mu, sigma, score, reason, bracket_id \
             FROM ranked_reports WHERE rank = 1 \
         ), game_stats AS ( \
             SELECT pairs.session_id, pairs.trial_id, COUNT(DISTINCT pairs.pair_id) AS pair_count, \
                    COALESCE(SUM(CASE WHEN games.outcome = 'candidate_win' THEN 1 ELSE 0 END), 0) AS wins, \
                    COALESCE(SUM(CASE WHEN games.outcome = 'baseline_win' THEN 1 ELSE 0 END), 0) AS losses, \
                    COALESCE(SUM(CASE WHEN games.outcome = 'draw' THEN 1 ELSE 0 END), 0) AS draws, \
                    COALESCE(SUM(games.elapsed_ms), 0) AS elapsed_ms, \
                    COALESCE(SUM( \
                        COALESCE(TRY_CAST(json_extract_string(games.candidate_metrics, '$.iterations_total') AS UBIGINT), 0) + \
                        COALESCE(TRY_CAST(json_extract_string(games.baseline_metrics, '$.iterations_total') AS UBIGINT), 0) \
                    ), 0) AS search_iterations_total, \
                    COALESCE(SUM( \
                        COALESCE(TRY_CAST(json_extract_string(games.candidate_metrics, '$.move_time_ms') AS UBIGINT), 0) + \
                        COALESCE(TRY_CAST(json_extract_string(games.baseline_metrics, '$.move_time_ms') AS UBIGINT), 0) \
                    ), 0) AS search_move_time_ms \
             FROM tuning_evaluation_pairs pairs \
             LEFT JOIN tuning_games games USING (session_id, pair_id) \
             WHERE pairs.session_id = ?1 \
             GROUP BY pairs.session_id, pairs.trial_id \
         ) \
         SELECT trials.trial_id, trials.trial_number, trials.attempt_id, trials.status, \
                CAST(trials.config AS TEXT), COALESCE(trials.score, reports.score), \
                COALESCE(trials.mu, reports.mu), COALESCE(trials.sigma, reports.sigma), trials.stop_reason, \
                reports.reason, reports.bracket_id, reports.completed_pairs, \
                COALESCE(stats.pair_count, 0), COALESCE(stats.wins, 0), COALESCE(stats.losses, 0), \
                COALESCE(stats.draws, 0), COALESCE(stats.elapsed_ms, 0), \
                COALESCE(stats.search_iterations_total, 0), COALESCE(stats.search_move_time_ms, 0) \
         FROM tuning_trials trials \
         LEFT JOIN last_reports reports USING (session_id, trial_id) \
         LEFT JOIN game_stats stats USING (session_id, trial_id) \
         WHERE trials.session_id = ?1",
    ).map_err(storage)?;
    query
        .query_map(params![session_id], |row| {
            Ok(TuningTrialPageRow {
                trial_id: row.get(0)?,
                trial_number: row.get(1)?,
                attempt_id: row.get(2)?,
                state: row.get(3)?,
                config: row.get(4)?,
                score: row.get(5)?,
                mu: row.get(6)?,
                sigma: row.get(7)?,
                stop_reason: row.get(8)?,
                last_reason: row.get(9)?,
                bracket_id: row.get(10)?,
                resource: row.get(11)?,
                pair_count: row.get(12)?,
                wins: row.get(13)?,
                losses: row.get(14)?,
                draws: row.get(15)?,
                elapsed_ms: row.get(16)?,
                search_iterations_total: row.get(17)?,
                search_move_time_ms: row.get(18)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn load_trial_detail_row(
    connection: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Option<TuningTrialDetailRow>, TuningTrialRepositoryError> {
    match connection.query_row(
        "SELECT trial_id, trial_number, attempt_id, status, CAST(config AS TEXT), score, mu, sigma, stop_reason, failure \
         FROM tuning_trials WHERE session_id = ?1 AND trial_id = ?2",
        params![session_id, trial_id],
        |row| Ok(TuningTrialDetailRow {
            trial_id: row.get(0)?, trial_number: row.get(1)?, attempt_id: row.get(2)?, state: row.get(3)?,
            config: row.get(4)?, score: row.get(5)?, mu: row.get(6)?, sigma: row.get(7)?,
            stop_reason: row.get(8)?, failure: row.get(9)?,
        }),
    ) {
        Ok(row) => Ok(Some(row)),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(storage(error)),
    }
}

fn load_trial_reports(
    connection: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Vec<TuningTrialReportRow>, TuningTrialRepositoryError> {
    let mut query = connection.prepare(
        "SELECT completed_pairs, CAST(reported_at AS TEXT), mu, sigma, score, score_formula_version, \
                conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource \
         FROM tuning_trial_reports WHERE session_id = ?1 AND trial_id = ?2 \
         ORDER BY completed_pairs ASC, event_id ASC",
    ).map_err(storage)?;
    query
        .query_map(params![session_id, trial_id], |row| {
            Ok(TuningTrialReportRow {
                completed_pairs: row.get(0)?,
                reported_at: row.get(1)?,
                mu: row.get(2)?,
                sigma: row.get(3)?,
                score: row.get(4)?,
                score_formula_version: row.get(5)?,
                conservative_k: row.get(6)?,
                outcome: row.get(7)?,
                reason: row.get(8)?,
                pruning_exempt: row.get(9)?,
                bracket_id: row.get(10)?,
                rung_resource: row.get(11)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn load_trial_pairs(
    connection: &Connection,
    session_id: &str,
    trial_id: &str,
) -> Result<Vec<TuningTrialPairRow>, TuningTrialRepositoryError> {
    let mut query = connection
        .prepare(
            "SELECT pair_id, pair_index, status, seed, round, CAST(opponent AS TEXT), \
                pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, rating_after_mu, \
                rating_after_sigma, score, failure, attempt_id \
         FROM tuning_evaluation_pairs WHERE session_id = ?1 AND trial_id = ?2 \
         ORDER BY pair_index ASC, pair_id ASC",
        )
        .map_err(storage)?;
    let pairs = query
        .query_map(params![session_id, trial_id], |row| {
            Ok((
                TuningTrialPairRow {
                    pair_id: row.get(0)?,
                    pair_index: row.get(1)?,
                    status: row.get(2)?,
                    seed: row.get(3)?,
                    round: row.get(4)?,
                    opponent: row.get(5)?,
                    pool_snapshot_fingerprint: row.get(6)?,
                    rating_before_mu: row.get(7)?,
                    rating_before_sigma: row.get(8)?,
                    rating_after_mu: row.get(9)?,
                    rating_after_sigma: row.get(10)?,
                    score: row.get(11)?,
                    failure: row.get(12)?,
                    games: Vec::new(),
                },
                row.get::<_, String>(13)?,
            ))
        })
        .map_err(storage)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage)?;
    pairs
        .into_iter()
        .map(|(mut pair, attempt_id)| {
            pair.games = load_trial_games(connection, session_id, &pair.pair_id, &attempt_id)?;
            Ok(pair)
        })
        .collect()
}

fn load_trial_games(
    connection: &Connection,
    session_id: &str,
    pair_id: &str,
    attempt_id: &str,
) -> Result<Vec<TuningTrialGameRow>, TuningTrialRepositoryError> {
    let mut query = connection.prepare(
        "SELECT games.game_id, games.candidate_side, games.outcome, games.seed, games.round, \
                games.trace_game_seq, games.plies, games.elapsed_ms, CAST(games.candidate_metrics AS TEXT), \
                CAST(games.baseline_metrics AS TEXT), attempts.bench_run_id, \
                EXISTS(SELECT 1 FROM game_moves moves WHERE moves.run_id = attempts.bench_run_id \
                       AND moves.game_seq = games.trace_game_seq AND moves.trace_schema_version = 1), \
                EXISTS(SELECT 1 FROM game_moves moves WHERE moves.run_id = attempts.bench_run_id \
                       AND moves.game_seq = games.trace_game_seq AND moves.search_report IS NOT NULL) \
         FROM tuning_games games \
         JOIN tuning_attempts attempts ON attempts.attempt_id = ?3 \
         WHERE games.session_id = ?1 AND games.pair_id = ?2 \
         ORDER BY games.candidate_side ASC, games.game_id ASC",
    ).map_err(storage)?;
    query
        .query_map(params![session_id, pair_id, attempt_id], |row| {
            Ok(TuningTrialGameRow {
                game_id: row.get(0)?,
                candidate_side: row.get(1)?,
                outcome: row.get(2)?,
                seed: row.get(3)?,
                round: row.get(4)?,
                trace_game_seq: row.get(5)?,
                plies: row.get(6)?,
                elapsed_ms: row.get(7)?,
                candidate_metrics: row.get(8)?,
                baseline_metrics: row.get(9)?,
                run_id: row.get(10)?,
                has_renderer_trace: row.get(11)?,
                has_search_reports: row.get(12)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<_, _>>()
        .map_err(storage)
}

fn storage(error: duckdb::Error) -> TuningTrialRepositoryError {
    TuningTrialRepositoryError::Storage(error.to_string())
}
