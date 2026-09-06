//! DuckDB implementation of [`crate::run_repository::RunRepository`].

use std::sync::{Arc, Mutex};

use duckdb::{params, Connection};

use crate::run_repository::{
    ExperimentCell, LeaderboardQuery, LeaderboardRow, RunDeletionInfo, RunDetail, RunGame,
    RunGameMove, RunGamesQuery, RunIncumbent, RunListQuery, RunRepository, RunRepositoryError,
    RunSummary, RunTrial, RunTrialsQuery,
};

/// A run repository backed by a DuckDB connection.
pub struct DuckDbRunRepository<'connection> {
    connection: &'connection Connection,
}

impl<'connection> DuckDbRunRepository<'connection> {
    pub fn new(connection: &'connection Connection) -> Self {
        Self { connection }
    }
}

impl RunRepository for DuckDbRunRepository<'_> {
    fn load_log_path(&self, run_id: &str) -> Result<String, RunRepositoryError> {
        load_log_path(self.connection, run_id)
    }

    fn list_runs(&self, query: &RunListQuery) -> Result<Vec<RunSummary>, RunRepositoryError> {
        list_runs(self.connection, query)
    }

    fn load_run(&self, run_id: &str) -> Result<RunDetail, RunRepositoryError> {
        load_run(self.connection, run_id)
    }

    fn load_leaderboard(
        &self,
        query: &LeaderboardQuery,
    ) -> Result<Vec<LeaderboardRow>, RunRepositoryError> {
        load_leaderboard(self.connection, query)
    }

    fn load_experiment_cells(
        &self,
        run_id: &str,
    ) -> Result<Vec<ExperimentCell>, RunRepositoryError> {
        load_experiment_cells(self.connection, run_id)
    }

    fn ensure_run_exists(&self, run_id: &str) -> Result<(), RunRepositoryError> {
        ensure_run_exists(self.connection, run_id)
    }

    fn load_trials(
        &self,
        run_id: &str,
        query: &RunTrialsQuery,
    ) -> Result<Vec<RunTrial>, RunRepositoryError> {
        load_trials(self.connection, run_id, query)
    }

    fn load_games(
        &self,
        run_id: &str,
        query: &RunGamesQuery,
    ) -> Result<Vec<RunGame>, RunRepositoryError> {
        load_games(self.connection, run_id, query)
    }

    fn load_game_moves(
        &self,
        run_id: &str,
        game_seq: i64,
        after_ply: Option<i64>,
    ) -> Result<Vec<RunGameMove>, RunRepositoryError> {
        load_game_moves(self.connection, run_id, game_seq, after_ply)
    }

    fn load_latest_game_seq(&self, run_id: &str) -> Result<Option<i64>, RunRepositoryError> {
        load_latest_game_seq(self.connection, run_id)
    }

    fn load_deletion_info(&self, run_id: &str) -> Result<RunDeletionInfo, RunRepositoryError> {
        load_deletion_info(self.connection, run_id)
    }

    fn delete_run_records(
        &self,
        run_id: &str,
        ingest_log_paths: &[String],
    ) -> Result<(), RunRepositoryError> {
        delete_run_records(self.connection, run_id, ingest_log_paths)
    }
}

/// A run repository backed by a shared DuckDB connection.
#[derive(Clone)]
pub struct SharedDuckDbRunRepository {
    connection: Arc<Mutex<Connection>>,
}

impl SharedDuckDbRunRepository {
    pub fn new(connection: Arc<Mutex<Connection>>) -> Self {
        Self { connection }
    }
}

impl RunRepository for SharedDuckDbRunRepository {
    fn load_log_path(&self, run_id: &str) -> Result<String, RunRepositoryError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| RunRepositoryError::Storage("benchmark database mutex poisoned".into()))?;
        load_log_path(&connection, run_id)
    }

    fn list_runs(&self, query: &RunListQuery) -> Result<Vec<RunSummary>, RunRepositoryError> {
        let connection = self.lock()?;
        list_runs(&connection, query)
    }

    fn load_run(&self, run_id: &str) -> Result<RunDetail, RunRepositoryError> {
        let connection = self.lock()?;
        load_run(&connection, run_id)
    }

    fn load_leaderboard(
        &self,
        query: &LeaderboardQuery,
    ) -> Result<Vec<LeaderboardRow>, RunRepositoryError> {
        let connection = self.lock()?;
        load_leaderboard(&connection, query)
    }

    fn load_experiment_cells(
        &self,
        run_id: &str,
    ) -> Result<Vec<ExperimentCell>, RunRepositoryError> {
        let connection = self.lock()?;
        load_experiment_cells(&connection, run_id)
    }

    fn ensure_run_exists(&self, run_id: &str) -> Result<(), RunRepositoryError> {
        let connection = self.lock()?;
        ensure_run_exists(&connection, run_id)
    }

    fn load_trials(
        &self,
        run_id: &str,
        query: &RunTrialsQuery,
    ) -> Result<Vec<RunTrial>, RunRepositoryError> {
        let connection = self.lock()?;
        load_trials(&connection, run_id, query)
    }

    fn load_games(
        &self,
        run_id: &str,
        query: &RunGamesQuery,
    ) -> Result<Vec<RunGame>, RunRepositoryError> {
        let connection = self.lock()?;
        load_games(&connection, run_id, query)
    }

    fn load_game_moves(
        &self,
        run_id: &str,
        game_seq: i64,
        after_ply: Option<i64>,
    ) -> Result<Vec<RunGameMove>, RunRepositoryError> {
        let connection = self.lock()?;
        load_game_moves(&connection, run_id, game_seq, after_ply)
    }

    fn load_latest_game_seq(&self, run_id: &str) -> Result<Option<i64>, RunRepositoryError> {
        let connection = self.lock()?;
        load_latest_game_seq(&connection, run_id)
    }

    fn load_deletion_info(&self, run_id: &str) -> Result<RunDeletionInfo, RunRepositoryError> {
        let connection = self.lock()?;
        load_deletion_info(&connection, run_id)
    }

    fn delete_run_records(
        &self,
        run_id: &str,
        ingest_log_paths: &[String],
    ) -> Result<(), RunRepositoryError> {
        let connection = self.lock()?;
        delete_run_records(&connection, run_id, ingest_log_paths)
    }
}

impl SharedDuckDbRunRepository {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, RunRepositoryError> {
        self.connection
            .lock()
            .map_err(|_| RunRepositoryError::Storage("benchmark database mutex poisoned".into()))
    }
}

fn load_log_path(connection: &Connection, run_id: &str) -> Result<String, RunRepositoryError> {
    connection
        .query_row(
            "SELECT log_path FROM runs WHERE run_id = ?1",
            params![run_id],
            |row| row.get(0),
        )
        .map_err(|error| match error {
            duckdb::Error::QueryReturnedNoRows => RunRepositoryError::NotFound,
            other => RunRepositoryError::Storage(other.to_string()),
        })
}

fn list_runs(
    connection: &Connection,
    query: &RunListQuery,
) -> Result<Vec<RunSummary>, RunRepositoryError> {
    let mut sql = String::from(
        "SELECT r.run_id, r.kind, r.game, r.label, r.git_sha, r.git_dirty, \
                r.host, r.pid, \
                CAST(r.started_at AS TEXT), \
                CAST(r.ended_at AS TEXT), \
                r.status, r.project_id, r.experiment_id, \
                COALESCE(m.match_count, 0), COALESCE(t.trial_count, 0) \
         FROM runs r \
         LEFT JOIN (SELECT run_id, COUNT(*) AS match_count FROM match_results GROUP BY run_id) m \
           ON r.run_id = m.run_id \
         LEFT JOIN (SELECT run_id, COUNT(*) AS trial_count FROM trials GROUP BY run_id) t \
           ON r.run_id = t.run_id \
         WHERE 1=1",
    );
    if let Some(game) = &query.game {
        sql.push_str(&format!(" AND r.game = '{}'", game.replace('\'', "''")));
    }
    if let Some(experiment_id) = &query.experiment_id {
        sql.push_str(&format!(
            " AND r.experiment_id = '{}'",
            experiment_id.replace('\'', "''")
        ));
    }
    if let Some(project_id) = &query.project_id {
        sql.push_str(&format!(
            " AND r.project_id = '{}'",
            project_id.replace('\'', "''")
        ));
    }
    sql.push_str(" ORDER BY CAST(r.started_at AS TEXT) DESC");

    let mut statement = connection.prepare(&sql).map_err(storage)?;
    statement
        .query_map([], |row| {
            Ok(RunSummary {
                run_id: row.get(0)?,
                kind: row.get(1)?,
                game: row.get(2)?,
                project_id: row.get(11)?,
                experiment_id: row.get(12)?,
                label: row.get(3)?,
                git_sha: row.get(4)?,
                git_dirty: row.get(5)?,
                host: row.get(6)?,
                pid: row.get(7)?,
                started_at: row.get(8)?,
                ended_at: row.get(9)?,
                status: row.get(10)?,
                match_count: row.get(13)?,
                trial_count: row.get(14)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage)
}

fn load_run(connection: &Connection, run_id: &str) -> Result<RunDetail, RunRepositoryError> {
    connection
        .query_row(
            "SELECT r.run_id, r.kind, r.game, r.label, \
                    CAST(r.config AS TEXT), r.project_id, r.experiment_id, CAST(r.experiment_spec AS TEXT), \
                    r.git_sha, r.git_dirty, r.host, r.pid, \
                    CAST(r.started_at AS TEXT), CAST(r.ended_at AS TEXT), \
                    r.status, r.log_path, r.exit_code, \
                    COALESCE(m.match_count, 0), COALESCE(t.trial_count, 0), \
                    CAST(i.config AS TEXT), i.cost \
             FROM runs r \
             LEFT JOIN (SELECT run_id, COUNT(*) AS match_count FROM match_results GROUP BY run_id) m ON r.run_id = m.run_id \
             LEFT JOIN (SELECT run_id, COUNT(*) AS trial_count FROM trials GROUP BY run_id) t ON r.run_id = t.run_id \
             LEFT JOIN incumbents i ON r.run_id = i.run_id \
             WHERE r.run_id = ?1",
            params![run_id],
            |row| {
                let incumbent_config: Option<String> = row.get::<_, Option<String>>(19)?;
                let incumbent_cost: Option<f64> = row.get(20)?;
                Ok(RunDetail {
                    run_id: row.get(0)?, kind: row.get(1)?, game: row.get(2)?,
                    project_id: row.get(5)?, experiment_id: row.get(6)?,
                    experiment_spec: json_column(row.get::<_, Option<String>>(7)?),
                    label: row.get(3)?, config: json_column(row.get::<_, Option<String>>(4)?),
                    git_sha: row.get(8)?, git_dirty: row.get(9)?, host: row.get(10)?, pid: row.get(11)?,
                    started_at: row.get(12)?, ended_at: row.get(13)?, status: row.get(14)?,
                    log_path: row.get(15)?, exit_code: row.get(16)?, match_count: row.get(17)?,
                    trial_count: row.get(18)?,
                    incumbent: incumbent_config.zip(incumbent_cost).map(|(config, cost)| RunIncumbent {
                        config: serde_json::from_str(&config).unwrap_or(serde_json::Value::Null), cost,
                    }),
                })
            },
        )
        .map_err(not_found_or_storage)
}

fn load_leaderboard(
    connection: &Connection,
    query: &LeaderboardQuery,
) -> Result<Vec<LeaderboardRow>, RunRepositoryError> {
    let mut conditions = String::from("r.status IN ('completed', 'crashed', 'stopped')");
    if let Some(game) = &query.game {
        conditions.push_str(&format!(" AND r.game = '{}'", game.replace('\'', "''")));
    }
    if let Some(git_sha) = &query.git_sha {
        conditions.push_str(&format!(
            " AND r.git_sha = '{}'",
            git_sha.replace('\'', "''")
        ));
    }
    if let Some(since) = &query.since {
        conditions.push_str(&format!(
            " AND r.started_at >= '{}'",
            since.replace('\'', "''")
        ));
    }
    let sql = format!(
        "WITH a_stats AS (
            SELECT mr.strategy_a AS strategy, COUNT(*) AS total,
                   SUM(CASE WHEN mr.outcome = 'win_a' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN mr.outcome = 'win_b' THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN mr.outcome = 'draw' THEN 1 ELSE 0 END) AS draws
            FROM match_results mr JOIN runs r ON mr.run_id = r.run_id
            WHERE {conditions} GROUP BY mr.strategy_a
        ), b_stats AS (
            SELECT mr.strategy_b AS strategy, COUNT(*) AS total,
                   SUM(CASE WHEN mr.outcome = 'win_b' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN mr.outcome = 'win_a' THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN mr.outcome = 'draw' THEN 1 ELSE 0 END) AS draws
            FROM match_results mr JOIN runs r ON mr.run_id = r.run_id
            WHERE {conditions} GROUP BY mr.strategy_b
        )
        SELECT COALESCE(a.strategy, b.strategy) AS strategy,
               COALESCE(a.total, 0) + COALESCE(b.total, 0) AS total,
               COALESCE(a.wins, 0) + COALESCE(b.wins, 0) AS wins,
               COALESCE(a.losses, 0) + COALESCE(b.losses, 0) AS losses,
               COALESCE(a.draws, 0) + COALESCE(b.draws, 0) AS draws
        FROM a_stats a FULL OUTER JOIN b_stats b ON a.strategy = b.strategy
        ORDER BY wins DESC, losses ASC"
    );
    let mut statement = connection.prepare(&sql).map_err(storage)?;
    statement
        .query_map([], |row| {
            Ok(LeaderboardRow {
                strategy: row.get(0)?,
                total: row.get(1)?,
                wins: row.get(2)?,
                losses: row.get(3)?,
                draws: row.get(4)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage)
}

fn load_experiment_cells(
    connection: &Connection,
    run_id: &str,
) -> Result<Vec<ExperimentCell>, RunRepositoryError> {
    let exists: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?1",
            params![run_id],
            |row| row.get(0),
        )
        .map_err(storage)?;
    if exists == 0 {
        return Err(RunRepositoryError::NotFound);
    }
    let mut cells = connection.prepare(
        "SELECT cell_id, cell_seed, game, CAST(game_config AS TEXT), variant_id, variant_label, \
         CAST(candidate_config AS TEXT), baseline_id, baseline_label, CAST(baseline_config AS TEXT), \
         CAST(budget AS TEXT), rounds, planned_games, completed_games, status, CAST(started_at AS TEXT), \
         CAST(ended_at AS TEXT), error FROM experiment_cells WHERE run_id = ?1 ORDER BY cell_id",
    ).map_err(storage)?;
    let rows = cells
        .query_map(params![run_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<u64>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
                row.get::<_, i64>(11)?,
                row.get::<_, u64>(12)?,
                row.get::<_, u64>(13)?,
                row.get::<_, String>(14)?,
                row.get::<_, Option<String>>(15)?,
                row.get::<_, Option<String>>(16)?,
                row.get::<_, Option<String>>(17)?,
            ))
        })
        .map_err(storage)?;
    let mut result = Vec::new();
    for row in rows {
        let (
            cell_id,
            cell_seed,
            game,
            game_config,
            variant_id,
            variant_label,
            candidate_config,
            baseline_id,
            baseline_label,
            baseline_config,
            budget,
            rounds,
            planned_games,
            completed_games,
            status,
            started_at,
            ended_at,
            error,
        ) = row.map_err(storage)?;
        let mut matches = connection.prepare(
            "SELECT CAST(metrics AS TEXT) FROM match_results WHERE run_id = ?1 AND cell_id = ?2 ORDER BY seq",
        ).map_err(storage)?;
        let match_outcomes = matches
            .query_map(params![run_id, &cell_id], |row| {
                let metrics: Option<String> = row.get(0)?;
                Ok(metrics
                    .and_then(|metrics| serde_json::from_str::<serde_json::Value>(&metrics).ok())
                    .and_then(|metrics| {
                        metrics
                            .get("outcome")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_owned)
                    }))
            })
            .map_err(storage)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(storage)?;
        result.push(ExperimentCell {
            cell_id,
            cell_seed,
            game,
            game_config: json_value(game_config),
            variant_id,
            variant_label,
            candidate_config: json_value(candidate_config),
            baseline_id,
            baseline_label,
            baseline_config: json_value(baseline_config),
            budget: json_value(budget),
            rounds,
            planned_games,
            completed_games,
            status,
            started_at,
            ended_at,
            error,
            match_outcomes,
        });
    }
    Ok(result)
}

fn ensure_run_exists(connection: &Connection, run_id: &str) -> Result<(), RunRepositoryError> {
    connection
        .query_row(
            "SELECT 1 FROM runs WHERE run_id = ?1",
            params![run_id],
            |_| Ok(()),
        )
        .map_err(not_found_or_storage)
}

fn load_trials(
    connection: &Connection,
    run_id: &str,
    query: &RunTrialsQuery,
) -> Result<Vec<RunTrial>, RunRepositoryError> {
    let mut sql = String::from(
        "SELECT trial_id, CAST(ts AS TEXT), CAST(config AS TEXT), seed, cost, CAST(extra AS TEXT) \
         FROM trials WHERE run_id = ?1 ORDER BY trial_id ASC",
    );
    if let Some(limit) = query.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }
    let mut statement = connection.prepare(&sql).map_err(storage)?;
    statement
        .query_map(params![run_id], |row| {
            let config: String = row.get(2)?;
            let extra: Option<String> = row.get(5)?;
            Ok(RunTrial {
                trial_id: row.get(0)?,
                ts: row.get(1)?,
                config: json_value(config),
                seed: row.get(3)?,
                cost: row.get(4)?,
                extra: json_column(extra),
            })
        })
        .map_err(storage)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage)
}

fn load_games(
    connection: &Connection,
    run_id: &str,
    query: &RunGamesQuery,
) -> Result<Vec<RunGame>, RunRepositoryError> {
    let mut sql = String::from(
        "SELECT g.game_seq, m.seq, m.cell_id, m.seed, CAST(m.metrics AS TEXT), COUNT(*), CAST(MIN(g.ts) AS TEXT), CAST(MAX(g.ts) AS TEXT), \
                m.strategy_a, m.strategy_b, m.outcome, m.winner \
         FROM game_moves g \
         LEFT JOIN match_results m ON m.run_id = g.run_id AND (m.trace_game_seq = g.game_seq OR (m.trace_game_seq IS NULL AND m.seq = g.game_seq)) \
         WHERE g.run_id = ?1 AND (?2 IS NULL OR m.cell_id = ?2) \
         GROUP BY g.game_seq, m.seq, m.cell_id, m.seed, m.metrics, m.strategy_a, m.strategy_b, m.outcome, m.winner \
         ORDER BY g.game_seq DESC",
    );
    if let Some(limit) = query.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }
    let mut statement = connection.prepare(&sql).map_err(storage)?;
    statement
        .query_map(params![run_id, query.cell_id.as_deref()], |row| {
            Ok(RunGame {
                game_seq: row.get(0)?,
                match_seq: row.get(1)?,
                cell_id: row.get(2)?,
                seed: row.get(3)?,
                metrics: json_column(row.get(4)?),
                ply_count: row.get(5)?,
                started_at: row.get(6)?,
                ended_at: row.get(7)?,
                strategy_a: row.get(8)?,
                strategy_b: row.get(9)?,
                outcome: row.get(10)?,
                winner: row.get(11)?,
            })
        })
        .map_err(storage)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage)
}

fn load_game_moves(
    connection: &Connection,
    run_id: &str,
    game_seq: i64,
    after_ply: Option<i64>,
) -> Result<Vec<RunGameMove>, RunRepositoryError> {
    let after_ply = after_ply.unwrap_or(-1);
    let mut statement = connection
        .prepare(
            "SELECT ply, CAST(ts AS TEXT), CAST(state AS TEXT), CAST(mv AS TEXT), player, CAST(search_report AS TEXT) \
             FROM game_moves WHERE run_id = ?1 AND game_seq = ?2 AND ply > ?3 ORDER BY ply ASC",
        )
        .map_err(storage)?;
    statement
        .query_map(params![run_id, game_seq, after_ply], |row| {
            let state: String = row.get(2)?;
            Ok(RunGameMove {
                game_seq,
                ply: row.get(0)?,
                ts: row.get(1)?,
                state: json_value(state),
                mv: json_column(row.get(3)?),
                player: row.get(4)?,
                search: json_column(row.get(5)?),
            })
        })
        .map_err(storage)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage)
}

fn load_latest_game_seq(
    connection: &Connection,
    run_id: &str,
) -> Result<Option<i64>, RunRepositoryError> {
    connection
        .query_row(
            "SELECT MAX(game_seq) FROM game_moves WHERE run_id = ?1",
            params![run_id],
            |row| row.get(0),
        )
        .map_err(storage)
}

fn load_deletion_info(
    connection: &Connection,
    run_id: &str,
) -> Result<RunDeletionInfo, RunRepositoryError> {
    connection
        .query_row(
            "SELECT run.status FROM runs run WHERE run.run_id = ?1",
            params![run_id],
            |row| {
                Ok(RunDeletionInfo {
                    status: row.get(0)?,
                })
            },
        )
        .map_err(not_found_or_storage)
}

fn delete_run_records(
    connection: &Connection,
    run_id: &str,
    ingest_log_paths: &[String],
) -> Result<(), RunRepositoryError> {
    for table in [
        "game_moves",
        "incumbents",
        "trials",
        "match_results",
        "experiment_cells",
    ] {
        connection
            .execute(
                &format!("DELETE FROM {table} WHERE run_id = ?1"),
                params![run_id],
            )
            .map_err(storage)?;
    }
    for table in [
        "_artifact_trace_cursor",
        "artifact_tasks",
        "artifact_descriptors",
        "artifact_roots",
    ] {
        connection
            .execute(
                &format!("DELETE FROM {table} WHERE physical_run_id = ?1"),
                params![run_id],
            )
            .map_err(storage)?;
    }
    for log_path in ingest_log_paths {
        connection
            .execute(
                "DELETE FROM _ingest_cursor WHERE log_path = ?1",
                params![log_path],
            )
            .map_err(storage)?;
    }
    connection
        .execute("DELETE FROM runs WHERE run_id = ?1", params![run_id])
        .map_err(storage)?;
    Ok(())
}

fn json_column(value: Option<String>) -> Option<serde_json::Value> {
    value.and_then(|value| serde_json::from_str(&value).ok())
}

fn json_value(value: String) -> serde_json::Value {
    serde_json::from_str(&value).unwrap_or(serde_json::Value::Null)
}

fn storage(error: duckdb::Error) -> RunRepositoryError {
    RunRepositoryError::Storage(error.to_string())
}

fn not_found_or_storage(error: duckdb::Error) -> RunRepositoryError {
    match error {
        duckdb::Error::QueryReturnedNoRows => RunRepositoryError::NotFound,
        error => storage(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_log_path_and_hides_duckdb_errors() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute("CREATE TABLE runs (run_id TEXT, log_path TEXT)", [])
            .unwrap();
        connection
            .execute("INSERT INTO runs VALUES ('known', '/tmp/known.jsonl')", [])
            .unwrap();
        let repository = DuckDbRunRepository::new(&connection);

        assert_eq!(
            repository.load_log_path("known"),
            Ok("/tmp/known.jsonl".into())
        );
        assert_eq!(
            repository.load_log_path("missing"),
            Err(RunRepositoryError::NotFound)
        );
    }
}
