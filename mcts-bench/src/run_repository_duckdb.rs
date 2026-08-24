//! DuckDB implementation of [`crate::run_repository::RunRepository`].

use std::sync::{Arc, Mutex};

use duckdb::{params, Connection};

use crate::run_repository::{RunRepository, RunRepositoryError};

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
