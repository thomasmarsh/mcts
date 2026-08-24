//! Schema-owned DuckDB composition for the benchmark server.
//!
//! This is the only production seam that turns a DuckDB connection into the
//! benchmark's repositories and ingest adapter.  Consumers receive traits,
//! never the connection itself.

use std::path::Path;
use std::sync::{Arc, Mutex};

use duckdb::Connection;

use crate::ingest::{self, IngestError};
use crate::project_repository::ProjectRepository;
use crate::projects_attempt::ProjectsRepository;
use crate::run_command_repository::RunCommandRepository;
use crate::run_repository::RunRepository;
use crate::tuning_analysis_repository::TuningAnalysisRepository;
use crate::tuning_command_repository::TuningCommandRepository;
use crate::tuning_session_repository::TuningSessionRepository;
use crate::tuning_trial_repository::TuningTrialRepository;

/// Runs one ingestion pass against the benchmark artifact directory.
pub trait BenchIngest: Send + Sync {
    fn ingest_once(&self, bench_runs_dir: &Path) -> Result<(), IngestError>;
}

#[derive(Clone)]
struct SharedDuckDbIngest {
    connection: Arc<Mutex<Connection>>,
}

impl BenchIngest for SharedDuckDbIngest {
    fn ingest_once(&self, bench_runs_dir: &Path) -> Result<(), IngestError> {
        let connection = self.connection.lock().map_err(|_| {
            IngestError::Io(std::io::Error::other("benchmark database mutex poisoned"))
        })?;
        ingest::ingest_once(&connection, bench_runs_dir)
    }
}

/// Logical benchmark adapters sharing one schema-initialized database.
pub struct BenchAdapters {
    pub project_repository: Arc<dyn ProjectRepository + Send + Sync>,
    pub projects_repository: Arc<dyn ProjectsRepository + Send + Sync>,
    pub run_repository: Arc<dyn RunRepository + Send + Sync>,
    pub run_command_repository: Arc<dyn RunCommandRepository + Send + Sync>,
    pub tuning_analysis_repository: Arc<dyn TuningAnalysisRepository + Send + Sync>,
    pub tuning_command_repository: Arc<dyn TuningCommandRepository + Send + Sync>,
    pub tuning_session_repository: Arc<dyn TuningSessionRepository + Send + Sync>,
    pub tuning_trial_repository: Arc<dyn TuningTrialRepository + Send + Sync>,
    pub ingest: Arc<dyn BenchIngest>,
}

impl BenchAdapters {
    /// Opens and fully upgrades a benchmark database before creating adapters.
    pub fn open(path: impl AsRef<Path>) -> duckdb::Result<Self> {
        Self::from_initialized_connection(crate::schema::open(path)?)
    }

    /// Builds adapters around a connection whose schema has already been
    /// initialized. Intended for narrow in-memory adapter fixtures.
    pub fn from_initialized_connection(connection: Connection) -> duckdb::Result<Self> {
        Self::from_initialized_shared_connection(Arc::new(Mutex::new(connection)))
    }

    /// Builds a fixture bundle around an already initialized shared in-memory
    /// connection. Production callers should use [`Self::open`].
    pub fn from_initialized_shared_connection(
        connection: Arc<Mutex<Connection>>,
    ) -> duckdb::Result<Self> {
        Ok(Self {
            project_repository: Arc::new(
                crate::project_repository_duckdb::SharedDuckDbProjectRepository::new(
                    connection.clone(),
                ),
            ),
            projects_repository: connection.clone(),
            run_repository: Arc::new(
                crate::run_repository_duckdb::SharedDuckDbRunRepository::new(connection.clone()),
            ),
            run_command_repository: Arc::new(
                crate::run_command_repository_duckdb::SharedDuckDbRunCommandRepository::new(
                    connection.clone(),
                ),
            ),
            tuning_analysis_repository: Arc::new(
                crate::tuning_analysis_repository_duckdb::SharedDuckDbTuningAnalysisRepository::new(
                    connection.clone(),
                ),
            ),
            tuning_command_repository: Arc::new(
                crate::tuning_command_repository_duckdb::SharedDuckDbTuningCommandRepository::new(
                    connection.clone(),
                ),
            ),
            tuning_session_repository: Arc::new(
                crate::tuning_session_repository_duckdb::SharedDuckDbTuningSessionRepository::new(
                    connection.clone(),
                ),
            ),
            tuning_trial_repository: Arc::new(
                crate::tuning_trial_repository_duckdb::SharedDuckDbTuningTrialRepository::new(
                    connection.clone(),
                ),
            ),
            ingest: Arc::new(SharedDuckDbIngest { connection }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project_repository::CreateProject;

    #[test]
    fn adapters_share_one_initialized_database() {
        let connection = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        crate::schema::ensure_schema(&connection.lock().unwrap()).unwrap();
        let adapters =
            BenchAdapters::from_initialized_shared_connection(connection.clone()).unwrap();

        adapters
            .project_repository
            .create_project(CreateProject {
                project_id: "project-a".into(),
                name: "Project A".into(),
                description: "shared database".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .unwrap();

        let projects = adapters.project_repository.list_active_projects().unwrap();
        assert_eq!(projects.len(), 1);
        let count: i64 = connection
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM projects", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn bootstrap_failure_prevents_adapter_startup() {
        let directory = std::env::temp_dir().join(format!(
            "mcts_bench_composition_failure_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("bench.duckdb");
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch("CREATE VIEW runs AS SELECT 1 AS run_id")
            .unwrap();
        drop(connection);

        assert!(BenchAdapters::open(&path).is_err());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn ingest_and_repository_mutation_share_the_same_lock() {
        let connection = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        crate::schema::ensure_schema(&connection.lock().unwrap()).unwrap();
        let adapters = BenchAdapters::from_initialized_shared_connection(connection).unwrap();
        let directory = std::env::temp_dir().join(format!(
            "mcts_bench_composition_serialization_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("registry.log"), "").unwrap();

        let ingest = adapters.ingest.clone();
        let repository = adapters.project_repository.clone();
        std::thread::scope(|scope| {
            let ingest_task = scope.spawn(|| ingest.ingest_once(&directory));
            let mutation_task = scope.spawn(|| {
                repository.create_project(CreateProject {
                    project_id: "project-b".into(),
                    name: "Project B".into(),
                    description: "serialized with ingest".into(),
                    created_at: "2026-01-01T00:00:00Z".into(),
                })
            });
            ingest_task.join().unwrap().unwrap();
            mutation_task.join().unwrap().unwrap();
        });
        assert_eq!(
            adapters
                .project_repository
                .list_active_projects()
                .unwrap()
                .len(),
            1
        );
        let _ = std::fs::remove_dir_all(directory);
    }
}
