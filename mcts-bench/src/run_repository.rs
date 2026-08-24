//! Logical reads over benchmark runs.
//!
//! Callers depend on this interface instead of a particular database driver,
//! so they can use an in-memory implementation or a test double.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunRepositoryError {
    NotFound,
    Storage(String),
}

impl std::fmt::Display for RunRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "benchmark run was not found"),
            Self::Storage(message) => write!(f, "benchmark run storage failure: {message}"),
        }
    }
}

impl std::error::Error for RunRepositoryError {}

/// Logical read operations over benchmark runs.
///
/// All arguments and results are ordinary application data. Implementations
/// may use DuckDB, a different durable store, or a test double.
pub trait RunRepository {
    fn load_log_path(&self, run_id: &str) -> Result<String, RunRepositoryError>;
}
