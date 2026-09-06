//! Ingests benchmark registry and run logs into the DuckDB database.
//!
//! Only the `server` process should call this module because DuckDB permits a
//! single writer.

mod artifacts;
mod cursor;
mod error;
mod liveness;
mod logs;
mod projects;
mod registry;

pub use error::IngestError;

use std::path::Path;

use duckdb::Connection;

#[cfg(test)]
use logs::process_runs as process_run_logs;
#[cfg(test)]
use registry::process as process_registry;

/// Incorporate newly-written registry, lifecycle, and run-log records.
pub fn ingest_once(conn: &Connection, bench_runs_dir: &Path) -> Result<(), IngestError> {
    registry::process(conn, &bench_runs_dir.join("registry.log"))?;
    artifacts::process(conn)?;
    let observation_error = projects::observe(conn).err();
    logs::process_runs(conn)?;
    liveness::reconcile(conn)?;
    if let Some(error) = observation_error {
        return Err(error);
    }

    Ok(())
}

#[cfg(test)]
#[path = "ingest_test_support.rs"]
mod test_support;

#[cfg(test)]
#[path = "ingest_projects_tests.rs"]
mod projects_tests;

#[cfg(test)]
#[path = "ingest_tests.rs"]
mod tests;
