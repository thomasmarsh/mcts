use duckdb::{params, Connection};

use crate::launch::{is_alive, iso_timestamp};
use crate::projects_attempt::ProjectsRepository;
use crate::projects_attempt_duckdb;

use super::IngestError;

pub(super) fn mark_experiment_crashed(
    conn: &Connection,
    run_id: &str,
    ended_at: &str,
    error: &str,
) -> Result<(), IngestError> {
    conn.execute(
        "UPDATE experiment_cells SET status = 'failed', error = COALESCE(error, ?1), ended_at = COALESCE(ended_at, ?2) WHERE run_id = ?3 AND status = 'running'",
        params![error, ended_at, run_id],
    )?;
    conn.execute(
        "UPDATE experiment_cells SET status = 'cancelled', error = COALESCE(error, ?1), ended_at = COALESCE(ended_at, ?2) WHERE run_id = ?3 AND status = 'pending'",
        params![error, ended_at, run_id],
    )?;
    conn.execute(
        "UPDATE runs SET ended_at = ?1, status = 'crashed' WHERE run_id = ?2 AND status IN ('running', 'completed')",
        params![ended_at, run_id],
    )?;
    Ok(())
}

pub(super) fn reconcile(conn: &Connection) -> Result<(), IngestError> {
    let mut stmt = conn.prepare(
        "SELECT run_id, pid, kind FROM runs WHERE pid IS NOT NULL AND status = 'running'",
    )?;
    let maybe_dead: Vec<(String, i64, String)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (run_id, pid, kind) in maybe_dead {
        if kind == "experiment"
            && projects_attempt_duckdb::Repository::new(conn)
                .load_if_initialized(&run_id)?
                .is_some()
        {
            continue;
        }
        if !is_alive(pid as u32) {
            let ended_at = iso_timestamp();
            if kind == "experiment" {
                mark_experiment_crashed(conn, &run_id, &ended_at, "coordinator disappeared")?;
            } else {
                conn.execute(
                    "UPDATE runs SET ended_at = ?1, status = 'crashed' \
                     WHERE run_id = ?2 AND status = 'running'",
                    params![ended_at, run_id],
                )?;
            }
        }
    }

    Ok(())
}
pub(super) fn hostname() -> String {
    if let Ok(host) = std::env::var("HOSTNAME") {
        return host;
    }
    if let Ok(output) = std::process::Command::new("hostname").output() {
        if output.status.success() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                return s.trim().to_owned();
            }
        }
    }
    "unknown".to_owned()
}
