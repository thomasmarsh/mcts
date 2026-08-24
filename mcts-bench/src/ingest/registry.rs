use duckdb::{params, Connection};
use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use super::cursor::{get_cursor, set_cursor};
use super::liveness::{hostname, mark_experiment_crashed};
use super::IngestError;
use crate::identity;
use crate::log::RegistryEvent;
use crate::projects_attempt::ProjectsRepository;
use crate::projects_attempt_duckdb;

/// Process registry entries written since the last recorded cursor.
pub(super) fn process(conn: &Connection, registry_path: &Path) -> Result<(), IngestError> {
    if !registry_path.exists() {
        return Ok(());
    }

    let cursor_key = registry_path.to_string_lossy().to_string();
    let offset = get_cursor(conn, &cursor_key)?;
    let file_len = fs::metadata(registry_path)?.len();

    if file_len <= offset {
        return Ok(());
    }

    let events = read_events_from_file(registry_path, offset)?;

    for event in events {
        match event {
            RegistryEvent::Start {
                run_id,
                kind,
                game,
                pid,
                log_path,
                git_sha,
                git_dirty,
                started_at,
                ..
            } => {
                handle_start_event(
                    conn, run_id, kind, game, pid, log_path, git_sha, git_dirty, started_at,
                )?;
            }
            RegistryEvent::Stop {
                run_id,
                exit_code,
                ended_at,
            } => {
                handle_stop_event(conn, run_id, exit_code, ended_at)?;
            }
        }
    }

    set_cursor(conn, &cursor_key, file_len)
}

/// Read parseable registry events after `offset`, ignoring malformed lines.
fn read_events_from_file(path: &Path, offset: u64) -> Result<Vec<RegistryEvent>, IngestError> {
    let mut file = fs::File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let reader = BufReader::new(file);

    let mut events = Vec::new();
    for line_result in reader.lines() {
        let line = line_result?;
        if let Ok(event) = serde_json::from_str(&line) {
            events.push(event);
        }
    }
    Ok(events)
}

/// Record a newly launched run and create its registry-owned root identity.
#[allow(clippy::too_many_arguments)]
fn handle_start_event(
    conn: &Connection,
    run_id: String,
    kind: String,
    game: String,
    pid: u32,
    log_path: String,
    git_sha: String,
    git_dirty: bool,
    started_at: String,
) -> Result<(), IngestError> {
    let host = hostname();
    let tx = conn.unchecked_transaction()?;

    let inserted = tx.execute(
        "INSERT INTO runs \
         (run_id, kind, game, config, git_sha, git_dirty, \
          host, pid, started_at, status, log_path) \
         VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6, ?7, ?8, 'running', ?9) \
         ON CONFLICT (run_id) DO NOTHING",
        params![run_id, kind, game, git_sha, git_dirty, host, pid as i64, started_at, log_path],
    )?;

    if inserted > 0 {
        identity::create_registry_root_identity(&tx, &run_id, &kind, &started_at)
            .map_err(|error| duckdb::Error::ToSqlConversionFailure(Box::new(error)))?;
    }

    tx.commit()?;
    Ok(())
}

/// Apply a terminal registry event unless a projects attempt owns the run.
fn handle_stop_event(
    conn: &Connection,
    run_id: String,
    exit_code: Option<i32>,
    ended_at: String,
) -> Result<(), IngestError> {
    if projects_attempt_duckdb::Repository::new(conn)
        .load_if_initialized(&run_id)?
        .is_some()
    {
        return Ok(());
    }

    let kind: Option<String> = conn
        .query_row(
            "SELECT kind FROM runs WHERE run_id = ?1",
            params![&run_id],
            |row| row.get(0),
        )
        .ok();

    if kind.as_deref() == Some("experiment") {
        process_experiment_stop(conn, &run_id, exit_code, &ended_at)?;
    } else {
        process_standard_stop(conn, &run_id, exit_code, &ended_at)?;
    }

    Ok(())
}

/// Mark an experiment terminal, preserving its per-cell failure outcome.
fn process_experiment_stop(
    conn: &Connection,
    run_id: &str,
    exit_code: Option<i32>,
    ended_at: &str,
) -> Result<(), IngestError> {
    if exit_code != Some(0) {
        mark_experiment_crashed(conn, run_id, ended_at, "coordinator exited")?;
    } else {
        let failed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM experiment_cells WHERE run_id = ?1 AND status = 'failed'",
            params![run_id],
            |row| row.get(0),
        )?;

        let status = if failed > 0 {
            "completed_with_errors"
        } else {
            "completed"
        };

        conn.execute(
            "UPDATE runs SET ended_at = ?1, exit_code = ?2, status = ?3 \
             WHERE run_id = ?4 AND status = 'running'",
            params![ended_at, exit_code, status, run_id],
        )?;
    }
    Ok(())
}

/// Mark a non-experiment run terminal.
fn process_standard_stop(
    conn: &Connection,
    run_id: &str,
    exit_code: Option<i32>,
    ended_at: &str,
) -> Result<(), IngestError> {
    let status = if exit_code == Some(0) {
        "completed"
    } else {
        "crashed"
    };

    conn.execute(
        "UPDATE runs SET ended_at = ?1, exit_code = ?2, status = ?3 \
         WHERE run_id = ?4 AND status = 'running'",
        params![ended_at, exit_code, status, run_id],
    )?;
    Ok(())
}
