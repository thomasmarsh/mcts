use std::fs;

use std::io::{BufRead, BufReader, Seek, SeekFrom};

use std::path::Path;

use duckdb::{params, Connection};

use crate::identity;

use crate::log::RegistryEvent;
use crate::projects_attempt::ProjectsRepository;
use crate::projects_attempt_duckdb;

use super::cursor::{get_cursor, set_cursor};

use super::liveness::{hostname, mark_experiment_crashed};

use super::IngestError;

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

    let mut file = fs::File::open(registry_path)?;
    file.seek(SeekFrom::Start(offset))?;
    let reader = BufReader::new(file);

    let mut events = Vec::new();
    for line_result in reader.lines() {
        let line = line_result?;

        let event: RegistryEvent = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        events.push(event);
    }
    for event in events {
        match event {
            RegistryEvent::Start {
                run_id,
                kind,
                game,
                pid,
                cmd: _,
                log_path,
                git_sha,
                git_dirty,
                started_at,
            } => {
                let host = hostname();
                let tx = conn.unchecked_transaction()?;
                let inserted = tx.execute(
                    "INSERT INTO runs \
                     (run_id, kind, game, config, git_sha, git_dirty, \
                      host, pid, started_at, status, log_path) \
                     VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6, ?7, ?8, 'running', ?9) \
                     ON CONFLICT (run_id) DO NOTHING",
                    params![
                        run_id, kind, game, git_sha, git_dirty, host, pid as i64, started_at,
                        log_path,
                    ],
                )?;
                if inserted > 0 {
                    identity::create_registry_root_identity(&tx, &run_id, &kind, &started_at)
                        .map_err(|error| duckdb::Error::ToSqlConversionFailure(Box::new(error)))?;
                }
                tx.commit()?;
            }
            RegistryEvent::Stop {
                run_id,
                exit_code,
                ended_at,
            } => {
                if projects_attempt_duckdb::Repository::new(conn)
                    .load_if_initialized(&run_id)?
                    .is_some()
                {
                    continue;
                }
                // Guard on `status = 'running'` so this can't clobber an
                // already-terminal status set by another path -- e.g.
                // `launch_and_record`'s own synchronous early-crash check
                // (a process that died within its 500ms post-spawn window
                // is marked `'crashed'` directly, before this event's
                // launcher-side reaper thread has necessarily even
                // observed the exit yet). Whichever path reaches a given
                // run first wins; the other becomes a no-op, matching
                // `reconcile_liveness`'s identical guard below.
                let kind: Option<String> = conn
                    .query_row(
                        "SELECT kind FROM runs WHERE run_id = ?1",
                        params![&run_id],
                        |row| row.get(0),
                    )
                    .ok();
                if kind.as_deref() == Some("experiment") {
                    if exit_code != Some(0) {
                        mark_experiment_crashed(conn, &run_id, &ended_at, "coordinator exited")?;
                    } else {
                        let failed: i64 = conn.query_row(
                            "SELECT COUNT(*) FROM experiment_cells WHERE run_id = ?1 AND status = 'failed'",
                            params![&run_id],
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
                } else {
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
                }
            }
        }
    }

    set_cursor(conn, &cursor_key, file_len)
}
