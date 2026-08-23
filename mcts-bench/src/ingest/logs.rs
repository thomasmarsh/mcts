use std::fs;

use std::io::{BufRead, BufReader, Seek, SeekFrom};

use std::path::Path;

use duckdb::{params, Connection};

use crate::launch::iso_timestamp;

use crate::log::LogRecord;
use crate::projects_attempt::ProjectsRepository;
use crate::projects_attempt_duckdb;

use super::cursor::{get_cursor, set_cursor};

use super::IngestError;

pub(super) fn process_runs(conn: &Connection) -> Result<(), IngestError> {
    let mut stmt = conn.prepare("SELECT run_id, log_path FROM runs WHERE status IN ('starting', 'running', 'completed', 'completed_with_errors', 'crashed', 'stopped')")?;
    let running_runs: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (run_id, log_path_str) in &running_runs {
        validate_typed_projects_attempt(conn, run_id)?;
        process_one_log_file(conn, run_id, Path::new(log_path_str))?;

        // Move-trace lines land in a dedicated `moves.jsonl` next to
        // `log.jsonl` (see `LogRecord::Move`'s doc comment for why they're
        // kept out of the main log) -- same directory, derived rather than
        // stored as its own `runs` column. Not every run kind writes one
        // (only round-robin/tuner launches that pass `--trace-path`;
        // ad hoc `bench round-robin` runs without it don't), so a missing
        // file here is normal, not an error.
        let moves_path = Path::new(log_path_str).with_file_name("moves.jsonl");
        process_one_log_file(conn, run_id, &moves_path)?;
        finalize_projects_attempt(conn, run_id)?;
    }

    Ok(())
}

fn validate_typed_projects_attempt(conn: &Connection, run_id: &str) -> Result<(), IngestError> {
    projects_attempt_duckdb::Repository::new(conn).load_if_initialized(run_id)?;
    Ok(())
}

/// Complete a typed Projects attempt only after both ordinary log cursors have
/// advanced successfully. Compatibility cells remain the source of the
/// completed-with-errors distinction; the typed phase itself never changes
/// after this event.
fn finalize_projects_attempt(conn: &Connection, run_id: &str) -> Result<(), IngestError> {
    let repo = projects_attempt_duckdb::Repository::new(conn);
    let Some(receipt) = repo.load_if_initialized(run_id)? else {
        return Ok(());
    };
    if !receipt.needs_final_output() {
        return Ok(());
    }
    repo.finalize_output(run_id, &iso_timestamp())?;
    Ok(())
}

fn process_one_log_file(
    conn: &Connection,
    run_id: &str,
    log_path: &Path,
) -> Result<(), IngestError> {
    if !log_path.exists() {
        return Ok(());
    }

    let cursor_key = log_path.to_string_lossy().to_string();
    let offset = get_cursor(conn, &cursor_key)?;
    let file_len = fs::metadata(log_path)?.len();

    if file_len <= offset {
        return Ok(());
    }

    let mut file = fs::File::open(log_path)?;
    file.seek(SeekFrom::Start(offset))?;
    let reader = BufReader::new(file);

    for line_result in reader.lines() {
        let line = line_result?;

        let record: LogRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };

        match record {
            LogRecord::MatchResult {
                seq,
                strategy_a,
                strategy_b,
                outcome,
                winner,
                extra,
                cell_id,
                seed,
                trace_game_seq,
                metrics,
            } => {
                let ts = iso_timestamp();
                let extra_json = extra
                    .as_ref()
                    .map(|v| serde_json::to_string(v).expect("Value -> String"));
                conn.execute(
                    "INSERT INTO match_results \
                         (run_id, seq, ts, strategy_a, strategy_b, \
                          outcome, winner, extra, cell_id, seed, trace_game_seq, metrics) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12) \
                         ON CONFLICT (run_id, seq) DO NOTHING",
                    params![
                        run_id,
                        seq as i64,
                        ts,
                        strategy_a,
                        strategy_b,
                        outcome,
                        winner,
                        extra_json,
                        cell_id,
                        seed,
                        trace_game_seq,
                        metrics
                            .as_ref()
                            .map(|v| serde_json::to_string(v).expect("Value -> String")),
                    ],
                )?;
                if let Some(ref cell_id) = cell_id {
                    conn.execute(
                        "UPDATE experiment_cells SET completed_games = (SELECT COUNT(*) FROM match_results WHERE run_id = ?1 AND cell_id = ?2) WHERE run_id = ?1 AND cell_id = ?2",
                        params![run_id, cell_id],
                    )?;
                }
            }
            LogRecord::CellStarted { cell_id } => {
                let belongs: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM experiment_cells WHERE run_id = ?1 AND cell_id = ?2",
                    params![run_id, cell_id],
                    |row| row.get(0),
                )?;
                if belongs == 0 {
                    return Err(IngestError::OrphanCell {
                        run_id: run_id.to_owned(),
                        cell_id,
                    });
                }
                conn.execute(
                    "UPDATE experiment_cells SET status = 'running', started_at = COALESCE(started_at, ?1) WHERE run_id = ?2 AND cell_id = ?3 AND status = 'pending'",
                    params![iso_timestamp(), run_id, cell_id],
                )?;
            }
            LogRecord::CellFinished {
                cell_id,
                completed_games,
            } => {
                ensure_cell_belongs(conn, run_id, &cell_id)?;
                conn.execute(
                    "UPDATE experiment_cells SET status = 'completed', completed_games = ?1, ended_at = ?2 WHERE run_id = ?3 AND cell_id = ?4 AND status IN ('pending', 'running')",
                    params![completed_games, iso_timestamp(), run_id, cell_id],
                )?;
            }
            LogRecord::CellFailed {
                cell_id,
                completed_games,
                error,
            } => {
                ensure_cell_belongs(conn, run_id, &cell_id)?;
                conn.execute(
                    "UPDATE experiment_cells SET status = 'failed', completed_games = ?1, error = ?2, ended_at = ?3 WHERE run_id = ?4 AND cell_id = ?5 AND status IN ('pending', 'running')",
                    params![completed_games, error, iso_timestamp(), run_id, cell_id],
                )?;
                conn.execute(
                    "UPDATE runs SET status = 'completed_with_errors', ended_at = ?1 WHERE kind = 'experiment' AND run_id = ?2 AND status = 'completed'",
                    params![iso_timestamp(), run_id],
                )?;
            }
            LogRecord::Trial {
                trial_id,
                config,
                seed,
                cost,
                extra,
            } => {
                let ts = iso_timestamp();
                let config_json = serde_json::to_string(&config).expect("Value -> String");
                let extra_json = extra
                    .as_ref()
                    .map(|v| serde_json::to_string(v).expect("Value -> String"));
                conn.execute(
                    "INSERT INTO trials \
                         (run_id, trial_id, ts, config, seed, cost, extra) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
                         ON CONFLICT (run_id, trial_id) DO NOTHING",
                    params![
                        run_id,
                        trial_id as i64,
                        ts,
                        config_json,
                        seed.map(|s| s as i64),
                        cost,
                        extra_json,
                    ],
                )?;
            }
            LogRecord::Incumbent {
                config,
                cost,
                extra,
            } => {
                let ts = iso_timestamp();
                let config_json = serde_json::to_string(&config).expect("Value -> String");
                let extra_json = extra
                    .as_ref()
                    .map(|v| serde_json::to_string(v).expect("Value -> String"));
                conn.execute(
                    "INSERT INTO incumbents (run_id, ts, config, cost, extra) \
                         VALUES (?1, ?2, ?3, ?4, ?5) \
                         ON CONFLICT (run_id) DO UPDATE SET \
                             ts = excluded.ts, config = excluded.config, \
                             cost = excluded.cost, extra = excluded.extra",
                    params![run_id, ts, config_json, cost, extra_json],
                )?;
            }
            LogRecord::Move {
                game_seq,
                ply,
                state,
                mv,
                player,
            } => {
                let ts = iso_timestamp();
                let state_json = serde_json::to_string(&state).expect("Value -> String");
                let mv_json = mv
                    .as_ref()
                    .map(|v| serde_json::to_string(v).expect("Value -> String"));
                conn.execute(
                    "INSERT INTO game_moves \
                         (run_id, game_seq, ply, ts, state, mv, player) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
                         ON CONFLICT (run_id, game_seq, ply) DO NOTHING",
                    params![
                        run_id,
                        game_seq as i64,
                        ply as i64,
                        ts,
                        state_json,
                        mv_json,
                        player
                    ],
                )?;
            }
            LogRecord::Heartbeat { .. } => {}
        }
    }

    set_cursor(conn, &cursor_key, file_len)?;

    Ok(())
}

fn ensure_cell_belongs(conn: &Connection, run_id: &str, cell_id: &str) -> Result<(), IngestError> {
    let belongs: i64 = conn.query_row(
        "SELECT COUNT(*) FROM experiment_cells WHERE run_id = ?1 AND cell_id = ?2",
        params![run_id, cell_id],
        |row| row.get(0),
    )?;
    if belongs == 0 {
        return Err(IngestError::OrphanCell {
            run_id: run_id.to_owned(),
            cell_id: cell_id.to_owned(),
        });
    }
    Ok(())
}
