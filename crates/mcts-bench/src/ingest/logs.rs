use std::fs;

use std::io::{BufRead, BufReader, Seek, SeekFrom};

use std::path::Path;

use duckdb::{params, Connection};
use game_host::{SearchReport, SearchReportReason, SearchReportStatus};

use crate::launch::iso_timestamp;

use crate::log::LogRecord;
use crate::projects_attempt::ProjectsRepository;
use crate::projects_attempt_duckdb;

use super::cursor::{get_cursor, set_cursor};

use super::IngestError;

pub(super) fn process_runs(conn: &Connection) -> Result<(), IngestError> {
    let mut stmt = conn.prepare("SELECT run_id, kind, log_path FROM runs WHERE status IN ('starting', 'running', 'completed', 'completed_with_errors', 'crashed', 'stopped')")?;
    let running_runs: Vec<(String, String, String)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (run_id, kind, log_path_str) in &running_runs {
        validate_typed_projects_attempt(conn, run_id)?;
        process_one_log_file(conn, run_id, Path::new(log_path_str))?;

        // Move-trace lines for non-tuner runs land in a dedicated `moves.jsonl` next to
        // `log.jsonl` (see `LogRecord::Move`'s doc comment for why they're
        // kept out of the main log) -- same directory, derived rather than
        // stored as its own `runs` column. Not every run kind writes one
        // (round-robin and experiment launches that pass `--trace-path`;
        // ad hoc runs without it don't). Tuner tasks are ingested from their
        // partitioned artifacts, so a missing sibling file is normal.
        if kind != "tuner" {
            let moves_path = Path::new(log_path_str).with_file_name("moves.jsonl");
            process_one_log_file(conn, run_id, &moves_path)?;
        }
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

        apply_record(conn, run_id, record)?;
    }

    set_cursor(conn, &cursor_key, file_len)?;

    Ok(())
}

/// Consume only newline-terminated task trace records. A game child can be
/// observed while it is writing, so EOF is not evidence of a complete record.
pub(super) fn process_complete_trace_file(
    conn: &Connection,
    run_id: &str,
    log_path: &Path,
    mut offset: u64,
) -> Result<u64, IngestError> {
    if !log_path.exists() {
        return Ok(offset);
    }
    let mut file = fs::File::open(log_path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        let count = reader.read_until(b'\n', &mut bytes)?;
        if count == 0 || bytes.last() != Some(&b'\n') {
            break;
        }
        bytes.pop();
        let record =
            serde_json::from_slice(&bytes).map_err(|error| IngestError::ArtifactIntegrity {
                run_id: run_id.to_owned(),
                artifact: log_path.to_string_lossy().into_owned(),
                message: format!("trace record is not valid JSON: {error}"),
            })?;
        if !matches!(&record, LogRecord::Move { .. }) {
            return Err(IngestError::ArtifactIntegrity {
                run_id: run_id.to_owned(),
                artifact: log_path.to_string_lossy().into_owned(),
                message: "task trace contains a non-move record".into(),
            });
        }
        apply_record(conn, run_id, record)?;
        offset += count as u64;
    }
    Ok(offset)
}

fn apply_record(conn: &Connection, run_id: &str, record: LogRecord) -> Result<(), IngestError> {
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
            trace_schema_version,
            game_seq,
            ply,
            state,
            mv,
            player,
            search,
        } => {
            let search = project_search_report(trace_schema_version, search.as_ref())?;
            let ts = iso_timestamp();
            let state_json = serde_json::to_string(&state).expect("Value -> String");
            let mv_json = mv
                .as_ref()
                .map(|v| serde_json::to_string(v).expect("Value -> String"));
            conn.execute(
                    "INSERT INTO game_moves \
                         (run_id, game_seq, ply, ts, trace_schema_version, state, mv, player, \
                          search_report, search_status, search_completed_iterations, search_elapsed_ms, \
                          search_nodes, search_mean_depth, search_max_depth, search_tt_hit_ratio) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16) \
                         ON CONFLICT (run_id, game_seq, ply) DO NOTHING",
                    params![
                        run_id,
                        game_seq as i64,
                        ply as i64,
                        ts,
                        trace_schema_version,
                        state_json,
                        mv_json,
                        player,
                        search.as_ref().map(|value| value.json.as_str()),
                        search.as_ref().map(|value| value.status),
                        search.as_ref().map(|value| value.completed_iterations),
                        search.as_ref().map(|value| value.elapsed_ms),
                        search.as_ref().map(|value| value.nodes),
                        search.as_ref().and_then(|value| value.mean_depth),
                        search.as_ref().and_then(|value| value.max_depth),
                        search.as_ref().and_then(|value| value.tt_hit_ratio),
                    ],
                )?;
        }
        LogRecord::Heartbeat { .. } => {}
    }
    Ok(())
}

struct SearchProjection {
    json: String,
    status: &'static str,
    completed_iterations: u64,
    elapsed_ms: Option<f64>,
    nodes: u64,
    mean_depth: Option<f64>,
    max_depth: Option<u64>,
    tt_hit_ratio: Option<f64>,
}

fn project_search_report(
    trace_schema_version: Option<u32>,
    search: Option<&serde_json::Value>,
) -> Result<Option<SearchProjection>, IngestError> {
    match (trace_schema_version, search) {
        (None, None) => return Ok(None),
        (Some(1), None) => return Ok(None),
        (Some(version), None) => {
            return invalid_report(format!("unsupported trace schema version {version}"))
        }
        (None, Some(_)) => {
            return invalid_report("a search report requires trace schema version 1")
        }
        (Some(1), Some(_)) => {}
        (Some(version), Some(_)) => {
            return invalid_report(format!("unsupported trace schema version {version}"));
        }
    }

    let search = search.expect("covered by the match above");
    let report: SearchReport =
        serde_json::from_value(search.clone()).map_err(|error| IngestError::InvalidMoveReport {
            message: format!("does not match the canonical wire shape: {error}"),
        })?;
    validate_search_report(&report)?;

    Ok(Some(SearchProjection {
        json: serde_json::to_string(search).expect("JSON value is serializable"),
        status: match report.status {
            SearchReportStatus::Available => "available",
            SearchReportStatus::Partial => "partial",
            SearchReportStatus::Unavailable => "unavailable",
        },
        completed_iterations: report.completed_iterations as u64,
        elapsed_ms: elapsed_milliseconds(report.elapsed_seconds)?,
        nodes: report.tree_nodes as u64,
        mean_depth: report.mean_depth,
        max_depth: report.max_depth.map(|depth| depth as u64),
        tt_hit_ratio: report.tt_hit_ratio,
    }))
}

fn validate_search_report(report: &SearchReport) -> Result<(), IngestError> {
    if report.schema_version != 1 {
        return invalid_report(format!(
            "unsupported search schema version {}",
            report.schema_version
        ));
    }
    match (report.status, report.reason) {
        (SearchReportStatus::Available, None)
        | (SearchReportStatus::Partial, Some(SearchReportReason::RootParallelPvSingleTree))
        | (
            SearchReportStatus::Unavailable,
            Some(SearchReportReason::StrategyUnsupported | SearchReportReason::SearchNotRun),
        ) => {}
        _ => return invalid_report("status and reason are not a valid combination"),
    }

    for (name, value) in [
        ("elapsed_seconds", report.elapsed_seconds),
        ("time_limit_seconds", report.time_limit_seconds),
        ("mean_depth", report.mean_depth),
        ("tt_hit_ratio", report.tt_hit_ratio),
        ("iterations_per_second", report.iterations_per_second),
    ] {
        if let Some(value) = value {
            if !value.is_finite() {
                return invalid_report(format!("{name} must be finite"));
            }
            if value < 0.0 {
                return invalid_report(format!("{name} must not be negative"));
            }
        }
    }
    for action in &report.actions {
        if !action.share.is_finite() || !(0.0..=1.0).contains(&action.share) {
            return invalid_report("action share must be finite and in [0, 1]");
        }
        if !action.mean_value.is_finite() || !(-1.0..=1.0).contains(&action.mean_value) {
            return invalid_report("action mean value must be finite and in [-1, 1]");
        }
    }
    let action_share_total: f64 = report.actions.iter().map(|action| action.share).sum();
    if action_share_total > 1.0 + 1e-9 {
        return invalid_report("action shares must not sum to more than one");
    }
    if report
        .iteration_limit
        .is_some_and(|limit| report.completed_iterations > limit)
    {
        return invalid_report("completed iterations must not exceed the iteration limit");
    }
    if let Some(max_depth) = report.max_depth {
        if report
            .mean_depth
            .is_some_and(|mean_depth| mean_depth > max_depth as f64)
        {
            return invalid_report("mean depth must not exceed max depth");
        }
    }
    if report.tt_hits > report.tt_reads {
        return invalid_report("TT hits must not exceed TT reads");
    }
    match (report.tt_reads, report.tt_hit_ratio) {
        (0, None) => {}
        (0, Some(_)) => return invalid_report("TT hit ratio requires TT reads"),
        (reads, Some(ratio)) => {
            if ratio > 1.0 {
                return invalid_report("TT hit ratio must be in [0, 1]");
            }
            let expected = report.tt_hits as f64 / reads as f64;
            if (ratio - expected).abs() > 1e-9 {
                return invalid_report("TT hit ratio does not match TT hits and reads");
            }
        }
        (_, None) => return invalid_report("TT reads require a TT hit ratio"),
    }
    Ok(())
}

fn elapsed_milliseconds(seconds: Option<f64>) -> Result<Option<f64>, IngestError> {
    let Some(seconds) = seconds else {
        return Ok(None);
    };
    let milliseconds = seconds * 1_000.0;
    if !milliseconds.is_finite() {
        return invalid_report("elapsed milliseconds must be finite");
    }
    Ok(Some(milliseconds))
}

fn invalid_report<T>(message: impl Into<String>) -> Result<T, IngestError> {
    Err(IngestError::InvalidMoveReport {
        message: message.into(),
    })
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
