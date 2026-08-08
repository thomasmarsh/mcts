//! Ingest loop: reads registry.log and running runs' log.jsonl files,
//! upserts into the DuckDB database.  Only the `server` process should
//! call this (DuckDB single-writer constraint).
//!
//! The single public entry point is `ingest_once`, which does one
//! pass: registry events → log-file records → liveness reconciliation.
//! It is designed to be called on a `tokio::time::interval` inside the
//! server's `main()`.

use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use duckdb::{params, Connection};

use super::launch::{is_alive, iso_timestamp};
use super::log::{LogRecord, RegistryEvent};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum IngestError {
    DuckDb(duckdb::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::DuckDb(e) => write!(f, "DuckDB error: {e}"),
            IngestError::Io(e) => write!(f, "I/O error: {e}"),
            IngestError::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for IngestError {}

impl From<duckdb::Error> for IngestError {
    fn from(e: duckdb::Error) -> Self {
        IngestError::DuckDb(e)
    }
}

impl From<std::io::Error> for IngestError {
    fn from(e: std::io::Error) -> Self {
        IngestError::Io(e)
    }
}

impl From<serde_json::Error> for IngestError {
    fn from(e: serde_json::Error) -> Self {
        IngestError::Json(e)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run one ingest pass: read registry.log, each running run's log.jsonl,
/// and perform liveness/crash reconciliation.
///
/// `bench_runs_dir` is the path to the `bench-runs/` directory (the same
/// constant [`super::launch::BENCH_RUNS_DIR`] that the launcher uses).
///
/// Idempotent: a second call immediately after the first is a no-op, because
/// every file's byte-offset cursor is persisted in `_ingest_cursor` before
/// returning.
pub fn ingest_once(conn: &Connection, bench_runs_dir: &Path) -> Result<(), IngestError> {
    let registry_path = bench_runs_dir.join("registry.log");

    // ---- 1. Process registry.log ----
    process_registry(conn, &registry_path)?;

    // ---- 2. Process each running run's log.jsonl ----
    process_run_logs(conn)?;

    // ---- 3. Liveness reconciliation ----
    reconcile_liveness(conn)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 1: registry.log
// ---------------------------------------------------------------------------

/// Read new lines from registry.log since the last cursor position and
/// upsert into `runs`.
fn process_registry(conn: &Connection, registry_path: &Path) -> Result<(), IngestError> {
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

    for line_result in reader.lines() {
        let line = line_result?;

        let event: RegistryEvent = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue, // skip unparseable lines (forward compat)
        };

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
                conn.execute(
                    "INSERT INTO runs \
                     (run_id, kind, game, config, git_sha, git_dirty, \
                      host, pid, started_at, status, log_path) \
                     VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6, ?7, ?8, 'running', ?9) \
                     ON CONFLICT (run_id) DO NOTHING",
                    params![
                        run_id,
                        kind,
                        game,
                        git_sha,
                        git_dirty,
                        host,
                        pid as i64,
                        started_at,
                        log_path,
                    ],
                )?;
            }
            RegistryEvent::Stop {
                run_id,
                exit_code,
                ended_at,
            } => {
                conn.execute(
                    "UPDATE runs SET ended_at = ?1, exit_code = ?2, status = 'completed' \
                     WHERE run_id = ?3",
                    params![ended_at, exit_code, run_id],
                )?;
            }
        }
    }

    set_cursor(conn, &cursor_key, file_len)
}

// ---------------------------------------------------------------------------
// Stage 2: per-run log.jsonl
// ---------------------------------------------------------------------------

/// Read new lines from each running run's log.jsonl and dispatch by type
/// into the appropriate DuckDB table.
fn process_run_logs(conn: &Connection) -> Result<(), IngestError> {
    let mut stmt = conn.prepare(
        "SELECT run_id, log_path FROM runs WHERE status = 'running'",
    )?;
    let running_runs: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (run_id, log_path_str) in &running_runs {
        let log_path = Path::new(log_path_str);
        if !log_path.exists() {
            continue;
        }

        let cursor_key = log_path.to_string_lossy().to_string();
        let offset = get_cursor(conn, &cursor_key)?;
        let file_len = fs::metadata(log_path)?.len();

        if file_len <= offset {
            continue;
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
                } => {
                    let ts = iso_timestamp();
                    let extra_json = extra
                        .as_ref()
                        .map(|v| serde_json::to_string(v).expect("Value -> String"));
                    conn.execute(
                        "INSERT INTO match_results \
                         (run_id, seq, ts, strategy_a, strategy_b, \
                          outcome, winner, extra) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
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
                        ],
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
                    let config_json =
                        serde_json::to_string(&config).expect("Value -> String");
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
                LogRecord::Heartbeat { .. } => {
                    // Heartbeats are informational only; no table row.
                }
            }
        }

        set_cursor(conn, &cursor_key, file_len)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 3: liveness reconciliation
// ---------------------------------------------------------------------------

/// For any `runs` row still `status = 'running'` whose PID is no longer
/// alive, mark it as `crashed`.  Runs whose Stop event was already
/// processed in Stage 1 will have `status = 'completed'` and won't appear
/// in the `WHERE status = 'running'` query.
fn reconcile_liveness(conn: &Connection) -> Result<(), IngestError> {
    let mut stmt = conn.prepare(
        "SELECT run_id, pid FROM runs \
         WHERE status = 'running' AND pid IS NOT NULL",
    )?;
    let maybe_dead: Vec<(String, i64)> = stmt
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?
        .filter_map(|r| r.ok())
        .collect();

    for (run_id, pid) in &maybe_dead {
        if !is_alive(*pid as u32) {
            let ended_at = iso_timestamp();
            conn.execute(
                "UPDATE runs SET ended_at = ?1, status = 'crashed' \
                 WHERE run_id = ?2 AND status = 'running'",
                params![ended_at, run_id],
            )?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers: cursor management
// ---------------------------------------------------------------------------

/// Get the current byte offset for a log file from `_ingest_cursor`.
fn get_cursor(conn: &Connection, log_path: &str) -> Result<u64, IngestError> {
    match conn.query_row(
        "SELECT byte_offset FROM _ingest_cursor WHERE log_path = ?1",
        params![log_path],
        |row| row.get::<_, i64>(0),
    ) {
        Ok(offset) => Ok(offset as u64),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(0),
        Err(e) => Err(IngestError::DuckDb(e)),
    }
}

/// Upsert the byte-offset cursor for a log file.
fn set_cursor(conn: &Connection, log_path: &str, byte_offset: u64) -> Result<(), IngestError> {
    let now = iso_timestamp();
    conn.execute(
        "INSERT INTO _ingest_cursor (log_path, byte_offset, updated_at) \
         VALUES (?1, ?2, ?3) \
         ON CONFLICT (log_path) DO UPDATE \
         SET byte_offset = ?2, updated_at = ?3",
        params![log_path, byte_offset as i64, now],
    )?;
    Ok(())
}

/// Get the hostname of the current machine.
fn hostname() -> String {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::schema::ensure_schema;

    /// Create an in-memory DuckDB and write fixture files under a temp dir.
    /// Thread-safe counter so parallel tests get unique temp directories.
    static FIXTURE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    struct TestFixture {
        _dir: std::path::PathBuf,
        bench_runs: std::path::PathBuf,
        db: Connection,
    }

    impl TestFixture {
        fn new(registry_events: &[RegistryEvent]) -> Self {
            let n = FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "mcts_bench_ingest_test_{}_{}",
                std::process::id(),
                n,
            ));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let mut content = String::new();
            for ev in registry_events {
                content.push_str(&ev.to_json_line());
                content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &content).unwrap();

            let db = duckdb::Connection::open_in_memory().unwrap();
            ensure_schema(&db).unwrap();

            TestFixture { _dir: dir, bench_runs, db }
        }

        /// Count rows in a table.
        fn count(&self, table: &str) -> i64 {
            self.db
                .query_row(
                    &format!("SELECT COUNT(*) FROM {table}"),
                    [],
                    |row| row.get(0),
                )
                .unwrap()
        }

        fn query_string(&self, sql: &str) -> String {
            self.db
                .query_row(sql, [], |row| row.get(0))
                .unwrap()
        }
    }

    fn start_event(
        run_id: &str,
        kind: &str,
        game: &str,
        pid: u32,
        log_path: &str,
    ) -> RegistryEvent {
        RegistryEvent::Start {
            run_id: run_id.to_owned(),
            kind: kind.to_owned(),
            game: game.to_owned(),
            pid,
            cmd: vec!["bench".into(), "round-robin".into()],
            log_path: log_path.to_owned(),
            git_sha: "abc1234".into(),
            git_dirty: false,
            started_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    fn stop_event(run_id: &str, exit_code: Option<i32>) -> RegistryEvent {
        RegistryEvent::Stop {
            run_id: run_id.to_owned(),
            exit_code,
            ended_at: "2026-01-01T01:00:00Z".into(),
        }
    }

    // -------------------------------------------------------------------
    // Registry processing
    // -------------------------------------------------------------------

    #[test]
    fn test_registry_start_creates_run_row() {
        let ev = start_event("run-1", "round_robin", "druid", 99999, "/tmp/nope/log.jsonl");
        let fix = TestFixture::new(&[ev]);

        ingest_once(&fix.db, &fix.bench_runs).unwrap();

        assert_eq!(fix.count("runs"), 1);
        // The fake PID 99999 doesn't exist, so liveness reconciliation marks
        // it as crashed — that's correct, not a test bug.
        assert_eq!(
            fix.query_string("SELECT status FROM runs WHERE run_id = 'run-1'"),
            "crashed",
        );
        // Cursor should be set for registry.log.
        assert_eq!(fix.count("_ingest_cursor"), 1);
    }

    #[test]
    fn test_registry_start_stop_marks_completed() {
        let ev_start = start_event("run-2", "round_robin", "druid", 99998, "/tmp/nope2/log.jsonl");
        let ev_stop = stop_event("run-2", Some(0));
        let fix = TestFixture::new(&[ev_start, ev_stop]);

        ingest_once(&fix.db, &fix.bench_runs).unwrap();

        assert_eq!(fix.query_string("SELECT status FROM runs WHERE run_id = 'run-2'"), "completed");
        let exit_code: i64 = fix
            .db
            .query_row(
                "SELECT exit_code FROM runs WHERE run_id = 'run-2'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(exit_code, 0);
    }

    #[test]
    fn test_registry_stop_without_start_is_benign() {
        // A Stop event for a run that has no Start row just does nothing
        // (ON CONFLICT DO NOTHING on Stop is a no-op update).
        let ev = stop_event("orphan-run", Some(1));
        let fix = TestFixture::new(&[ev]);

        ingest_once(&fix.db, &fix.bench_runs).unwrap();

        assert_eq!(fix.count("runs"), 0);
    }

    // -------------------------------------------------------------------
    // Match result ingestion
    // -------------------------------------------------------------------

    #[test]
    fn test_ingest_match_results() {
        let fix = {
            let dir = std::env::temp_dir().join(format!(
                "mcts_bench_ingest_mr_{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "mr-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(run_id, "round_robin", "druid", 99997, &log_path_str)];

            let mut reg_content = String::new();
            for ev in &reg_events {
                reg_content.push_str(&ev.to_json_line());
                reg_content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

            let records = vec![
                LogRecord::MatchResult {
                    seq: 1,
                    strategy_a: "strong".into(),
                    strategy_b: "master".into(),
                    outcome: "win_a".into(),
                    winner: Some("strong".into()),
                    extra: None,
                },
                LogRecord::MatchResult {
                    seq: 2,
                    strategy_a: "master".into(),
                    strategy_b: "strong".into(),
                    outcome: "win_b".into(),
                    winner: Some("strong".into()),
                    extra: None,
                },
                LogRecord::MatchResult {
                    seq: 3,
                    strategy_a: "easy".into(),
                    strategy_b: "master".into(),
                    outcome: "draw".into(),
                    winner: None,
                    extra: None,
                },
            ];
            let mut log_content = String::new();
            for rec in &records {
                log_content.push_str(&rec.to_json_line());
                log_content.push('\n');
            }
            fs::write(&log_path, &log_content).unwrap();

            let db = duckdb::Connection::open_in_memory().unwrap();
            ensure_schema(&db).unwrap();

            (bench_runs, db)
        };

        ingest_once(&fix.1, &fix.0).unwrap();

        let count: i64 = fix
            .1
            .query_row("SELECT COUNT(*) FROM match_results", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 3);

        // Every row has a non-null ts and outcome.
        let null_outcomes: i64 = fix
            .1
            .query_row(
                "SELECT COUNT(*) FROM match_results WHERE outcome IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(null_outcomes, 0);
    }

    // -------------------------------------------------------------------
    // Idempotency
    // -------------------------------------------------------------------

    #[test]
    fn test_ingest_idempotent() {
        let (bench_runs, db) = {
            let dir = std::env::temp_dir().join(format!(
                "mcts_bench_idemp_{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "idem-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(run_id, "round_robin", "druid", 99996, &log_path_str)];
            let mut reg_content = String::new();
            for ev in &reg_events {
                reg_content.push_str(&ev.to_json_line());
                reg_content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

            let records = vec![LogRecord::MatchResult {
                seq: 1,
                strategy_a: "a".into(),
                strategy_b: "b".into(),
                outcome: "win_a".into(),
                winner: Some("a".into()),
                extra: None,
            }];
            let mut log_content = String::new();
            for rec in &records {
                log_content.push_str(&rec.to_json_line());
                log_content.push('\n');
            }
            fs::write(&log_path, &log_content).unwrap();

            let db = duckdb::Connection::open_in_memory().unwrap();
            ensure_schema(&db).unwrap();

            (bench_runs, db)
        };

        // First ingest.
        ingest_once(&db, &bench_runs).unwrap();
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );

        // Second ingest — must be a no-op.
        ingest_once(&db, &bench_runs).unwrap();
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1,
            "second ingest should not duplicate match results"
        );
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM runs", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1,
            "second ingest should not duplicate runs"
        );
    }

    // -------------------------------------------------------------------
    // Error handling: garbage lines, missing files
    // -------------------------------------------------------------------

    #[test]
    fn test_ingest_skips_unparseable_log_lines() {
        let (bench_runs, db) = {
            let dir = std::env::temp_dir().join(format!(
                "mcts_bench_garbage_{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "garbage-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(run_id, "round_robin", "druid", 99995, &log_path_str)];
            let mut reg_content = String::new();
            for ev in &reg_events {
                reg_content.push_str(&ev.to_json_line());
                reg_content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

            let good = LogRecord::MatchResult {
                seq: 1,
                strategy_a: "x".into(),
                strategy_b: "y".into(),
                outcome: "draw".into(),
                winner: None,
                extra: None,
            };
            let mut log_content = String::new();
            log_content.push_str(&good.to_json_line());
            log_content.push('\n');
            log_content.push_str("this is not json\n");
            log_content.push_str("{\"type\": \"unknown_thing\", \"data\": 42}\n");
            fs::write(&log_path, &log_content).unwrap();

            let db = duckdb::Connection::open_in_memory().unwrap();
            ensure_schema(&db).unwrap();

            (bench_runs, db)
        };

        ingest_once(&db, &bench_runs).unwrap();
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1,
            "should have ingested the one valid record past garbage"
        );
    }

    #[test]
    fn test_ingest_missing_log_jsonl_is_not_fatal() {
        // A run whose log.jsonl doesn't exist on disk should be silently
        // skipped, not crash the entire ingest.
        // Use the current process's PID so liveness reconciliation doesn't
        // kill the run (it's still alive while this test runs).
        let alive_pid = std::process::id();
        let ev = start_event(
            "ghost-run",
            "round_robin",
            "druid",
            alive_pid,
            "/tmp/nope/log.jsonl",
        );
        let fix = TestFixture::new(&[ev]);

        ingest_once(&fix.db, &fix.bench_runs).unwrap();

        assert_eq!(fix.count("runs"), 1);
        assert_eq!(
            fix.query_string("SELECT status FROM runs WHERE run_id = 'ghost-run'"),
            "running",
        );
    }

    // -------------------------------------------------------------------
    // Heartbeat records are silently ignored
    // -------------------------------------------------------------------

    #[test]
    fn test_heartbeats_are_skipped() {
        let (bench_runs, db) = {
            let dir = std::env::temp_dir().join(format!(
                "mcts_bench_hb_{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "hb-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(run_id, "round_robin", "druid", 99993, &log_path_str)];
            let mut reg_content = String::new();
            for ev in &reg_events {
                reg_content.push_str(&ev.to_json_line());
                reg_content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

            let records = vec![
                LogRecord::Heartbeat { games_played: 10 },
                LogRecord::MatchResult {
                    seq: 1,
                    strategy_a: "a".into(),
                    strategy_b: "b".into(),
                    outcome: "win_a".into(),
                    winner: Some("a".into()),
                    extra: None,
                },
                LogRecord::Heartbeat { games_played: 20 },
            ];
            let mut log_content = String::new();
            for rec in &records {
                log_content.push_str(&rec.to_json_line());
                log_content.push('\n');
            }
            fs::write(&log_path, &log_content).unwrap();

            let db = duckdb::Connection::open_in_memory().unwrap();
            ensure_schema(&db).unwrap();

            (bench_runs, db)
        };

        ingest_once(&db, &bench_runs).unwrap();

        // Only 1 match_result, no table for heartbeats.
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
        // No trials table populated either.
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM trials", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    // -------------------------------------------------------------------
    // Trial ingestion
    // -------------------------------------------------------------------

    #[test]
    fn test_ingest_trials() {
        let (bench_runs, db) = {
            let dir = std::env::temp_dir().join(format!(
                "mcts_bench_trial_{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "smac3-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(run_id, "smac3", "druid", 99992, &log_path_str)];
            let mut reg_content = String::new();
            for ev in &reg_events {
                reg_content.push_str(&ev.to_json_line());
                reg_content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

            let records = vec![
                LogRecord::Trial {
                    trial_id: 1,
                    config: serde_json::json!({"lr": 0.001, "iterations": 100}),
                    seed: Some(42),
                    cost: 0.375,
                    extra: None,
                },
                LogRecord::Trial {
                    trial_id: 2,
                    config: serde_json::json!({"lr": 0.01, "iterations": 200}),
                    seed: None,
                    cost: 0.512,
                    extra: Some(serde_json::json!({"note": "second trial"})),
                },
            ];
            let mut log_content = String::new();
            for rec in &records {
                log_content.push_str(&rec.to_json_line());
                log_content.push('\n');
            }
            fs::write(&log_path, &log_content).unwrap();

            let db = duckdb::Connection::open_in_memory().unwrap();
            ensure_schema(&db).unwrap();

            (bench_runs, db)
        };

        ingest_once(&db, &bench_runs).unwrap();

        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM trials", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            2
        );

        // Verify first trial's fields.
        let cost: f64 = db
            .query_row(
                "SELECT cost FROM trials WHERE run_id = 'smac3-run' AND trial_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!((cost - 0.375).abs() < 1e-9);

        let seed: Option<i64> = db
            .query_row(
                "SELECT seed FROM trials WHERE run_id = 'smac3-run' AND trial_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(seed, Some(42));

        // Second trial has extra metadata.
        let extra: Option<String> = db
            .query_row(
                "SELECT extra FROM trials WHERE run_id = 'smac3-run' AND trial_id = 2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(extra.is_some());
        let parsed: serde_json::Value =
            serde_json::from_str(&extra.unwrap()).unwrap();
        assert_eq!(parsed["note"], "second trial");
    }

    // -------------------------------------------------------------------
    // Registry garbage lines are skipped
    // -------------------------------------------------------------------

    #[test]
    fn test_registry_garbage_lines_are_skipped() {
        let dir = std::env::temp_dir().join(format!(
            "mcts_bench_reg_garbage_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let ev = start_event("garb-run", "round_robin", "druid", 99991, "/tmp/nope/log.jsonl");
        let mut content = String::new();
        content.push_str("totally not json\n");
        content.push_str(&ev.to_json_line());
        content.push('\n');
        content.push_str("also not json\n");
        fs::write(bench_runs.join("registry.log"), &content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        ingest_once(&db, &bench_runs).unwrap();

        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM runs", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1,
            "should parse the valid Start event past garbage lines"
        );
    }
}