//! Ingest loop: reads registry.log and running runs' log.jsonl files,
//! upserts into the DuckDB database.  Only the `server` process should
//! call this (DuckDB single-writer constraint).

use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use duckdb::{params, Connection};

use crate::launch::{is_alive, iso_timestamp};
use crate::log::{LogRecord, RegistryEvent};

#[derive(Debug)]
pub enum IngestError {
    DuckDb(duckdb::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    OrphanCell { run_id: String, cell_id: String },
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::DuckDb(e) => write!(f, "DuckDB error: {e}"),
            IngestError::Io(e) => write!(f, "I/O error: {e}"),
            IngestError::Json(e) => write!(f, "JSON error: {e}"),
            IngestError::OrphanCell { run_id, cell_id } => {
                write!(f, "cell '{cell_id}' does not belong to run '{run_id}'")
            }
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

pub fn ingest_once(conn: &Connection, bench_runs_dir: &Path) -> Result<(), IngestError> {
    let registry_path = bench_runs_dir.join("registry.log");

    process_registry(conn, &registry_path)?;
    process_run_logs(conn)?;
    reconcile_liveness(conn)?;

    Ok(())
}

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
            Err(_) => continue,
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
                        run_id, kind, game, git_sha, git_dirty, host, pid as i64, started_at,
                        log_path,
                    ],
                )?;
            }
            RegistryEvent::Stop {
                run_id,
                exit_code,
                ended_at,
            } => {
                // Guard on `status = 'running'` so this can't clobber an
                // already-terminal status set by another path -- e.g.
                // `launch_and_record`'s own synchronous early-crash check
                // (a process that died within its 500ms post-spawn window
                // is marked `'crashed'` directly, before this event's
                // launcher-side reaper thread has necessarily even
                // observed the exit yet). Whichever path reaches a given
                // run first wins; the other becomes a no-op, matching
                // `reconcile_liveness`'s identical guard below.
                conn.execute(
                    "UPDATE runs SET ended_at = ?1, exit_code = ?2, status = 'completed' \
                     WHERE run_id = ?3 AND status = 'running'",
                    params![ended_at, exit_code, run_id],
                )?;
            }
        }
    }

    set_cursor(conn, &cursor_key, file_len)
}

fn process_run_logs(conn: &Connection) -> Result<(), IngestError> {
    let mut stmt = conn.prepare("SELECT run_id, log_path FROM runs WHERE status IN ('running', 'completed', 'crashed', 'stopped')")?;
    let running_runs: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (run_id, log_path_str) in &running_runs {
        process_one_log_file(conn, run_id, Path::new(log_path_str))?;

        // Move-trace lines land in a dedicated `moves.jsonl` next to
        // `log.jsonl` (see `LogRecord::Move`'s doc comment for why they're
        // kept out of the main log) -- same directory, derived rather than
        // stored as its own `runs` column. Not every run kind writes one
        // (only round-robin/SMAC3 launches that pass `--trace-path`;
        // ad hoc `bench round-robin` runs without it don't), so a missing
        // file here is normal, not an error.
        let moves_path = Path::new(log_path_str).with_file_name("moves.jsonl");
        process_one_log_file(conn, run_id, &moves_path)?;
    }

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
                    "UPDATE experiment_cells SET status = 'running', started_at = COALESCE(started_at, ?1) WHERE run_id = ?2 AND cell_id = ?3",
                    params![iso_timestamp(), run_id, cell_id],
                )?;
            }
            LogRecord::CellFinished {
                cell_id,
                completed_games,
            } => {
                ensure_cell_belongs(conn, run_id, &cell_id)?;
                conn.execute(
                    "UPDATE experiment_cells SET status = 'completed', completed_games = ?1, ended_at = ?2 WHERE run_id = ?3 AND cell_id = ?4",
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
                    "UPDATE experiment_cells SET status = 'failed', completed_games = ?1, error = ?2, ended_at = ?3 WHERE run_id = ?4 AND cell_id = ?5",
                    params![completed_games, error, iso_timestamp(), run_id, cell_id],
                )?;
                conn.execute(
                    "UPDATE runs SET status = 'crashed', ended_at = ?1 WHERE run_id = ?2 AND status IN ('running', 'completed')",
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

fn reconcile_liveness(conn: &Connection) -> Result<(), IngestError> {
    let mut stmt =
        conn.prepare("SELECT run_id, pid FROM runs WHERE status = 'running' AND pid IS NOT NULL")?;
    let maybe_dead: Vec<(String, i64)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ensure_schema;

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

            TestFixture {
                _dir: dir,
                bench_runs,
                db,
            }
        }

        fn count(&self, table: &str) -> i64 {
            self.db
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap()
        }

        fn query_string(&self, sql: &str) -> String {
            self.db.query_row(sql, [], |row| row.get(0)).unwrap()
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

    #[test]
    fn test_registry_start_creates_run_row() {
        let ev = start_event(
            "run-1",
            "round_robin",
            "druid",
            99999,
            "/tmp/nope/log.jsonl",
        );
        let fix = TestFixture::new(&[ev]);

        ingest_once(&fix.db, &fix.bench_runs).unwrap();

        assert_eq!(fix.count("runs"), 1);
        assert_eq!(
            fix.query_string("SELECT status FROM runs WHERE run_id = 'run-1'"),
            "crashed",
        );
        assert_eq!(fix.count("_ingest_cursor"), 1);
    }

    #[test]
    fn test_registry_start_stop_marks_completed() {
        let ev_start = start_event(
            "run-2",
            "round_robin",
            "druid",
            99998,
            "/tmp/nope2/log.jsonl",
        );
        let ev_stop = stop_event("run-2", Some(0));
        let fix = TestFixture::new(&[ev_start, ev_stop]);

        ingest_once(&fix.db, &fix.bench_runs).unwrap();

        assert_eq!(
            fix.query_string("SELECT status FROM runs WHERE run_id = 'run-2'"),
            "completed"
        );
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
    fn test_registry_stop_does_not_clobber_an_already_terminal_status() {
        // A run already marked 'stopped' (e.g. by the explicit stop_run
        // handler) must stay 'stopped' when a Stop registry event for it
        // is later ingested -- the launcher's own reaper thread writes one
        // for every exit, including a process that was SIGTERM'd, and it
        // races against (may land before or after) whatever else already
        // set the terminal status. See the `AND status = 'running'` guard
        // this test exercises.
        let ev_start = start_event("run-3", "smac3", "nim", 99997, "/tmp/nope3/log.jsonl");
        let fix = TestFixture::new(&[ev_start]);
        ingest_once(&fix.db, &fix.bench_runs).unwrap();
        fix.db
            .execute(
                "UPDATE runs SET status = 'stopped' WHERE run_id = 'run-3'",
                [],
            )
            .unwrap();

        // Now a Stop event for the same run arrives on a later ingest pass.
        fs::write(
            fix.bench_runs.join("registry.log"),
            format!(
                "{}\n{}\n",
                start_event("run-3", "smac3", "nim", 99997, "/tmp/nope3/log.jsonl").to_json_line(),
                stop_event("run-3", Some(0)).to_json_line(),
            ),
        )
        .unwrap();
        ingest_once(&fix.db, &fix.bench_runs).unwrap();

        assert_eq!(
            fix.query_string("SELECT status FROM runs WHERE run_id = 'run-3'"),
            "stopped",
        );
    }

    #[test]
    fn test_registry_stop_without_start_is_benign() {
        let ev = stop_event("orphan-run", Some(1));
        let fix = TestFixture::new(&[ev]);

        ingest_once(&fix.db, &fix.bench_runs).unwrap();
        assert_eq!(fix.count("runs"), 0);
    }

    #[test]
    fn test_ingest_match_results() {
        let fix = {
            let dir =
                std::env::temp_dir().join(format!("mcts_bench_ingest_mr_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "mr-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(
                run_id,
                "round_robin",
                "druid",
                99997,
                &log_path_str,
            )];

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
                    cell_id: None,
                    seed: None,
                    trace_game_seq: None,
                    metrics: None,
                },
                LogRecord::MatchResult {
                    seq: 2,
                    strategy_a: "master".into(),
                    strategy_b: "strong".into(),
                    outcome: "win_b".into(),
                    winner: Some("strong".into()),
                    extra: None,
                    cell_id: None,
                    seed: None,
                    trace_game_seq: None,
                    metrics: None,
                },
                LogRecord::MatchResult {
                    seq: 3,
                    strategy_a: "easy".into(),
                    strategy_b: "master".into(),
                    outcome: "draw".into(),
                    winner: None,
                    extra: None,
                    cell_id: None,
                    seed: None,
                    trace_game_seq: None,
                    metrics: None,
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

    #[test]
    fn test_ingest_idempotent() {
        let (bench_runs, db) = {
            let dir = std::env::temp_dir().join(format!("mcts_bench_idemp_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "idem-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(
                run_id,
                "round_robin",
                "druid",
                99996,
                &log_path_str,
            )];
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
                cell_id: None,
                seed: None,
                trace_game_seq: None,
                metrics: None,
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

        ingest_once(&db, &bench_runs).unwrap();
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );

        ingest_once(&db, &bench_runs).unwrap();
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1,
            "second ingest should not duplicate match results"
        );
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM runs", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            1,
            "second ingest should not duplicate runs"
        );
    }

    #[test]
    fn test_ingest_skips_unparseable_log_lines() {
        let (bench_runs, db) = {
            let dir =
                std::env::temp_dir().join(format!("mcts_bench_garbage_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "garbage-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(
                run_id,
                "round_robin",
                "druid",
                99995,
                &log_path_str,
            )];
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
                cell_id: None,
                seed: None,
                trace_game_seq: None,
                metrics: None,
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
        );
    }

    #[test]
    fn test_ingest_missing_log_jsonl_is_not_fatal() {
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

    #[test]
    fn test_heartbeats_are_skipped() {
        let (bench_runs, db) = {
            let dir = std::env::temp_dir().join(format!("mcts_bench_hb_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "hb-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(
                run_id,
                "round_robin",
                "druid",
                99993,
                &log_path_str,
            )];
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
                    cell_id: None,
                    seed: None,
                    trace_game_seq: None,
                    metrics: None,
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

        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM trials", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_ingest_trials() {
        let (bench_runs, db) = {
            let dir = std::env::temp_dir().join(format!("mcts_bench_trial_{}", std::process::id()));
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

        let extra: Option<String> = db
            .query_row(
                "SELECT extra FROM trials WHERE run_id = 'smac3-run' AND trial_id = 2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(extra.is_some());
        let parsed: serde_json::Value = serde_json::from_str(&extra.unwrap()).unwrap();
        assert_eq!(parsed["note"], "second trial");
    }

    #[test]
    fn test_ingest_incumbent_upserts_latest() {
        let (bench_runs, db) = {
            let dir =
                std::env::temp_dir().join(format!("mcts_bench_incumbent_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "smac3-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(run_id, "smac3", "druid", 99993, &log_path_str)];
            let mut reg_content = String::new();
            for ev in &reg_events {
                reg_content.push_str(&ev.to_json_line());
                reg_content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

            // Two incumbent records for the same run -- the intensifier
            // found a better config partway through, so the second should
            // overwrite the first rather than both landing as separate rows.
            let records = vec![
                LogRecord::Incumbent {
                    config: serde_json::json!({"family": "ucb1", "c": 1.0}),
                    cost: 0.5,
                    extra: None,
                },
                LogRecord::Incumbent {
                    config: serde_json::json!({"family": "rave", "c": 0.7}),
                    cost: 0.2,
                    extra: Some(serde_json::json!({"hash": "abc123"})),
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
            db.query_row("SELECT COUNT(*) FROM incumbents", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1,
            "a run's incumbent should overwrite in place, not accumulate rows"
        );

        let (cost, config_str, extra_str): (f64, String, Option<String>) = db
            .query_row(
                "SELECT cost, CAST(config AS TEXT), CAST(extra AS TEXT) \
                 FROM incumbents WHERE run_id = 'smac3-run'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert!((cost - 0.2).abs() < 1e-9);
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        assert_eq!(config["family"], "rave");
        let extra: serde_json::Value = serde_json::from_str(&extra_str.unwrap()).unwrap();
        assert_eq!(extra["hash"], "abc123");
    }

    #[test]
    fn test_ingest_tails_moves_jsonl_sibling_of_log_jsonl() {
        // Move traces land in a `moves.jsonl` next to `log.jsonl`, not
        // inside it (see `LogRecord::Move`'s doc comment) -- this proves
        // `process_run_logs` derives and tails that sibling path too, not
        // just the registered `log_path` itself.
        let (bench_runs, db) = {
            let dir = std::env::temp_dir()
                .join(format!("mcts_bench_moves_sibling_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "sibling-moves-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();
            let moves_path = run_dir.join("moves.jsonl");

            let reg_events = vec![start_event(
                run_id,
                "round_robin",
                "druid",
                99990,
                &log_path_str,
            )];
            let mut reg_content = String::new();
            for ev in &reg_events {
                reg_content.push_str(&ev.to_json_line());
                reg_content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

            // The main log only carries the match result...
            let match_result = LogRecord::MatchResult {
                seq: 1,
                strategy_a: "a".into(),
                strategy_b: "b".into(),
                outcome: "win_a".into(),
                winner: Some("a".into()),
                extra: None,
                cell_id: None,
                seed: None,
                trace_game_seq: None,
                metrics: None,
            };
            fs::write(&log_path, format!("{}\n", match_result.to_json_line())).unwrap();

            // ...while the moves live in the sibling file.
            let mv = LogRecord::Move {
                game_seq: 1,
                ply: 0,
                state: serde_json::json!({"board": []}),
                mv: None,
                player: None,
            };
            fs::write(&moves_path, format!("{}\n", mv.to_json_line())).unwrap();

            let db = duckdb::Connection::open_in_memory().unwrap();
            ensure_schema(&db).unwrap();

            (bench_runs, db)
        };

        ingest_once(&db, &bench_runs).unwrap();

        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM game_moves", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_ingest_moves() {
        let (bench_runs, db) = {
            let dir = std::env::temp_dir().join(format!("mcts_bench_moves_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let bench_runs = dir.join("bench-runs");
            fs::create_dir_all(&bench_runs).unwrap();

            let run_id = "moves-run";
            let run_dir = bench_runs.join(run_id);
            fs::create_dir_all(&run_dir).unwrap();
            let log_path = run_dir.join("log.jsonl");
            let log_path_str = log_path.to_string_lossy().to_string();

            let reg_events = vec![start_event(run_id, "smac3", "druid", 99994, &log_path_str)];
            let mut reg_content = String::new();
            for ev in &reg_events {
                reg_content.push_str(&ev.to_json_line());
                reg_content.push('\n');
            }
            fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

            let records = vec![
                LogRecord::Move {
                    game_seq: 7,
                    ply: 0,
                    state: serde_json::json!({"board": []}),
                    mv: None,
                    player: None,
                },
                LogRecord::Move {
                    game_seq: 7,
                    ply: 1,
                    state: serde_json::json!({"board": [1]}),
                    mv: Some(serde_json::json!({"cell": 0})),
                    player: Some("strong".into()),
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
            db.query_row("SELECT COUNT(*) FROM game_moves", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            2
        );

        let (state_str, mv_str, player): (String, Option<String>, Option<String>) = db
            .query_row(
                "SELECT CAST(state AS TEXT), CAST(mv AS TEXT), player \
                 FROM game_moves WHERE run_id = 'moves-run' AND game_seq = 7 AND ply = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        let state: serde_json::Value = serde_json::from_str(&state_str).unwrap();
        assert_eq!(state["board"][0], 1);
        let mv: serde_json::Value = serde_json::from_str(&mv_str.unwrap()).unwrap();
        assert_eq!(mv["cell"], 0);
        assert_eq!(player.as_deref(), Some("strong"));

        // Idempotent re-ingest should not duplicate rows.
        ingest_once(&db, &bench_runs).unwrap();
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM game_moves", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
    }

    #[test]
    fn test_registry_garbage_lines_are_skipped() {
        let dir =
            std::env::temp_dir().join(format!("mcts_bench_reg_garbage_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let ev = start_event(
            "garb-run",
            "round_robin",
            "druid",
            99991,
            "/tmp/nope/log.jsonl",
        );
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
            db.query_row("SELECT COUNT(*) FROM runs", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            1,
        );
    }

    #[test]
    fn test_experiment_cell_ingestion_is_idempotent_and_keeps_trace_mapping() {
        let dir = std::env::temp_dir().join(format!(
            "mcts_bench_experiment_ingest_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        let run_dir = bench_runs.join("experiment-run");
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        fs::write(
            bench_runs.join("registry.log"),
            format!(
                "{}\n",
                start_event(
                    "experiment-run",
                    "experiment",
                    "nim",
                    99991,
                    &log_path.to_string_lossy()
                )
                .to_json_line()
            ),
        )
        .unwrap();
        let records = [
            LogRecord::CellStarted {
                cell_id: "cell-1".into(),
            },
            LogRecord::MatchResult {
                seq: 1,
                strategy_a: "Candidate".into(),
                strategy_b: "Baseline".into(),
                outcome: "win_a".into(),
                winner: Some("Candidate".into()),
                extra: None,
                cell_id: Some("cell-1".into()),
                seed: Some(42),
                trace_game_seq: Some(177),
                metrics: Some(serde_json::json!({"outcome":"candidate_win","plies":3})),
            },
            LogRecord::MatchResult {
                seq: 2,
                strategy_a: "Baseline".into(),
                strategy_b: "Candidate".into(),
                outcome: "draw".into(),
                winner: None,
                extra: None,
                cell_id: Some("cell-1".into()),
                seed: Some(43),
                trace_game_seq: Some(178),
                metrics: Some(serde_json::json!({"outcome":"draw","plies":4})),
            },
            LogRecord::CellFinished {
                cell_id: "cell-1".into(),
                completed_games: 2,
            },
        ];
        fs::write(
            &log_path,
            records
                .iter()
                .map(|record| format!("{}\n", record.to_json_line()))
                .collect::<String>(),
        )
        .unwrap();
        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();
        db.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) VALUES ('experiment-run', 'experiment', 'nim', 'test', false, 'test', NULL, CURRENT_TIMESTAMP, 'running', ?1)", [&log_path.to_string_lossy()]).unwrap();
        db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games) VALUES ('experiment-run', 'cell-1', 'nim', 'null', 'candidate', 'Candidate', '{}', 'base', 'Baseline', '{}', '{\"kind\":\"iterations\",\"value\":1}', 1, 2)", []).unwrap();
        ingest_once(&db, &bench_runs).unwrap();
        ingest_once(&db, &bench_runs).unwrap();
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            db.query_row(
                "SELECT completed_games FROM experiment_cells WHERE run_id = 'experiment-run'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
            2
        );
        assert_eq!(
            db.query_row(
                "SELECT trace_game_seq FROM match_results WHERE seq = 1",
                [],
                |row| row.get::<_, u64>(0)
            )
            .unwrap(),
            177
        );
    }

    #[test]
    fn test_cell_failure_is_ingested_after_registry_stop() {
        let dir =
            std::env::temp_dir().join(format!("mcts_bench_cell_failure_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        let run_dir = bench_runs.join("failed-run");
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        fs::write(
            &log_path,
            format!(
                "{}\n",
                LogRecord::CellFailed {
                    cell_id: "cell-1".into(),
                    completed_games: 1,
                    error: "child failed".into()
                }
                .to_json_line()
            ),
        )
        .unwrap();
        fs::write(
            bench_runs.join("registry.log"),
            format!(
                "{}\n{}\n",
                start_event(
                    "failed-run",
                    "experiment",
                    "nim",
                    99991,
                    &log_path.to_string_lossy()
                )
                .to_json_line(),
                stop_event("failed-run", Some(1)).to_json_line()
            ),
        )
        .unwrap();
        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();
        process_registry(&db, &bench_runs.join("registry.log")).unwrap();
        db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games) VALUES ('failed-run', 'cell-1', 'nim', 'null', 'variant', 'Variant', '{}', 'baseline', 'Baseline', '{}', '{\"kind\":\"iterations\",\"value\":1}', 1, 2)", []).unwrap();
        process_run_logs(&db).unwrap();
        assert_eq!(
            db.query_row(
                "SELECT status FROM runs WHERE run_id = 'failed-run'",
                [],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
            "crashed"
        );
        assert_eq!(
            db.query_row(
                "SELECT status FROM experiment_cells WHERE run_id = 'failed-run'",
                [],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
            "failed"
        );
    }
}
