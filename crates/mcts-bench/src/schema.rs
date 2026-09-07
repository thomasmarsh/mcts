//! SQLite schema, connection helpers, and row types for the benchmark
//! database.  Only the `server` process ever opens `bench.sqlite` read-write;
//! `bin/bench` and the Python tuner harness never link against SQLite at all.

pub const CREATE_TABLES: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS logical_runs (
        logical_run_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        project_id TEXT,
        experiment_id TEXT,
        created_at TEXT NOT NULL,
        current_attempt_id TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 0
    )",
    "CREATE TABLE IF NOT EXISTS runs (
        run_id      TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        game        TEXT,
        project_id  TEXT,
        experiment_id TEXT,
        experiment_spec TEXT,
        label       TEXT,
        config      TEXT,
        git_sha     TEXT NOT NULL,
        git_dirty   INTEGER NOT NULL,
        host        TEXT NOT NULL,
        pid         INTEGER,
        started_at  TEXT NOT NULL,
        ended_at    TEXT,
        status      TEXT NOT NULL DEFAULT 'running',
        log_path    TEXT NOT NULL,
        exit_code   INTEGER,
        logical_run_id TEXT,
        parent_attempt_id TEXT,
        attempt_ordinal INTEGER,
        attempt_phase TEXT,
        attempt_stop_reason TEXT,
        attempt_process_observed INTEGER,
        attempt_signal_observed INTEGER,
        attempt_exit_kind TEXT,
        attempt_exit_code INTEGER,
        attempt_version INTEGER
    )",
    "CREATE TABLE IF NOT EXISTS match_results (
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        seq         INTEGER NOT NULL,
        ts          TEXT NOT NULL,
        strategy_a  TEXT NOT NULL,
        strategy_b  TEXT NOT NULL,
        outcome     TEXT NOT NULL,
        winner      TEXT,
        extra       TEXT,
        cell_id     TEXT,
        seed        INTEGER,
        trace_game_seq INTEGER,
        metrics     TEXT,
        PRIMARY KEY (run_id, seq)
    )",
    "CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        archived INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(project_id),
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        spec TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS experiment_cells (
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        cell_id TEXT NOT NULL,
        cell_seed INTEGER,
        game TEXT NOT NULL,
        game_config TEXT NOT NULL,
        variant_id TEXT NOT NULL,
        variant_label TEXT NOT NULL,
        candidate_config TEXT NOT NULL,
        baseline_id TEXT NOT NULL,
        baseline_label TEXT NOT NULL,
        baseline_config TEXT NOT NULL,
        budget TEXT NOT NULL,
        rounds INTEGER NOT NULL,
        planned_games INTEGER NOT NULL,
        completed_games INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        started_at TEXT,
        ended_at TEXT,
        error TEXT,
        PRIMARY KEY (run_id, cell_id)
    )",
    "CREATE TABLE IF NOT EXISTS trials (
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        trial_id    INTEGER NOT NULL,
        ts          TEXT NOT NULL,
        config      TEXT NOT NULL,
        seed        INTEGER,
        cost        REAL,
        extra       TEXT,
        PRIMARY KEY (run_id, trial_id)
    )",
    "CREATE TABLE IF NOT EXISTS incumbents (
        run_id      TEXT PRIMARY KEY REFERENCES runs(run_id),
        ts          TEXT NOT NULL,
        config      TEXT NOT NULL,
        cost        REAL NOT NULL,
        extra       TEXT
    )",
    "CREATE TABLE IF NOT EXISTS game_moves (
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        game_seq    INTEGER NOT NULL,
        ply         INTEGER NOT NULL,
        ts          TEXT NOT NULL,
        trace_schema_version INTEGER,
        state       TEXT NOT NULL,
        mv          TEXT,
        player      TEXT,
        search_report TEXT,
        search_status TEXT,
        search_completed_iterations INTEGER,
        search_elapsed_ms REAL,
        search_nodes INTEGER,
        search_mean_depth REAL,
        search_max_depth INTEGER,
        search_tt_hit_ratio REAL,
        PRIMARY KEY (run_id, game_seq, ply)
    )",
    "CREATE TABLE IF NOT EXISTS _ingest_cursor (
        log_path    TEXT PRIMARY KEY,
        byte_offset INTEGER NOT NULL DEFAULT 0,
        updated_at  TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS artifact_roots (
        physical_run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
        artifact_root TEXT NOT NULL UNIQUE,
        attempt_id TEXT,
        attempt_digest TEXT,
        descriptor_watermark TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'active',
        integrity_error TEXT,
        updated_at TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS artifact_descriptors (
        physical_run_id TEXT NOT NULL REFERENCES runs(run_id),
        descriptor_filename TEXT NOT NULL,
        descriptor_path TEXT NOT NULL,
        task_id TEXT,
        task_sequence INTEGER,
        descriptor_digest TEXT,
        task_root TEXT,
        status TEXT NOT NULL,
        integrity_error TEXT,
        PRIMARY KEY (physical_run_id, descriptor_filename)
    )",
    "CREATE TABLE IF NOT EXISTS artifact_tasks (
        physical_run_id TEXT NOT NULL REFERENCES runs(run_id),
        task_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL,
        task_sequence INTEGER NOT NULL,
        descriptor_path TEXT NOT NULL,
        task_root TEXT NOT NULL,
        trace_path TEXT NOT NULL,
        descriptor_digest TEXT NOT NULL,
        completion_digest TEXT,
        status TEXT NOT NULL,
        integrity_error TEXT,
        completed_at TEXT,
        PRIMARY KEY (physical_run_id, task_id)
    )",
    "CREATE TABLE IF NOT EXISTS _artifact_trace_cursor (
        physical_run_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        trace_path TEXT NOT NULL,
        byte_offset INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (physical_run_id, task_id)
    )",
    "CREATE TABLE IF NOT EXISTS attempt_events (
        attempt_id TEXT NOT NULL REFERENCES runs(run_id),
        attempt_version INTEGER NOT NULL,
        event_key TEXT NOT NULL,
        event_type TEXT NOT NULL,
        stop_reason TEXT,
        exit_kind TEXT,
        exit_code INTEGER,
        observed_at TEXT NOT NULL,
        PRIMARY KEY (attempt_id, event_key),
        UNIQUE (attempt_id, attempt_version)
    )",
    "CREATE TABLE IF NOT EXISTS projects_launches (
        attempt_id TEXT PRIMARY KEY REFERENCES runs(run_id),
        logical_run_id TEXT NOT NULL,
        parent_attempt_id TEXT,
        launch_nonce TEXT NOT NULL,
        workload_argv TEXT NOT NULL,
        lifecycle_path TEXT NOT NULL,
        stdout_path TEXT NOT NULL,
        stderr_path TEXT NOT NULL,
        wrapper_pid INTEGER,
        process_group_id INTEGER,
        launch_result TEXT,
        launch_diagnostic TEXT
    )",
];

pub fn ensure_schema(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "foreign_keys", "ON")?;
    for ddl in CREATE_TABLES {
        conn.execute_batch(ddl)?;
    }
    // Check each legacy column before altering its table. Ignoring duplicate
    // `ALTER TABLE` failures left partially upgraded databases behind when a
    // previous migration had added only an earlier column in this list.
    ensure_columns(
        conn,
        "runs",
        &[
            ("project_id", "TEXT"),
            ("experiment_id", "TEXT"),
            ("experiment_spec", "TEXT"),
            ("logical_run_id", "TEXT"),
            ("parent_attempt_id", "TEXT"),
            ("attempt_ordinal", "INTEGER"),
            ("attempt_phase", "TEXT"),
            ("attempt_stop_reason", "TEXT"),
            ("attempt_process_observed", "INTEGER"),
            ("attempt_signal_observed", "INTEGER"),
            ("attempt_exit_kind", "TEXT"),
            ("attempt_exit_code", "INTEGER"),
            ("attempt_version", "INTEGER"),
        ],
    )?;
    ensure_columns(
        conn,
        "match_results",
        &[
            ("cell_id", "TEXT"),
            ("seed", "INTEGER"),
            ("trace_game_seq", "INTEGER"),
            ("metrics", "TEXT"),
        ],
    )?;
    ensure_columns(conn, "experiment_cells", &[("cell_seed", "INTEGER")])?;
    ensure_columns(
        conn,
        "game_moves",
        &[
            ("trace_schema_version", "INTEGER"),
            ("search_report", "TEXT"),
            ("search_status", "TEXT"),
            ("search_completed_iterations", "INTEGER"),
            ("search_elapsed_ms", "REAL"),
            ("search_nodes", "INTEGER"),
            ("search_mean_depth", "REAL"),
            ("search_max_depth", "INTEGER"),
            ("search_tt_hit_ratio", "REAL"),
        ],
    )?;
    Ok(())
}

fn ensure_columns(
    conn: &rusqlite::Connection,
    table: &str,
    columns: &[(&str, &str)],
) -> rusqlite::Result<()> {
    for (column, definition) in columns {
        let mut statement = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let exists = statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .iter()
            .any(|name| name == column);
        if !exists {
            conn.execute_batch(&format!(
                "ALTER TABLE {table} ADD COLUMN {column} {definition}"
            ))?;
        }
    }
    Ok(())
}

pub fn open(path: impl AsRef<std::path::Path>) -> rusqlite::Result<rusqlite::Connection> {
    let conn = rusqlite::Connection::open(path.as_ref())?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    ensure_schema(&conn)?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    type LegacyShape = (
        String,
        Option<String>,
        Option<String>,
        Option<u64>,
        Option<u64>,
        Option<String>,
    );
    type TypedProjection = (
        Option<String>,
        Option<String>,
        Option<bool>,
        Option<bool>,
        Option<String>,
        Option<i32>,
        Option<u64>,
    );

    #[test]
    fn fresh_in_memory_db_creates_all_tables() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();

        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();

        for want in &[
            "runs",
            "logical_runs",
            "projects",
            "experiments",
            "experiment_cells",
            "match_results",
            "trials",
            "incumbents",
            "game_moves",
            "_ingest_cursor",
            "artifact_roots",
            "artifact_descriptors",
            "artifact_tasks",
            "_artifact_trace_cursor",
            "attempt_events",
        ] {
            assert!(tables.iter().any(|t| t == want), "missing table: {want}");
        }
        let experiment_columns = table_columns(&conn, "experiment_cells");
        assert!(experiment_columns.contains(&"cell_seed".into()));
        let move_columns = table_columns(&conn, "game_moves");
        for column in [
            "search_completed_iterations",
            "search_elapsed_ms",
            "search_max_depth",
            "search_mean_depth",
            "search_nodes",
            "search_report",
            "search_status",
            "search_tt_hit_ratio",
            "trace_schema_version",
        ] {
            assert!(
                move_columns.contains(&column.into()),
                "missing column: {column}"
            );
        }
        let run_columns = table_columns(&conn, "runs");
        assert!(run_columns.contains(&"logical_run_id".into()));
        assert!(run_columns.contains(&"parent_attempt_id".into()));
        assert!(run_columns.contains(&"attempt_ordinal".into()));
        for column in [
            "attempt_exit_code",
            "attempt_exit_kind",
            "attempt_phase",
            "attempt_process_observed",
            "attempt_signal_observed",
            "attempt_stop_reason",
            "attempt_version",
        ] {
            assert!(
                run_columns.contains(&column.into()),
                "missing column: {column}"
            );
        }
        assert_eq!(
            table_columns(&conn, "attempt_events"),
            vec![
                "attempt_id",
                "attempt_version",
                "event_key",
                "event_type",
                "stop_reason",
                "exit_kind",
                "exit_code",
                "observed_at",
            ]
        );
    }

    fn table_columns(conn: &rusqlite::Connection, table: &str) -> Vec<String> {
        conn.prepare(&format!("PRAGMA table_info({table})"))
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    #[test]
    fn open_creates_on_disk_and_is_idempotent() {
        let dir = std::env::temp_dir().join("mcts_bench_test_open");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let db_path = dir.join("test.sqlite");

        let conn1 = open(&db_path).unwrap();
        let row_count: i64 = conn1
            .query_row("SELECT COUNT(*) FROM runs", [], |row| row.get(0))
            .unwrap();
        assert_eq!(row_count, 0);

        let conn2 = open(&db_path).unwrap();
        let journal_mode: String = conn2
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(journal_mode, "wal");
        let foreign_keys: i64 = conn2
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .unwrap();
        assert_eq!(foreign_keys, 1);
        let row_count: i64 = conn2
            .query_row("SELECT COUNT(*) FROM runs", [], |row| row.get(0))
            .unwrap();
        assert_eq!(row_count, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn foreign_keys_and_text_payloads_round_trip() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();
        let error = conn
            .execute(
                "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome) \
                 VALUES ('missing', 1, '2026-01-01T00:00:00Z', 'a', 'b', 'draw')",
                [],
            )
            .unwrap_err();
        assert!(matches!(error, rusqlite::Error::SqliteFailure(_, _)));

        conn.execute(
            "INSERT INTO runs (run_id, kind, git_sha, git_dirty, host, started_at, log_path, config) \
             VALUES ('run', 'tuner', 'sha', 0, 'host', '2026-01-01T00:00:00Z', '/tmp/log', ?1)",
            rusqlite::params![r#"{"budget": 3}"#],
        )
        .unwrap();
        let payload: String = conn
            .query_row("SELECT config FROM runs WHERE run_id = 'run'", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(payload, r#"{"budget": 3}"#);
    }

    #[test]
    fn migrates_the_legacy_run_and_match_shapes_without_rewriting_rows() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, game TEXT NOT NULL,
                label TEXT, config TEXT, git_sha TEXT NOT NULL, git_dirty INTEGER NOT NULL,
                host TEXT NOT NULL, pid INTEGER, started_at TEXT NOT NULL,
                ended_at TEXT, status TEXT NOT NULL, log_path TEXT NOT NULL, exit_code INTEGER
            );
            CREATE TABLE match_results (
                run_id TEXT NOT NULL REFERENCES runs(run_id), seq INTEGER NOT NULL,
                ts TEXT NOT NULL, strategy_a TEXT NOT NULL, strategy_b TEXT NOT NULL,
                outcome TEXT NOT NULL, winner TEXT, extra TEXT, PRIMARY KEY (run_id, seq)
            );
            INSERT INTO runs VALUES ('legacy', 'round_robin', 'nim', 'old', NULL, 'sha', false, 'host', NULL, CURRENT_TIMESTAMP, NULL, 'completed', '/tmp/log', 0);
            INSERT INTO match_results VALUES ('legacy', 1, CURRENT_TIMESTAMP, 'a', 'b', 'draw', NULL, NULL);",
        ).unwrap();

        ensure_schema(&conn).unwrap();

        let row: LegacyShape = conn
            .query_row(
                "SELECT game, project_id, experiment_id, seed, trace_game_seq, metrics FROM runs LEFT JOIN match_results USING (run_id)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
            )
            .unwrap();
        assert_eq!(row.0, "nim");
        assert_eq!(row.1, None);
        assert_eq!(row.2, None);
        assert_eq!(row.3, None);
        assert_eq!(row.4, None);
        assert_eq!(row.5, None);

        let linkage: (Option<String>, Option<String>, Option<u64>) = conn
            .query_row(
                "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = 'legacy'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(linkage, (None, None, None));
        let typed: TypedProjection = conn
            .query_row(
                "SELECT attempt_phase, attempt_stop_reason, attempt_process_observed, attempt_signal_observed, attempt_exit_kind, attempt_exit_code, attempt_version FROM runs WHERE run_id = 'legacy'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?)),
            )
            .unwrap();
        assert_eq!(typed, (None, None, None, None, None, None, None));
    }

    #[test]
    fn migrates_legacy_experiment_cells_with_a_null_seed() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, game TEXT, git_sha TEXT NOT NULL, git_dirty INTEGER NOT NULL, host TEXT NOT NULL, started_at TEXT NOT NULL, status TEXT NOT NULL, log_path TEXT NOT NULL);
             CREATE TABLE experiment_cells (run_id TEXT NOT NULL, cell_id TEXT NOT NULL, game TEXT NOT NULL, game_config TEXT NOT NULL, variant_id TEXT NOT NULL, variant_label TEXT NOT NULL, candidate_config TEXT NOT NULL, baseline_id TEXT NOT NULL, baseline_label TEXT NOT NULL, baseline_config TEXT NOT NULL, budget TEXT NOT NULL, rounds INTEGER NOT NULL, planned_games INTEGER NOT NULL, completed_games INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending', PRIMARY KEY(run_id, cell_id));
             INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games) VALUES ('run', 'cell-000001', 'nim', '{}', 'v', 'V', '{}', 'b', 'B', '{}', '{}', 1, 2);",
        )
        .unwrap();
        ensure_schema(&conn).unwrap();
        let seed: Option<u64> = conn
            .query_row(
                "SELECT cell_seed FROM experiment_cells WHERE run_id = 'run'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(seed, None);
    }

    #[test]
    fn upgrades_legacy_move_rows_without_backfilling_search_evidence() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, game TEXT, git_sha TEXT NOT NULL, git_dirty INTEGER NOT NULL, host TEXT NOT NULL, started_at TEXT NOT NULL, status TEXT NOT NULL, log_path TEXT NOT NULL);
             CREATE TABLE game_moves (run_id TEXT NOT NULL, game_seq INTEGER NOT NULL, ply INTEGER NOT NULL, ts TEXT NOT NULL, state TEXT NOT NULL, mv TEXT, player TEXT, PRIMARY KEY (run_id, game_seq, ply));
             INSERT INTO runs VALUES ('legacy', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'completed', '/tmp/log');
             INSERT INTO game_moves VALUES ('legacy', 1, 0, CURRENT_TIMESTAMP, '{}', NULL, NULL);",
        )
        .unwrap();

        ensure_schema(&conn).unwrap();

        let row: (Option<u32>, Option<String>, Option<u64>) = conn
            .query_row(
                "SELECT trace_schema_version, search_status, search_completed_iterations FROM game_moves WHERE run_id = 'legacy'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(row, (None, None, None));
    }
}
