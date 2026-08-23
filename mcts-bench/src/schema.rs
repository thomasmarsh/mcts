//! DuckDB schema, connection helpers, and row types for the benchmark
//! database.  Only the `server` process ever opens `bench.duckdb` read-write;
//! `bin/bench` and the Python tuner harness never link against DuckDB at all.

pub const CREATE_TABLES: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS logical_runs (
        logical_run_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        project_id TEXT,
        experiment_id TEXT,
        created_at TIMESTAMP NOT NULL,
        current_attempt_id TEXT NOT NULL,
        version UINTEGER NOT NULL DEFAULT 0
    )",
    "CREATE TABLE IF NOT EXISTS runs (
        run_id      TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        game        TEXT,
        project_id  TEXT,
        experiment_id TEXT,
        experiment_spec JSON,
        label       TEXT,
        config      JSON,
        git_sha     TEXT NOT NULL,
        git_dirty   BOOLEAN NOT NULL,
        host        TEXT NOT NULL,
        pid         INTEGER,
        started_at  TIMESTAMP NOT NULL,
        ended_at    TIMESTAMP,
        status      TEXT NOT NULL DEFAULT 'running',
        log_path    TEXT NOT NULL,
        exit_code   INTEGER,
        logical_run_id TEXT,
        parent_attempt_id TEXT,
        attempt_ordinal UINTEGER,
        attempt_phase TEXT,
        attempt_stop_reason TEXT,
        attempt_process_observed BOOLEAN,
        attempt_signal_observed BOOLEAN,
        attempt_exit_kind TEXT,
        attempt_exit_code INTEGER,
        attempt_version UINTEGER
    )",
    "CREATE TABLE IF NOT EXISTS match_results (
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        seq         INTEGER NOT NULL,
        ts          TIMESTAMP NOT NULL,
        strategy_a  TEXT NOT NULL,
        strategy_b  TEXT NOT NULL,
        outcome     TEXT NOT NULL,
        winner      TEXT,
        extra       JSON,
        cell_id     TEXT,
        seed        UBIGINT,
        trace_game_seq UBIGINT,
        metrics     JSON,
        PRIMARY KEY (run_id, seq)
    )",
    "CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        archived BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(project_id),
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        spec JSON NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS experiment_cells (
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        cell_id TEXT NOT NULL,
        cell_seed UBIGINT,
        game TEXT NOT NULL,
        game_config JSON NOT NULL,
        variant_id TEXT NOT NULL,
        variant_label TEXT NOT NULL,
        candidate_config JSON NOT NULL,
        baseline_id TEXT NOT NULL,
        baseline_label TEXT NOT NULL,
        baseline_config JSON NOT NULL,
        budget JSON NOT NULL,
        rounds INTEGER NOT NULL,
        planned_games UBIGINT NOT NULL,
        completed_games UBIGINT NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        started_at TIMESTAMP,
        ended_at TIMESTAMP,
        error TEXT,
        PRIMARY KEY (run_id, cell_id)
    )",
    "CREATE TABLE IF NOT EXISTS trials (
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        trial_id    INTEGER NOT NULL,
        ts          TIMESTAMP NOT NULL,
        config      JSON NOT NULL,
        seed        INTEGER,
        cost        DOUBLE,
        extra       JSON,
        PRIMARY KEY (run_id, trial_id)
    )",
    "CREATE TABLE IF NOT EXISTS incumbents (
        run_id      TEXT PRIMARY KEY REFERENCES runs(run_id),
        ts          TIMESTAMP NOT NULL,
        config      JSON NOT NULL,
        cost        DOUBLE NOT NULL,
        extra       JSON
    )",
    "CREATE TABLE IF NOT EXISTS game_moves (
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        game_seq    BIGINT NOT NULL,
        ply         INTEGER NOT NULL,
        ts          TIMESTAMP NOT NULL,
        state       JSON NOT NULL,
        mv          JSON,
        player      TEXT,
        PRIMARY KEY (run_id, game_seq, ply)
    )",
    "CREATE TABLE IF NOT EXISTS _ingest_cursor (
        log_path    TEXT PRIMARY KEY,
        byte_offset BIGINT NOT NULL DEFAULT 0,
        updated_at  TIMESTAMP NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS attempt_events (
        attempt_id TEXT NOT NULL REFERENCES runs(run_id),
        attempt_version UINTEGER NOT NULL,
        event_key TEXT NOT NULL,
        event_type TEXT NOT NULL,
        stop_reason TEXT,
        exit_kind TEXT,
        exit_code INTEGER,
        observed_at TIMESTAMP NOT NULL,
        PRIMARY KEY (attempt_id, event_key),
        UNIQUE (attempt_id, attempt_version)
    )",
    "CREATE TABLE IF NOT EXISTS projects_launches (
        attempt_id TEXT PRIMARY KEY REFERENCES runs(run_id),
        logical_run_id TEXT NOT NULL,
        parent_attempt_id TEXT,
        launch_nonce TEXT NOT NULL,
        workload_argv JSON NOT NULL,
        lifecycle_path TEXT NOT NULL,
        stdout_path TEXT NOT NULL,
        stderr_path TEXT NOT NULL,
        wrapper_pid UBIGINT,
        process_group_id UBIGINT,
        launch_result TEXT,
        launch_diagnostic TEXT
    )",
    "CREATE TABLE IF NOT EXISTS tuning_sessions (
        session_id TEXT PRIMARY KEY,
        status TEXT NOT NULL CHECK (status IN ('active', 'idle')),
        manifest JSON NOT NULL,
        manifest_fingerprint TEXT,
        target_trial_count BIGINT,
        created_at TIMESTAMP NOT NULL,
        last_sequence BIGINT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS tuning_attempts (
        attempt_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES tuning_sessions(session_id),
        bench_run_id TEXT REFERENCES runs(run_id),
        status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'stopped')),
        started_at TIMESTAMP NOT NULL,
        ended_at TIMESTAMP,
        failure TEXT
    )",
    "CREATE TABLE IF NOT EXISTS tuning_trials (
        session_id TEXT NOT NULL REFERENCES tuning_sessions(session_id),
        trial_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL REFERENCES tuning_attempts(attempt_id),
        trial_number BIGINT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'complete', 'failed', 'pruned', 'cancelled')),
        config JSON,
        created_at TIMESTAMP NOT NULL,
        started_at TIMESTAMP,
        ended_at TIMESTAMP,
        score DOUBLE,
        mu DOUBLE,
        sigma DOUBLE,
        failure TEXT,
        PRIMARY KEY (session_id, trial_id)
    )",
    "CREATE TABLE IF NOT EXISTS tuning_lifecycle_events (
        event_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL,
        session_sequence BIGINT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        event_type TEXT NOT NULL,
        payload JSON NOT NULL,
        raw JSON NOT NULL,
        source_path TEXT NOT NULL,
        source_offset BIGINT NOT NULL,
        accepted BOOLEAN,
        rejection_reason TEXT
    )",
];

pub fn ensure_schema(conn: &duckdb::Connection) -> duckdb::Result<()> {
    for ddl in CREATE_TABLES {
        conn.execute_batch(ddl)?;
    }
    // ALTERs are intentionally idempotent and keep existing legacy rows
    // untouched. CREATE TABLE above covers fresh databases.
    for ddl in [
        "ALTER TABLE runs ADD COLUMN project_id TEXT",
        "ALTER TABLE runs ADD COLUMN experiment_id TEXT",
        "ALTER TABLE runs ADD COLUMN experiment_spec JSON",
        "ALTER TABLE runs ALTER COLUMN game DROP NOT NULL",
        "ALTER TABLE match_results ADD COLUMN cell_id TEXT",
        "ALTER TABLE match_results ADD COLUMN seed UBIGINT",
        "ALTER TABLE match_results ADD COLUMN trace_game_seq UBIGINT",
        "ALTER TABLE match_results ADD COLUMN metrics JSON",
        "ALTER TABLE experiment_cells ADD COLUMN cell_seed UBIGINT",
        "ALTER TABLE runs ADD COLUMN logical_run_id TEXT",
        "ALTER TABLE runs ADD COLUMN parent_attempt_id TEXT",
        "ALTER TABLE runs ADD COLUMN attempt_ordinal UINTEGER",
        "ALTER TABLE runs ADD COLUMN attempt_phase TEXT",
        "ALTER TABLE runs ADD COLUMN attempt_stop_reason TEXT",
        "ALTER TABLE runs ADD COLUMN attempt_process_observed BOOLEAN",
        "ALTER TABLE runs ADD COLUMN attempt_signal_observed BOOLEAN",
        "ALTER TABLE runs ADD COLUMN attempt_exit_kind TEXT",
        "ALTER TABLE runs ADD COLUMN attempt_exit_code INTEGER",
        "ALTER TABLE runs ADD COLUMN attempt_version UINTEGER",
    ] {
        let _ = conn.execute_batch(ddl);
    }
    Ok(())
}

pub fn open(path: impl AsRef<std::path::Path>) -> duckdb::Result<duckdb::Connection> {
    let conn = duckdb::Connection::open(path.as_ref())?;
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
        let conn = duckdb::Connection::open_in_memory().unwrap();
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
            "attempt_events",
            "tuning_sessions",
            "tuning_attempts",
            "tuning_trials",
            "tuning_lifecycle_events",
        ] {
            assert!(tables.iter().any(|t| t == want), "missing table: {want}");
        }
        let cell_seed: (String, bool) = conn
            .query_row(
                "SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = 'experiment_cells' AND column_name = 'cell_seed'",
                [],
                |row| Ok((row.get(0)?, row.get::<_, String>(1)? == "YES")),
            )
            .unwrap();
        assert_eq!(cell_seed, ("cell_seed".into(), true));
        let identity_columns: Vec<(String, String)> = conn
            .prepare(
                "SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = 'runs' AND column_name IN ('logical_run_id', 'parent_attempt_id', 'attempt_ordinal') ORDER BY column_name",
            )
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(
            identity_columns,
            vec![
                ("attempt_ordinal".into(), "YES".into()),
                ("logical_run_id".into(), "YES".into()),
                ("parent_attempt_id".into(), "YES".into()),
            ]
        );
        let attempt_columns: Vec<String> = conn
            .prepare(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'runs' AND column_name IN ('attempt_phase', 'attempt_stop_reason', 'attempt_process_observed', 'attempt_signal_observed', 'attempt_exit_kind', 'attempt_exit_code', 'attempt_version') ORDER BY column_name",
            )
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(
            attempt_columns,
            vec![
                "attempt_exit_code",
                "attempt_exit_kind",
                "attempt_phase",
                "attempt_process_observed",
                "attempt_signal_observed",
                "attempt_stop_reason",
                "attempt_version",
            ]
        );
        let event_columns: Vec<(String, String)> = conn
            .prepare(
                "SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = 'attempt_events' ORDER BY ordinal_position",
            )
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(
            event_columns,
            vec![
                ("attempt_id".into(), "NO".into()),
                ("attempt_version".into(), "NO".into()),
                ("event_key".into(), "NO".into()),
                ("event_type".into(), "NO".into()),
                ("stop_reason".into(), "YES".into()),
                ("exit_kind".into(), "YES".into()),
                ("exit_code".into(), "YES".into()),
                ("observed_at".into(), "NO".into()),
            ]
        );
    }

    #[test]
    fn open_creates_on_disk_and_is_idempotent() {
        let dir = std::env::temp_dir().join("mcts_bench_test_open");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let db_path = dir.join("test.duckdb");

        let conn1 = open(&db_path).unwrap();
        let row_count: i64 = conn1
            .query_row("SELECT COUNT(*) FROM runs", [], |row| row.get(0))
            .unwrap();
        assert_eq!(row_count, 0);

        let conn2 = open(&db_path).unwrap();
        let row_count: i64 = conn2
            .query_row("SELECT COUNT(*) FROM runs", [], |row| row.get(0))
            .unwrap();
        assert_eq!(row_count, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrates_the_legacy_run_and_match_shapes_without_rewriting_rows() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, game TEXT NOT NULL,
                label TEXT, config JSON, git_sha TEXT NOT NULL, git_dirty BOOLEAN NOT NULL,
                host TEXT NOT NULL, pid INTEGER, started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP, status TEXT NOT NULL, log_path TEXT NOT NULL, exit_code INTEGER
            );
            CREATE TABLE match_results (
                run_id TEXT NOT NULL REFERENCES runs(run_id), seq INTEGER NOT NULL,
                ts TIMESTAMP NOT NULL, strategy_a TEXT NOT NULL, strategy_b TEXT NOT NULL,
                outcome TEXT NOT NULL, winner TEXT, extra JSON, PRIMARY KEY (run_id, seq)
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
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, game TEXT, git_sha TEXT NOT NULL, git_dirty BOOLEAN NOT NULL, host TEXT NOT NULL, started_at TIMESTAMP NOT NULL, status TEXT NOT NULL, log_path TEXT NOT NULL);
             CREATE TABLE experiment_cells (run_id TEXT NOT NULL, cell_id TEXT NOT NULL, game TEXT NOT NULL, game_config JSON NOT NULL, variant_id TEXT NOT NULL, variant_label TEXT NOT NULL, candidate_config JSON NOT NULL, baseline_id TEXT NOT NULL, baseline_label TEXT NOT NULL, baseline_config JSON NOT NULL, budget JSON NOT NULL, rounds INTEGER NOT NULL, planned_games UBIGINT NOT NULL, completed_games UBIGINT NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending', PRIMARY KEY(run_id, cell_id));
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
}
