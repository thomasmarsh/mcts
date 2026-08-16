//! DuckDB schema, connection helpers, and row types for the benchmark
//! database.  Only the `server` process ever opens `bench.duckdb` read-write;
//! `bin/bench` and the Python SMAC3 harness never link against DuckDB at all.

pub const CREATE_TABLES: &[&str] = &[
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
        exit_code   INTEGER
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
            "projects",
            "experiments",
            "experiment_cells",
            "match_results",
            "trials",
            "incumbents",
            "game_moves",
            "_ingest_cursor",
        ] {
            assert!(tables.iter().any(|t| t == want), "missing table: {want}");
        }
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

        let row: (String, Option<String>, Option<String>, Option<u64>, Option<u64>, Option<String>) = conn
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
    }
}
