//! DuckDB schema, connection helpers, and row types for the benchmark
//! database.  Only the `server` process ever opens `bench.duckdb` read-write;
//! `bin/bench` and the Python SMAC3 harness never link against DuckDB at all.

pub const CREATE_TABLES: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS runs (
        run_id      TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        game        TEXT NOT NULL,
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
        PRIMARY KEY (run_id, seq)
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

        for want in &["runs", "match_results", "trials", "_ingest_cursor"] {
            assert!(
                tables.iter().any(|t| t == want),
                "missing table: {want}"
            );
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
}