use duckdb::{params, Connection};

use crate::launch::iso_timestamp;

use super::IngestError;

pub(crate) fn get_cursor(conn: &Connection, log_path: &str) -> Result<u64, IngestError> {
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

pub(crate) fn set_cursor(
    conn: &Connection,
    log_path: &str,
    byte_offset: u64,
) -> Result<(), IngestError> {
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
