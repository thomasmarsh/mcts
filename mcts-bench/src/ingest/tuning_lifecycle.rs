use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use duckdb::Connection;

use crate::tuning_lifecycle::TuningLifecycleEvent;
use crate::tuning_store;

use super::cursor::{get_cursor, set_cursor};
use super::IngestError;

pub(super) fn process(conn: &Connection) -> Result<(), IngestError> {
    let mut statement =
        conn.prepare("SELECT source_path, bench_run_id FROM tuning_lifecycle_sources")?;
    let mut sources: Vec<(String, Option<String>)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(Result::ok)
        .collect();
    let registered: std::collections::HashSet<String> =
        sources.iter().map(|(path, _)| path.clone()).collect();
    let mut legacy = conn.prepare(
        "SELECT run_id, log_path FROM runs WHERE kind = 'tuner' AND status IN ('starting', 'running', 'completed', 'crashed', 'stopped')",
    )?;
    for (run_id, log_path) in legacy
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(Result::ok)
    {
        let path = Path::new(&log_path).with_file_name("lifecycle.jsonl");
        let source_path = canonical_source_path(&path);
        if !registered.contains(&source_path) {
            sources.push((source_path, Some(run_id)));
        }
    }
    for (source_path, fallback_bench_run_id) in sources {
        process_one(
            conn,
            fallback_bench_run_id.as_deref(),
            &PathBuf::from(source_path),
        )?;
    }
    Ok(())
}

fn canonical_source_path(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
        })
        .to_string_lossy()
        .into_owned()
}

fn process_one(
    conn: &Connection,
    fallback_bench_run_id: Option<&str>,
    path: &Path,
) -> Result<(), IngestError> {
    if !path.exists() {
        return Ok(());
    }
    let source_path = canonical_source_path(path);
    let mut offset = get_cursor(conn, &source_path)?;
    let file_len = fs::metadata(path)?.len();
    if file_len <= offset {
        return Ok(());
    }
    let mut file = fs::File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(file);
    offset = process_complete_records(
        conn,
        fallback_bench_run_id,
        &source_path,
        &mut reader,
        offset,
    )?;
    set_cursor(conn, &source_path, offset)
}

fn process_complete_records(
    conn: &Connection,
    fallback_bench_run_id: Option<&str>,
    source_path: &str,
    reader: &mut BufReader<fs::File>,
    mut offset: u64,
) -> Result<u64, IngestError> {
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        let count = reader.read_until(b'\n', &mut bytes)?;
        if count == 0 {
            break;
        }
        if bytes.last() != Some(&b'\n') {
            break;
        }
        let record_offset = offset;
        offset += count as u64;
        bytes.pop();
        apply_record(
            conn,
            fallback_bench_run_id,
            source_path,
            record_offset,
            &bytes,
        )?;
    }
    Ok(offset)
}

fn apply_record(
    conn: &Connection,
    fallback_bench_run_id: Option<&str>,
    source_path: &str,
    record_offset: u64,
    bytes: &[u8],
) -> Result<(), IngestError> {
    let Ok(event) = serde_json::from_slice::<TuningLifecycleEvent>(bytes) else {
        return Ok(());
    };
    let transaction = conn.unchecked_transaction()?;
    let disposition = tuning_store::apply_event(
        &transaction,
        &event,
        fallback_bench_run_id,
        source_path,
        record_offset,
    )
    .map_err(|error| duckdb::Error::ToSqlConversionFailure(Box::new(error)))?;
    report_disposition(&event, source_path, disposition);
    transaction.commit()?;
    Ok(())
}

fn report_disposition(
    event: &TuningLifecycleEvent,
    source_path: &str,
    disposition: tuning_store::ApplyDisposition,
) {
    match disposition {
        tuning_store::ApplyDisposition::Rejected => eprintln!(
            "rejected tuning lifecycle event {} from {}",
            event.event_id.as_str(),
            source_path
        ),
        tuning_store::ApplyDisposition::Conflict => eprintln!(
            "conflicting tuning lifecycle event {} from {}",
            event.event_id.as_str(),
            source_path
        ),
        tuning_store::ApplyDisposition::Applied | tuning_store::ApplyDisposition::Replay => {}
    }
}
