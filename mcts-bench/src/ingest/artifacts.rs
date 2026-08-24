//! Discovery and projection of immutable, task-partitioned tuner artifacts.
//!
//! These tables are a rebuildable view of files owned by the tuner. They do
//! not participate in scheduling: a descriptor is merely observed after its
//! coordinator has committed it, and a task becomes terminal only after its
//! worker's completion manifest validates every immutable member.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use duckdb::{params, Connection};
use serde_json::Value;

use crate::launch::iso_timestamp;

use super::logs::process_complete_trace_file;
use super::IngestError;

const DISCOVERY_BATCH: usize = 256;
const TASK_ID_LENGTH: usize = "task-".len() + 32;

pub(super) fn process(conn: &Connection) -> Result<(), IngestError> {
    let mut statement = conn.prepare("SELECT run_id, log_path FROM runs WHERE kind = 'tuner'")?;
    let sources: Vec<(String, String)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(Result::ok)
        .collect();
    let mut first_error = None;
    for (run_id, log_path) in sources {
        let Some(parent) = Path::new(&log_path).parent() else {
            continue;
        };
        let root = parent.join("tuning-artifacts");
        if !root.exists() {
            continue;
        }
        if let Err(error) = process_root(conn, &run_id, &root) {
            eprintln!("{error}");
            first_error.get_or_insert(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn process_root(conn: &Connection, run_id: &str, root: &Path) -> Result<(), IngestError> {
    let root = real_directory(root).map_err(|message| {
        record_root_failure(conn, run_id, root, &message);
        integrity(run_id, root, message)
    })?;
    let attempt_path = root.join("attempt.json");
    let attempt = read_regular(&attempt_path).map_err(|message| {
        record_root_failure(conn, run_id, &root, &message);
        integrity(run_id, &attempt_path, message)
    })?;
    let attempt_digest = digest(&attempt);
    let attempt_id = validate_attempt(&attempt, run_id).map_err(|message| {
        record_root_failure(conn, run_id, &root, &message);
        integrity(run_id, &attempt_path, message)
    })?;
    conn.execute(
        "INSERT INTO artifact_roots \
         (physical_run_id, artifact_root, attempt_id, attempt_digest, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5) \
         ON CONFLICT (physical_run_id) DO UPDATE SET \
         artifact_root = excluded.artifact_root, attempt_id = excluded.attempt_id, \
         attempt_digest = excluded.attempt_digest, status = 'active', integrity_error = NULL, \
         updated_at = excluded.updated_at",
        params![
            run_id,
            path_string(&root),
            attempt_id,
            attempt_digest,
            iso_timestamp()
        ],
    )?;

    let watermark: String = conn.query_row(
        "SELECT descriptor_watermark FROM artifact_roots WHERE physical_run_id = ?1",
        params![run_id],
        |row| row.get(0),
    )?;
    let descriptors = descriptor_batch(&root, &watermark).map_err(|message| {
        record_root_failure(conn, run_id, &root, &message);
        integrity(run_id, &root, message)
    })?;
    let mut first_error = None;
    for filename in descriptors {
        let result = process_descriptor(conn, run_id, &root, &attempt_id, &filename);
        conn.execute(
            "UPDATE artifact_roots SET descriptor_watermark = ?1, updated_at = ?2 \
             WHERE physical_run_id = ?3",
            params![filename, iso_timestamp(), run_id],
        )?;
        if let Err(error) = result {
            first_error.get_or_insert(error);
        }
    }
    if let Err(error) = process_tasks(conn, run_id, &root) {
        first_error.get_or_insert(error);
    }
    first_error.map_or(Ok(()), Err)
}

fn descriptor_batch(root: &Path, watermark: &str) -> Result<Vec<String>, String> {
    let directory = root.join("descriptors");
    if !directory.exists() {
        return Ok(Vec::new());
    }
    real_directory(&directory)?;
    let mut names = Vec::new();
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name.as_str() <= watermark {
            continue;
        }
        names.push(name);
    }
    names.sort();
    names.truncate(DISCOVERY_BATCH);
    Ok(names)
}

fn process_descriptor(
    conn: &Connection,
    run_id: &str,
    root: &Path,
    attempt_id: &str,
    filename: &str,
) -> Result<(), IngestError> {
    let path = root.join("descriptors").join(filename);
    let descriptor = (|| -> Result<Descriptor, String> {
        let (task_sequence, filename_task_id) = parse_descriptor_filename(filename)?;
        let bytes = read_regular(&path)?;
        let payload: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        let object = payload
            .as_object()
            .ok_or_else(|| "descriptor must be a JSON object".to_owned())?;
        require_keys(
            object,
            &[
                "artifact_layout_schema_version",
                "attempt_id",
                "bench_run_id",
                "binary",
                "candidate_config",
                "created_at",
                "game",
                "game_ids",
                "manifest_fingerprint",
                "opponent",
                "optimizer_id",
                "pair_id",
                "pair_index",
                "pool_snapshot",
                "pool_snapshot_fingerprint",
                "rating_before",
                "schema_version",
                "search_budget",
                "seed",
                "session_id",
                "task_directory",
                "task_id",
                "task_sequence",
                "trace_game_sequences",
                "trial_id",
            ],
        )?;
        required_version(object, "artifact_layout_schema_version")?;
        required_version(object, "schema_version")?;
        let task_id = required_string(object, "task_id")?;
        if task_id != filename_task_id {
            return Err("descriptor task_id does not match its filename".into());
        }
        if required_string(object, "attempt_id")? != attempt_id {
            return Err("descriptor attempt_id does not match attempt.json".into());
        }
        if required_string(object, "bench_run_id")? != run_id {
            return Err("descriptor bench_run_id does not match physical run".into());
        }
        let sequence = required_u64(object, "task_sequence")?;
        if sequence != task_sequence {
            return Err("descriptor task_sequence does not match its filename".into());
        }
        let expected_directory = format!("tasks/{task_id}");
        if required_string(object, "task_directory")? != expected_directory {
            return Err("descriptor task_directory does not match task identity".into());
        }
        Ok(Descriptor {
            task_id,
            task_sequence,
            digest: digest(&bytes),
            task_root: root.join(expected_directory),
        })
    })();
    let descriptor = match descriptor {
        Ok(descriptor) => descriptor,
        Err(message) => {
            record_descriptor_failure(conn, run_id, filename, &path, &message)?;
            return Err(integrity(run_id, &path, message));
        }
    };
    let existing: Option<String> = conn
        .query_row(
            "SELECT descriptor_digest FROM artifact_descriptors \
             WHERE physical_run_id = ?1 AND descriptor_filename = ?2",
            params![run_id, filename],
            |row| row.get(0),
        )
        .ok();
    if existing
        .as_deref()
        .is_some_and(|value| value != descriptor.digest)
    {
        let message = "descriptor bytes changed after discovery".to_owned();
        record_descriptor_failure(conn, run_id, filename, &path, &message)?;
        return Err(integrity(run_id, &path, message));
    }
    conn.execute(
        "INSERT INTO artifact_descriptors \
         (physical_run_id, descriptor_filename, descriptor_path, task_id, task_sequence, descriptor_digest, task_root, status) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'registered') \
         ON CONFLICT (physical_run_id, descriptor_filename) DO NOTHING",
        params![run_id, filename, path_string(&path), descriptor.task_id, descriptor.task_sequence as i64, descriptor.digest, path_string(&descriptor.task_root)],
    )?;
    conn.execute(
        "INSERT INTO artifact_tasks \
         (physical_run_id, task_id, attempt_id, task_sequence, descriptor_path, task_root, trace_path, descriptor_digest, status) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'incomplete') \
         ON CONFLICT (physical_run_id, task_id) DO NOTHING",
        params![run_id, descriptor.task_id, attempt_id, descriptor.task_sequence as i64, path_string(&path), path_string(&descriptor.task_root), path_string(&descriptor.task_root.join("trace.jsonl")), descriptor.digest],
    )?;
    Ok(())
}

fn process_tasks(conn: &Connection, run_id: &str, root: &Path) -> Result<(), IngestError> {
    let mut statement = conn.prepare(
        "SELECT task_id, attempt_id, descriptor_digest, task_root, trace_path \
         FROM artifact_tasks WHERE physical_run_id = ?1 \
         AND status NOT IN ('completed', 'failed', 'integrity_failure') ORDER BY task_sequence",
    )?;
    let tasks: Vec<TaskRow> = statement
        .query_map(params![run_id], |row| {
            Ok(TaskRow {
                task_id: row.get(0)?,
                attempt_id: row.get(1)?,
                descriptor_digest: row.get(2)?,
                task_root: PathBuf::from(row.get::<_, String>(3)?),
                trace_path: PathBuf::from(row.get::<_, String>(4)?),
            })
        })?
        .filter_map(Result::ok)
        .collect();
    let mut first_error = None;
    for task in tasks {
        if let Err(error) = process_task(conn, run_id, root, &task) {
            first_error.get_or_insert(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn process_task(
    conn: &Connection,
    run_id: &str,
    root: &Path,
    task: &TaskRow,
) -> Result<(), IngestError> {
    let task_root = match contained_real_directory(root, &task.task_root) {
        Ok(path) => path,
        Err(_) if !task.task_root.exists() => return Ok(()),
        Err(message) => return task_failure(conn, run_id, task, &task.task_root, message),
    };
    let trace = task_root.join("trace.jsonl");
    if trace != task.trace_path {
        return task_failure(conn, run_id, task, &trace, "task trace path changed".into());
    }
    if trace.exists() {
        if let Err(message) = regular_file(&trace) {
            return task_failure(conn, run_id, task, &trace, message);
        }
        let offset = trace_cursor(conn, run_id, &task.task_id)?;
        let next = match process_complete_trace_file(conn, run_id, &trace, offset) {
            Ok(next) => next,
            Err(error) => return task_failure(conn, run_id, task, &trace, error.to_string()),
        };
        set_trace_cursor(conn, run_id, &task.task_id, &trace, next)?;
    }
    let complete = task_root.join("complete.json");
    if !complete.exists() {
        return Ok(());
    }
    let completion = match validate_completion(run_id, &task_root, task) {
        Ok(completion) => completion,
        Err(error) => return task_failure(conn, run_id, task, &complete, error.to_string()),
    };
    if completion.has_trace != trace.exists() {
        return task_failure(
            conn,
            run_id,
            task,
            &complete,
            "completion trace member does not match task files".into(),
        );
    }
    if trace.exists() {
        let offset = trace_cursor(conn, run_id, &task.task_id)?;
        let length = fs::metadata(&trace)?.len();
        if offset != length {
            return task_failure(
                conn,
                run_id,
                task,
                &trace,
                "completed trace ends with a partial record".into(),
            );
        }
    }
    conn.execute(
        "UPDATE artifact_tasks SET status = ?1, completion_digest = ?2, integrity_error = NULL, \
         completed_at = ?3 WHERE physical_run_id = ?4 AND task_id = ?5",
        params![
            completion.outcome,
            completion.digest,
            iso_timestamp(),
            run_id,
            task.task_id
        ],
    )?;
    Ok(())
}

fn validate_attempt(bytes: &[u8], run_id: &str) -> Result<String, String> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "attempt.json must be a JSON object".to_owned())?;
    require_keys(
        object,
        &[
            "artifact_layout_schema_version",
            "attempt_id",
            "bench_run_id",
            "created_at",
            "manifest_fingerprint",
            "optimizer_id",
            "schema_version",
            "session_id",
        ],
    )?;
    required_version(object, "artifact_layout_schema_version")?;
    required_version(object, "schema_version")?;
    if required_string(object, "bench_run_id")? != run_id {
        return Err("attempt.json bench_run_id does not match physical run".into());
    }
    for key in [
        "attempt_id",
        "session_id",
        "optimizer_id",
        "manifest_fingerprint",
        "created_at",
    ] {
        required_string(object, key)?;
    }
    required_string(object, "attempt_id")
}

fn validate_completion(
    run_id: &str,
    task_root: &Path,
    task: &TaskRow,
) -> Result<Completion, IngestError> {
    let complete_path = task_root.join("complete.json");
    let bytes = read_regular(&complete_path)
        .map_err(|message| integrity(run_id, &complete_path, message))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| integrity(run_id, &complete_path, error.to_string()))?;
    let object = value.as_object().ok_or_else(|| {
        integrity(
            run_id,
            &complete_path,
            "complete.json must be a JSON object".into(),
        )
    })?;
    let completion_keys = [
        "attempt_id",
        "descriptor_digest",
        "outcome",
        "schema_version",
        "stderr",
        "stdout",
        "task_id",
        "terminal",
    ];
    let mut expected = completion_keys.to_vec();
    if object.contains_key("trace") {
        expected.push("trace");
    }
    require_keys(object, &expected)
        .map_err(|message| integrity(run_id, &complete_path, message))?;
    required_version(object, "schema_version")
        .map_err(|message| integrity(run_id, &complete_path, message))?;
    if required_string(object, "task_id")
        .map_err(|message| integrity(run_id, &complete_path, message))?
        != task.task_id
        || required_string(object, "attempt_id")
            .map_err(|message| integrity(run_id, &complete_path, message))?
            != task.attempt_id
        || required_string(object, "descriptor_digest")
            .map_err(|message| integrity(run_id, &complete_path, message))?
            != task.descriptor_digest
    {
        return Err(integrity(
            run_id,
            &complete_path,
            "completion identity or descriptor digest conflicts".into(),
        ));
    }
    let outcome = required_string(object, "outcome")
        .map_err(|message| integrity(run_id, &complete_path, message))?;
    if outcome != "completed" && outcome != "failed" {
        return Err(integrity(
            run_id,
            &complete_path,
            "completion outcome is invalid".into(),
        ));
    }
    let terminal = if outcome == "completed" {
        "result.json"
    } else {
        "failure.json"
    };
    validate_member(task_root, object.get("terminal"), terminal)
        .map_err(|message| integrity(run_id, &complete_path, message))?;
    validate_member(task_root, object.get("stdout"), "stdout.log")
        .map_err(|message| integrity(run_id, &complete_path, message))?;
    validate_member(task_root, object.get("stderr"), "stderr.log")
        .map_err(|message| integrity(run_id, &complete_path, message))?;
    let has_trace = match object.get("trace") {
        Some(member) => {
            validate_member(task_root, Some(member), "trace.jsonl")
                .map_err(|message| integrity(run_id, &complete_path, message))?;
            true
        }
        None => false,
    };
    Ok(Completion {
        outcome,
        digest: digest(&bytes),
        has_trace,
    })
}

fn validate_member(
    task_root: &Path,
    value: Option<&Value>,
    expected_filename: &str,
) -> Result<(), String> {
    let object = value
        .and_then(Value::as_object)
        .ok_or_else(|| format!("completion member {expected_filename} is missing or invalid"))?;
    if object.len() != 3 || required_string(object, "filename")? != expected_filename {
        return Err(format!(
            "completion member {expected_filename} has an invalid schema"
        ));
    }
    let expected_digest = required_string(object, "digest")?;
    if !is_digest(&expected_digest) {
        return Err(format!(
            "completion member {expected_filename} has an invalid digest"
        ));
    }
    let length = required_u64(object, "byte_length")?;
    let contents = read_regular(&task_root.join(expected_filename))?;
    if contents.len() as u64 != length || digest(&contents) != expected_digest {
        return Err(format!("completion member differs: {expected_filename}"));
    }
    Ok(())
}

fn task_failure(
    conn: &Connection,
    run_id: &str,
    task: &TaskRow,
    artifact: &Path,
    message: String,
) -> Result<(), IngestError> {
    conn.execute(
        "UPDATE artifact_tasks SET status = 'integrity_failure', integrity_error = ?1 \
         WHERE physical_run_id = ?2 AND task_id = ?3",
        params![message, run_id, task.task_id],
    )?;
    Err(integrity(run_id, artifact, message))
}

fn record_root_failure(conn: &Connection, run_id: &str, root: &Path, message: &str) {
    let _ = conn.execute(
        "INSERT INTO artifact_roots (physical_run_id, artifact_root, status, integrity_error, updated_at) \
         VALUES (?1, ?2, 'integrity_failure', ?3, ?4) \
         ON CONFLICT (physical_run_id) DO UPDATE SET status = 'integrity_failure', integrity_error = excluded.integrity_error, updated_at = excluded.updated_at",
        params![run_id, path_string(root), message, iso_timestamp()],
    );
}

fn record_descriptor_failure(
    conn: &Connection,
    run_id: &str,
    filename: &str,
    path: &Path,
    message: &str,
) -> Result<(), IngestError> {
    conn.execute(
        "INSERT INTO artifact_descriptors (physical_run_id, descriptor_filename, descriptor_path, status, integrity_error) \
         VALUES (?1, ?2, ?3, 'integrity_failure', ?4) \
         ON CONFLICT (physical_run_id, descriptor_filename) DO UPDATE SET status = 'integrity_failure', integrity_error = excluded.integrity_error",
        params![run_id, filename, path_string(path), message],
    )?;
    Ok(())
}

fn trace_cursor(conn: &Connection, run_id: &str, task_id: &str) -> Result<u64, IngestError> {
    match conn.query_row(
        "SELECT byte_offset FROM _artifact_trace_cursor WHERE physical_run_id = ?1 AND task_id = ?2",
        params![run_id, task_id],
        |row| row.get::<_, i64>(0),
    ) {
        Ok(offset) => Ok(offset as u64),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(0),
        Err(error) => Err(error.into()),
    }
}

fn set_trace_cursor(
    conn: &Connection,
    run_id: &str,
    task_id: &str,
    path: &Path,
    offset: u64,
) -> Result<(), IngestError> {
    conn.execute(
        "INSERT INTO _artifact_trace_cursor (physical_run_id, task_id, trace_path, byte_offset, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT (physical_run_id, task_id) \
         DO UPDATE SET trace_path = excluded.trace_path, byte_offset = excluded.byte_offset, updated_at = excluded.updated_at",
        params![run_id, task_id, path_string(path), offset as i64, iso_timestamp()],
    )?;
    Ok(())
}

fn parse_descriptor_filename(filename: &str) -> Result<(u64, String), String> {
    let Some((sequence_text, task)) = filename
        .strip_suffix(".json")
        .and_then(|name| name.split_once('-'))
    else {
        return Err("descriptor filename is not canonical".into());
    };
    if sequence_text.len() != 19
        || !sequence_text.bytes().all(|byte| byte.is_ascii_digit())
        || !valid_task_id(task)
    {
        return Err("descriptor filename is not canonical".into());
    }
    let sequence = sequence_text
        .parse::<u64>()
        .map_err(|_| "descriptor sequence is invalid")?;
    if sequence == 0 || format!("{sequence:019}") != sequence_text {
        return Err("descriptor sequence is not canonical".into());
    }
    Ok((sequence, task.to_owned()))
}

fn valid_task_id(value: &str) -> bool {
    value.len() == TASK_ID_LENGTH
        && value.starts_with("task-")
        && value["task-".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn required_string(object: &serde_json::Map<String, Value>, key: &str) -> Result<String, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("{key} must be a nonempty string"))
}

fn required_u64(object: &serde_json::Map<String, Value>, key: &str) -> Result<u64, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}

fn required_version(object: &serde_json::Map<String, Value>, key: &str) -> Result<(), String> {
    if required_u64(object, key)? != 1 {
        return Err(format!("unsupported {key}"));
    }
    Ok(())
}

fn require_keys(object: &serde_json::Map<String, Value>, expected: &[&str]) -> Result<(), String> {
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err("artifact has an invalid schema".into());
    }
    Ok(())
}

fn real_directory(path: &Path) -> Result<PathBuf, String> {
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !metadata.file_type().is_dir() {
        return Err("path must be a real directory".into());
    }
    path.canonicalize().map_err(|error| error.to_string())
}

fn contained_real_directory(root: &Path, path: &Path) -> Result<PathBuf, String> {
    let path = real_directory(path)?;
    if !path.starts_with(root) {
        return Err("task root escapes artifact root".into());
    }
    Ok(path)
}

fn regular_file(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !metadata.file_type().is_file() {
        return Err("artifact must be a regular file".into());
    }
    Ok(())
}

fn read_regular(path: &Path) -> Result<Vec<u8>, String> {
    regular_file(path)?;
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

pub(super) fn digest(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while !(padded.len() + 8).is_multiple_of(64) {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, word) in words[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(
                chunk[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("chunk word"),
            );
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let mut work = state;
        for index in 0..64 {
            let choice = (work[4] & work[5]) ^ ((!work[4]) & work[6]);
            let major = (work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]);
            let sum1 =
                work[4].rotate_right(6) ^ work[4].rotate_right(11) ^ work[4].rotate_right(25);
            let sum0 =
                work[0].rotate_right(2) ^ work[0].rotate_right(13) ^ work[0].rotate_right(22);
            let first = work[7]
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let second = sum0.wrapping_add(major);
            work = [
                first.wrapping_add(second),
                work[0],
                work[1],
                work[2],
                work[3].wrapping_add(first),
                work[4],
                work[5],
                work[6],
            ];
        }
        for (value, update) in state.iter_mut().zip(work) {
            *value = value.wrapping_add(update);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}
fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
fn integrity(run_id: &str, path: &Path, message: String) -> IngestError {
    IngestError::ArtifactIntegrity {
        run_id: run_id.to_owned(),
        artifact: path_string(path),
        message,
    }
}

struct Descriptor {
    task_id: String,
    task_sequence: u64,
    digest: String,
    task_root: PathBuf,
}
struct TaskRow {
    task_id: String,
    attempt_id: String,
    descriptor_digest: String,
    task_root: PathBuf,
    trace_path: PathBuf,
}
struct Completion {
    outcome: String,
    digest: String,
    has_trace: bool,
}

#[cfg(test)]
mod tests {
    use super::digest;

    #[test]
    fn sha256_matches_the_artifact_protocol() {
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
