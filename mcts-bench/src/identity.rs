//! Durable logical-run and physical-attempt identity helpers.
//!
//! These helpers deliberately know only the additive identity columns. The
//! rest of a `runs` row remains owned by the server and registry ingestion.

use duckdb::{params, Connection, Transaction};
use serde_json::Value;

#[derive(Debug)]
pub enum IdentityError {
    DuckDb(duckdb::Error),
    MissingRun(String),
    InvalidLinkage(String),
    Contradiction(String),
}

impl std::fmt::Display for IdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuckDb(error) => write!(f, "DuckDB error: {error}"),
            Self::MissingRun(run_id) => write!(f, "run '{run_id}' not found"),
            Self::InvalidLinkage(message) => write!(f, "invalid run identity linkage: {message}"),
            Self::Contradiction(message) => write!(f, "contradictory run identity: {message}"),
        }
    }
}

impl std::error::Error for IdentityError {}

impl From<duckdb::Error> for IdentityError {
    fn from(error: duckdb::Error) -> Self {
        Self::DuckDb(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentIdentity {
    pub logical_run_id: String,
    pub parent_attempt_id: String,
    pub attempt_ordinal: u64,
}

struct RunIdentityRow {
    run_id: String,
    kind: String,
    project_id: Option<String>,
    experiment_id: Option<String>,
    config: Option<String>,
    logical_run_id: Option<String>,
    parent_attempt_id: Option<String>,
    attempt_ordinal: Option<u64>,
}

fn run_identity(tx: &Transaction<'_>, run_id: &str) -> Result<RunIdentityRow, IdentityError> {
    tx.query_row(
        "SELECT run_id, kind, project_id, experiment_id, CAST(config AS TEXT), logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = ?1",
        params![run_id],
        |row| {
            Ok(RunIdentityRow {
                run_id: row.get(0)?,
                kind: row.get(1)?,
                project_id: row.get(2)?,
                experiment_id: row.get(3)?,
                config: row.get(4)?,
                logical_run_id: row.get(5)?,
                parent_attempt_id: row.get(6)?,
                attempt_ordinal: row.get(7)?,
            })
        },
    )
    .map_err(|error| match error {
        duckdb::Error::QueryReturnedNoRows => IdentityError::MissingRun(run_id.to_owned()),
        other => IdentityError::DuckDb(other),
    })
}

fn logical_exists(tx: &Transaction<'_>, logical_run_id: &str) -> Result<bool, IdentityError> {
    Ok(tx.query_row(
        "SELECT COUNT(*) FROM logical_runs WHERE logical_run_id = ?1",
        params![logical_run_id],
        |row| row.get::<_, i64>(0),
    )? > 0)
}

fn config_string(row: &RunIdentityRow) -> Option<&str> {
    row.config.as_deref()
}

fn resumed_from(row: &RunIdentityRow) -> Option<String> {
    config_string(row)
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .and_then(|value| {
            value
                .get("resumed_from")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn ladder_root(row: &RunIdentityRow) -> Option<String> {
    config_string(row)
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .and_then(|value| {
            value
                .get("ladder_root")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn validate_linkage(tx: &Transaction<'_>, row: &RunIdentityRow) -> Result<(), IdentityError> {
    let has_logical = row.logical_run_id.is_some();
    let has_parent = row.parent_attempt_id.is_some();
    let has_ordinal = row.attempt_ordinal.is_some();
    if has_parent && !has_logical || has_ordinal != has_logical {
        return Err(IdentityError::InvalidLinkage(row.run_id.clone()));
    }
    if let Some(logical_run_id) = row.logical_run_id.as_deref() {
        if !logical_exists(tx, logical_run_id)? {
            return Err(IdentityError::InvalidLinkage(format!(
                "logical run {logical_run_id} is missing"
            )));
        }
        if !has_parent && row.attempt_ordinal != Some(1) {
            return Err(IdentityError::InvalidLinkage(format!(
                "root attempt {} has ordinal {:?}",
                row.run_id, row.attempt_ordinal
            )));
        }
        if let Some(recorded_parent) = row.parent_attempt_id.as_deref() {
            if !run_exists(tx, recorded_parent)? {
                return Err(IdentityError::InvalidLinkage(format!(
                    "parent attempt {recorded_parent} is missing"
                )));
            }
        }
    }
    Ok(())
}

fn resolve_parent_in_tx(
    tx: &Transaction<'_>,
    parent_attempt_id: &str,
) -> Result<ParentIdentity, IdentityError> {
    let parent = run_identity(tx, parent_attempt_id)?;
    validate_linkage(tx, &parent)?;

    if let (Some(logical_run_id), Some(attempt_ordinal)) =
        (parent.logical_run_id.clone(), parent.attempt_ordinal)
    {
        if let Some(hint) = ladder_root(&parent) {
            if !run_exists(tx, &hint)? {
                return Err(IdentityError::Contradiction(format!(
                    "ladder_root hint {hint} is missing"
                )));
            }
            if hint != logical_run_id {
                return Err(IdentityError::Contradiction(format!(
                    "ladder_root hint {hint} disagrees with logical run {logical_run_id}"
                )));
            }
        }
        return Ok(ParentIdentity {
            logical_run_id,
            parent_attempt_id: parent_attempt_id.to_owned(),
            attempt_ordinal,
        });
    }

    // Follow the explicit historical resume links read-only. Only the
    // concrete parent is anchored below; ancestors are never rewritten.
    let mut chain = vec![parent];
    let mut cursor = parent_attempt_id.to_owned();
    while let Some(previous) = resumed_from(chain.last().expect("chain has parent")) {
        if previous == cursor || chain.iter().any(|row| row.run_id == previous) {
            return Err(IdentityError::Contradiction(format!(
                "cycle in historical ancestry at {previous}"
            )));
        }
        let row = run_identity(tx, &previous)?;
        validate_linkage(tx, &row)?;
        cursor = previous;
        chain.push(row);
    }

    let hint = chain.iter().find_map(ladder_root);
    if let Some(hint) = &hint {
        if !run_exists(tx, hint)? {
            return Err(IdentityError::Contradiction(format!(
                "ladder_root hint {hint} is missing"
            )));
        }
        if !chain.iter().any(|row| row.run_id == *hint) {
            return Err(IdentityError::Contradiction(format!(
                "ladder_root hint {hint} is not an ancestor of {parent_attempt_id}"
            )));
        }
        if chain
            .iter()
            .filter_map(ladder_root)
            .any(|value| value != *hint)
        {
            return Err(IdentityError::Contradiction(
                "historical ladder_root hints disagree".into(),
            ));
        }
    }

    let linked_anchor = chain.iter().enumerate().find_map(|(offset, row)| {
        row.logical_run_id
            .clone()
            .zip(row.attempt_ordinal)
            .map(|(logical_run_id, attempt_ordinal)| (offset, logical_run_id, attempt_ordinal))
    });
    if let Some((anchor_position, logical_run_id, anchor_ordinal)) = linked_anchor {
        if let Some(hint) = &hint {
            if hint != &logical_run_id {
                return Err(IdentityError::Contradiction(format!(
                    "ladder_root hint {hint} disagrees with linked logical run {logical_run_id}"
                )));
            }
        }
        let attempt_ordinal = anchor_ordinal
            .checked_add(anchor_position as u64)
            .ok_or_else(|| {
                IdentityError::Contradiction("historical ancestry ordinal overflow".into())
            })?;
        for (offset, row) in chain.iter().enumerate() {
            if let Some(existing_logical) = &row.logical_run_id {
                if existing_logical != &logical_run_id
                    || row.attempt_ordinal
                        != Some(attempt_ordinal.checked_sub(offset as u64).ok_or_else(|| {
                            IdentityError::Contradiction(
                                "historical ancestry ordinal underflow".into(),
                            )
                        })?)
                {
                    return Err(IdentityError::Contradiction(format!(
                        "historical attempt {} disagrees with linked ancestry",
                        row.run_id
                    )));
                }
            }
        }
        return Ok(ParentIdentity {
            logical_run_id,
            parent_attempt_id: parent_attempt_id.to_owned(),
            attempt_ordinal,
        });
    }

    let logical_run_id = hint
        .clone()
        .unwrap_or_else(|| chain.last().expect("chain has parent").run_id.clone());
    let attempt_ordinal = if hint.is_some() {
        chain
            .iter()
            .position(|row| row.run_id == logical_run_id)
            .map(|position| position as u64 + 1)
            .expect("validated ladder root is in chain")
    } else {
        chain.len() as u64
    };

    Ok(ParentIdentity {
        logical_run_id,
        parent_attempt_id: parent_attempt_id.to_owned(),
        attempt_ordinal,
    })
}

fn run_exists(tx: &Transaction<'_>, run_id: &str) -> Result<bool, IdentityError> {
    Ok(tx.query_row(
        "SELECT COUNT(*) FROM runs WHERE run_id = ?1",
        params![run_id],
        |row| row.get::<_, i64>(0),
    )? > 0)
}

/// Validate a continuation parent before any child process is spawned. If a
/// legacy parent is unlinked, anchor only that parent and its logical record.
pub fn prepare_continuation(
    conn: &Connection,
    parent_attempt_id: &str,
) -> Result<ParentIdentity, IdentityError> {
    let tx = conn.unchecked_transaction()?;
    let identity = resolve_parent_in_tx(&tx, parent_attempt_id)?;
    let parent = run_identity(&tx, parent_attempt_id)?;
    let needs_anchor = parent.logical_run_id.is_none();
    if needs_anchor {
        let anchor_parent_attempt_id = resumed_from(&parent);
        tx.execute(
            "INSERT INTO logical_runs (logical_run_id, kind, project_id, experiment_id, created_at, current_attempt_id) VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP, ?5) ON CONFLICT (logical_run_id) DO NOTHING",
            params![
                &identity.logical_run_id,
                parent.kind,
                parent.project_id,
                parent.experiment_id,
                parent_attempt_id,
            ],
        )?;
        let updated = tx.execute(
            "UPDATE runs SET logical_run_id = ?1, parent_attempt_id = ?2, attempt_ordinal = ?3 WHERE run_id = ?4 AND logical_run_id IS NULL",
            params![
                &identity.logical_run_id,
                anchor_parent_attempt_id,
                identity.attempt_ordinal,
                parent_attempt_id
            ],
        )?;
        if updated != 1 {
            return Err(IdentityError::InvalidLinkage(format!(
                "attempt {parent_attempt_id} changed while it was being anchored"
            )));
        }
        let linked: (Option<String>, Option<String>, Option<u64>) = tx.query_row(
            "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = ?1",
            params![parent_attempt_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        if linked.0.as_deref() != Some(identity.logical_run_id.as_str())
            || linked.1 != anchor_parent_attempt_id
            || linked.2 != Some(identity.attempt_ordinal)
        {
            return Err(IdentityError::Contradiction(format!(
                "attempt {parent_attempt_id} was anchored with a different identity"
            )));
        }
        let logical_kind: String = tx.query_row(
            "SELECT kind FROM logical_runs WHERE logical_run_id = ?1",
            params![&identity.logical_run_id],
            |row| row.get(0),
        )?;
        if logical_kind != parent.kind {
            return Err(IdentityError::Contradiction(format!(
                "logical run {} has kind {logical_kind}, not {}",
                identity.logical_run_id, parent.kind
            )));
        }
        let advanced = tx.execute(
            "UPDATE logical_runs SET current_attempt_id = ?1, version = version + 1 WHERE logical_run_id = ?2 AND current_attempt_id <> ?1",
            params![parent_attempt_id, &identity.logical_run_id],
        )?;
        if advanced > 1 {
            return Err(IdentityError::Contradiction(format!(
                "multiple logical anchors exist for {}",
                identity.logical_run_id
            )));
        }
    }
    tx.commit()?;
    Ok(identity)
}

/// Link a newly inserted physical row as a logical root. The caller must
/// keep the row insert and this helper in the same transaction.
pub fn create_root_identity(
    tx: &Transaction<'_>,
    attempt_id: &str,
    kind: &str,
    project_id: Option<&str>,
    experiment_id: Option<&str>,
    created_at: &str,
) -> Result<(), IdentityError> {
    let attempt = run_identity(tx, attempt_id)?;
    if attempt.kind != kind
        || attempt.project_id.as_deref() != project_id
        || attempt.experiment_id.as_deref() != experiment_id
    {
        return Err(IdentityError::Contradiction(format!(
            "attempt {attempt_id} metadata disagrees with its logical root"
        )));
    }
    tx.execute(
        "INSERT INTO logical_runs (logical_run_id, kind, project_id, experiment_id, created_at, current_attempt_id) VALUES (?1, ?2, ?3, ?4, ?5, ?1) ON CONFLICT (logical_run_id) DO NOTHING",
        params![attempt_id, kind, project_id, experiment_id, created_at],
    )?;
    let updated = tx.execute(
        "UPDATE runs SET logical_run_id = ?1, parent_attempt_id = NULL, attempt_ordinal = 1 WHERE run_id = ?1 AND logical_run_id IS NULL",
        params![attempt_id],
    )?;
    if updated == 0 {
        let existing: (Option<String>, Option<String>, Option<u64>) = tx.query_row(
            "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = ?1",
            params![attempt_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        if existing != (Some(attempt_id.to_owned()), None, Some(1)) {
            return Err(IdentityError::Contradiction(format!(
                "attempt {attempt_id} already has different identity"
            )));
        }
    }
    let logical: (String, Option<String>, Option<String>, String) = tx.query_row(
        "SELECT kind, project_id, experiment_id, current_attempt_id FROM logical_runs WHERE logical_run_id = ?1",
        params![attempt_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;
    if logical
        != (
            kind.to_owned(),
            project_id.map(str::to_owned),
            experiment_id.map(str::to_owned),
            attempt_id.to_owned(),
        )
    {
        return Err(IdentityError::Contradiction(format!(
            "logical root {attempt_id} already has different metadata"
        )));
    }
    Ok(())
}

/// Link a newly inserted physical row as a continuation and advance the
/// logical run's current attempt in the same transaction.
pub fn create_child_identity(
    tx: &Transaction<'_>,
    child_attempt_id: &str,
    parent: &ParentIdentity,
) -> Result<(), IdentityError> {
    if !run_exists(tx, child_attempt_id)? {
        return Err(IdentityError::MissingRun(child_attempt_id.to_owned()));
    }
    let parent_row = run_identity(tx, &parent.parent_attempt_id)?;
    if parent_row.logical_run_id.as_deref() != Some(parent.logical_run_id.as_str())
        || parent_row.attempt_ordinal != Some(parent.attempt_ordinal)
    {
        return Err(IdentityError::InvalidLinkage(
            "parent identity changed before child insertion".into(),
        ));
    }
    let child_ordinal = parent.attempt_ordinal + 1;
    let existing_child: (Option<String>, Option<String>, Option<u64>) = tx.query_row(
        "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = ?1",
        params![child_attempt_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    let expected_child = (
        Some(parent.logical_run_id.clone()),
        Some(parent.parent_attempt_id.clone()),
        Some(child_ordinal),
    );
    if existing_child == expected_child {
        let current: String = tx.query_row(
            "SELECT current_attempt_id FROM logical_runs WHERE logical_run_id = ?1",
            params![&parent.logical_run_id],
            |row| row.get(0),
        )?;
        if current == child_attempt_id {
            return Ok(());
        }
        return Err(IdentityError::Contradiction(format!(
            "child attempt {child_attempt_id} is linked but is not current"
        )));
    }
    if existing_child == (Some(child_attempt_id.to_owned()), None, Some(1)) {
        let removed = tx.execute(
            "DELETE FROM logical_runs WHERE logical_run_id = ?1 AND current_attempt_id = ?1 AND version = 0",
            params![child_attempt_id],
        )?;
        if removed != 1 {
            return Err(IdentityError::Contradiction(format!(
                "attempt {child_attempt_id} is an established logical root, not a provisional registry root"
            )));
        }
        tx.execute(
            "UPDATE runs SET logical_run_id = NULL, parent_attempt_id = NULL, attempt_ordinal = NULL WHERE run_id = ?1",
            params![child_attempt_id],
        )?;
    }
    let updated = tx.execute(
        "UPDATE runs SET logical_run_id = ?1, parent_attempt_id = ?2, attempt_ordinal = ?3 WHERE run_id = ?4 AND logical_run_id IS NULL",
        params![&parent.logical_run_id, &parent.parent_attempt_id, child_ordinal, child_attempt_id],
    )?;
    let (actual_logical, actual_parent, actual_ordinal): (
        Option<String>,
        Option<String>,
        Option<u64>,
    ) = tx.query_row(
        "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = ?1",
        params![child_attempt_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    if actual_logical.as_deref() != Some(parent.logical_run_id.as_str())
        || actual_parent.as_deref() != Some(parent.parent_attempt_id.as_str())
        || actual_ordinal != Some(child_ordinal)
    {
        return Err(IdentityError::Contradiction(format!(
            "child attempt {child_attempt_id} already has different identity"
        )));
    }
    if updated == 0 && actual_logical.is_none() {
        return Err(IdentityError::Contradiction(format!(
            "child attempt {child_attempt_id} was not linked"
        )));
    }
    let advanced = tx.execute(
        "UPDATE logical_runs SET current_attempt_id = ?1, version = version + 1 WHERE logical_run_id = ?2",
        params![child_attempt_id, &parent.logical_run_id],
    )?;
    if advanced != 1 {
        return Err(IdentityError::InvalidLinkage(format!(
            "logical run {} is missing while linking child {child_attempt_id}",
            parent.logical_run_id
        )));
    }
    Ok(())
}

/// Create the self-root identity for a detached registry Start event. The
/// caller invokes this only when it inserted the physical row, so replay of
/// the event cannot overwrite server-recorded identity.
pub fn create_registry_root_identity(
    tx: &Transaction<'_>,
    attempt_id: &str,
    kind: &str,
    created_at: &str,
) -> Result<(), IdentityError> {
    create_root_identity(tx, attempt_id, kind, None, None, created_at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ensure_schema;

    fn db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();
        conn
    }

    fn insert_run(conn: &Connection, run_id: &str, config: Option<&str>) {
        conn.execute(
            "INSERT INTO runs (run_id, kind, game, config, git_sha, git_dirty, host, started_at, status, log_path) VALUES (?1, 'smac3', 'nim', ?2, 'sha', false, 'host', CURRENT_TIMESTAMP, 'completed', '/tmp/log')",
            params![run_id, config],
        )
        .unwrap();
    }

    #[test]
    fn root_and_child_identity_are_atomic() {
        let conn = db();
        insert_run(&conn, "root", None);
        let tx = conn.unchecked_transaction().unwrap();
        create_root_identity(&tx, "root", "smac3", None, None, "2026-01-01T00:00:00Z").unwrap();
        tx.commit().unwrap();
        insert_run(&conn, "child", Some(r#"{"resumed_from":"root"}"#));
        let parent = prepare_continuation(&conn, "root").unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        create_child_identity(&tx, "child", &parent).unwrap();
        tx.commit().unwrap();
        let row: (String, String, u64, String) = conn.query_row("SELECT runs.logical_run_id, runs.parent_attempt_id, runs.attempt_ordinal, logical_runs.current_attempt_id FROM runs JOIN logical_runs ON runs.logical_run_id = logical_runs.logical_run_id WHERE runs.run_id = 'child'", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))).unwrap();
        assert_eq!(row, ("root".into(), "root".into(), 2, "child".into()));
    }

    #[test]
    fn registry_root_can_be_adopted_as_a_server_child_once() {
        let conn = db();
        insert_run(&conn, "root", None);
        let tx = conn.unchecked_transaction().unwrap();
        create_root_identity(&tx, "root", "smac3", None, None, "2026-01-01T00:00:00Z").unwrap();
        tx.commit().unwrap();

        insert_run(&conn, "child", None);
        let tx = conn.unchecked_transaction().unwrap();
        create_registry_root_identity(&tx, "child", "smac3", "2026-01-01T00:00:01Z").unwrap();
        tx.commit().unwrap();

        let parent = prepare_continuation(&conn, "root").unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        create_child_identity(&tx, "child", &parent).unwrap();
        tx.commit().unwrap();
        let identity: (String, String, u64, String) = conn
            .query_row(
                "SELECT r.logical_run_id, r.parent_attempt_id, r.attempt_ordinal, l.current_attempt_id FROM runs r JOIN logical_runs l ON l.logical_run_id = r.logical_run_id WHERE r.run_id = 'child'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(identity, ("root".into(), "root".into(), 2, "child".into()));
        let provisional_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM logical_runs WHERE logical_run_id = 'child'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(provisional_count, 0);
    }

    #[test]
    fn missing_hint_has_no_partial_anchor() {
        let conn = db();
        insert_run(&conn, "parent", Some(r#"{"ladder_root":"missing"}"#));
        assert!(prepare_continuation(&conn, "parent").is_err());
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM logical_runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn historical_anchor_follows_ancestry_without_rewriting_ancestors() {
        let conn = db();
        insert_run(&conn, "root", Some(r#"{"ladder_root":"root"}"#));
        insert_run(
            &conn,
            "parent",
            Some(r#"{"ladder_root":"root","resumed_from":"root"}"#),
        );

        let identity = prepare_continuation(&conn, "parent").unwrap();

        assert_eq!(
            identity,
            ParentIdentity {
                logical_run_id: "root".into(),
                parent_attempt_id: "parent".into(),
                attempt_ordinal: 2,
            }
        );
        let parent_link: (String, u64) = conn
            .query_row(
                "SELECT logical_run_id, attempt_ordinal FROM runs WHERE run_id = 'parent'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(parent_link, ("root".into(), 2));
        let root_link: (Option<String>, Option<u64>) = conn
            .query_row(
                "SELECT logical_run_id, attempt_ordinal FROM runs WHERE run_id = 'root'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(root_link, (None, None));
    }

    #[test]
    fn historical_anchor_is_repeatable_and_keeps_resumed_from_parent() {
        let conn = db();
        insert_run(&conn, "root", Some(r#"{"ladder_root":"root"}"#));
        insert_run(
            &conn,
            "parent",
            Some(r#"{"ladder_root":"root","resumed_from":"root"}"#),
        );

        let first = prepare_continuation(&conn, "parent").unwrap();
        let second = prepare_continuation(&conn, "parent").unwrap();
        assert_eq!(first, second);
        let linkage: (Option<String>, Option<String>, Option<u64>) = conn
            .query_row(
                "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = 'parent'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(linkage, (Some("root".into()), Some("root".into()), Some(2)));
    }

    #[test]
    fn historical_parent_uses_fully_linked_ancestor_identity() {
        let conn = db();
        insert_run(&conn, "root", None);
        let tx = conn.unchecked_transaction().unwrap();
        create_root_identity(&tx, "root", "smac3", None, None, "2026-01-01T00:00:00Z").unwrap();
        tx.commit().unwrap();
        insert_run(&conn, "parent", Some(r#"{"resumed_from":"root"}"#));

        let identity = prepare_continuation(&conn, "parent").unwrap();
        assert_eq!(identity.logical_run_id, "root");
        assert_eq!(identity.attempt_ordinal, 2);
        let root_link: (Option<String>, Option<u64>) = conn
            .query_row(
                "SELECT logical_run_id, attempt_ordinal FROM runs WHERE run_id = 'root'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(root_link, (Some("root".into()), Some(1)));
        let current: String = conn
            .query_row(
                "SELECT current_attempt_id FROM logical_runs WHERE logical_run_id = 'root'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(current, "parent");
    }

    #[test]
    fn partial_identity_on_an_ancestor_is_rejected() {
        let conn = db();
        insert_run(&conn, "root", None);
        insert_run(&conn, "parent", Some(r#"{"resumed_from":"root"}"#));
        conn.execute(
            "UPDATE runs SET logical_run_id = 'root' WHERE run_id = 'root'",
            [],
        )
        .unwrap();
        assert!(matches!(
            prepare_continuation(&conn, "parent"),
            Err(IdentityError::InvalidLinkage(_))
        ));
        let parent_link: (Option<String>, Option<String>, Option<u64>) = conn
            .query_row(
                "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = 'parent'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(parent_link, (None, None, None));
    }

    #[test]
    fn registry_root_replay_is_idempotent() {
        let conn = db();
        insert_run(&conn, "r", None);
        let tx = conn.unchecked_transaction().unwrap();
        create_registry_root_identity(&tx, "r", "smac3", "2026-01-01T00:00:00Z").unwrap();
        tx.commit().unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        create_registry_root_identity(&tx, "r", "smac3", "2026-01-01T00:00:00Z").unwrap();
        tx.commit().unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM logical_runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }
}
