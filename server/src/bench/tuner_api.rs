//! Read-only HTTP API over the SQLite projection of version-4 tuner runs.
//!
//! Every handler is a total function of the projection file built by the
//! `tuner-project` tool (the tuner crate's `tuner_projection` package): it
//! opens the file read-only, issues parameterized `SELECT`s, and shapes typed
//! JSON. No handler replays evidence, opens a run directory, or calls into the
//! tuner CLI. The one write-shaped route, `POST /projection/refresh`, shells
//! the projector out of band and returns its counts; it writes only the read
//! model, never scientific authority.
//!
//! The `run-dir` triple is scientific authority, the SQLite file is the
//! rebuildable read model, and this API is a thin projection of that file --
//! three layers, one direction.

use std::path::Path;
use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, Query, State as AxumState},
    http::StatusCode,
    response::Json,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{BenchError, BenchState};

// ---------------------------------------------------------------------------
// Connection + error plumbing
// ---------------------------------------------------------------------------

fn sql_error(error: rusqlite::Error) -> BenchError {
    BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("tuner projection query failed: {error}"),
    }
}

/// Open the projection file read-only. A missing or unreadable file is a 500 --
/// the projection is server infrastructure, not user input; `POST
/// /projection/refresh` (or the `tuner-project` CLI) is how it comes to exist.
fn open(state: &BenchState) -> Result<Connection, BenchError> {
    Connection::open_with_flags(
        &state.tuner_projection_db,
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(|error| BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!(
            "tuner projection db unavailable at {}: {error}",
            state.tuner_projection_db.display()
        ),
    })
}

fn require_run(conn: &Connection, run_id: &str) -> Result<(), BenchError> {
    let found: Option<i64> = conn
        .query_row("SELECT 1 FROM runs WHERE run_id = ?1", [run_id], |row| {
            row.get(0)
        })
        .optional()
        .map_err(sql_error)?;
    if found.is_some() {
        Ok(())
    } else {
        Err(BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' is not in the projection"),
        })
    }
}

/// Parse a canonical-JSON `TEXT` column that stores a string array.
fn string_array(raw: &str) -> Result<Vec<String>, BenchError> {
    serde_json::from_str(raw).map_err(|error| BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("projection stored a malformed string array: {error}"),
    })
}

fn json_value(raw: &str) -> Result<Value, BenchError> {
    serde_json::from_str(raw).map_err(|error| BenchError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("projection stored malformed JSON: {error}"),
    })
}

// ---------------------------------------------------------------------------
// Query parameters
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub(crate) struct Page {
    limit: Option<i64>,
    offset: Option<i64>,
}

impl Page {
    fn limit(&self) -> i64 {
        self.limit.unwrap_or(100).clamp(1, 1000)
    }
    fn offset(&self) -> i64 {
        self.offset.unwrap_or(0).max(0)
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct PairFilter {
    candidate: Option<String>,
    cohort: Option<i64>,
    limit: Option<i64>,
    offset: Option<i64>,
}

// ---------------------------------------------------------------------------
// Response shapes
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(crate) struct RunListItem {
    run_id: String,
    terminal_status: Option<String>,
    report_available: bool,
    ingest_error: Option<String>,
    game_kind: Option<String>,
    objective_id: Option<String>,
    shadow_policy_kind: Option<String>,
    active_elimination: Option<bool>,
    report_status: Option<String>,
    validation_claim: Option<String>,
    total_pair_attempts: i64,
    total_completed_pairs: i64,
}

#[derive(Serialize)]
pub(crate) struct ManifestSummary {
    manifest_run_id: Option<String>,
    manifest_fingerprint: Option<String>,
    game_kind: String,
    objective_id: String,
    cohort_size: i64,
    finalists: i64,
    seed: i64,
    task_seed: i64,
    shadow_policy_kind: String,
    active_elimination: bool,
}

#[derive(Serialize)]
pub(crate) struct ReportSummary {
    schema_version: i64,
    status: String,
    validation_claim: String,
}

#[derive(Serialize)]
pub(crate) struct ComputePhase {
    phase: String,
    pair_attempts: i64,
    completed_pairs: i64,
    failed_attempts: i64,
    censored_attempts: i64,
    physical_games: i64,
    search_iterations: i64,
    wall_time_ms: i64,
}

#[derive(Serialize)]
pub(crate) struct RunDetail {
    run_id: String,
    terminal_status: Option<String>,
    report_available: bool,
    ingest_error: Option<String>,
    manifest: Option<ManifestSummary>,
    report: Option<ReportSummary>,
    compute: Vec<ComputePhase>,
}

#[derive(Serialize)]
pub(crate) struct Cohort {
    cohort_index: i64,
    candidate_ids: Vec<String>,
    retained_candidate_ids: Vec<String>,
}

#[derive(Serialize)]
pub(crate) struct Candidate {
    candidate_id: String,
    fingerprint: String,
    canonical_config: Value,
    cohort_index: i64,
    cohort_slot: i64,
    source: String,
    parent_candidate_id: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct PairRow {
    pair_id: String,
    phase: String,
    candidate_id: String,
    task_id: String,
    opponent_id: String,
    pair_utility: f64,
}

#[derive(Serialize)]
pub(crate) struct GameRow {
    game_id: String,
    pair_id: String,
    candidate_side: String,
    outcome: String,
    plies: i64,
    elapsed_ms: i64,
    candidate_iterations_total: i64,
    opponent_iterations_total: i64,
}

#[derive(Serialize)]
pub(crate) struct ValidationRow {
    candidate_id: String,
    rank: i64,
    estimate: f64,
    lower: f64,
    upper: f64,
    wins: i64,
    draws: i64,
    losses: i64,
}

#[derive(Serialize)]
pub(crate) struct Validation {
    rows: Vec<ValidationRow>,
    /// The `unresolved_ties` array lifted verbatim from the stored
    /// `report.json`; `null` for a run with no report.
    unresolved_ties: Value,
}

#[derive(Serialize)]
pub(crate) struct RefreshResult {
    projected: i64,
    skipped: i64,
    ingest_errors: i64,
    pruned: i64,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /api/bench/tuner/projection/runs`
pub(crate) async fn list_runs(
    AxumState(state): AxumState<Arc<BenchState>>,
    Query(page): Query<Page>,
) -> Result<Json<Vec<RunListItem>>, BenchError> {
    let conn = open(&state)?;
    let mut stmt = conn
        .prepare(
            "SELECT r.run_id, r.terminal_status, r.report_available, r.ingest_error, \
                    m.game_kind, m.objective_id, m.shadow_policy_kind, m.active_elimination, \
                    rep.status, rep.validation_claim, \
                    COALESCE(c.attempts, 0), COALESCE(c.completed, 0) \
             FROM runs r \
             LEFT JOIN run_manifest m ON m.run_id = r.run_id \
             LEFT JOIN run_report rep ON rep.run_id = r.run_id \
             LEFT JOIN (SELECT run_id, SUM(pair_attempts) AS attempts, \
                               SUM(completed_pairs) AS completed \
                        FROM compute_phases GROUP BY run_id) c ON c.run_id = r.run_id \
             ORDER BY r.run_id LIMIT ?1 OFFSET ?2",
        )
        .map_err(sql_error)?;
    let rows = stmt
        .query_map([page.limit(), page.offset()], |row| {
            Ok(RunListItem {
                run_id: row.get(0)?,
                terminal_status: row.get(1)?,
                report_available: row.get::<_, i64>(2)? != 0,
                ingest_error: row.get(3)?,
                game_kind: row.get(4)?,
                objective_id: row.get(5)?,
                shadow_policy_kind: row.get(6)?,
                active_elimination: row.get::<_, Option<i64>>(7)?.map(|v| v != 0),
                report_status: row.get(8)?,
                validation_claim: row.get(9)?,
                total_pair_attempts: row.get(10)?,
                total_completed_pairs: row.get(11)?,
            })
        })
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;
    Ok(Json(rows))
}

/// `GET /api/bench/tuner/projection/runs/{run_id}`
pub(crate) async fn run_detail(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<RunDetail>, BenchError> {
    let conn = open(&state)?;

    let (terminal_status, report_available, ingest_error, manifest_run_id, manifest_fingerprint) =
        conn.query_row(
            "SELECT terminal_status, report_available, ingest_error, \
                    manifest_run_id, manifest_fingerprint FROM runs WHERE run_id = ?1",
            [&run_id],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, i64>(1)? != 0,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            },
        )
        .optional()
        .map_err(sql_error)?
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("tuner run '{run_id}' is not in the projection"),
        })?;

    let manifest = conn
        .query_row(
            "SELECT game_kind, objective_id, cohort_size, finalists, seed, task_seed, \
                    shadow_policy_kind, active_elimination FROM run_manifest WHERE run_id = ?1",
            [&run_id],
            |row| {
                Ok(ManifestSummary {
                    manifest_run_id: manifest_run_id.clone(),
                    manifest_fingerprint: manifest_fingerprint.clone(),
                    game_kind: row.get(0)?,
                    objective_id: row.get(1)?,
                    cohort_size: row.get(2)?,
                    finalists: row.get(3)?,
                    seed: row.get(4)?,
                    task_seed: row.get(5)?,
                    shadow_policy_kind: row.get(6)?,
                    active_elimination: row.get::<_, i64>(7)? != 0,
                })
            },
        )
        .optional()
        .map_err(sql_error)?;

    let report = conn
        .query_row(
            "SELECT schema_version, status, validation_claim FROM run_report WHERE run_id = ?1",
            [&run_id],
            |row| {
                Ok(ReportSummary {
                    schema_version: row.get(0)?,
                    status: row.get(1)?,
                    validation_claim: row.get(2)?,
                })
            },
        )
        .optional()
        .map_err(sql_error)?;

    let mut stmt = conn
        .prepare(
            "SELECT phase, pair_attempts, completed_pairs, failed_attempts, censored_attempts, \
                    physical_games, search_iterations, wall_time_ms \
             FROM compute_phases WHERE run_id = ?1 ORDER BY phase",
        )
        .map_err(sql_error)?;
    let compute = stmt
        .query_map([&run_id], |row| {
            Ok(ComputePhase {
                phase: row.get(0)?,
                pair_attempts: row.get(1)?,
                completed_pairs: row.get(2)?,
                failed_attempts: row.get(3)?,
                censored_attempts: row.get(4)?,
                physical_games: row.get(5)?,
                search_iterations: row.get(6)?,
                wall_time_ms: row.get(7)?,
            })
        })
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;

    Ok(Json(RunDetail {
        run_id,
        terminal_status,
        report_available,
        ingest_error,
        manifest,
        report,
        compute,
    }))
}

/// `GET /api/bench/tuner/projection/runs/{run_id}/cohorts`
pub(crate) async fn cohorts(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Vec<Cohort>>, BenchError> {
    let conn = open(&state)?;
    require_run(&conn, &run_id)?;
    let mut stmt = conn
        .prepare(
            "SELECT cohort_index, candidate_ids, retained_candidate_ids \
             FROM cohorts WHERE run_id = ?1 ORDER BY cohort_index",
        )
        .map_err(sql_error)?;
    let raw = stmt
        .query_map([&run_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;
    let cohorts = raw
        .into_iter()
        .map(|(cohort_index, ids, retained)| {
            Ok(Cohort {
                cohort_index,
                candidate_ids: string_array(&ids)?,
                retained_candidate_ids: string_array(&retained)?,
            })
        })
        .collect::<Result<Vec<_>, BenchError>>()?;
    Ok(Json(cohorts))
}

/// The `candidates` row exactly as stored, before `canonical_config` is parsed
/// from its canonical-JSON `TEXT` column into a `Value`.
struct RawCandidate {
    candidate_id: String,
    fingerprint: String,
    canonical_config: String,
    cohort_index: i64,
    cohort_slot: i64,
    source: String,
    parent_candidate_id: Option<String>,
}

impl RawCandidate {
    fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        Ok(Self {
            candidate_id: row.get(0)?,
            fingerprint: row.get(1)?,
            canonical_config: row.get(2)?,
            cohort_index: row.get(3)?,
            cohort_slot: row.get(4)?,
            source: row.get(5)?,
            parent_candidate_id: row.get(6)?,
        })
    }
}

fn candidate_shape(raw: RawCandidate) -> Result<Candidate, BenchError> {
    Ok(Candidate {
        candidate_id: raw.candidate_id,
        fingerprint: raw.fingerprint,
        canonical_config: json_value(&raw.canonical_config)?,
        cohort_index: raw.cohort_index,
        cohort_slot: raw.cohort_slot,
        source: raw.source,
        parent_candidate_id: raw.parent_candidate_id,
    })
}

const CANDIDATE_COLUMNS: &str = "candidate_id, fingerprint, canonical_config, cohort_index, \
                                 cohort_slot, source, parent_candidate_id";

/// `GET /api/bench/tuner/projection/runs/{run_id}/candidates`
pub(crate) async fn candidates(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Vec<Candidate>>, BenchError> {
    let conn = open(&state)?;
    require_run(&conn, &run_id)?;
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {CANDIDATE_COLUMNS} FROM candidates WHERE run_id = ?1 \
             ORDER BY cohort_index, cohort_slot"
        ))
        .map_err(sql_error)?;
    let raw = stmt
        .query_map([&run_id], RawCandidate::from_row)
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;
    let out = raw
        .into_iter()
        .map(candidate_shape)
        .collect::<Result<Vec<_>, BenchError>>()?;
    Ok(Json(out))
}

/// `GET /api/bench/tuner/projection/runs/{run_id}/candidates/{candidate_id}`
pub(crate) async fn candidate(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath((run_id, candidate_id)): AxumPath<(String, String)>,
) -> Result<Json<Candidate>, BenchError> {
    let conn = open(&state)?;
    require_run(&conn, &run_id)?;
    let raw = conn
        .query_row(
            &format!(
                "SELECT {CANDIDATE_COLUMNS} FROM candidates \
                 WHERE run_id = ?1 AND candidate_id = ?2"
            ),
            [&run_id, &candidate_id],
            RawCandidate::from_row,
        )
        .optional()
        .map_err(sql_error)?
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("candidate '{candidate_id}' is not in run '{run_id}'"),
        })?;
    Ok(Json(candidate_shape(raw)?))
}

/// `GET /api/bench/tuner/projection/runs/{run_id}/pairs`
///
/// Filterable by `candidate` (exact id) and `cohort` (the candidate's cohort
/// index), with `limit` / `offset` pagination.
pub(crate) async fn pairs(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
    Query(filter): Query<PairFilter>,
) -> Result<Json<Vec<PairRow>>, BenchError> {
    let conn = open(&state)?;
    require_run(&conn, &run_id)?;

    let mut sql = String::from(
        "SELECT p.pair_id, p.phase, p.candidate_id, p.task_id, p.opponent_id, p.pair_utility \
         FROM pairs p",
    );
    if filter.cohort.is_some() {
        sql.push_str(
            " JOIN candidates c ON c.run_id = p.run_id AND c.candidate_id = p.candidate_id",
        );
    }
    sql.push_str(" WHERE p.run_id = ?1");
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(run_id.clone())];
    if let Some(candidate) = &filter.candidate {
        params.push(Box::new(candidate.clone()));
        sql.push_str(&format!(" AND p.candidate_id = ?{}", params.len()));
    }
    if let Some(cohort) = filter.cohort {
        params.push(Box::new(cohort));
        sql.push_str(&format!(" AND c.cohort_index = ?{}", params.len()));
    }
    let limit = filter.limit.unwrap_or(500).clamp(1, 5000);
    let offset = filter.offset.unwrap_or(0).max(0);
    params.push(Box::new(limit));
    params.push(Box::new(offset));
    sql.push_str(&format!(
        " ORDER BY p.pair_id LIMIT ?{} OFFSET ?{}",
        params.len() - 1,
        params.len()
    ));

    let mut stmt = conn.prepare(&sql).map_err(sql_error)?;
    let rows = stmt
        .query_map(
            rusqlite::params_from_iter(params.iter()),
            |row| {
                Ok(PairRow {
                    pair_id: row.get(0)?,
                    phase: row.get(1)?,
                    candidate_id: row.get(2)?,
                    task_id: row.get(3)?,
                    opponent_id: row.get(4)?,
                    pair_utility: row.get(5)?,
                })
            },
        )
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;
    Ok(Json(rows))
}

/// `GET /api/bench/tuner/projection/runs/{run_id}/pairs/{pair_id}/games`
///
/// The two seat-swapped game summaries recorded for one pair. Per-ply move
/// traces are not projected for v4 tuner runs (the tuner passes no
/// `--trace-path` to `compare eval`); only these summaries are available.
pub(crate) async fn pair_games(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath((run_id, pair_id)): AxumPath<(String, String)>,
) -> Result<Json<Vec<GameRow>>, BenchError> {
    let conn = open(&state)?;
    require_run(&conn, &run_id)?;
    let mut stmt = conn
        .prepare(
            "SELECT game_id, pair_id, candidate_side, outcome, plies, elapsed_ms, \
                    candidate_iterations_total, opponent_iterations_total \
             FROM games WHERE run_id = ?1 AND pair_id = ?2 ORDER BY game_id",
        )
        .map_err(sql_error)?;
    let rows = stmt
        .query_map([&run_id, &pair_id], |row| {
            Ok(GameRow {
                game_id: row.get(0)?,
                pair_id: row.get(1)?,
                candidate_side: row.get(2)?,
                outcome: row.get(3)?,
                plies: row.get(4)?,
                elapsed_ms: row.get(5)?,
                candidate_iterations_total: row.get(6)?,
                opponent_iterations_total: row.get(7)?,
            })
        })
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;
    Ok(Json(rows))
}

/// `GET /api/bench/tuner/projection/runs/{run_id}/validation`
pub(crate) async fn validation(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Validation>, BenchError> {
    let conn = open(&state)?;
    require_run(&conn, &run_id)?;
    let mut stmt = conn
        .prepare(
            "SELECT candidate_id, rank, estimate, lower, upper, wins, draws, losses \
             FROM validation_rows WHERE run_id = ?1 ORDER BY rank",
        )
        .map_err(sql_error)?;
    let rows = stmt
        .query_map([&run_id], |row| {
            Ok(ValidationRow {
                candidate_id: row.get(0)?,
                rank: row.get(1)?,
                estimate: row.get(2)?,
                lower: row.get(3)?,
                upper: row.get(4)?,
                wins: row.get(5)?,
                draws: row.get(6)?,
                losses: row.get(7)?,
            })
        })
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;

    let unresolved_ties = conn
        .query_row(
            "SELECT report_json FROM run_report WHERE run_id = ?1",
            [&run_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(sql_error)?
        .map(|report_json| {
            json_value(&report_json).map(|report| {
                report
                    .get("unresolved_ties")
                    .cloned()
                    .unwrap_or(Value::Null)
            })
        })
        .transpose()?
        .unwrap_or(Value::Null);

    Ok(Json(Validation {
        rows,
        unresolved_ties,
    }))
}

/// `GET /api/bench/tuner/projection/runs/{run_id}/report`
///
/// Returns the stored `report.json` verbatim (a 404 if the run has no report).
pub(crate) async fn report(
    AxumState(state): AxumState<Arc<BenchState>>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<Value>, BenchError> {
    let conn = open(&state)?;
    require_run(&conn, &run_id)?;
    let report_json = conn
        .query_row(
            "SELECT report_json FROM run_report WHERE run_id = ?1",
            [&run_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(sql_error)?
        .ok_or_else(|| BenchError {
            status: StatusCode::NOT_FOUND,
            message: format!("run '{run_id}' has no report in the projection"),
        })?;
    Ok(Json(json_value(&report_json)?))
}

/// `POST /api/bench/tuner/projection/refresh`
///
/// Re-runs the `tuner-project` projector (incremental) against the bench runs
/// root and returns its counts. This is the only write-shaped route and it
/// writes only the read model.
pub(crate) async fn refresh(
    AxumState(state): AxumState<Arc<BenchState>>,
) -> Result<Json<RefreshResult>, BenchError> {
    let [projected, skipped, ingest_errors, pruned] =
        (state.tuner_projection_refresh)(&state.bench_runs_dir, &state.tuner_projection_db)
            .map_err(|error| BenchError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("tuner projection refresh failed: {error}"),
            })?;
    Ok(Json(RefreshResult {
        projected,
        skipped,
        ingest_errors,
        pruned,
    }))
}

// ---------------------------------------------------------------------------
// Default projector shell-out
// ---------------------------------------------------------------------------

/// Parse `tuner-project`'s one summary line
/// (`projected=N skipped=N ingest_errors=N pruned=N`) into the four counts.
pub(crate) fn parse_refresh_counts(stdout: &str) -> std::io::Result<[i64; 4]> {
    let mut fields = std::collections::HashMap::new();
    for token in stdout.split_whitespace() {
        if let Some((key, value)) = token.split_once('=') {
            if let Ok(parsed) = value.parse::<i64>() {
                fields.insert(key, parsed);
            }
        }
    }
    let get = |key: &str| {
        fields.get(key).copied().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("projector output missing '{key}': {stdout:?}"),
            )
        })
    };
    Ok([
        get("projected")?,
        get("skipped")?,
        get("ingest_errors")?,
        get("pruned")?,
    ])
}

/// Shell `uv run --project tuner tuner-project --runs-root <root> --db <db>`
/// from the repository root and parse its summary line. This is the production
/// [`BenchState::tuner_projection_refresh`]; tests inject a stub.
pub fn shell_refresh(runs_root: &Path, db: &Path) -> std::io::Result<[i64; 4]> {
    let output = std::process::Command::new("uv")
        .args(["run", "--project", "tuner", "tuner-project", "--runs-root"])
        .arg(runs_root)
        .arg("--db")
        .arg(db)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "tuner-project exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    parse_refresh_counts(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_projector_summary_line() {
        assert_eq!(
            parse_refresh_counts("projected=3 skipped=1 ingest_errors=1 pruned=0\n").unwrap(),
            [3, 1, 1, 0]
        );
    }

    #[test]
    fn rejects_a_summary_line_missing_a_field() {
        assert!(parse_refresh_counts("projected=3 skipped=1").is_err());
    }
}
