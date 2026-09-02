#![allow(unused_imports)]
use std::convert::Infallible;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path as AxumPath, Query, State as AxumState},
    http::{HeaderValue, Method, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer};

use game_host::TunerInfo;
use mcts_bench::identity;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::log::RegistryEvent;
use mcts_bench::projects_attempt::{CellRequest, ProjectsError, StartRequest};
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::StrategyInfo;

use super::super::*;
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode as HttpStatusCode};
use mcts_bench::schema::ensure_schema;
use mcts_bench::supervised_launch::WrapperIdentity;
use tower::ServiceExt;

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

pub(super) const DEFAULT_RUN_ID: &str = "rr-druid-20260101T000000-abc1234";

/// The checked-in read-only tuner projection fixture (three runs: `version4`,
/// `version4-active-halving`, and a garbage-manifest `broken` run). Rebuild it
/// with `tests/fixtures/regenerate_tuner_projection_fixture.sh`.
pub(super) fn tuner_projection_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/bench/tests/fixtures/tuner-projection.sqlite")
}

pub(super) static FIXTURE_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Build a fully seeded test app: creates a temp dir with bench-runs/
/// subdirectory, opens an in-memory DuckDB, seeds it, then moves the
/// connection into the `BenchState`.  Returns the Router and the temp
/// dir (kept alive for the test's duration).
pub(super) fn seeded_app(seed_fn: impl FnOnce(&duckdb::Connection, &Path)) -> (Router, PathBuf) {
    seeded_app_with(seed_fn)
}

#[test]
fn adapter_fixture_ingests_and_reads_a_registry_run() {
    use mcts_bench::duckdb_composition::{BenchAdapters, BenchIngest};
    use mcts_bench::run_repository::{RunListQuery, RunRepository};

    let directory = std::env::temp_dir().join(format!(
        "mcts_bench_adapter_fixture_{}_{}",
        std::process::id(),
        FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let connection = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&connection).unwrap();
    let adapters = BenchAdapters::from_initialized_connection(connection).unwrap();
    let event = RegistryEvent::Start {
        run_id: "adapter-round-trip".into(),
        kind: "round_robin".into(),
        game: "druid".into(),
        pid: 999_999_999,
        cmd: vec!["bench".into()],
        log_path: directory.join("run.log").display().to_string(),
        git_sha: "test".into(),
        git_dirty: false,
        started_at: "2026-01-01T00:00:00Z".into(),
    };
    std::fs::write(
        directory.join("registry.log"),
        format!("{}\n", event.to_json_line()),
    )
    .unwrap();

    adapters.ingest.ingest_once(&directory).unwrap();
    let runs = adapters
        .run_repository
        .list_runs(&RunListQuery::default())
        .unwrap();
    assert_eq!(
        runs.iter()
            .map(|run| run.run_id.as_str())
            .collect::<Vec<_>>(),
        ["adapter-round-trip"]
    );
    let _ = std::fs::remove_dir_all(directory);
}

pub(super) fn seeded_app_with(
    seed_fn: impl FnOnce(&duckdb::Connection, &Path),
) -> (Router, PathBuf) {
    let (app, path, _) = seeded_app_with_state(seed_fn);
    (app, path)
}

pub(super) fn seeded_app_with_state(
    seed_fn: impl FnOnce(&duckdb::Connection, &Path),
) -> (Router, PathBuf, Arc<BenchState>) {
    seeded_app_with_state_and_signaller(
        seed_fn,
        Arc::new(|_| {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "injected missing process",
            ))
        }),
    )
}

pub(super) fn seeded_app_with_state_and_signaller(
    seed_fn: impl FnOnce(&duckdb::Connection, &Path),
    process_group_signaller: ProcessGroupSignaller,
) -> (Router, PathBuf, Arc<BenchState>) {
    let n = FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_dir =
        std::env::temp_dir().join(format!("mcts_bench_api_test_{}_{}", std::process::id(), n,));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let bench_runs_dir = tmp_dir.join("bench-runs");
    std::fs::create_dir_all(&bench_runs_dir).unwrap();

    // Create, seed, and keep the same connection — no file intermediary.
    let conn = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&conn).unwrap();
    seed_fn(&conn, &bench_runs_dir);

    let db = Arc::new(Mutex::new(conn));
    let adapters =
        mcts_bench::duckdb_composition::BenchAdapters::from_initialized_shared_connection(
            db.clone(),
        )
        .unwrap();
    let state = Arc::new(BenchState {
        db: TestDatabase::shared(db.clone()),
        projects_repository: adapters.projects_repository,
        run_repository: adapters.run_repository,
        run_command_repository: adapters.run_command_repository,
        bench_runs_dir,
        tuner_objectives_dir: tmp_dir.join("objectives"),
        process_group_signaller,
        tuner_projection_db: tuner_projection_fixture(),
        // Stub: the endpoint test asserts the handler shapes these counts; the
        // real projector shell-out is covered by `tuner_api`'s parser unit test
        // and by code review (see the plan's `refresh_is_only_spawn` claim).
        tuner_projection_refresh: Arc::new(|_, _| Ok([2, 1, 0, 0])),
    });

    (bench_router(state.clone()), tmp_dir, state)
}

/// Default seed: one completed run with two match results and one trial.
pub(super) fn default_seed(conn: &duckdb::Connection, _bench_runs_dir: &Path) {
    conn.execute(
        "INSERT INTO runs \
         (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
         VALUES (?1, 'round_robin', 'druid', 'abc1234', false, 'testhost', NULL, \
                 '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/nope/log.jsonl')",
        duckdb::params![DEFAULT_RUN_ID],
    ).unwrap();

    // Two matches: "strong" beats "master", "master" beats "strong" (1-1).
    conn.execute(
        "INSERT INTO match_results (run_id, seq, ts, strategy_a, strategy_b, outcome, winner) \
         VALUES (?1, 1, '2026-01-01T00:00:10Z', 'strong', 'master', 'win_a', 'strong'),\
                (?1, 2, '2026-01-01T00:00:20Z', 'master', 'strong', 'win_a', 'master')",
        duckdb::params![DEFAULT_RUN_ID],
    )
    .unwrap();

    // One trial for good measure.
    conn.execute(
        "INSERT INTO trials (run_id, trial_id, ts, config, cost) \
         VALUES (?1, 1, '2026-01-01T00:00:30Z', '{}', 0.375)",
        duckdb::params![DEFAULT_RUN_ID],
    )
    .unwrap();
}

/// Seed a run that is still `running` (no ended_at, no stop event).
pub(super) fn running_run_seed(conn: &duckdb::Connection, _bench_runs_dir: &Path) {
    conn.execute(
        "INSERT INTO runs \
         (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) \
         VALUES ('running-run', 'round_robin', 'druid', 'def5678', false, 'testhost', 12345, \
                 '2026-02-01T00:00:00Z', 'running', '/tmp/running/log.jsonl')",
        duckdb::params![],
    )
    .unwrap();
}

/// Seed with the default run plus a second run for multi-run queries.
pub(super) fn multi_run_seed(conn: &duckdb::Connection, _bench_runs_dir: &Path) {
    default_seed(conn, _bench_runs_dir);
    conn.execute(
        "INSERT INTO runs \
         (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, ended_at, status, log_path) \
         VALUES ('rr-ttt-20260201T000000-def5678', 'round_robin', 'ttt', 'def5678', false, 'testhost', \
                 NULL, '2026-02-01T00:00:00Z', '2026-02-01T02:00:00Z', 'completed', '/tmp/ttt/log.jsonl')",
        duckdb::params![],
    ).unwrap();
}

/// Default seed plus a two-ply trace for `match_results.seq = 1` (game
/// 1: "strong" beats "master") -- exercises the join between
/// `game_moves` and `match_results` on `(run_id, seq == game_seq)`.
pub(super) fn game_moves_seed(conn: &duckdb::Connection, bench_runs_dir: &Path) {
    default_seed(conn, bench_runs_dir);
    conn.execute(
        "INSERT INTO game_moves (run_id, game_seq, ply, ts, state, mv, player) \
         VALUES \
         (?1, 1, 0, '2026-01-01T00:00:10Z', '{\"board\":[]}', NULL, NULL), \
         (?1, 1, 1, '2026-01-01T00:00:11Z', '{\"board\":[1]}', '4', 'strong')",
        duckdb::params![DEFAULT_RUN_ID],
    )
    .unwrap();
}

pub(super) fn body_json(body: &axum::body::Bytes) -> Value {
    serde_json::from_slice(body).unwrap()
}

pub(super) async fn http_get(app: Router, uri: &str) -> (HttpStatusCode, axum::body::Bytes) {
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, body)
}

pub(super) async fn http_post_json(
    app: Router,
    uri: &str,
    json: Value,
) -> (HttpStatusCode, axum::body::Bytes) {
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, body)
}

pub(super) async fn http_delete(app: Router, uri: &str) -> (HttpStatusCode, axum::body::Bytes) {
    let resp = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, body)
}
