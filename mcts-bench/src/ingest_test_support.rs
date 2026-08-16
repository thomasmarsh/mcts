use std::fs;

use duckdb::Connection;

use crate::attempt_store;
use crate::log::RegistryEvent;
use crate::orchestration::AttemptEvent;
use crate::projects_attempt;
use crate::schema::ensure_schema;

static FIXTURE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) struct TestFixture {
    pub(crate) _dir: std::path::PathBuf,
    pub(crate) bench_runs: std::path::PathBuf,
    pub(crate) db: Connection,
}

impl TestFixture {
    pub(crate) fn new(registry_events: &[RegistryEvent]) -> Self {
        let n = FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "mcts_bench_ingest_test_{}_{}",
            std::process::id(),
            n,
        ));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let mut content = String::new();
        for ev in registry_events {
            content.push_str(&ev.to_json_line());
            content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        TestFixture {
            _dir: dir,
            bench_runs,
            db,
        }
    }

    pub(crate) fn count(&self, table: &str) -> i64 {
        self.db
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap()
    }

    pub(crate) fn query_string(&self, sql: &str) -> String {
        self.db.query_row(sql, [], |row| row.get(0)).unwrap()
    }
}

pub(crate) fn typed_projects_fixture(run_id: &str) -> (TestFixture, std::path::PathBuf) {
    typed_projects_fixture_with_process(run_id, true)
}

pub(crate) fn typed_projects_starting_fixture(run_id: &str) -> (TestFixture, std::path::PathBuf) {
    typed_projects_fixture_with_process(run_id, false)
}

fn typed_projects_fixture_with_process(
    run_id: &str,
    process_observed: bool,
) -> (TestFixture, std::path::PathBuf) {
    let fixture = TestFixture::new(&[]);
    let run_dir = fixture.bench_runs.join(run_id);
    fs::create_dir_all(&run_dir).unwrap();
    let log_path = run_dir.join("log.jsonl");
    fs::write(&log_path, "").unwrap();
    fixture.db.execute("INSERT INTO logical_runs (logical_run_id, kind, created_at, current_attempt_id) VALUES (?1, 'experiment', CURRENT_TIMESTAMP, ?1)", duckdb::params![run_id]).unwrap();
    fixture.db.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path, logical_run_id, attempt_ordinal) VALUES (?1, 'experiment', 'nim', 'sha', false, 'host', ?2, CURRENT_TIMESTAMP, 'running', ?3, ?1, 1)", duckdb::params![run_id, std::process::id() as i64, log_path.to_string_lossy().to_string()]).unwrap();
    fixture.db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games, status) VALUES (?1, 'cell-1', 'nim', '{}', 'v', 'V', '{}', 'b', 'B', '{}', '{}', 1, 2, 'pending')", duckdb::params![run_id]).unwrap();
    let tx = fixture.db.unchecked_transaction().unwrap();
    attempt_store::initialize_attempt(&tx, run_id).unwrap();
    attempt_store::record_attempt_event(
        &tx,
        run_id,
        0,
        projects_attempt::START_REQUESTED_KEY,
        AttemptEvent::StartRequested,
        "2026-01-01T00:00:00Z",
    )
    .unwrap();
    if process_observed {
        attempt_store::record_attempt_event(
            &tx,
            run_id,
            1,
            projects_attempt::PROCESS_OBSERVED_KEY,
            AttemptEvent::ProcessObserved,
            "2026-01-01T00:00:01Z",
        )
        .unwrap();
    }
    tx.commit().unwrap();
    (fixture, log_path)
}

pub(crate) fn start_event(
    run_id: &str,
    kind: &str,
    game: &str,
    pid: u32,
    log_path: &str,
) -> RegistryEvent {
    RegistryEvent::Start {
        run_id: run_id.to_owned(),
        kind: kind.to_owned(),
        game: game.to_owned(),
        pid,
        cmd: vec!["bench".into(), "round-robin".into()],
        log_path: log_path.to_owned(),
        git_sha: "abc1234".into(),
        git_dirty: false,
        started_at: "2026-01-01T00:00:00Z".into(),
    }
}

pub(crate) fn stop_event(run_id: &str, exit_code: Option<i32>) -> RegistryEvent {
    RegistryEvent::Stop {
        run_id: run_id.to_owned(),
        exit_code,
        ended_at: "2026-01-01T01:00:00Z".into(),
    }
}
