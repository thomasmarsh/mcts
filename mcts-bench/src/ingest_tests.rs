use super::test_support::*;
use std::fs;
use std::io::Write;

use crate::log::LogRecord;
use crate::schema::ensure_schema;

use super::*;

#[test]
fn test_registry_start_creates_run_row() {
    let ev = start_event(
        "run-1",
        "round_robin",
        "druid",
        99999,
        "/tmp/nope/log.jsonl",
    );
    let fix = TestFixture::new(&[ev]);

    ingest_once(&fix.db, &fix.bench_runs).unwrap();

    assert_eq!(fix.count("runs"), 1);
    assert_eq!(
        fix.query_string("SELECT status FROM runs WHERE run_id = 'run-1'"),
        "crashed",
    );
    assert_eq!(fix.count("_ingest_cursor"), 1);
}

#[test]
fn registry_replay_does_not_clobber_server_identity() {
    let ev = start_event("child-run", "tuner", "nim", 99996, "/tmp/child/log.jsonl");
    let fix = TestFixture::new(&[ev]);
    fix.db
            .execute(
                "INSERT INTO logical_runs (logical_run_id, kind, created_at, current_attempt_id) VALUES ('logical-root', 'tuner', CURRENT_TIMESTAMP, 'child-run');
                 INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path, logical_run_id, parent_attempt_id, attempt_ordinal) VALUES ('child-run', 'tuner', 'nim', 'server', false, 'server', CURRENT_TIMESTAMP, 'running', '/tmp/server/log.jsonl', 'logical-root', 'parent-run', 2)",
                [],
            )
            .unwrap();

    ingest_once(&fix.db, &fix.bench_runs).unwrap();

    let identity: (String, String, u64) = fix
            .db
            .query_row(
                "SELECT logical_run_id, parent_attempt_id, attempt_ordinal FROM runs WHERE run_id = 'child-run'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
    assert_eq!(identity, ("logical-root".into(), "parent-run".into(), 2));
}

#[test]
fn test_registry_start_stop_marks_completed() {
    let ev_start = start_event(
        "run-2",
        "round_robin",
        "druid",
        99998,
        "/tmp/nope2/log.jsonl",
    );
    let ev_stop = stop_event("run-2", Some(0));
    let fix = TestFixture::new(&[ev_start, ev_stop]);

    ingest_once(&fix.db, &fix.bench_runs).unwrap();

    assert_eq!(
        fix.query_string("SELECT status FROM runs WHERE run_id = 'run-2'"),
        "completed"
    );
    let exit_code: i64 = fix
        .db
        .query_row(
            "SELECT exit_code FROM runs WHERE run_id = 'run-2'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(exit_code, 0);
    let identity: (String, Option<String>, u64, String) = fix
            .db
            .query_row(
                "SELECT r.logical_run_id, r.parent_attempt_id, r.attempt_ordinal, l.current_attempt_id FROM runs r JOIN logical_runs l ON l.logical_run_id = r.logical_run_id WHERE r.run_id = 'run-2'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
    assert_eq!(identity, ("run-2".into(), None, 1, "run-2".into()));
}

#[test]
fn test_registry_start_stop_marks_tuner_crashed_on_nonzero_exit() {
    // A tuner (or other non-experiment) run whose process exits nonzero
    // -- e.g. it dies during the tuner's preflight check before spawning any
    // trials -- must land as 'crashed', not silently as 'completed'.
    // Only the 'experiment' kind used to check exit_code here; every
    // other kind unconditionally marked itself 'completed'.
    let ev_start = start_event(
        "run-crash",
        "tuner",
        "traffic-lights",
        99996,
        "/tmp/nope4/log.jsonl",
    );
    let ev_stop = stop_event("run-crash", Some(1));
    let fix = TestFixture::new(&[ev_start, ev_stop]);

    ingest_once(&fix.db, &fix.bench_runs).unwrap();

    assert_eq!(
        fix.query_string("SELECT status FROM runs WHERE run_id = 'run-crash'"),
        "crashed"
    );
    let exit_code: i64 = fix
        .db
        .query_row(
            "SELECT exit_code FROM runs WHERE run_id = 'run-crash'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(exit_code, 1);
}

#[test]
fn test_registry_stop_does_not_clobber_an_already_terminal_status() {
    // A run already marked 'stopped' (e.g. by the explicit stop_run
    // handler) must stay 'stopped' when a Stop registry event for it
    // is later ingested -- the launcher's own reaper thread writes one
    // for every exit, including a process that was SIGTERM'd, and it
    // races against (may land before or after) whatever else already
    // set the terminal status. See the `AND status = 'running'` guard
    // this test exercises.
    let ev_start = start_event("run-3", "tuner", "nim", 99997, "/tmp/nope3/log.jsonl");
    let fix = TestFixture::new(&[ev_start]);
    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    fix.db
        .execute(
            "UPDATE runs SET status = 'stopped' WHERE run_id = 'run-3'",
            [],
        )
        .unwrap();

    // Now a Stop event for the same run arrives on a later ingest pass.
    fs::write(
        fix.bench_runs.join("registry.log"),
        format!(
            "{}\n{}\n",
            start_event("run-3", "tuner", "nim", 99997, "/tmp/nope3/log.jsonl").to_json_line(),
            stop_event("run-3", Some(0)).to_json_line(),
        ),
    )
    .unwrap();
    ingest_once(&fix.db, &fix.bench_runs).unwrap();

    assert_eq!(
        fix.query_string("SELECT status FROM runs WHERE run_id = 'run-3'"),
        "stopped",
    );
}

#[test]
fn test_registry_stop_without_start_is_benign() {
    let ev = stop_event("orphan-run", Some(1));
    let fix = TestFixture::new(&[ev]);

    ingest_once(&fix.db, &fix.bench_runs).unwrap();
    assert_eq!(fix.count("runs"), 0);
}

#[test]
fn test_ingest_match_results() {
    let fix = {
        let dir = std::env::temp_dir().join(format!("mcts_bench_ingest_mr_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let run_id = "mr-run";
        let run_dir = bench_runs.join(run_id);
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();

        let reg_events = vec![start_event(
            run_id,
            "round_robin",
            "druid",
            99997,
            &log_path_str,
        )];

        let mut reg_content = String::new();
        for ev in &reg_events {
            reg_content.push_str(&ev.to_json_line());
            reg_content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

        let records = vec![
            LogRecord::MatchResult {
                seq: 1,
                strategy_a: "strong".into(),
                strategy_b: "master".into(),
                outcome: "win_a".into(),
                winner: Some("strong".into()),
                extra: None,
                cell_id: None,
                seed: None,
                trace_game_seq: None,
                metrics: None,
            },
            LogRecord::MatchResult {
                seq: 2,
                strategy_a: "master".into(),
                strategy_b: "strong".into(),
                outcome: "win_b".into(),
                winner: Some("strong".into()),
                extra: None,
                cell_id: None,
                seed: None,
                trace_game_seq: None,
                metrics: None,
            },
            LogRecord::MatchResult {
                seq: 3,
                strategy_a: "easy".into(),
                strategy_b: "master".into(),
                outcome: "draw".into(),
                winner: None,
                extra: None,
                cell_id: None,
                seed: None,
                trace_game_seq: None,
                metrics: None,
            },
        ];
        let mut log_content = String::new();
        for rec in &records {
            log_content.push_str(&rec.to_json_line());
            log_content.push('\n');
        }
        fs::write(&log_path, &log_content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        (bench_runs, db)
    };

    ingest_once(&fix.1, &fix.0).unwrap();

    let count: i64 = fix
        .1
        .query_row("SELECT COUNT(*) FROM match_results", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 3);

    let null_outcomes: i64 = fix
        .1
        .query_row(
            "SELECT COUNT(*) FROM match_results WHERE outcome IS NULL",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(null_outcomes, 0);
}

#[test]
fn test_ingest_idempotent() {
    let (bench_runs, db) = {
        let dir = std::env::temp_dir().join(format!("mcts_bench_idemp_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let run_id = "idem-run";
        let run_dir = bench_runs.join(run_id);
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();

        let reg_events = vec![start_event(
            run_id,
            "round_robin",
            "druid",
            99996,
            &log_path_str,
        )];
        let mut reg_content = String::new();
        for ev in &reg_events {
            reg_content.push_str(&ev.to_json_line());
            reg_content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

        let records = vec![LogRecord::MatchResult {
            seq: 1,
            strategy_a: "a".into(),
            strategy_b: "b".into(),
            outcome: "win_a".into(),
            winner: Some("a".into()),
            extra: None,
            cell_id: None,
            seed: None,
            trace_game_seq: None,
            metrics: None,
        }];
        let mut log_content = String::new();
        for rec in &records {
            log_content.push_str(&rec.to_json_line());
            log_content.push('\n');
        }
        fs::write(&log_path, &log_content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        (bench_runs, db)
    };

    ingest_once(&db, &bench_runs).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        1
    );

    ingest_once(&db, &bench_runs).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        1,
        "second ingest should not duplicate match results"
    );
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM runs", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        1,
        "second ingest should not duplicate runs"
    );
}

#[test]
fn test_ingest_skips_unparseable_log_lines() {
    let (bench_runs, db) = {
        let dir = std::env::temp_dir().join(format!("mcts_bench_garbage_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let run_id = "garbage-run";
        let run_dir = bench_runs.join(run_id);
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();

        let reg_events = vec![start_event(
            run_id,
            "round_robin",
            "druid",
            99995,
            &log_path_str,
        )];
        let mut reg_content = String::new();
        for ev in &reg_events {
            reg_content.push_str(&ev.to_json_line());
            reg_content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

        let good = LogRecord::MatchResult {
            seq: 1,
            strategy_a: "x".into(),
            strategy_b: "y".into(),
            outcome: "draw".into(),
            winner: None,
            extra: None,
            cell_id: None,
            seed: None,
            trace_game_seq: None,
            metrics: None,
        };
        let mut log_content = String::new();
        log_content.push_str(&good.to_json_line());
        log_content.push('\n');
        log_content.push_str("this is not json\n");
        log_content.push_str("{\"type\": \"unknown_thing\", \"data\": 42}\n");
        fs::write(&log_path, &log_content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        (bench_runs, db)
    };

    ingest_once(&db, &bench_runs).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        1,
    );
}

#[test]
fn test_ingest_missing_log_jsonl_is_not_fatal() {
    let alive_pid = std::process::id();
    let ev = start_event(
        "ghost-run",
        "round_robin",
        "druid",
        alive_pid,
        "/tmp/nope/log.jsonl",
    );
    let fix = TestFixture::new(&[ev]);

    ingest_once(&fix.db, &fix.bench_runs).unwrap();

    assert_eq!(fix.count("runs"), 1);
    assert_eq!(
        fix.query_string("SELECT status FROM runs WHERE run_id = 'ghost-run'"),
        "running",
    );
}

#[test]
fn test_heartbeats_are_skipped() {
    let (bench_runs, db) = {
        let dir = std::env::temp_dir().join(format!("mcts_bench_hb_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let run_id = "hb-run";
        let run_dir = bench_runs.join(run_id);
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();

        let reg_events = vec![start_event(
            run_id,
            "round_robin",
            "druid",
            99993,
            &log_path_str,
        )];
        let mut reg_content = String::new();
        for ev in &reg_events {
            reg_content.push_str(&ev.to_json_line());
            reg_content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

        let records = vec![
            LogRecord::Heartbeat { games_played: 10 },
            LogRecord::MatchResult {
                seq: 1,
                strategy_a: "a".into(),
                strategy_b: "b".into(),
                outcome: "win_a".into(),
                winner: Some("a".into()),
                extra: None,
                cell_id: None,
                seed: None,
                trace_game_seq: None,
                metrics: None,
            },
            LogRecord::Heartbeat { games_played: 20 },
        ];
        let mut log_content = String::new();
        for rec in &records {
            log_content.push_str(&rec.to_json_line());
            log_content.push('\n');
        }
        fs::write(&log_path, &log_content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        (bench_runs, db)
    };

    ingest_once(&db, &bench_runs).unwrap();

    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM trials", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn test_ingest_trials() {
    let (bench_runs, db) = {
        let dir = std::env::temp_dir().join(format!("mcts_bench_trial_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let run_id = "tuner-run";
        let run_dir = bench_runs.join(run_id);
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();

        let reg_events = vec![start_event(run_id, "tuner", "druid", 99992, &log_path_str)];
        let mut reg_content = String::new();
        for ev in &reg_events {
            reg_content.push_str(&ev.to_json_line());
            reg_content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

        let records = vec![
            LogRecord::Trial {
                trial_id: 1,
                config: serde_json::json!({"lr": 0.001, "iterations": 100}),
                seed: Some(42),
                cost: 0.375,
                extra: None,
            },
            LogRecord::Trial {
                trial_id: 2,
                config: serde_json::json!({"lr": 0.01, "iterations": 200}),
                seed: None,
                cost: 0.512,
                extra: Some(serde_json::json!({"note": "second trial"})),
            },
        ];
        let mut log_content = String::new();
        for rec in &records {
            log_content.push_str(&rec.to_json_line());
            log_content.push('\n');
        }
        fs::write(&log_path, &log_content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        (bench_runs, db)
    };

    ingest_once(&db, &bench_runs).unwrap();

    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM trials", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        2
    );

    let cost: f64 = db
        .query_row(
            "SELECT cost FROM trials WHERE run_id = 'tuner-run' AND trial_id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!((cost - 0.375).abs() < 1e-9);

    let seed: Option<i64> = db
        .query_row(
            "SELECT seed FROM trials WHERE run_id = 'tuner-run' AND trial_id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(seed, Some(42));

    let extra: Option<String> = db
        .query_row(
            "SELECT extra FROM trials WHERE run_id = 'tuner-run' AND trial_id = 2",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(extra.is_some());
    let parsed: serde_json::Value = serde_json::from_str(&extra.unwrap()).unwrap();
    assert_eq!(parsed["note"], "second trial");
}

#[test]
fn test_ingest_incumbent_upserts_latest() {
    let (bench_runs, db) = {
        let dir = std::env::temp_dir().join(format!("mcts_bench_incumbent_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let run_id = "tuner-run";
        let run_dir = bench_runs.join(run_id);
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();

        let reg_events = vec![start_event(run_id, "tuner", "druid", 99993, &log_path_str)];
        let mut reg_content = String::new();
        for ev in &reg_events {
            reg_content.push_str(&ev.to_json_line());
            reg_content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

        // Two incumbent records for the same run -- the intensifier
        // found a better config partway through, so the second should
        // overwrite the first rather than both landing as separate rows.
        let records = vec![
            LogRecord::Incumbent {
                config: serde_json::json!({"family": "ucb1", "c": 1.0}),
                cost: 0.5,
                extra: None,
            },
            LogRecord::Incumbent {
                config: serde_json::json!({"family": "rave", "c": 0.7}),
                cost: 0.2,
                extra: Some(serde_json::json!({"hash": "abc123"})),
            },
        ];
        let mut log_content = String::new();
        for rec in &records {
            log_content.push_str(&rec.to_json_line());
            log_content.push('\n');
        }
        fs::write(&log_path, &log_content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        (bench_runs, db)
    };

    ingest_once(&db, &bench_runs).unwrap();

    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM incumbents", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        1,
        "a run's incumbent should overwrite in place, not accumulate rows"
    );

    let (cost, config_str, extra_str): (f64, String, Option<String>) = db
        .query_row(
            "SELECT cost, CAST(config AS TEXT), CAST(extra AS TEXT) \
                 FROM incumbents WHERE run_id = 'tuner-run'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert!((cost - 0.2).abs() < 1e-9);
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    assert_eq!(config["family"], "rave");
    let extra: serde_json::Value = serde_json::from_str(&extra_str.unwrap()).unwrap();
    assert_eq!(extra["hash"], "abc123");
}

#[test]
fn test_ingest_tails_moves_jsonl_sibling_of_log_jsonl() {
    // Move traces land in a `moves.jsonl` next to `log.jsonl`, not
    // inside it (see `LogRecord::Move`'s doc comment) -- this proves
    // `process_run_logs` derives and tails that sibling path too, not
    // just the registered `log_path` itself.
    let (bench_runs, db) = {
        let dir =
            std::env::temp_dir().join(format!("mcts_bench_moves_sibling_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let run_id = "sibling-moves-run";
        let run_dir = bench_runs.join(run_id);
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();
        let moves_path = run_dir.join("moves.jsonl");

        let reg_events = vec![start_event(
            run_id,
            "round_robin",
            "druid",
            99990,
            &log_path_str,
        )];
        let mut reg_content = String::new();
        for ev in &reg_events {
            reg_content.push_str(&ev.to_json_line());
            reg_content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

        // The main log only carries the match result...
        let match_result = LogRecord::MatchResult {
            seq: 1,
            strategy_a: "a".into(),
            strategy_b: "b".into(),
            outcome: "win_a".into(),
            winner: Some("a".into()),
            extra: None,
            cell_id: None,
            seed: None,
            trace_game_seq: None,
            metrics: None,
        };
        fs::write(&log_path, format!("{}\n", match_result.to_json_line())).unwrap();

        // ...while the moves live in the sibling file.
        let mv = LogRecord::Move {
            game_seq: 1,
            ply: 0,
            state: serde_json::json!({"board": []}),
            mv: None,
            player: None,
        };
        fs::write(&moves_path, format!("{}\n", mv.to_json_line())).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        (bench_runs, db)
    };

    ingest_once(&db, &bench_runs).unwrap();

    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM game_moves", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[test]
fn test_ingest_moves() {
    let (bench_runs, db) = {
        let dir = std::env::temp_dir().join(format!("mcts_bench_moves_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bench_runs = dir.join("bench-runs");
        fs::create_dir_all(&bench_runs).unwrap();

        let run_id = "moves-run";
        let run_dir = bench_runs.join(run_id);
        fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        let log_path_str = log_path.to_string_lossy().to_string();

        let reg_events = vec![start_event(run_id, "tuner", "druid", 99994, &log_path_str)];
        let mut reg_content = String::new();
        for ev in &reg_events {
            reg_content.push_str(&ev.to_json_line());
            reg_content.push('\n');
        }
        fs::write(bench_runs.join("registry.log"), &reg_content).unwrap();

        let records = vec![
            LogRecord::Move {
                game_seq: 7,
                ply: 0,
                state: serde_json::json!({"board": []}),
                mv: None,
                player: None,
            },
            LogRecord::Move {
                game_seq: 7,
                ply: 1,
                state: serde_json::json!({"board": [1]}),
                mv: Some(serde_json::json!({"cell": 0})),
                player: Some("strong".into()),
            },
        ];
        let mut log_content = String::new();
        for rec in &records {
            log_content.push_str(&rec.to_json_line());
            log_content.push('\n');
        }
        fs::write(&log_path, &log_content).unwrap();

        let db = duckdb::Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();

        (bench_runs, db)
    };

    ingest_once(&db, &bench_runs).unwrap();

    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM game_moves", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        2
    );

    let (state_str, mv_str, player): (String, Option<String>, Option<String>) = db
        .query_row(
            "SELECT CAST(state AS TEXT), CAST(mv AS TEXT), player \
                 FROM game_moves WHERE run_id = 'moves-run' AND game_seq = 7 AND ply = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    let state: serde_json::Value = serde_json::from_str(&state_str).unwrap();
    assert_eq!(state["board"][0], 1);
    let mv: serde_json::Value = serde_json::from_str(&mv_str.unwrap()).unwrap();
    assert_eq!(mv["cell"], 0);
    assert_eq!(player.as_deref(), Some("strong"));

    // Idempotent re-ingest should not duplicate rows.
    ingest_once(&db, &bench_runs).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM game_moves", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn test_registry_garbage_lines_are_skipped() {
    let dir = std::env::temp_dir().join(format!("mcts_bench_reg_garbage_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let bench_runs = dir.join("bench-runs");
    fs::create_dir_all(&bench_runs).unwrap();

    let ev = start_event(
        "garb-run",
        "round_robin",
        "druid",
        99991,
        "/tmp/nope/log.jsonl",
    );
    let mut content = String::new();
    content.push_str("totally not json\n");
    content.push_str(&ev.to_json_line());
    content.push('\n');
    content.push_str("also not json\n");
    fs::write(bench_runs.join("registry.log"), &content).unwrap();

    let db = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&db).unwrap();

    ingest_once(&db, &bench_runs).unwrap();

    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM runs", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        1,
    );
}

#[test]
fn test_experiment_cell_ingestion_is_idempotent_and_keeps_trace_mapping() {
    let dir = std::env::temp_dir().join(format!(
        "mcts_bench_experiment_ingest_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let bench_runs = dir.join("bench-runs");
    let run_dir = bench_runs.join("experiment-run");
    fs::create_dir_all(&run_dir).unwrap();
    let log_path = run_dir.join("log.jsonl");
    fs::write(
        bench_runs.join("registry.log"),
        format!(
            "{}\n",
            start_event(
                "experiment-run",
                "experiment",
                "nim",
                99991,
                &log_path.to_string_lossy()
            )
            .to_json_line()
        ),
    )
    .unwrap();
    let records = [
        LogRecord::CellStarted {
            cell_id: "cell-1".into(),
        },
        LogRecord::MatchResult {
            seq: 1,
            strategy_a: "Candidate".into(),
            strategy_b: "Baseline".into(),
            outcome: "win_a".into(),
            winner: Some("Candidate".into()),
            extra: None,
            cell_id: Some("cell-1".into()),
            seed: Some(42),
            trace_game_seq: Some(177),
            metrics: Some(serde_json::json!({"outcome":"candidate_win","plies":3})),
        },
        LogRecord::MatchResult {
            seq: 2,
            strategy_a: "Baseline".into(),
            strategy_b: "Candidate".into(),
            outcome: "draw".into(),
            winner: None,
            extra: None,
            cell_id: Some("cell-1".into()),
            seed: Some(43),
            trace_game_seq: Some(178),
            metrics: Some(serde_json::json!({"outcome":"draw","plies":4})),
        },
        LogRecord::CellFinished {
            cell_id: "cell-1".into(),
            completed_games: 2,
        },
    ];
    fs::write(
        &log_path,
        records
            .iter()
            .map(|record| format!("{}\n", record.to_json_line()))
            .collect::<String>(),
    )
    .unwrap();
    let db = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&db).unwrap();
    db.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) VALUES ('experiment-run', 'experiment', 'nim', 'test', false, 'test', NULL, CURRENT_TIMESTAMP, 'running', ?1)", [&log_path.to_string_lossy()]).unwrap();
    db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games) VALUES ('experiment-run', 'cell-1', 'nim', 'null', 'candidate', 'Candidate', '{}', 'base', 'Baseline', '{}', '{\"kind\":\"iterations\",\"value\":1}', 1, 2)", []).unwrap();
    ingest_once(&db, &bench_runs).unwrap();
    ingest_once(&db, &bench_runs).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM match_results", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        db.query_row(
            "SELECT completed_games FROM experiment_cells WHERE run_id = 'experiment-run'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    assert_eq!(
        db.query_row(
            "SELECT trace_game_seq FROM match_results WHERE seq = 1",
            [],
            |row| row.get::<_, u64>(0)
        )
        .unwrap(),
        177
    );
}

#[test]
fn live_cell_failure_waits_for_coordinator_and_later_logs_are_ingested() {
    let dir = std::env::temp_dir().join(format!(
        "mcts_bench_live_cell_failure_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let bench_runs = dir.join("bench-runs");
    let run_dir = bench_runs.join("live-failure-run");
    fs::create_dir_all(&run_dir).unwrap();
    let log_path = run_dir.join("log.jsonl");
    fs::write(
        &log_path,
        format!(
            "{}\n",
            LogRecord::CellFailed {
                cell_id: "cell-000001".into(),
                completed_games: 3,
                error: "candidate rejected".into(),
            }
            .to_json_line()
        ),
    )
    .unwrap();
    fs::write(
        bench_runs.join("registry.log"),
        format!(
            "{}\n",
            start_event(
                "live-failure-run",
                "experiment",
                "nim",
                std::process::id(),
                &log_path.to_string_lossy()
            )
            .to_json_line()
        ),
    )
    .unwrap();

    let db = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&db).unwrap();
    process_registry(&db, &bench_runs.join("registry.log")).unwrap();
    db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games, status) VALUES ('live-failure-run', 'cell-000001', 'nim', '{}', 'v1', 'V1', '{}', 'b', 'B', '{}', '{}', 2, 4, 'pending'), ('live-failure-run', 'cell-000002', 'nim', '{}', 'v2', 'V2', '{}', 'b', 'B', '{}', '{}', 2, 4, 'pending')", []).unwrap();

    ingest_once(&db, &bench_runs).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT status FROM runs WHERE run_id = 'live-failure-run'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "running"
    );
    assert_eq!(
        db.query_row(
            "SELECT status FROM experiment_cells WHERE cell_id = 'cell-000001'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "failed"
    );
    assert_eq!(
        db.query_row(
            "SELECT completed_games FROM experiment_cells WHERE cell_id = 'cell-000001'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        3
    );

    let successful_match = LogRecord::MatchResult {
        seq: 1,
        strategy_a: "V2".into(),
        strategy_b: "B".into(),
        outcome: "win_a".into(),
        winner: Some("V2".into()),
        extra: None,
        cell_id: Some("cell-000002".into()),
        seed: Some(7),
        trace_game_seq: None,
        metrics: None,
    };
    let successful_finish = LogRecord::CellFinished {
        cell_id: "cell-000002".into(),
        completed_games: 1,
    };
    let mut log = fs::OpenOptions::new().append(true).open(&log_path).unwrap();
    writeln!(log, "{}", successful_match.to_json_line()).unwrap();
    writeln!(log, "{}", successful_finish.to_json_line()).unwrap();
    let mut registry = fs::OpenOptions::new()
        .append(true)
        .open(bench_runs.join("registry.log"))
        .unwrap();
    writeln!(
        registry,
        "{}",
        stop_event("live-failure-run", Some(0)).to_json_line()
    )
    .unwrap();

    ingest_once(&db, &bench_runs).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT status FROM runs WHERE run_id = 'live-failure-run'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "completed_with_errors"
    );
    assert_eq!(
        db.query_row(
            "SELECT COUNT(*) FROM match_results WHERE run_id = 'live-failure-run'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(
        db.query_row(
            "SELECT status FROM experiment_cells WHERE cell_id = 'cell-000002'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "completed"
    );
}

#[test]
fn late_cell_events_do_not_change_stopped_or_cancelled_state() {
    let dir = std::env::temp_dir().join(format!(
        "mcts_bench_stopped_late_cell_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    let bench_runs = dir.join("bench-runs");
    let run_dir = bench_runs.join("stopped-run");
    fs::create_dir_all(&run_dir).unwrap();
    let log_path = run_dir.join("log.jsonl");
    fs::write(&log_path, "").unwrap();
    fs::write(
        bench_runs.join("registry.log"),
        format!(
            "{}\n",
            start_event(
                "stopped-run",
                "experiment",
                "nim",
                std::process::id(),
                &log_path.to_string_lossy()
            )
            .to_json_line()
        ),
    )
    .unwrap();
    let db = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&db).unwrap();
    process_registry(&db, &bench_runs.join("registry.log")).unwrap();
    db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games, completed_games, status) VALUES ('stopped-run', 'cell-000001', 'nim', '{}', 'v1', 'V1', '{}', 'b', 'B', '{}', '{}', 1, 2, 2, 'completed'), ('stopped-run', 'cell-000002', 'nim', '{}', 'v2', 'V2', '{}', 'b', 'B', '{}', '{}', 1, 2, 1, 'failed'), ('stopped-run', 'cell-000003', 'nim', '{}', 'v3', 'V3', '{}', 'b', 'B', '{}', '{}', 1, 2, 0, 'cancelled')", []).unwrap();
    db.execute(
        "UPDATE runs SET status = 'stopped' WHERE run_id = 'stopped-run'",
        [],
    )
    .unwrap();

    let late = [
        LogRecord::CellFinished {
            cell_id: "cell-000003".into(),
            completed_games: 2,
        },
        LogRecord::CellFailed {
            cell_id: "cell-000001".into(),
            completed_games: 2,
            error: "late failure".into(),
        },
    ];
    fs::write(
        &log_path,
        late.iter()
            .map(|record| format!("{}\n", record.to_json_line()))
            .collect::<String>(),
    )
    .unwrap();
    ingest_once(&db, &bench_runs).unwrap();

    assert_eq!(
        db.query_row(
            "SELECT status FROM runs WHERE run_id = 'stopped-run'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "stopped"
    );
    let statuses: Vec<String> = db
        .prepare(
            "SELECT status FROM experiment_cells WHERE run_id = 'stopped-run' ORDER BY cell_id",
        )
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(statuses, vec!["completed", "failed", "cancelled"]);
}

#[test]
fn test_cell_failure_is_ingested_after_registry_stop() {
    let dir = std::env::temp_dir().join(format!("mcts_bench_cell_failure_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let bench_runs = dir.join("bench-runs");
    let run_dir = bench_runs.join("failed-run");
    fs::create_dir_all(&run_dir).unwrap();
    let log_path = run_dir.join("log.jsonl");
    fs::write(
        &log_path,
        format!(
            "{}\n",
            LogRecord::CellFailed {
                cell_id: "cell-1".into(),
                completed_games: 1,
                error: "child failed".into()
            }
            .to_json_line()
        ),
    )
    .unwrap();
    fs::write(
        bench_runs.join("registry.log"),
        format!(
            "{}\n{}\n",
            start_event(
                "failed-run",
                "experiment",
                "nim",
                99991,
                &log_path.to_string_lossy()
            )
            .to_json_line(),
            stop_event("failed-run", Some(1)).to_json_line()
        ),
    )
    .unwrap();
    let db = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&db).unwrap();
    process_registry(&db, &bench_runs.join("registry.log")).unwrap();
    db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games) VALUES ('failed-run', 'cell-1', 'nim', 'null', 'variant', 'Variant', '{}', 'baseline', 'Baseline', '{}', '{\"kind\":\"iterations\",\"value\":1}', 1, 2)", []).unwrap();
    process_run_logs(&db).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT status FROM runs WHERE run_id = 'failed-run'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "crashed"
    );
    assert_eq!(
        db.query_row(
            "SELECT status FROM experiment_cells WHERE run_id = 'failed-run'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "failed"
    );
}

#[test]
fn experiment_stop_then_late_failure_upgrades_completed_status() {
    let dir = std::env::temp_dir().join(format!("mcts_bench_late_failure_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let bench_runs = dir.join("bench-runs");
    let run_dir = bench_runs.join("late-run");
    fs::create_dir_all(&run_dir).unwrap();
    let log_path = run_dir.join("log.jsonl");
    fs::write(&log_path, "").unwrap();
    fs::write(
        bench_runs.join("registry.log"),
        format!(
            "{}\n",
            start_event(
                "late-run",
                "experiment",
                "nim",
                std::process::id(),
                &log_path.to_string_lossy()
            )
            .to_json_line()
        ),
    )
    .unwrap();
    let db = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&db).unwrap();
    process_registry(&db, &bench_runs.join("registry.log")).unwrap();
    db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games) VALUES ('late-run', 'cell-000001', 'nim', '{}', 'v', 'V', '{}', 'b', 'B', '{}', '{}', 1, 2)", []).unwrap();
    fs::write(
        &log_path,
        format!(
            "{}\n",
            LogRecord::CellFailed {
                cell_id: "cell-000001".into(),
                completed_games: 1,
                error: "late child failure".into()
            }
            .to_json_line()
        ),
    )
    .unwrap();
    let mut registry = fs::OpenOptions::new()
        .append(true)
        .open(bench_runs.join("registry.log"))
        .unwrap();
    writeln!(
        registry,
        "{}",
        stop_event("late-run", Some(0)).to_json_line()
    )
    .unwrap();
    ingest_once(&db, &bench_runs).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT status FROM runs WHERE run_id = 'late-run'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "completed_with_errors"
    );
}

#[test]
fn nonzero_experiment_exit_cleans_running_and_pending_cells() {
    let dir = std::env::temp_dir().join(format!("mcts_bench_crash_cleanup_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let bench_runs = dir.join("bench-runs");
    let run_dir = bench_runs.join("crashed-run");
    fs::create_dir_all(&run_dir).unwrap();
    let log_path = run_dir.join("log.jsonl");
    fs::write(&log_path, "").unwrap();
    fs::write(
        bench_runs.join("registry.log"),
        format!(
            "{}\n",
            start_event(
                "crashed-run",
                "experiment",
                "nim",
                std::process::id(),
                &log_path.to_string_lossy()
            )
            .to_json_line()
        ),
    )
    .unwrap();
    let db = duckdb::Connection::open_in_memory().unwrap();
    ensure_schema(&db).unwrap();
    process_registry(&db, &bench_runs.join("registry.log")).unwrap();
    db.execute("INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games, status) VALUES ('crashed-run', 'cell-000001', 'nim', '{}', 'v1', 'V1', '{}', 'b', 'B', '{}', '{}', 1, 2, 'running'), ('crashed-run', 'cell-000002', 'nim', '{}', 'v2', 'V2', '{}', 'b', 'B', '{}', '{}', 1, 2, 'pending')", []).unwrap();
    let mut registry = fs::OpenOptions::new()
        .append(true)
        .open(bench_runs.join("registry.log"))
        .unwrap();
    writeln!(
        registry,
        "{}",
        stop_event("crashed-run", Some(1)).to_json_line()
    )
    .unwrap();
    ingest_once(&db, &bench_runs).unwrap();
    let statuses: Vec<String> = db
        .prepare(
            "SELECT status FROM experiment_cells WHERE run_id = 'crashed-run' ORDER BY cell_id",
        )
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(statuses, vec!["failed", "cancelled"]);
    assert_eq!(
        db.query_row(
            "SELECT status FROM runs WHERE run_id = 'crashed-run'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "crashed"
    );
}
