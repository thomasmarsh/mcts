#![allow(unused_imports)]
use super::support::*;
use crate::bench::*;
use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode as HttpStatusCode;
use mcts_bench::experiment::ExperimentSpecV1;
use mcts_bench::launch::LaunchedRun;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

#[test]
fn test_inject_ladder_root_sets_self_reference_on_a_new_ladder_launch() {
    let config = Some(json!({
        "overrides": ["optimizer.n_trials=10"],
        "ladder": {"max_rungs": 3, "saturation_threshold": 0.0},
    }));
    let config = inject_ladder_root_if_new_ladder(config, "root-run-1").unwrap();
    assert_eq!(config["ladder_root"], json!("root-run-1"));
}

#[test]
fn test_inject_ladder_root_leaves_non_ladder_config_untouched() {
    let config = Some(json!({ "overrides": ["optimizer.n_trials=10"] }));
    let config = inject_ladder_root_if_new_ladder(config, "some-run").unwrap();
    assert!(config.get("ladder_root").is_none());
}

#[test]
fn test_inject_ladder_root_does_not_override_a_carried_forward_root() {
    // A resumed rung's config already has `ladder_root` pointing at the
    // *original* root (via `build_resume_config`) -- this must not be
    // clobbered with the resumed rung's own id.
    let config = Some(json!({
        "ladder": {"max_rungs": 3, "saturation_threshold": 0.0},
        "ladder_root": "original-root",
    }));
    let config = inject_ladder_root_if_new_ladder(config, "rung-2-run").unwrap();
    assert_eq!(config["ladder_root"], json!("original-root"));
}

#[test]
fn test_inject_ladder_root_handles_none_config() {
    assert_eq!(inject_ladder_root_if_new_ladder(None, "some-run"), None);
}

// -------------------------------------------------------------------
// record_floor_baseline_settings
// -------------------------------------------------------------------

#[test]
fn test_record_floor_baseline_settings_persists_flat_mc_params() {
    let config = Some(json!({
        "overrides": ["optimizer.n_trials=10", "target.baselines=[\"flat_mc\"]"],
    }));
    let config = record_floor_baseline_settings(config).unwrap();
    assert_eq!(
        config["baseline_settings"]["flat_mc"],
        json!({"family": "flat_mc", "q_init": "Infinity"})
    );
    assert!(config.get("baseline_configs").is_none());
}

#[test]
fn test_record_floor_baseline_settings_preserves_existing_settings() {
    let config = Some(json!({
        "overrides": ["target.baselines=[\"random\"]"],
        "baseline_settings": {"chosen": {"family": "custom"}},
    }));
    let config = record_floor_baseline_settings(config).unwrap();
    assert_eq!(
        config["baseline_settings"],
        json!({"chosen": {"family": "custom"}})
    );
}

// -------------------------------------------------------------------
// plan_ladder_advances
// -------------------------------------------------------------------

fn ladder_root_run(run_id: &str, max_rungs: i64, saturation_threshold: f64) -> LadderRunRow {
    LadderRunRow {
        run_id: run_id.to_string(),
        game: "nim".to_string(),
        status: "completed".to_string(),
        exit_code: Some(0),
        config: Some(json!({
            "overrides": ["optimizer.n_trials=10"],
            "ladder": {"max_rungs": max_rungs, "saturation_threshold": saturation_threshold},
            "ladder_root": run_id,
        })),
    }
}

#[test]
fn test_plan_ladder_advances_widens_a_saturated_root_with_budget_left() {
    let runs = vec![ladder_root_run("root-1", 3, 0.0)];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([(
        "root-1".to_string(),
        (json!({"family": "ucb1", "c": 1.4}), 0.0),
    )]);

    let advances = plan_ladder_advances(&runs, &trial_counts, &incumbents);
    assert_eq!(advances.len(), 1);
    let advance = &advances[0];
    assert_eq!(advance.parent_run_id, "root-1");
    assert_eq!(advance.game, "nim");
    assert_eq!(advance.label, "ladder rung 2 of root-1");
    assert_eq!(advance.widened_config["resumed_from"], json!("root-1"));
    assert_eq!(advance.widened_config["ladder_root"], json!("root-1"));
    // Cumulative budget: root's own 10 trials + another 10 for the new
    // rung, plus a trailing `target.baselines=[]` neutralizing whatever
    // named baseline the root started against -- see
    // `replace_baseline_with_incumbent`'s doc comment.
    assert_eq!(
        advance.widened_config["overrides"],
        json!([
            "optimizer.n_trials=10",
            "optimizer.n_trials=10",
            "target.baselines=[]"
        ])
    );
    // rung_count is 1 (the root itself) before this widen, so the new
    // rung being created is rung 2 -- its baseline id is "ladder2".
    assert_eq!(
        advance.widened_config["baseline_configs"]["ladder2"],
        json!({"family": "ucb1", "c": 1.4})
    );
}

#[test]
fn test_plan_ladder_advances_widens_a_running_rung_at_threshold() {
    let mut run = ladder_root_run("root-1", 3, 0.15);
    run.status = "running".to_string();
    run.exit_code = None;
    let runs = vec![run];
    let trial_counts = HashMap::from([("root-1".to_string(), 3)]);
    let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.025))]);

    let advances = plan_ladder_advances(&runs, &trial_counts, &incumbents);
    assert_eq!(advances.len(), 1);
    assert_eq!(advances[0].parent_run_id, "root-1");
    assert_eq!(
        advances[0].widened_config["overrides"],
        json!([
            "optimizer.n_trials=10",
            "optimizer.n_trials=10",
            "target.baselines=[]"
        ])
    );
}

#[test]
fn test_plan_ladder_advances_replaces_rather_than_accumulates_baseline_configs() {
    // The parent rung already carries a `baseline_configs` entry from a
    // prior widen (or a hand-launched `--baseline-config`) -- the new
    // widen must *replace* it with just the new incumbent, not merge
    // alongside it, matching "always face the current incumbent" rather
    // than tuner's multi-instance averaging.
    let mut root = ladder_root_run("root-1", 5, 0.0);
    root.config.as_mut().unwrap()["baseline_configs"] =
        json!({"ladder1": {"family": "ucb1", "c": 0.5}});
    let runs = vec![root];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([(
        "root-1".to_string(),
        (json!({"family": "rave", "threshold": 700}), 0.0),
    )]);

    let advances = plan_ladder_advances(&runs, &trial_counts, &incumbents);
    assert_eq!(advances.len(), 1);
    let baseline_configs = advances[0].widened_config["baseline_configs"]
        .as_object()
        .unwrap();
    assert_eq!(baseline_configs.len(), 1);
    assert_eq!(
        baseline_configs.get("ladder2"),
        Some(&json!({"family": "rave", "threshold": 700}))
    );
    assert!(!baseline_configs.contains_key("ladder1"));
}

#[test]
fn test_plan_ladder_advances_does_not_widen_when_not_saturated() {
    let runs = vec![ladder_root_run("root-1", 3, 0.0)];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([(
        "root-1".to_string(),
        (json!({"family": "ucb1"}), 0.2), // above the 0.0 threshold
    )]);

    assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
}

#[test]
fn test_plan_ladder_advances_does_not_widen_without_an_incumbent() {
    let runs = vec![ladder_root_run("root-1", 3, 0.0)];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::new();

    assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
}

#[test]
fn test_plan_ladder_advances_stops_at_max_rungs() {
    // Two rungs already exist for this ladder and max_rungs is 2 --
    // no third rung should be proposed even though the second is
    // saturated with budget nominally available.
    let mut rung2 = ladder_root_run("root-1", 2, 0.0);
    rung2.run_id = "root-1-rung2".to_string();
    rung2.config.as_mut().unwrap()["resumed_from"] = json!("root-1");
    let root = ladder_root_run("root-1", 2, 0.0);
    // root already has a child (rung2), so it wouldn't be reconsidered
    // either -- but the rung-count check is what should stop rung2.
    let runs = vec![root, rung2];
    let trial_counts =
        HashMap::from([("root-1".to_string(), 10), ("root-1-rung2".to_string(), 10)]);
    let incumbents =
        HashMap::from([("root-1-rung2".to_string(), (json!({"family": "ucb1"}), 0.0))]);

    assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
}

#[test]
fn test_plan_ladder_advances_skips_a_rung_that_already_has_a_child() {
    let root = ladder_root_run("root-1", 5, 0.0);
    let mut child = ladder_root_run("root-1", 5, 0.0);
    child.run_id = "root-1-rung2".to_string();
    child.config.as_mut().unwrap()["resumed_from"] = json!("root-1");
    let runs = vec![root, child];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

    assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
}

#[test]
fn test_plan_ladder_advances_ignores_stopped_run() {
    let mut run = ladder_root_run("root-1", 3, 0.0);
    run.status = "stopped".to_string();
    let runs = vec![run];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

    assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
}

#[test]
fn test_plan_ladder_advances_ignores_crashed_exit_code() {
    let mut run = ladder_root_run("root-1", 3, 0.0);
    run.exit_code = Some(1);
    let runs = vec![run];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

    assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
}

#[test]
fn test_plan_ladder_advances_ignores_non_ladder_run() {
    let run = LadderRunRow {
        run_id: "plain-run".to_string(),
        game: "nim".to_string(),
        status: "completed".to_string(),
        exit_code: Some(0),
        config: Some(json!({"overrides": ["optimizer.n_trials=10"]})),
    };
    let runs = vec![run];
    let trial_counts = HashMap::from([("plain-run".to_string(), 10)]);
    let incumbents = HashMap::from([("plain-run".to_string(), (json!({"family": "ucb1"}), 0.0))]);

    assert!(plan_ladder_advances(&runs, &trial_counts, &incumbents).is_empty());
}

// -------------------------------------------------------------------
// plan_manual_advance
// -------------------------------------------------------------------

fn plain_run(run_id: &str, trials: i64) -> LadderRunRow {
    LadderRunRow {
        run_id: run_id.to_string(),
        game: "nim".to_string(),
        status: "completed".to_string(),
        exit_code: Some(0),
        config: Some(json!({"overrides": [format!("optimizer.n_trials={trials}")]})),
    }
}

#[test]
fn test_plan_manual_advance_starts_a_new_chain_from_a_plain_run() {
    let runs = vec![plain_run("root-1", 10)];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([(
        "root-1".to_string(),
        (json!({"family": "ucb1", "c": 1.4}), 0.0),
    )]);

    let advance =
        plan_manual_advance(&runs, &trial_counts, &incumbents, "root-1", None, None).unwrap();
    assert_eq!(advance.game, "nim");
    assert_eq!(advance.label, "baseline advance from root-1");
    assert_eq!(advance.widened_config["resumed_from"], json!("root-1"));
    assert_eq!(advance.widened_config["ladder_root"], json!("root-1"));
    // No pre-existing "ladder" block -- this is a manual-only chain,
    // so the automated driver must never pick it up.
    assert!(advance.widened_config.get("ladder").is_none());
    // The baseline changes within the original total trial budget.
    assert_eq!(
        advance.widened_config["overrides"],
        json!([
            "optimizer.n_trials=10",
            "optimizer.n_trials=10",
            "target.baselines=[]"
        ])
    );
    assert_eq!(
        advance.widened_config["baseline_configs"]["ladder2"],
        json!({"family": "ucb1", "c": 1.4})
    );
    // The root itself never had `ladder_root` set -- the caller must
    // retroactively tag it so a later advance (or the UI) can find the
    // chain by `ladder_root` alone.
    let (root_id, root_config) = advance.root_patch.expect("expected a root patch");
    assert_eq!(root_id, "root-1");
    assert_eq!(root_config["ladder_root"], json!("root-1"));
}

#[test]
fn test_plan_manual_advance_respects_an_explicit_n_trials() {
    let runs = vec![plain_run("root-1", 10)];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

    let advance = plan_manual_advance(
        &runs,
        &trial_counts,
        &incumbents,
        "root-1",
        Some(500),
        Some(4),
    )
    .unwrap();
    assert_eq!(
        advance.widened_config["overrides"],
        json!([
            "optimizer.n_trials=10",
            "optimizer.n_trials=500",
            "optimizer.n_workers=4",
            "target.baselines=[]"
        ])
    );
}

#[test]
fn test_plan_manual_advance_continues_an_existing_chain_without_re_patching_the_root() {
    // root-1 already has ladder_root=root-1 (a prior manual or automated
    // advance already tagged it) and one child rung already exists.
    let mut root = plain_run("root-1", 10);
    root.config.as_mut().unwrap()["ladder_root"] = json!("root-1");
    let mut rung2 = plain_run("root-1-rung2", 10);
    rung2.config.as_mut().unwrap()["ladder_root"] = json!("root-1");
    rung2.config.as_mut().unwrap()["resumed_from"] = json!("root-1");
    let runs = vec![root, rung2];
    let trial_counts =
        HashMap::from([("root-1".to_string(), 10), ("root-1-rung2".to_string(), 10)]);
    let incumbents =
        HashMap::from([("root-1-rung2".to_string(), (json!({"family": "ucb1"}), 0.0))]);

    let advance = plan_manual_advance(
        &runs,
        &trial_counts,
        &incumbents,
        "root-1-rung2",
        None,
        None,
    )
    .unwrap();
    assert!(advance.root_patch.is_none());
    assert_eq!(advance.widened_config["ladder_root"], json!("root-1"));
    // rung_count is 2 (root + rung2) before this widen -> next id "ladder3".
    assert_eq!(
        advance.widened_config["baseline_configs"]["ladder3"],
        json!({"family": "ucb1"})
    );
    // A later baseline change still preserves the logical run's budget.
    assert_eq!(
        advance.widened_config["overrides"],
        json!([
            "optimizer.n_trials=10",
            "optimizer.n_trials=10",
            "target.baselines=[]"
        ])
    );
}

#[test]
fn test_plan_manual_advance_replaces_rather_than_accumulates_baseline_configs() {
    let mut root = plain_run("root-1", 10);
    root.config.as_mut().unwrap()["baseline_configs"] =
        json!({"ladder1": {"family": "ucb1", "c": 0.5}});
    let runs = vec![root];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([(
        "root-1".to_string(),
        (json!({"family": "rave", "threshold": 700}), 0.0),
    )]);

    let advance =
        plan_manual_advance(&runs, &trial_counts, &incumbents, "root-1", None, None).unwrap();
    let baseline_configs = advance.widened_config["baseline_configs"]
        .as_object()
        .unwrap();
    assert_eq!(baseline_configs.len(), 1);
    assert_eq!(
        baseline_configs.get("ladder2"),
        Some(&json!({"family": "rave", "threshold": 700}))
    );
    assert!(!baseline_configs.contains_key("ladder1"));
}

#[test]
fn test_plan_manual_advance_errors_without_an_incumbent() {
    let runs = vec![plain_run("root-1", 10)];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::new();

    let err =
        plan_manual_advance(&runs, &trial_counts, &incumbents, "root-1", None, None).unwrap_err();
    assert!(err.contains("no incumbent"));
}

#[test]
fn test_plan_manual_advance_errors_for_unknown_run() {
    let runs = vec![plain_run("root-1", 10)];
    let trial_counts = HashMap::from([("root-1".to_string(), 10)]);
    let incumbents = HashMap::from([("root-1".to_string(), (json!({"family": "ucb1"}), 0.0))]);

    let err =
        plan_manual_advance(&runs, &trial_counts, &incumbents, "nope", None, None).unwrap_err();
    assert!(err.contains("not found"));
}

// -------------------------------------------------------------------
// POST /api/bench/runs/{run_id}/advance-baseline
// -------------------------------------------------------------------

#[tokio::test]
async fn test_advance_baseline_returns_404_for_unknown_run() {
    let app = seeded_app(|_, _| {}).0;
    let (status, body) = http_post_json(
        app,
        "/api/bench/runs/nonexistent/advance-baseline",
        json!({}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);
}

#[tokio::test]
async fn test_advance_baseline_rejects_non_tuner_run() {
    // DEFAULT_RUN_ID is seeded as a 'round_robin' run.
    let app = seeded_app(default_seed).0;
    let (status, body) = http_post_json(
        app,
        &format!("/api/bench/runs/{DEFAULT_RUN_ID}/advance-baseline"),
        json!({}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    let body = body_json(&body);
    assert!(body["error"].as_str().unwrap().contains("round_robin"));
}

#[tokio::test]
async fn test_advance_baseline_rejects_a_run_with_no_incumbent() {
    let app = seeded_app(|conn, dir| {
        std::fs::create_dir_all(dir).ok();
        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, config, git_sha, git_dirty, host, pid, \
              started_at, ended_at, status, log_path) \
             VALUES ('tuner-no-incumbent', 'tuner', 'traffic-lights', \
                     '{\"config\": \"tuner/config/default.yaml\", \"overrides\": []}', \
                     'abc1234', false, 'testhost', NULL, \
                     '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/nope/log.jsonl')",
            duckdb::params![],
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_post_json(
        app,
        "/api/bench/runs/tuner-no-incumbent/advance-baseline",
        json!({}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    let body = body_json(&body);
    assert!(body["error"].as_str().unwrap().contains("no incumbent"));
}

#[tokio::test]
async fn test_advance_baseline_tuner_reaches_the_launcher() {
    // Same "reaches the launcher, doesn't get rejected as a bad
    // request" shape as test_resume_tuner_reaches_the_launcher: a
    // completed (non-running) run with a recorded incumbent should sail
    // past the stop-and-wait step (a no-op for a non-running run) and
    // the plan_manual_advance validation, reaching launch_and_record.
    let app = seeded_app(|conn, dir| {
        std::fs::create_dir_all(dir).ok();
        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, config, git_sha, git_dirty, host, pid, \
              started_at, ended_at, status, log_path) \
             VALUES ('tuner-advance-src', 'tuner', 'traffic-lights', \
                     '{\"config\": \"tuner/config/default.yaml\", \"overrides\": []}', \
                     'abc1234', false, 'testhost', NULL, \
                     '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/nope/log.jsonl')",
            duckdb::params![],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO incumbents (run_id, ts, config, cost) \
             VALUES ('tuner-advance-src', '2026-01-01T00:30:00Z', '{\"family\": \"ucb1\"}', 0.02)",
            duckdb::params![],
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_post_json(
        app,
        "/api/bench/runs/tuner-advance-src/advance-baseline",
        json!({}),
    )
    .await;

    assert!(
        status == HttpStatusCode::OK || status == HttpStatusCode::INTERNAL_SERVER_ERROR,
        "advance-baseline returned unexpected status {status}: body={}",
        String::from_utf8_lossy(&body),
    );
}

// -------------------------------------------------------------------
// POST /api/bench/runs/{run_id}/resume
// -------------------------------------------------------------------

#[tokio::test]
async fn test_resume_returns_404_for_unknown_run() {
    let app = seeded_app(|_, _| {}).0;
    let (status, body) = http_post_json(
        app,
        "/api/bench/runs/nonexistent/resume",
        json!({ "n_trials": 500 }),
    )
    .await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);
}

#[tokio::test]
async fn test_resume_rejects_non_tuner_run() {
    // DEFAULT_RUN_ID is seeded as a 'round_robin' run.
    let app = seeded_app(default_seed).0;
    let (status, body) = http_post_json(
        app,
        &format!("/api/bench/runs/{DEFAULT_RUN_ID}/resume"),
        json!({ "n_trials": 500 }),
    )
    .await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    let body = body_json(&body);
    assert!(body["error"].as_str().unwrap().contains("round_robin"));
}

#[tokio::test]
async fn test_resume_tuner_reaches_the_launcher() {
    // Same "reaches the launcher, doesn't get rejected as a bad
    // request" shape as test_launch_tuner_reaches_the_launcher: proves
    // the old run's kind/config are read back out of the DB and turned
    // into a launch the handler forwards, rather than being rejected
    // before ever reaching launch::launch_with_run_id.
    let app = seeded_app(|conn, dir| {
        std::fs::create_dir_all(dir).ok();
        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, config, git_sha, git_dirty, host, pid, \
              started_at, ended_at, status, log_path) \
             VALUES ('tuner-resume-src', 'tuner', 'traffic-lights', \
                     '{\"config\": \"tuner/config/default.yaml\", \"overrides\": []}', \
                     'abc1234', false, 'testhost', NULL, \
                     '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 'completed', '/tmp/nope/log.jsonl')",
            duckdb::params![],
        )
        .unwrap();
    })
    .0;

    let (status, body) = http_post_json(
        app,
        "/api/bench/runs/tuner-resume-src/resume",
        json!({ "n_trials": 500 }),
    )
    .await;

    assert!(
        status == HttpStatusCode::OK || status == HttpStatusCode::INTERNAL_SERVER_ERROR,
        "resume returned unexpected status {status}: body={}",
        String::from_utf8_lossy(&body),
    );
}

// -------------------------------------------------------------------
// POST /api/bench/runs/{run_id}/stop
// -------------------------------------------------------------------

#[tokio::test]
async fn test_stop_returns_404_for_unknown_run() {
    let app = seeded_app(|_, _| {}).0;
    let (status, body) = http_post_json(app, "/api/bench/runs/nonexistent/stop", json!({})).await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    let body = body_json(&body);
    assert_eq!(body["code"], 404);
}

#[tokio::test]
async fn test_stop_returns_ok_for_non_running_run_without_signalling() {
    let signal_calls = Arc::new(Mutex::new(0_u32));
    let signal_calls_for_handler = signal_calls.clone();
    let app = seeded_app_with_state_and_signaller(
        default_seed,
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
        Arc::new(move |_| {
            *signal_calls_for_handler.lock().unwrap() += 1;
            Ok(())
        }),
    )
    .0;
    let (status, body) = http_post_json(
        app,
        &format!("/api/bench/runs/{DEFAULT_RUN_ID}/stop"),
        json!({}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let body = body_json(&body);
    // Completed run — no signal sent, but still succeeds.
    assert_eq!(body["status"], "completed");
    assert_eq!(*signal_calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn test_stop_marks_running_run_as_stopped() {
    let app = seeded_app(|conn, bench_runs_dir| {
        let run_dir = bench_runs_dir.join("stoppable-run");
        std::fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        std::fs::write(&log_path, "").unwrap();
        let log_path_str = log_path.to_string_lossy().to_string();

        // Use a non-existent PID so the test doesn't accidentally
        // signal the current process (which would kill the test runner).
        // The stop handler gracefully handles missing PIDs.
        conn.execute(
            "INSERT INTO runs \
             (run_id, kind, game, git_sha, git_dirty, host, pid, started_at, status, log_path) \
             VALUES ('stoppable-run', 'round_robin', 'druid', 'abc', false, 'h', 999999999, \
                     '2026-03-01T00:00:00Z', 'running', ?1)",
            duckdb::params![log_path_str],
        )
        .unwrap();
    })
    .0;

    let (status, body) =
        http_post_json(app.clone(), "/api/bench/runs/stoppable-run/stop", json!({})).await;
    assert_eq!(status, HttpStatusCode::OK);
    let body = body_json(&body);
    // No signal was sent (PID doesn't exist), but the run should still
    // be marked as stopped in the database.
    assert_eq!(
        body["message"].as_str().unwrap_or(""),
        "run marked as stopped (PID was no longer alive or had no PID)"
    );

    // Verify the DB was updated.
    let (_, check_body) = http_get(app, "/api/bench/runs/stoppable-run").await;
    let detail = body_json(&check_body);
    assert_eq!(detail["status"], "stopped");
    assert!(detail["ended_at"].as_str().unwrap_or("").len() >= 10);
}

#[tokio::test]
async fn test_stop_preserves_terminal_cells_and_cancels_pending_and_running_cells() {
    let app = seeded_app(|conn, bench_runs_dir| {
        let run_dir = bench_runs_dir.join("stoppable-experiment");
        std::fs::create_dir_all(&run_dir).unwrap();
        let log_path = run_dir.join("log.jsonl");
        std::fs::write(&log_path, "").unwrap();
        conn.execute(
            "INSERT INTO runs (run_id, kind, project_id, experiment_id, experiment_spec, label, git_sha, git_dirty, host, pid, started_at, status, log_path) VALUES ('stoppable-experiment', 'experiment', 'p-route', 'e-route', '{}', 'Grid', 'abc', false, 'h', 999999999, '2026-03-01T00:00:00Z', 'running', ?1)",
            duckdb::params![log_path.to_string_lossy().to_string()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO experiment_cells (run_id, cell_id, game, game_config, variant_id, variant_label, candidate_config, baseline_id, baseline_label, baseline_config, budget, rounds, planned_games, completed_games, status, error) VALUES ('stoppable-experiment', 'cell-000001', 'nim', '{}', 'v1', 'V1', '{}', 'b', 'B', '{}', '{}', 1, 2, 2, 'completed', NULL), ('stoppable-experiment', 'cell-000002', 'nim', '{}', 'v2', 'V2', '{}', 'b', 'B', '{}', '{}', 1, 2, 1, 'failed', 'child failed'), ('stoppable-experiment', 'cell-000003', 'nim', '{}', 'v3', 'V3', '{}', 'b', 'B', '{}', '{}', 1, 2, 0, 'pending', NULL), ('stoppable-experiment', 'cell-000004', 'nim', '{}', 'v4', 'V4', '{}', 'b', 'B', '{}', '{}', 1, 2, 1, 'running', NULL)",
            [],
        )
        .unwrap();
    })
    .0;

    let (status, _) = http_post_json(
        app.clone(),
        "/api/bench/runs/stoppable-experiment/stop",
        json!({}),
    )
    .await;
    assert_eq!(status, HttpStatusCode::OK);
    let (status, body) = http_get(app, "/api/bench/runs/stoppable-experiment/cells").await;
    assert_eq!(status, HttpStatusCode::OK);
    let response = body_json(&body);
    let cells = response.as_array().unwrap();
    assert_eq!(
        cells
            .iter()
            .map(|cell| cell["status"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["completed", "failed", "cancelled", "cancelled"]
    );
}
// Error formatting
// -------------------------------------------------------------------

#[tokio::test]
async fn test_bench_error_has_structured_body() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app, "/api/bench/runs/nope").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    let body = body_json(&body);
    assert_eq!(body["code"], 404);
    assert!(body["error"].as_str().unwrap().contains("nope"));
}
