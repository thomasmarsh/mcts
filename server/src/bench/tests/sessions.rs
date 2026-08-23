use super::support::*;
use axum::http::StatusCode as HttpStatusCode;
use std::sync::Arc;

#[tokio::test]
async fn tuning_sessions_list_projects_counts_attempts_capabilities_and_order() {
    let app = seeded_app(|conn, dir| {
        default_seed(conn, dir);
        conn.execute(
            "INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES \
             ('tuner-projected', 'tuner', 'nim', 'abc', false, 'test', '2026-01-01T00:00:00Z', 'running', '/tmp/projected.log'), \
             ('tuner-legacy', 'tuner', 'nim', 'abc', false, 'test', '2026-01-01T00:00:00Z', 'completed', '/tmp/legacy.log')",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_sessions (session_id, status, manifest, target_trial_count, created_at, last_sequence) VALUES \
             ('session-a', 'idle', '{\"schema_version\":1}', NULL, '2026-01-02T00:00:00Z', 1), \
             ('session-z', 'idle', '{\"schema_version\":1}', NULL, '2026-01-02T00:00:00Z', 1), \
             ('session-main', 'active', '{\"schema_version\":1,\"semantic_inputs\":{\"game\":{\"kind\":\"nim\",\"label\":\"Nim tuning\"}}}', 8, '2026-01-01T00:00:00Z', 9)",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at, ended_at, failure) VALUES \
             ('attempt-old', 'session-main', NULL, 'stopped', '2026-01-01T00:00:00Z', '2026-01-01T00:01:00Z', 'stopped by user'), \
             ('attempt-live', 'session-main', 'tuner-projected', 'running', '2026-01-03T00:00:00Z', NULL, NULL)",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at) VALUES \
             ('session-main', 'trial-queued', 'attempt-live', 0, 'queued', '{}', '2026-01-03T00:00:00Z'), \
             ('session-main', 'trial-running', 'attempt-live', 1, 'running', '{}', '2026-01-03T00:00:00Z'), \
             ('session-main', 'trial-complete', 'attempt-old', 2, 'complete', '{}', '2026-01-01T00:00:00Z'), \
             ('session-main', 'trial-failed', 'attempt-old', 3, 'failed', '{}', '2026-01-01T00:00:00Z'), \
             ('session-main', 'trial-pruned', 'attempt-old', 4, 'pruned', '{}', '2026-01-01T00:00:00Z')",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, started_at) VALUES \
             ('session-main', 'pair-1', 'trial-running', 'attempt-live', 0, 'running', 7, 1, '{\"anchor_id\":\"a\",\"config\":{},\"mu\":25.0,\"sigma\":1.0}', 'pool', 25.0, 1.0, '2026-01-03T00:01:00Z')",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO tuning_games (session_id, pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, candidate_metrics, baseline_metrics, finished_at) VALUES \
             ('session-main', 'pair-1', 'game-1', 'first', 'candidate_win', 7, 1, 11, 10, 20, '{\"iterations_total\":1,\"iterations_first_half\":1,\"move_time_ms\":1}', '{\"iterations_total\":1,\"iterations_first_half\":1,\"move_time_ms\":1}', '2026-01-03T00:02:00Z')",
            [],
        ).unwrap();
    }).0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    let sessions = value["sessions"].as_array().unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        sessions
            .iter()
            .map(|session| &session["session_id"])
            .collect::<Vec<_>>(),
        vec!["session-main", "session-z", "session-a"]
    );
    assert_eq!(sessions[0]["game"], "nim");
    assert_eq!(sessions[0]["label"], "Nim tuning");
    assert_eq!(
        sessions[0]["counts"],
        serde_json::json!({"total": 5, "queued": 1, "running": 1, "terminal": 3, "completed": 1, "failed": 1, "pruned": 1, "cancelled": 0})
    );
    assert_eq!(sessions[0]["attempts"][0]["attempt_id"], "attempt-old");
    assert_eq!(
        sessions[0]["attempts"][1]["bench_run_id"],
        "tuner-projected"
    );
    assert_eq!(
        sessions[0]["attempts"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|attempt| attempt["bench_run_id"].as_str())
            .collect::<Vec<_>>(),
        vec!["tuner-projected"]
    );
    assert_eq!(sessions[0]["capabilities"]["has_pairs"], true);
    assert_eq!(sessions[0]["capabilities"]["has_renderer_trace"], false);
    assert_eq!(sessions[0]["capabilities"]["has_search_reports"], false);
    assert_eq!(sessions[0]["capabilities"]["has_trial_reports"], false);
    assert_ne!(sessions[0]["last_activity_at"], sessions[0]["created_at"]);
    assert!(sessions[1]["game"].is_null());
}

#[tokio::test]
async fn tuning_sessions_list_rejects_a_malformed_manifest() {
    let app = seeded_app(|conn, _| {
        conn.execute(
            "INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('broken', 'idle', '{\"semantic_inputs\":{\"game\":\"nim\"}}', CURRENT_TIMESTAMP, 1)",
            [],
        ).unwrap();
    }).0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    assert_eq!(body_json(&body)["code"], 400);
}

#[tokio::test]
async fn tuning_sessions_list_returns_a_structured_storage_error() {
    let (app, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute(
                "INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'idle', '{}', CURRENT_TIMESTAMP, 1)",
                [],
            ).unwrap();
        },
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    );
    state
        .db
        .lock()
        .unwrap()
        .execute_batch("DROP TABLE tuning_trials")
        .unwrap();

    let (status, body) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["code"], 500);
}

#[tokio::test]
async fn tuning_session_detail_projects_counts_attempts_and_capabilities() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, manifest_fingerprint, target_trial_count, created_at, last_sequence) VALUES ('session-1', 'idle', '{\"game\":\"nim\"}', 'fp', 2, CURRENT_TIMESTAMP, 7)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at, ended_at) VALUES ('attempt-1', 'session-1', NULL, 'stopped', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at, score) VALUES ('session-1', 'trial-1', 'attempt-1', 0, 'complete', '{\"c\":1}', CURRENT_TIMESTAMP, 0.5), ('session-1', 'trial-2', 'attempt-1', 1, 'failed', '{\"c\":2}', CURRENT_TIMESTAMP, NULL)", []).unwrap();
    }).0;
    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(value["summary"]["session_id"], "session-1");
    assert_eq!(value["summary"]["status"], "idle");
    assert_eq!(value["summary"]["counts"]["total"], 2);
    assert_eq!(value["summary"]["counts"]["completed"], 1);
    assert_eq!(value["summary"]["counts"]["failed"], 1);
    assert_eq!(value["attempts"][0]["attempt_id"], "attempt-1");
    assert_eq!(value["trials"][1]["status"], "failed");
    assert_eq!(value["fingerprint"], "fp");
    assert_eq!(value["capabilities"]["has_lifecycle"], true);
    assert_eq!(value["capabilities"]["has_pairs"], false);
    assert_eq!(value["capabilities"]["has_search_reports"], false);
    assert_eq!(value["capabilities"]["has_trial_reports"], false);
    assert!(value["policy"].is_null());
    assert_eq!(value["trials"][0]["stop_reason"], serde_json::Value::Null);
    assert_eq!(value["trials"][0]["reports"], serde_json::json!([]));
    assert_eq!(value["cursor"]["session_sequence"], 7);
}

#[tokio::test]
async fn tuning_session_detail_loads_all_reports_without_per_trial_queries() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'idle', '{\"schema_version\":1,\"semantic_inputs\":{\"game\":{\"kind\":\"nim\"},\"optimizer\":{\"resource\":{\"min_pairs\":2,\"max_pairs\":6},\"sampler\":{\"kind\":\"tpe\",\"seed\":4,\"deterministic\":true,\"startup_trials\":3},\"pruning\":{\"enabled\":true,\"kind\":\"hyperband\",\"reduction_factor\":3.0,\"startup_terminal_trials\":5}},\"rating\":{\"model\":\"ThurstoneMostellerPart\",\"score\":\"mu_minus_k_sigma\",\"sigma_stop\":2.0,\"conservative_k\":3.0}}}', '2026-01-01T00:00:00Z', 4)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at) VALUES ('attempt-1', 'session-1', 'completed', '2026-01-01T00:00:00Z')", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at, stop_reason) VALUES ('session-1', 'trial-1', 'attempt-1', 1, 'complete', '{}', '2026-01-01T00:00:00Z', 'max_pairs'), ('session-1', 'trial-2', 'attempt-1', 2, 'pruned', '{}', '2026-01-01T00:00:00Z', 'hyperband_prune')", []).unwrap();
        conn.execute("INSERT INTO tuning_trial_reports (session_id, trial_id, trial_number, completed_pairs, event_id, reported_at, mu, sigma, score, score_formula_version, conservative_k, outcome, reason, pruning_exempt, bracket_id, rung_resource) VALUES ('session-1', 'trial-1', 1, 4, 'report-4', '2026-01-01T00:04:00Z', 26.0, 1.0, 23.0, 1, 3.0, 'complete', 'max_pairs', false, 'bracket-a', 4), ('session-1', 'trial-2', 2, 2, 'report-2', '2026-01-01T00:02:00Z', 24.0, 2.0, 18.0, 1, 3.0, 'prune', 'hyperband_prune', false, NULL, NULL), ('session-1', 'trial-1', 1, 2, 'report-1', '2026-01-01T00:02:00Z', 25.0, 2.0, 19.0, 1, 3.0, 'continue', 'startup_exempt', true, NULL, NULL)", []).unwrap();
    }).0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(
        value["policy"],
        serde_json::json!({
            "resource": {"min_pairs": 2, "max_pairs": 6},
            "rating": {"model": "ThurstoneMostellerPart", "score": "mu_minus_k_sigma", "sigma_stop": 2.0, "conservative_k": 3.0},
            "sampler": {"kind": "tpe", "seed": 4, "deterministic": true, "startup_trials": 3},
            "pruning": {"enabled": true, "kind": "hyperband", "reduction_factor": 3.0, "startup_terminal_trials": 5}
        })
    );
    assert_eq!(
        value["trials"],
        serde_json::json!([
            {
                "trial_id": "trial-1", "trial_number": 1, "attempt_id": "attempt-1", "status": "complete", "config": {}, "score": null, "mu": null, "sigma": null, "stop_reason": "max_pairs", "failure": null, "pairs": [],
                "reports": [
                    {"completed_pairs": 2, "rating": {"mu": 25.0, "sigma": 2.0}, "score": 19.0, "score_formula_version": 1, "conservative_k": 3.0, "decision": {"outcome": "continue", "reason": "startup_exempt", "pruning_exempt": true, "bracket_id": null, "rung_resource": null}, "reported_at": "2026-01-01 00:02:00"},
                    {"completed_pairs": 4, "rating": {"mu": 26.0, "sigma": 1.0}, "score": 23.0, "score_formula_version": 1, "conservative_k": 3.0, "decision": {"outcome": "complete", "reason": "max_pairs", "pruning_exempt": false, "bracket_id": "bracket-a", "rung_resource": 4}, "reported_at": "2026-01-01 00:04:00"}
                ]
            },
            {
                "trial_id": "trial-2", "trial_number": 2, "attempt_id": "attempt-1", "status": "pruned", "config": {}, "score": null, "mu": null, "sigma": null, "stop_reason": "hyperband_prune", "failure": null, "pairs": [],
                "reports": [
                    {"completed_pairs": 2, "rating": {"mu": 24.0, "sigma": 2.0}, "score": 18.0, "score_formula_version": 1, "conservative_k": 3.0, "decision": {"outcome": "prune", "reason": "hyperband_prune", "pruning_exempt": false, "bracket_id": null, "rung_resource": null}, "reported_at": "2026-01-01 00:02:00"}
                ]
            }
        ])
    );
    assert_eq!(value["capabilities"]["has_search_reports"], false);
    assert_eq!(value["capabilities"]["has_trial_reports"], true);
}

#[tokio::test]
async fn tuning_session_detail_rejects_a_malformed_new_policy() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'idle', '{\"semantic_inputs\":{\"optimizer\":{}}}', CURRENT_TIMESTAMP, 1)", []).unwrap();
    }).0;

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    assert_eq!(body_json(&body)["code"], 400);
}

#[tokio::test]
async fn tuning_session_detail_returns_a_storage_error_when_reports_are_unavailable() {
    let (app, _, state) = seeded_app_with_state(
        |conn, _| {
            conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'idle', '{}', CURRENT_TIMESTAMP, 1)", []).unwrap();
        },
        Arc::new(|spec| spec.expand().map(|_| ()).map_err(|error| error.fields)),
        injected_general_launcher(),
    );
    state
        .db
        .lock()
        .unwrap()
        .execute_batch("DROP TABLE tuning_trial_reports")
        .unwrap();

    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body_json(&body)["code"], 500);
}

#[tokio::test]
async fn tuning_session_detail_returns_404_without_affecting_legacy_run_endpoint() {
    let app = seeded_app(default_seed).0;
    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/missing").await;
    assert_eq!(status, HttpStatusCode::NOT_FOUND);
    assert_eq!(body_json(&body)["code"], 404);
    let (status, body) = http_get(app, &format!("/api/bench/runs/{DEFAULT_RUN_ID}")).await;
    assert_eq!(status, HttpStatusCode::OK);
    assert_eq!(body_json(&body)["run_id"], DEFAULT_RUN_ID);
}

#[tokio::test]
async fn tuning_session_detail_nests_projected_pairs_games_and_trace_capability() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-1', 'active', '{}', CURRENT_TIMESTAMP, 8)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, status, started_at) VALUES ('attempt-1', 'session-1', 'running', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at) VALUES ('session-1', 'trial-1', 'attempt-1', 0, 'running', '{}', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, rating_after_mu, rating_after_sigma, score, started_at, ended_at) VALUES ('session-1', 'pair-1', 'trial-1', 'attempt-1', 0, 'complete', 7, 1, '{\"anchor_id\":\"a\",\"config\":{},\"mu\":25.0,\"sigma\":1.0}', 'pool', 24.0, 2.0, 25.0, 1.5, 20.5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_games (session_id, pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, candidate_metrics, baseline_metrics, finished_at) VALUES ('session-1', 'pair-1', 'game-1', 'first', 'candidate_win', 7, 1, 99, 12, 30, '{\"iterations_total\":20,\"iterations_first_half\":9,\"move_time_ms\":14}', '{\"iterations_total\":18,\"iterations_first_half\":8,\"move_time_ms\":13}', CURRENT_TIMESTAMP), ('session-1', 'pair-1', 'game-2', 'second', 'draw', 7, 1, NULL, 13, 31, '{\"iterations_total\":21,\"iterations_first_half\":10,\"move_time_ms\":15}', '{\"iterations_total\":19,\"iterations_first_half\":9,\"move_time_ms\":14}', CURRENT_TIMESTAMP)", []).unwrap();
    }).0;
    let (status, body) = http_get(app, "/api/bench/tuner/sessions/session-1").await;
    assert_eq!(status, HttpStatusCode::OK);
    let value = body_json(&body);
    assert_eq!(value["trials"][0]["pairs"][0]["pair_id"], "pair-1");
    assert_eq!(
        value["trials"][0]["pairs"][0]["games"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        value["trials"][0]["pairs"][0]["games"][0]["trace_game_seq"],
        99
    );
    assert_eq!(value["capabilities"]["has_pairs"], true);
    assert_eq!(value["capabilities"]["has_renderer_trace"], false);
}

#[tokio::test]
async fn tuning_capabilities_require_the_authoritative_run_move_join() {
    let app = seeded_app(|conn, _| {
        conn.execute("INSERT INTO runs (run_id, kind, game, git_sha, git_dirty, host, started_at, status, log_path) VALUES ('bench-run', 'tuner', 'nim', 'sha', false, 'host', CURRENT_TIMESTAMP, 'completed', '/tmp/bench-run.log')", []).unwrap();
        conn.execute("INSERT INTO tuning_sessions (session_id, status, manifest, created_at, last_sequence) VALUES ('session-join', 'idle', '{}', CURRENT_TIMESTAMP, 1)", []).unwrap();
        conn.execute("INSERT INTO tuning_attempts (attempt_id, session_id, bench_run_id, status, started_at) VALUES ('attempt-join', 'session-join', 'bench-run', 'completed', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_trials (session_id, trial_id, attempt_id, trial_number, status, config, created_at) VALUES ('session-join', 'trial-join', 'attempt-join', 1, 'complete', '{}', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_evaluation_pairs (session_id, pair_id, trial_id, attempt_id, pair_index, status, seed, round, opponent, pool_snapshot_fingerprint, rating_before_mu, rating_before_sigma, started_at) VALUES ('session-join', 'pair-join', 'trial-join', 'attempt-join', 1, 'complete', 7, 1, '{\"anchor_id\":\"a\",\"config\":{},\"mu\":25.0,\"sigma\":1.0}', 'pool', 25.0, 1.0, CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO tuning_games (session_id, pair_id, game_id, candidate_side, outcome, seed, round, trace_game_seq, plies, elapsed_ms, candidate_metrics, baseline_metrics, finished_at) VALUES ('session-join', 'pair-join', 'game-join', 'first', 'draw', 7, 1, 41, 1, 1, '{\"iterations_total\":1,\"iterations_first_half\":1,\"move_time_ms\":1}', '{\"iterations_total\":1,\"iterations_first_half\":1,\"move_time_ms\":1}', CURRENT_TIMESTAMP)", []).unwrap();
        conn.execute("INSERT INTO game_moves (run_id, game_seq, ply, ts, trace_schema_version, state, search_report) VALUES ('bench-run', 41, 0, CURRENT_TIMESTAMP, 1, '{}', NULL), ('bench-run', 41, 1, CURRENT_TIMESTAMP, 1, '{}', '{}')", []).unwrap();
    }).0;

    let (status, body) = http_get(app.clone(), "/api/bench/tuner/sessions/session-join").await;
    assert_eq!(status, HttpStatusCode::OK);
    let capabilities = &body_json(&body)["capabilities"];
    assert_eq!(capabilities["has_renderer_trace"], true);
    assert_eq!(capabilities["has_search_reports"], true);
    assert_eq!(capabilities["has_trial_reports"], false);

    let (status, body) = http_get(app, "/api/bench/tuner/sessions").await;
    assert_eq!(status, HttpStatusCode::OK);
    let capabilities = &body_json(&body)["sessions"][0]["capabilities"];
    assert_eq!(capabilities["has_renderer_trace"], true);
    assert_eq!(capabilities["has_search_reports"], true);
    assert_eq!(capabilities["has_trial_reports"], false);
}
