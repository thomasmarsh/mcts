use super::support::*;
use axum::http::StatusCode as HttpStatusCode;

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
    assert_eq!(value["cursor"]["session_sequence"], 7);
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
    assert_eq!(value["capabilities"]["has_renderer_trace"], true);
}
