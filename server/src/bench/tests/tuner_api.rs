//! Endpoint tests for the read-only tuner projection API, driven against the
//! checked-in `tuner-projection.sqlite` fixture (see `support::
//! tuner_projection_fixture`).

use axum::http::StatusCode;
use serde_json::Value;

use super::support::{
    body_json, default_seed, http_get, http_post_json, seeded_app,
    seeded_app_with_state_signaller_and_tuner_db,
};

const V4: &str = "/api/bench/tuner/projection/runs/version4";
const CAND0: &str = "candidate-130051c1c73a2aa1f25731bb5f9bf9fad38bd5f2852406cef837c5b14cc8fd90";

#[tokio::test]
async fn projection_meta_exposes_the_last_pass_stamp() {
    let (app, _root) = seeded_app(default_seed);
    let (status, body) = http_get(app, "/api/bench/tuner/projection/meta").await;
    assert_eq!(status, StatusCode::OK);
    // The key is always present; its value is a string once the projector has
    // stamped a pass, or null for a projection built before that stamp existed.
    let meta = body_json(&body);
    assert!(meta.as_object().unwrap().contains_key("last_pass_at"));
}

#[tokio::test]
async fn lists_runs() {
    let (app, root) = seeded_app(default_seed);
    let (status, body) = http_get(app, "/api/bench/tuner/projection/runs").await;
    assert_eq!(status, StatusCode::OK);
    let rows = body_json(&body);
    let ids: Vec<&str> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["run_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        [
            "broken",
            "version4",
            "version4-active-halving",
            "version4-partial"
        ]
    );

    let v4 = &rows[1];
    assert_eq!(v4["terminal_status"], "complete");
    assert_eq!(v4["report_available"], true);
    assert_eq!(v4["game_kind"], "druid");
    assert_eq!(v4["shadow_policy_kind"], "paired_bootstrap");
    assert_eq!(v4["report_status"], "complete");
    assert_eq!(v4["total_pair_attempts"], 88);
    assert_eq!(v4["total_completed_pairs"], 88);

    // Pagination.
    let (_, body) = http_get(
        seeded_app(default_seed).0,
        "/api/bench/tuner/projection/runs?limit=1&offset=1",
    )
    .await;
    let rows = body_json(&body);
    assert_eq!(rows.as_array().unwrap().len(), 1);
    assert_eq!(rows[0]["run_id"], "version4");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn run_detail() {
    let (app, root) = seeded_app(default_seed);
    let (status, body) = http_get(app, V4).await;
    assert_eq!(status, StatusCode::OK);
    let detail = body_json(&body);
    assert_eq!(detail["manifest"]["game_kind"], "druid");
    assert_eq!(detail["manifest"]["cohort_size"], 4);
    assert_eq!(detail["manifest"]["finalists"], 2);
    assert_eq!(detail["manifest"]["active_elimination"], false);
    assert_eq!(detail["report"]["schema_version"], 5);
    assert_eq!(detail["report"]["status"], "complete");
    let phases: Vec<&str> = detail["compute"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["phase"].as_str().unwrap())
        .collect();
    assert_eq!(phases, ["diagnostic", "tuning", "validation"]);
    let tuning = &detail["compute"][1];
    assert_eq!(tuning["pair_attempts"], 84);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn cohorts_candidates_and_pairs_match_the_fixture() {
    let (app, root) = seeded_app(default_seed);

    let (status, body) = http_get(app.clone(), &format!("{V4}/cohorts")).await;
    assert_eq!(status, StatusCode::OK);
    let cohorts = body_json(&body);
    assert_eq!(cohorts.as_array().unwrap().len(), 2);
    assert_eq!(cohorts[0]["cohort_index"], 0);
    assert_eq!(cohorts[0]["candidate_ids"].as_array().unwrap().len(), 4);

    let (status, body) = http_get(app.clone(), &format!("{V4}/candidates")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body).as_array().unwrap().len(), 6);

    let (status, body) = http_get(app.clone(), &format!("{V4}/candidates/{CAND0}")).await;
    assert_eq!(status, StatusCode::OK);
    let cand = body_json(&body);
    assert_eq!(cand["candidate_id"], CAND0);
    assert_eq!(cand["cohort_index"], 0);
    assert!(cand["canonical_config"].is_object());

    let (status, _) = http_get(app.clone(), &format!("{V4}/candidates/nope")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, body) = http_get(app, &format!("{V4}/pairs")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body).as_array().unwrap().len(), 88);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn pair_games_match_the_fixture() {
    let (app, root) = seeded_app(default_seed);

    let (_, body) = http_get(app.clone(), &format!("{V4}/pairs?candidate={CAND0}&limit=1")).await;
    let pair_id = body_json(&body)[0]["pair_id"].as_str().unwrap().to_string();

    let (status, body) = http_get(app.clone(), &format!("{V4}/pairs/{pair_id}/games")).await;
    assert_eq!(status, StatusCode::OK);
    let games = body_json(&body);
    let rows = games.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|g| g["pair_id"] == pair_id.as_str()));
    assert!(rows.iter().any(|g| g["candidate_side"] == "first"));
    assert!(rows.iter().any(|g| g["candidate_side"] == "second"));
    assert!(rows[0]["plies"].is_number());

    let (status, _) = http_get(app.clone(), &format!("{V4}/pairs/nope/games")).await;
    assert_eq!(status, StatusCode::OK);

    let (status, _) = http_get(app, "/api/bench/tuner/projection/runs/nope/pairs/x/games").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn pairs_filtered() {
    let (app, root) = seeded_app(default_seed);

    let (status, body) = http_get(app.clone(), &format!("{V4}/pairs?candidate={CAND0}")).await;
    assert_eq!(status, StatusCode::OK);
    let rows = body_json(&body);
    assert!(!rows.as_array().unwrap().is_empty());
    assert!(rows
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["candidate_id"] == CAND0));

    let (_, body) = http_get(app.clone(), &format!("{V4}/pairs?cohort=0")).await;
    assert_eq!(body_json(&body).as_array().unwrap().len(), 56);

    let (_, body) = http_get(app, &format!("{V4}/pairs?cohort=0&limit=5")).await;
    assert_eq!(body_json(&body).as_array().unwrap().len(), 5);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn validation_rows_and_ties() {
    let (app, root) = seeded_app(default_seed);
    let (status, body) = http_get(app, &format!("{V4}/validation")).await;
    assert_eq!(status, StatusCode::OK);
    let v = body_json(&body);
    let rows = v["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["rank"], 0);
    assert_eq!(rows[0]["wins"], 4);
    assert_eq!(v["unresolved_ties"].as_array().unwrap().len(), 1);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn report_verbatim() {
    let (app, root) = seeded_app(default_seed);
    let (status, body) = http_get(app, &format!("{V4}/report")).await;
    assert_eq!(status, StatusCode::OK);
    let report: Value = body_json(&body);
    assert_eq!(report["schema_version"], 5);
    assert_eq!(report["status"], "complete");
    assert!(report["validation_order"].is_array());
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn live_science_rows_serve_from_a_partial_run() {
    // `version4-partial` has evidence truncated just after its first cohort
    // completes: `terminal_status` is "open", there is no report, yet the
    // science row tables are populated. These four endpoints are the live
    // science source (12e-7).
    let (app, root) = seeded_app(default_seed);
    const P: &str = "/api/bench/tuner/projection/runs/version4-partial";

    let (status, body) = http_get(app.clone(), P).await;
    assert_eq!(status, StatusCode::OK);
    let detail = body_json(&body);
    assert_eq!(detail["terminal_status"], "open");
    assert!(detail["report"].is_null());

    let (status, body) = http_get(app.clone(), &format!("{P}/proposals")).await;
    assert_eq!(status, StatusCode::OK);
    let proposals = body_json(&body);
    assert_eq!(proposals.as_array().unwrap().len(), 4);
    assert_eq!(proposals[0]["disposition"], "accepted");
    assert!(proposals[0]["candidate_id"].is_string());

    let (status, body) = http_get(app.clone(), &format!("{P}/observations")).await;
    assert_eq!(status, StatusCode::OK);
    let observations = body_json(&body);
    assert_eq!(observations.as_array().unwrap().len(), 28);
    assert!(observations[0]["mean"].is_number());
    assert!(observations[0]["prefix_id"].is_string());

    let (status, body) = http_get(app.clone(), &format!("{P}/shadow-decisions")).await;
    assert_eq!(status, StatusCode::OK);
    let shadow = body_json(&body);
    assert_eq!(shadow.as_array().unwrap().len(), 4);
    assert_eq!(shadow[0]["policy_kind"], "paired_bootstrap");
    assert!(shadow[0]["boundary_candidate_id"].is_string());

    let (status, body) = http_get(app.clone(), &format!("{P}/active-eliminations")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body_json(&body).as_array().unwrap().is_empty());

    // The report overlay is genuinely absent for a live run.
    let (status, _) = http_get(app.clone(), &format!("{P}/report")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Unknown run -> 404 on every new route.
    for suffix in ["proposals", "observations", "shadow-decisions", "active-eliminations"] {
        let (status, _) = http_get(
            app.clone(),
            &format!("/api/bench/tuner/projection/runs/nope/{suffix}"),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{suffix}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn proposals_observations_and_shadow_decisions_respect_limit_and_offset() {
    // `version4-partial` has 4 proposals, 28 observations, and 4 shadow
    // decisions -- enough to exercise limit/offset without needing a
    // synthetic fixture for the "under the default cap" claim.
    let (app, root) = seeded_app(default_seed);
    const P: &str = "/api/bench/tuner/projection/runs/version4-partial";

    // Omitting limit/offset returns everything (all four cases are well
    // under the 500 default).
    let (_, body) = http_get(app.clone(), &format!("{P}/proposals")).await;
    assert_eq!(body_json(&body).as_array().unwrap().len(), 4);

    let (_, body) = http_get(app.clone(), &format!("{P}/proposals?limit=2")).await;
    let first_two = body_json(&body);
    assert_eq!(first_two.as_array().unwrap().len(), 2);

    let (_, body) = http_get(app.clone(), &format!("{P}/proposals?limit=2&offset=2")).await;
    let next_two = body_json(&body);
    assert_eq!(next_two.as_array().unwrap().len(), 2);
    assert_ne!(
        first_two[0]["proposal_index"], next_two[0]["proposal_index"],
        "offset should skip the first page"
    );

    let (_, body) = http_get(app.clone(), &format!("{P}/proposals?offset=10")).await;
    assert!(body_json(&body).as_array().unwrap().is_empty());

    let (_, body) = http_get(app.clone(), &format!("{P}/observations?limit=10")).await;
    assert_eq!(body_json(&body).as_array().unwrap().len(), 10);
    let (_, body) = http_get(app.clone(), &format!("{P}/observations?limit=10&offset=25")).await;
    assert_eq!(body_json(&body).as_array().unwrap().len(), 3);

    let (_, body) = http_get(app.clone(), &format!("{P}/shadow-decisions?limit=1")).await;
    let first = body_json(&body);
    assert_eq!(first.as_array().unwrap().len(), 1);
    let (_, body) = http_get(app, &format!("{P}/shadow-decisions?limit=1&offset=1")).await;
    let second = body_json(&body);
    assert_eq!(second.as_array().unwrap().len(), 1);
    assert_ne!(
        first[0]["candidate_id"], second[0]["candidate_id"],
        "offset should skip the first page"
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn science_row_pagination_clamps_like_pairs() {
    let (app, root) = seeded_app(default_seed);
    const P: &str = "/api/bench/tuner/projection/runs/version4-partial";

    for (suffix, total) in [
        ("proposals", 4),
        ("observations", 28),
        ("shadow-decisions", 4),
        ("active-eliminations", 0),
    ] {
        // limit=0 clamps up to 1, so the response is never empty just because
        // of a degenerate limit (except when the table itself is empty).
        let (status, body) = http_get(app.clone(), &format!("{P}/{suffix}?limit=0")).await;
        assert_eq!(status, StatusCode::OK, "{suffix}");
        let rows = body_json(&body);
        assert_eq!(rows.as_array().unwrap().len(), total.min(1), "{suffix}");

        // A limit above the 5000 clamp still returns at most the table's rows.
        let (status, body) = http_get(app.clone(), &format!("{P}/{suffix}?limit=999999")).await;
        assert_eq!(status, StatusCode::OK, "{suffix}");
        assert_eq!(
            body_json(&body).as_array().unwrap().len(),
            total,
            "{suffix}"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn science_row_endpoints_paginate_a_large_synthetic_run_without_blocking() {
    // Build a purpose-made SQLite projection with one run carrying 6000
    // observations -- well past the 5000 clamp -- and confirm a paginated
    // request returns a bounded page while a concurrent, unrelated small
    // request against the same connection pool completes promptly (a proxy
    // for "the big scan doesn't serialize behind the async runtime thread").
    use rusqlite::Connection;

    let db_path = std::env::temp_dir().join(format!(
        "mcts_tuner_projection_scale_test_{}_{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, manifest_run_id TEXT, \
             manifest_fingerprint TEXT, terminal_status TEXT, report_available INTEGER NOT NULL, \
             ingest_error TEXT);
             CREATE TABLE observations (run_id TEXT NOT NULL, observation_id TEXT NOT NULL, \
             candidate_id TEXT NOT NULL, phase TEXT NOT NULL, prefix_id TEXT NOT NULL, \
             mean REAL NOT NULL, lower REAL NOT NULL, upper REAL NOT NULL, \
             PRIMARY KEY (run_id, observation_id));
             INSERT INTO runs (run_id, terminal_status, report_available, ingest_error) \
             VALUES ('big', 'open', 0, NULL);",
        )
        .unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO observations \
                     (run_id, observation_id, candidate_id, phase, prefix_id, mean, lower, upper) \
                     VALUES ('big', ?1, 'cand', 'tuning', 'prefix', 0.5, 0.4, 0.6)",
                )
                .unwrap();
            for i in 0..6000 {
                stmt.execute([format!("obs-{i:06}")]).unwrap();
            }
        }
        tx.commit().unwrap();
    }

    let (app, root, _state) = seeded_app_with_state_signaller_and_tuner_db(
        default_seed,
        std::sync::Arc::new(|_| {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "injected missing process",
            ))
        }),
        db_path.clone(),
    );

    // A bounded page over the 6000-row table.
    let (status, body) = http_get(
        app.clone(),
        "/api/bench/tuner/projection/runs/big/observations",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body).as_array().unwrap().len(), 500);

    let (status, body) = http_get(
        app.clone(),
        "/api/bench/tuner/projection/runs/big/observations?limit=50&offset=5990",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body_json(&body).as_array().unwrap().len(), 10);

    // A concurrent unrelated request (against the small default fixture-free
    // `meta` route) finishes -- exercising that the big-table handler runs on
    // a blocking thread rather than starving the runtime.
    let meta_app = app.clone();
    let (big_status, (meta_status, _)) = tokio::join!(
        async {
            http_get(
                app,
                "/api/bench/tuner/projection/runs/big/observations?limit=5000",
            )
            .await
            .0
        },
        http_get(meta_app, "/api/bench/tuner/projection/meta"),
    );
    assert_eq!(big_status, StatusCode::OK);
    assert_eq!(meta_status, StatusCode::OK);

    std::fs::remove_dir_all(root).unwrap();
    let _ = std::fs::remove_file(&db_path);
}

#[tokio::test]
async fn active_eliminations_serve_from_the_halving_run() {
    let (app, root) = seeded_app(default_seed);
    let (status, body) = http_get(
        app,
        "/api/bench/tuner/projection/runs/version4-active-halving/active-eliminations",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let rows = body_json(&body);
    assert!(!rows.as_array().unwrap().is_empty());
    assert!(rows[0]["action"].is_string());
    assert!(rows[0]["margin_kind"].is_string());
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn missing_and_errored_runs() {
    let (app, root) = seeded_app(default_seed);

    let (status, _) = http_get(app.clone(), "/api/bench/tuner/projection/runs/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = http_get(app.clone(), "/api/bench/tuner/projection/runs/nope/cohorts").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // The garbage-manifest run is projected with its ingest error and empty
    // child collections -- never a 500.
    let (status, body) =
        http_get(app.clone(), "/api/bench/tuner/projection/runs/broken").await;
    assert_eq!(status, StatusCode::OK);
    let detail = body_json(&body);
    assert!(detail["ingest_error"].as_str().unwrap().contains("ValueError"));
    assert!(detail["manifest"].is_null());
    assert!(detail["compute"].as_array().unwrap().is_empty());

    let (status, body) =
        http_get(app.clone(), "/api/bench/tuner/projection/runs/broken/candidates").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body_json(&body).as_array().unwrap().is_empty());

    let (status, _) =
        http_get(app, "/api/bench/tuner/projection/runs/broken/report").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn refresh_endpoint_reports_counts() {
    let (app, root) = seeded_app(default_seed);
    let (status, body) =
        http_post_json(app, "/api/bench/tuner/projection/refresh", serde_json::json!({})).await;
    assert_eq!(status, StatusCode::OK);
    let counts = body_json(&body);
    // The support harness injects a stub returning [2, 1, 0, 0].
    assert_eq!(counts["projected"], 2);
    assert_eq!(counts["skipped"], 1);
    assert_eq!(counts["ingest_errors"], 0);
    assert_eq!(counts["pruned"], 0);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn legacy_routes_absent() {
    let (app, root) = seeded_app(default_seed);
    for uri in [
        "/api/bench/tuner/sessions",
        "/api/bench/tuner/sessions/x/analysis",
        "/api/bench/tuner/sessions/x/trials",
    ] {
        let (status, _) = http_get(app.clone(), uri).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{uri} should be gone");
    }
    std::fs::remove_dir_all(root).unwrap();
}
