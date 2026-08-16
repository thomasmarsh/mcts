// Local web server for playing board games (Druid today) in a browser.
//
// Stateless per request: every route that needs a position takes the full
// game state as JSON and hands back the result -- there is no server-side
// session, no mutable game-in-progress, and no auth.
// The client is expected to hold the authoritative game tree; the server
// only ever computes (never remembers) a position's legal moves, successor
// state, AI move, or analysis.

mod adapter;
mod bench;

const BUILD_INFO: mcts_bench::launch::BuildInfo<'static> = mcts_bench::launch::BuildInfo {
    git_sha: env!("GIT_SHA"),
    git_dirty: matches!(env!("GIT_DIRTY").as_bytes(), b"true"),
};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use mcts_bench::{ingest, schema};

use axum::{
    extract::DefaultBodyLimit,
    extract::Path,
    extract::State as AxumState,
    http::{HeaderValue, Method, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::{cors::CorsLayer, services::ServeDir, timeout::TimeoutLayer};

use adapter::{AdapterError, AiPresetInfo, Analysis, GameAdapter};

// A stateless server still shouldn't let a client tie up a `spawn_blocking`
// thread indefinitely -- this bounds `ai_move`/`analyze` well above the
// slowest preset's default budget (Druid's Master, 8s) so it only ever
// fires on a genuinely stuck/abusive request, not normal use.
const AI_ROUTE_TIMEOUT: Duration = Duration::from_secs(30);

// No request body this API accepts is legitimately large -- every route
// takes a single game state/move, and Druid's board (this repo's biggest
// state) is capped at ~100 cells by `Size::is_supported` (see
// `src/games/druid.rs`). 1 MiB is generous headroom over any real payload.
const MAX_BODY_BYTES: usize = 1024 * 1024;

struct AppState {
    games: Arc<HashMap<&'static str, Arc<dyn GameAdapter>>>,
}

fn find_adapter(app: &AppState, kind: &str) -> Result<Arc<dyn GameAdapter>, AdapterError> {
    app.games
        .get(kind)
        .cloned()
        .ok_or_else(|| AdapterError::not_found(format!("unknown game kind {kind:?}")))
}

#[derive(Serialize)]
struct GameInfo {
    kind: &'static str,
    label: &'static str,
    description: &'static str,
    config_schema: Value,
}

async fn get_games(AxumState(app): AxumState<Arc<AppState>>) -> Json<Vec<GameInfo>> {
    let mut games: Vec<GameInfo> = app
        .games
        .values()
        .map(|a| GameInfo {
            kind: a.kind(),
            label: a.label(),
            description: a.description(),
            config_schema: a.default_config(),
        })
        .collect();
    games.sort_by_key(|g| g.kind);
    Json(games)
}

#[derive(Deserialize)]
struct NewRequest {
    #[serde(default)]
    config: Option<Value>,
}

async fn post_new(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(req): Json<NewRequest>,
) -> Result<Json<Value>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    let adapter2 = adapter.clone();
    let config = req.config.unwrap_or_else(|| adapter.default_config());
    let state = tokio::task::spawn_blocking(move || adapter.new_state(config))
        .await
        .map_err(|e| AdapterError::internal(e.to_string()))??;
    let state2 = state.clone();
    let view = tokio::task::spawn_blocking(move || adapter2.view(&state2))
        .await
        .map_err(|e| AdapterError::internal(e.to_string()))??;
    Ok(Json(json!({ "state": state, "view": view })))
}

#[derive(Deserialize)]
struct StateRequest {
    state: Value,
}

async fn post_legal_moves(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(req): Json<StateRequest>,
) -> Result<Json<Value>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    let state = req.state;
    let moves = tokio::task::spawn_blocking(move || adapter.legal_moves(&state))
        .await
        .map_err(|e| AdapterError::internal(e.to_string()))??;
    Ok(Json(json!({ "moves": moves })))
}

async fn post_view(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(req): Json<StateRequest>,
) -> Result<Json<Value>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    let state = req.state;
    let view = tokio::task::spawn_blocking(move || adapter.view(&state))
        .await
        .map_err(|e| AdapterError::internal(e.to_string()))??;
    Ok(Json(view))
}

#[derive(Deserialize)]
struct ApplyRequest {
    state: Value,
    #[serde(rename = "move")]
    mv: Value,
}

async fn post_apply(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(req): Json<ApplyRequest>,
) -> Result<Json<Value>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    let adapter2 = adapter.clone();
    let state = req.state;
    let mv = req.mv;
    let new_state = tokio::task::spawn_blocking(move || adapter.apply(&state, &mv))
        .await
        .map_err(|e| AdapterError::internal(e.to_string()))??;
    let ns_clone = new_state.clone();
    let view = tokio::task::spawn_blocking(move || adapter2.view(&ns_clone))
        .await
        .map_err(|e| AdapterError::internal(e.to_string()))??;
    Ok(Json(json!({ "state": new_state, "view": view })))
}

async fn get_ai_presets(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
) -> Result<Json<Vec<AiPresetInfo>>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    Ok(Json(adapter.ai_presets()))
}

#[derive(Deserialize)]
struct AiMoveRequest {
    state: Value,
    preset: String,
}

// Runs on a blocking thread -- the search is CPU-bound for its whole
// thinking budget (up to Master's 8s) and would otherwise stall the async
// executor. Unlike the old session-`Mutex`-based server, there's no shared
// game state that could change out from under this call while it runs: the
// state came in as a request body, not a session read, so there's no "board
// changed while the AI was thinking" race to handle here anymore.
async fn post_ai_move(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(req): Json<AiMoveRequest>,
) -> Result<Json<Value>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    let search_adapter = adapter.clone();
    let result =
        tokio::task::spawn_blocking(move || search_adapter.ai_move(&req.state, &req.preset))
            .await
            .map_err(|e| AdapterError::internal(e.to_string()))??;
    let view = adapter.view(&result.state)?;
    Ok(Json(
        json!({ "move": result.mv, "state": result.state, "view": view }),
    ))
}

#[derive(Deserialize)]
struct AnalyzeRequest {
    state: Value,
    preset: String,
    #[serde(default)]
    budget_ms: Option<u64>,
}

async fn post_analyze(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(req): Json<AnalyzeRequest>,
) -> Result<Json<Analysis>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    let analysis = tokio::task::spawn_blocking(move || {
        adapter.analyze(&req.state, &req.preset, req.budget_ms)
    })
    .await
    .map_err(|e| AdapterError::internal(e.to_string()))??;
    Ok(Json(analysis))
}

// Split out from `main` so tests can exercise the API surface directly
// (`tower::ServiceExt::oneshot`) without binding a real socket or serving
// static files.
fn api_router(app_state: Arc<AppState>) -> Router {
    // `ai_move`/`analyze` get their own `TimeoutLayer` -- they're the only
    // routes that run a CPU-bound search on a `spawn_blocking` thread, so
    // they're the only ones that can legitimately run long enough to need
    // one. `tower_http`'s `TimeoutLayer` (unlike `tower::timeout`'s) returns
    // an empty response with the given status directly on elapse rather than
    // erroring, so it needs no separate error-handling layer to stay
    // `Infallible` for axum's `Router`.
    let ai_routes = Router::new()
        .route("/api/games/{kind}/ai_move", post(post_ai_move))
        .route("/api/games/{kind}/analyze", post(post_analyze))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            AI_ROUTE_TIMEOUT,
        ));

    let other_routes = Router::new()
        .route("/api/games", get(get_games))
        .route("/api/games/{kind}/new", post(post_new))
        .route("/api/games/{kind}/legal_moves", post(post_legal_moves))
        .route("/api/games/{kind}/view", post(post_view))
        .route("/api/games/{kind}/apply", post(post_apply))
        .route("/api/games/{kind}/ai_presets", get(get_ai_presets));

    // Explicitly scoped, not wildcard -- there's no cross-origin need today
    // (the Vite dev proxy and the production `ServeDir` both serve the API
    // same-origin), but that stops being true the moment `pnpm dev`'s proxy
    // setup changes, and an explicit allow-list costs nothing now.
    let cors = CorsLayer::new()
        .allow_origin([
            "http://127.0.0.1:7878".parse::<HeaderValue>().unwrap(),
            "http://localhost:5173".parse::<HeaderValue>().unwrap(),
            "http://127.0.0.1:5173".parse::<HeaderValue>().unwrap(),
        ])
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    other_routes
        .merge(ai_routes)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .with_state(app_state)
}

#[tokio::main]
async fn main() {
    let app_state = Arc::new(AppState {
        games: Arc::new(adapter::registry()),
    });

    // Open (or create) the benchmark database.  Only the server process ever
    // opens `bench.duckdb` read-write; `bin/bench` and the Python SMAC3
    // harness communicate via JSONL files and the registry log instead.
    let bench_runs_dir = PathBuf::from(mcts_bench::launch::BENCH_RUNS_DIR);
    let bench_db_path = bench_runs_dir.join("bench.duckdb");
    let bench_conn = schema::open(&bench_db_path).expect("failed to open benchmark database");
    let bench_state = Arc::new(bench::BenchState {
        db: std::sync::Mutex::new(bench_conn),
        bench_runs_dir,
        experiment_validator: Arc::new(bench::validate_experiment_spec),
        run_launcher: Arc::new(|run_id, command, kind, game, label| {
            mcts_bench::launch::launch_with_run_id(
                run_id,
                command,
                &kind,
                &game,
                label.as_deref(),
                crate::BUILD_INFO,
            )
        }),
        process_group_signaller: Arc::new(bench::signal_process_group),
    });

    // Start the background ingest loop.  Every 5 seconds it reads
    // registry.log and running runs' log.jsonl files, upserts into the
    // DuckDB, and runs PID liveness reconciliation — so runs launched
    // via the API have their match results and terminal status appear
    // within one polling cycle of the child process exiting.
    {
        let ingest_state = bench_state.clone();
        let bench_runs = bench_state.bench_runs_dir.clone();
        tokio::spawn(async move {
            // Wait a few seconds before the first poll so short-lived
            // processes have time to finish, letting the first ingest
            // pass catch them in one shot rather than waiting for a
            // second tick.
            tokio::time::sleep(Duration::from_secs(3)).await;
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                if let Ok(db) = ingest_state.db.lock() {
                    if let Err(e) = ingest::ingest_once(&db, &bench_runs) {
                        eprintln!("bench ingest error: {e}");
                    }
                }
            }
        });
    }

    // Start the background ladder driver.  Same cadence and shape as the
    // ingest loop above -- every 5 seconds it scans completed SMAC3 runs
    // for a saturated, ladder-enabled rung with budget left and, if found,
    // launches the next one.  A no-op for every run that never opted into
    // `config.ladder`.
    {
        let ladder_state = bench_state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(3)).await;
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                bench::advance_ladders_once(&ladder_state).await;
            }
        });
    }

    // `ui/`'s Vite build (`pnpm build`, or `pnpm dev`'s proxy in
    // development -- see ui/README.md) is the only frontend now; the old
    // hand-rolled `server/static/app.js` was retired once it stopped
    // matching the stateless API.
    let static_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("static/dist");

    let app = api_router(app_state)
        .merge(bench::bench_router(bench_state))
        .fallback_service(ServeDir::new(static_dir));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878")
        .await
        .expect("failed to bind 127.0.0.1:7878");
    println!("Game server listening on http://127.0.0.1:7878");
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode as HttpStatusCode};
    use std::time::Instant;
    use tower::ServiceExt;

    static TEST_GAMES: std::sync::OnceLock<Arc<HashMap<&'static str, Arc<dyn GameAdapter>>>> =
        std::sync::OnceLock::new();

    fn test_app() -> Router {
        api_router(Arc::new(AppState {
            games: TEST_GAMES
                .get_or_init(|| Arc::new(adapter::registry()))
                .clone(),
        }))
    }

    async fn http_get(app: Router, uri: &str) -> (HttpStatusCode, axum::body::Bytes) {
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

    async fn http_post_json(
        app: Router,
        uri: &str,
        json: serde_json::Value,
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

    fn body_json(body: &axum::body::Bytes) -> serde_json::Value {
        serde_json::from_slice(body).unwrap()
    }

    /// Like `http_post_json`, but takes a pre-built raw body instead of a
    /// `serde_json::Value` -- needed for the malformed/oversized-input tests
    /// below, which deliberately send bodies that aren't (or aren't only)
    /// valid JSON.
    async fn http_post_raw(
        app: Router,
        uri: &str,
        body: Vec<u8>,
    ) -> (HttpStatusCode, axum::body::Bytes) {
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, body)
    }

    async fn new_druid_state(app: Router, w: u8, h: u8) -> serde_json::Value {
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/druid/new",
            json!({ "config": { "size": { "w": w, "h": h } } }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        body_json(&body)["state"].clone()
    }

    /// Build a forced-win position (Black has 2 of 3 in two columns on a
    /// 3×3 board, giving two winning threats after White is forced to
    /// block one) through the subprocess-backed test API.
    async fn forced_win_state(app: Router) -> serde_json::Value {
        let mut state = new_druid_state(app.clone(), 3, 3).await;
        // Apply 7 Sarsen moves: Black@0, White@1, Black@3, White@4,
        // Black@2, White@7, Black@5.
        for cell in [0, 1, 3, 4, 2, 7, 5] {
            let (status, body) = http_post_json(
                app.clone(),
                "/api/games/druid/apply",
                json!({ "state": state, "move": ["Sarsen", cell] }),
            )
            .await;
            assert_eq!(status, HttpStatusCode::OK);
            state = body_json(&body)["state"].clone();
        }
        state
    }

    #[tokio::test]
    async fn test_get_games_lists_both_kinds() {
        let (status, body) = http_get(test_app(), "/api/games").await;
        assert_eq!(status, HttpStatusCode::OK);
        let games = body_json(&body);
        let kinds: Vec<&str> = games
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["kind"].as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            vec![
                "atarigo",
                "breakthrough",
                "congo",
                "druid",
                "gonnect",
                "hex-gen",
                "knightthrough",
                "othello",
                "tak",
                "tanbo",
                "traffic-lights",
                "ttt"
            ]
        );
    }

    #[tokio::test]
    async fn test_unknown_game_kind_is_404() {
        let (status, body) = http_post_json(
            test_app(),
            "/api/games/nope/new",
            json!({ "config": { "size": { "w": 5, "h": 5 } } }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::NOT_FOUND);

        // Every error response is a structured
        // `{error, code}` JSON body, not a bare string.
        let body = body_json(&body);
        assert_eq!(body["code"], 404);
        assert!(body["error"].as_str().unwrap().contains("nope"));
    }

    #[tokio::test]
    async fn test_oversized_body_is_413() {
        // One byte over `MAX_BODY_BYTES` -- content doesn't matter, the
        // `DefaultBodyLimit` layer rejects it before the handler (or even
        // the JSON parser) ever sees it.
        let oversized = vec![b'a'; MAX_BODY_BYTES + 1];
        let (status, _) = http_post_raw(test_app(), "/api/games/druid/new", oversized).await;
        assert_eq!(status, HttpStatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_malformed_json_body_is_400_not_500() {
        let (status, _) = http_post_raw(
            test_app(),
            "/api/games/druid/new",
            b"{not valid json".to_vec(),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_analyze_clamps_out_of_range_budget_ms_instead_of_rejecting() {
        // A `budget_ms` far outside the binary's clamp range must not
        // fail the request -- it's silently bounded to a sane value.
        // Using the forced-win position keeps this fast regardless of how
        // large a budget was actually honored, since MCTS-Solver stops the
        // moment the root is proven.
        let app = test_app();
        let state = forced_win_state(app.clone()).await;

        let (status, body) = http_post_json(
            app,
            "/api/games/druid/analyze",
            json!({ "state": state, "preset": "easy", "budget_ms": u64::MAX }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        assert_ne!(body_json(&body)["suggested_move"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn test_new_game_rejects_unsupported_size() {
        let (status, _) = http_post_json(
            test_app(),
            "/api/games/druid/new",
            json!({ "config": { "size": { "w": 2, "h": 2 } } }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_apply_rejects_illegal_move() {
        let app = test_app();
        let state = new_druid_state(app.clone(), 5, 5).await;

        let (status, _) = http_post_json(
            app,
            "/api/games/druid/apply",
            json!({ "state": state, "move": [ "Sarsen", 255 ] }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_apply_accepts_a_legal_move_and_view_reflects_it() {
        let app = test_app();
        let state = new_druid_state(app.clone(), 5, 5).await;

        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/druid/legal_moves",
            json!({ "state": state }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let moves = body_json(&body)["moves"].clone();
        let first_move = moves.as_array().unwrap()[0].clone();

        let (status, body) = http_post_json(
            app,
            "/api/games/druid/apply",
            json!({ "state": state, "move": first_move }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        assert_eq!(body["view"]["player"], "White");
        assert_eq!(body["view"]["terminal"], false);
    }

    #[tokio::test]
    async fn test_druid_new_game_returns_consistent_state() {
        // A state the server just emitted must round-trip through
        // legal_moves and view without errors -- the subprocess handles
        // the internal state reconstruction.
        let app = test_app();
        let state = new_druid_state(app.clone(), 5, 5).await;

        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/druid/legal_moves",
            json!({ "state": state }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let moves = body_json(&body)["moves"].as_array().unwrap().len();
        assert!(moves > 0, "new 5x5 game should have legal moves");

        let (status, _) =
            http_post_json(app, "/api/games/druid/view", json!({ "state": state })).await;
        assert_eq!(status, HttpStatusCode::OK);
    }

    #[tokio::test]
    async fn test_ai_move_converts_forced_win() {
        let app = test_app();
        let state = forced_win_state(app.clone()).await;

        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/druid/ai_move",
            json!({ "state": state, "preset": "easy" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let after_white = body_json(&body)["state"].clone();

        let (status, body) = http_post_json(
            app,
            "/api/games/druid/ai_move",
            json!({ "state": after_white, "preset": "easy" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        assert_eq!(body["view"]["winner"], "Black");
    }

    #[tokio::test]
    async fn test_analyze_on_forced_win_proves_the_winning_move() {
        let app = test_app();
        let state = forced_win_state(app.clone()).await;

        let (status, body) = http_post_json(
            app,
            "/api/games/druid/analyze",
            json!({ "state": state, "preset": "easy" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let analysis = body_json(&body);

        let suggested = analysis["suggested_move"].clone();
        assert_ne!(suggested, serde_json::Value::Null);

        let actions = analysis["actions"].as_array().unwrap();
        let suggested_report = actions
            .iter()
            .find(|a| a["action"] == suggested)
            .expect("suggested move should be an explored root action");
        assert_eq!(
            suggested_report["is_proven"], true,
            "the forced win should be reported as proven: {analysis}"
        );
    }

    // The AI's thinking budget runs on a `spawn_blocking` thread with
    // `num_tree_threads` `thread::scope` workers underneath it. This
    // confirms that pattern keeps the async executor free -- other requests
    // to a *different* game kind (with its own subprocess and mutex) should
    // complete quickly, not queue up behind the AI's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_ai_move_does_not_stall_other_requests() {
        let app = test_app();
        let druid_state = new_druid_state(app.clone(), 5, 5).await;
        let (status, body) = http_post_json(app.clone(), "/api/games/ttt/new", json!({})).await;
        assert_eq!(status, HttpStatusCode::OK);
        let ttt_state = body_json(&body)["state"].clone();

        let ai_app = app.clone();
        let ai_state = druid_state.clone();
        let ai_task = tokio::spawn(async move {
            let (status, _) = http_post_json(
                ai_app,
                "/api/games/druid/ai_move",
                json!({ "state": ai_state, "preset": "strong" }),
            )
            .await;
            status
        });

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let mut probe_latencies = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let (status, _) = http_post_json(
                app.clone(),
                "/api/games/ttt/legal_moves",
                json!({ "state": ttt_state }),
            )
            .await;
            assert_eq!(status, HttpStatusCode::OK);
            probe_latencies.push(start.elapsed());
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        assert_eq!(ai_task.await.unwrap(), HttpStatusCode::OK);

        for latency in probe_latencies {
            assert!(
                latency < std::time::Duration::from_secs(2),
                "a ttt legal_moves request took {latency:?} while a Druid AI move was in flight -- \
                 looks like the async runtime stalled instead of handling concurrent requests"
            );
        }
    }

    // Tic-tac-toe: the second game proving the
    // `GameAdapter` contract generalizes. Deliberately lighter than Druid's
    // suite above -- no engine-cache or concurrency tests, since
    // `SimpleAdapter::<TicTacToe>` has neither.

    async fn new_ttt_state(app: Router) -> Value {
        let (status, body) = http_post_json(app, "/api/games/ttt/new", json!({})).await;
        assert_eq!(status, HttpStatusCode::OK);
        body_json(&body)["state"].clone()
    }

    // Same forced-block position `src/games/ttt.rs`'s own
    // `must_block_position` test helper uses: after X plays 0/4/8 and O
    // plays 1, O threatens to complete column 1 (cells 1, 4, 7) -- cell 7 is
    // X's only non-losing reply.
    async fn forced_block_state(app: Router) -> Value {
        let mut state = new_ttt_state(app.clone()).await;
        for mv in [0, 4, 8, 1] {
            let (status, body) = http_post_json(
                app.clone(),
                "/api/games/ttt/apply",
                json!({ "state": state, "move": mv }),
            )
            .await;
            assert_eq!(status, HttpStatusCode::OK);
            state = body_json(&body)["state"].clone();
        }
        state
    }

    #[tokio::test]
    async fn test_ttt_new_game_has_nine_legal_moves() {
        let app = test_app();
        let state = new_ttt_state(app.clone()).await;

        let (status, body) =
            http_post_json(app, "/api/games/ttt/legal_moves", json!({ "state": state })).await;
        assert_eq!(status, HttpStatusCode::OK);
        assert_eq!(body_json(&body)["moves"].as_array().unwrap().len(), 9);
    }

    #[tokio::test]
    async fn test_ttt_apply_accepts_a_legal_move_and_view_reflects_it() {
        let app = test_app();
        let state = new_ttt_state(app.clone()).await;

        let (status, body) = http_post_json(
            app,
            "/api/games/ttt/apply",
            json!({ "state": state, "move": 4 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        assert_eq!(body["view"]["turn"], "O");
        assert_eq!(body["view"]["cells"][4], "X");
        assert_eq!(body["view"]["terminal"], false);
    }

    #[tokio::test]
    async fn test_ttt_apply_rejects_illegal_move() {
        let app = test_app();
        let state = new_ttt_state(app.clone()).await;

        // Cell 9 is out of a 3x3 board's 0..9 range.
        let (status, _) = http_post_json(
            app.clone(),
            "/api/games/ttt/apply",
            json!({ "state": state, "move": 9 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);

        // Playing an already-occupied cell.
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/ttt/apply",
            json!({ "state": state, "move": 0 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let state = body_json(&body)["state"].clone();

        let (status, _) = http_post_json(
            app,
            "/api/games/ttt/apply",
            json!({ "state": state, "move": 0 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_ttt_ai_move_finds_the_forced_block() {
        let app = test_app();
        let state = forced_block_state(app.clone()).await;

        let (status, body) = http_post_json(
            app,
            "/api/games/ttt/ai_move",
            json!({ "state": state, "preset": "strong" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        assert_eq!(
            body["move"], 7,
            "expected the forced block at cell 7: {body}"
        );
    }

    #[tokio::test]
    async fn test_ttt_analyze_proves_the_forced_block() {
        let app = test_app();
        let state = forced_block_state(app.clone()).await;

        let (status, body) = http_post_json(
            app,
            "/api/games/ttt/analyze",
            json!({ "state": state, "preset": "strong" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let analysis = body_json(&body);

        assert_eq!(
            analysis["suggested_move"], 7,
            "expected the forced block at cell 7: {analysis}"
        );
        let actions = analysis["actions"].as_array().unwrap();
        let suggested_report = actions
            .iter()
            .find(|a| a["action"] == 7)
            .expect("suggested move should be an explored root action");
        assert_eq!(
            suggested_report["is_proven"], true,
            "the forced block should be reported as proven: {analysis}"
        );
    }

    // Traffic Lights: the third game proving the
    // `GameAdapter` contract generalizes. Verifies in particular that
    // board state accumulates across sequential `apply` calls (a bug
    // where `value_to_state` used raw `Piece` discriminants instead of
    // the board-bit encoding would silently drop all R cells, making
    // every move look like a reset to the initial position).

    async fn new_tl_state(app: Router) -> Value {
        let (status, body) = http_post_json(app, "/api/games/traffic-lights/new", json!({})).await;
        assert_eq!(status, HttpStatusCode::OK);
        body_json(&body)["state"].clone()
    }

    #[tokio::test]
    async fn test_tl_new_game_has_nine_legal_moves() {
        let app = test_app();
        let state = new_tl_state(app.clone()).await;

        let (status, body) = http_post_json(
            app,
            "/api/games/traffic-lights/legal_moves",
            json!({ "state": state }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let json = body_json(&body);
        let moves = json["moves"].as_array().unwrap();
        // Every cell is empty → one legal move per cell (place R).
        assert_eq!(moves.len(), 9);
    }

    #[tokio::test]
    async fn test_tl_apply_preserves_state_across_moves() {
        // The core regression test: apply moves sequentially and confirm
        // the board state is not lost between calls.
        let app = test_app();
        let mut state = new_tl_state(app.clone()).await;

        // Player A places R at cell 0 → move encoding: (0 << 2) | 0 = 0
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": 0 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        state = body["state"].clone();
        let cells = body["view"]["cells"].as_array().unwrap();
        assert_eq!(cells[0], "R", "cell 0 should be R after first move");
        assert_eq!(body["view"]["turn"], "B", "turn should switch to B");
        assert_eq!(body["view"]["terminal"], false);

        // Player B places R at cell 1 → move encoding: (1 << 2) | 0 = 4
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": 4 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        state = body["state"].clone();
        let cells = body["view"]["cells"].as_array().unwrap();
        assert_eq!(cells[0], "R", "cell 0 should still be R");
        assert_eq!(cells[1], "R", "cell 1 should be R");
        assert_eq!(body["view"]["turn"], "A");

        // Player A advances cell 0 from R → Y → move encoding: (0 << 2) | 1 = 1
        let (status, step3_body) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": 1 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let step3 = body_json(&step3_body);
        let cells = step3["view"]["cells"].as_array().unwrap();
        assert_eq!(cells[0], "Y", "cell 0 should be Y after advance");
        assert_eq!(cells[1], "R", "cell 1 should still be R");
        assert_eq!(step3["view"]["turn"], "B");
        state = step3["state"].clone();

        // Player B advances cell 1 from R → Y → move encoding: (1 << 2) | 1 = 5
        let (status, body) = http_post_json(
            app,
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": 5 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        let cells = body["view"]["cells"].as_array().unwrap();
        assert_eq!(cells[0], "Y", "cell 0 should still be Y");
        assert_eq!(cells[1], "Y", "cell 1 should be Y");
        assert_eq!(body["view"]["turn"], "A");
    }

    #[tokio::test]
    async fn test_tl_apply_rejects_illegal_move() {
        let app = test_app();
        let state = new_tl_state(app.clone()).await;

        // Move index 9 is out of the 3×3 board (cells 0..8).
        let (status, _) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": 36 }), // (9 << 2) | 0 = 36
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);

        // Play cell 0 then try to play it again with the wrong move
        // (same move again, which would mean placing R on an already-R
        // cell — the engine sees a different piece encoding than what
        // the cell's current state expects).
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": 0 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let state = body_json(&body)["state"].clone();

        // Same raw move again — cell 0 is now R, so placing R (move 0)
        // is illegal; the only legal move for cell 0 now is Y (move 1).
        let (status, _) = http_post_json(
            app,
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": 0 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_tl_ai_move_finds_a_legal_action() {
        let app = test_app();
        let state = new_tl_state(app.clone()).await;

        let (status, body) = http_post_json(
            app,
            "/api/games/traffic-lights/ai_move",
            json!({ "state": state, "preset": "easy" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        let mv = body["move"].as_u64().unwrap() as u8;
        let index = (mv >> 2) as usize;
        assert!(
            index < 9,
            "AI move index {index} should be within the board"
        );
        // AI chose a legal move and returned a new state with that cell occupied.
        let cells = body["view"]["cells"].as_array().unwrap();
        assert_eq!(cells[index], "R", "AI-placed cell {index} should be R");
    }

    #[tokio::test]
    async fn test_tl_user_center_ai_then_advance_center() {
        // Reproduction of the user-reported scenario: user plays
        // center, AI plays, then user tries to play again.
        // Confirms the server stays consistent throughout.
        let app = test_app();
        let mut state = new_tl_state(app.clone()).await;

        // Play cell 4 (center) with R: move = (4 << 2) | 0 = 16
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": 16 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let resp = body_json(&body);
        assert_eq!(resp["view"]["cells"][4], "R");
        assert_eq!(resp["view"]["turn"], "B");
        state = resp["state"].clone();

        // Get legal moves before the AI plays.
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/legal_moves",
            json!({ "state": state }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let legal = body_json(&body);
        let pre_ai: Vec<u8> = serde_json::from_value(legal["moves"].clone()).unwrap();
        // Cell 4 is R → must advance to Y (move 17)
        assert!(
            pre_ai.contains(&17),
            "advance cell 4 R->Y should be legal: {pre_ai:?}"
        );

        // AI plays as B.
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/ai_move",
            json!({ "state": state, "preset": "strong" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let resp = body_json(&body);
        let ai_move: u8 = serde_json::from_value(resp["move"].clone()).unwrap();
        assert!(
            pre_ai.contains(&ai_move),
            "AI move {ai_move} must be in legal set {pre_ai:?}"
        );
        state = resp["state"].clone();

        // The state after AI is self-consistent: generate legal moves
        // and verify every cell has exactly one legal move.
        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/traffic-lights/legal_moves",
            json!({ "state": state }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let legal = body_json(&body);
        let post_ai: Vec<u8> = serde_json::from_value(legal["moves"].clone()).unwrap();
        // Exactly 9 moves (one per cell)
        assert_eq!(post_ai.len(), 9, "should have 9 legal moves: {post_ai:?}");

        // The cell the AI played on must have advanced (not regressed).
        let view = &resp["view"];
        let cells = view["cells"].as_array().unwrap();
        let ai_idx = (ai_move >> 2) as usize;
        let val = cells[ai_idx].as_str().map(|s| s.to_owned());
        assert!(
            val.is_some(),
            "AI-played cell {ai_idx} should not be empty after move"
        );

        // Any user cell they can click will work.
        // Pick the first non-AI cell and play it.
        let user_idx = if ai_idx == 4 { 0usize } else { 4usize };
        let user_moves: Vec<u8> = post_ai
            .iter()
            .copied()
            .filter(|m| (m >> 2) as usize == user_idx)
            .collect();
        assert_eq!(
            user_moves.len(),
            1,
            "cell {user_idx} must have one legal move"
        );
        let (status, _body) = http_post_json(
            app,
            "/api/games/traffic-lights/apply",
            json!({ "state": state, "move": user_moves[0] }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
    }

    #[tokio::test]
    async fn test_ai_move_rebuilds_on_preset_switch() {
        // A client can request a different preset on the next `ai_move`
        // call for the same state. This confirms it doesn't panic or
        // deadlock, and that both presets produce a valid response.
        let app = test_app();
        let state = forced_win_state(app.clone()).await;

        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/druid/ai_move",
            json!({ "state": state, "preset": "easy" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        assert_eq!(body_json(&body)["view"]["winner"], serde_json::Value::Null);

        let (status, body) = http_post_json(
            app,
            "/api/games/druid/ai_move",
            json!({ "state": state, "preset": "medium" }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        assert_eq!(body_json(&body)["view"]["winner"], serde_json::Value::Null);
    }
}
