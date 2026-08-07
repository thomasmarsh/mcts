// Local web server for playing board games (Druid today) in a browser.
//
// Stateless per request: every route that needs a position takes the full
// game state as JSON and hands back the result -- there is no server-side
// session, no mutable game-in-progress, and no auth (PLAN-UI.md session 2).
// The client is expected to hold the authoritative game tree; the server
// only ever computes (never remembers) a position's legal moves, successor
// state, AI move, or analysis. The one server-side mutable state is each
// `GameAdapter`'s own AI-engine reuse cache (see `adapters::druid::EngineCache`),
// which is a performance cache only -- safe to evict or miss at any time,
// never consulted for correctness.

mod adapters;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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

use adapters::{AdapterError, GameAdapter};

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
    games: HashMap<&'static str, Arc<dyn GameAdapter>>,
}

fn registry() -> HashMap<&'static str, Arc<dyn GameAdapter>> {
    let all: Vec<Arc<dyn GameAdapter>> = vec![
        Arc::new(adapters::druid::DruidAdapter::default()),
        Arc::new(adapters::ttt::TttAdapter),
    ];
    all.into_iter().map(|a| (a.kind(), a)).collect()
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
    let config = req.config.unwrap_or_else(|| adapter.default_config());
    let state = adapter.new_state(config)?;
    let view = adapter.view(&state)?;
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
    let moves = adapter.legal_moves(&req.state)?;
    Ok(Json(json!({ "moves": moves })))
}

async fn post_view(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(req): Json<StateRequest>,
) -> Result<Json<Value>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    let view = adapter.view(&req.state)?;
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
    let state = adapter.apply(&req.state, &req.mv)?;
    let view = adapter.view(&state)?;
    Ok(Json(json!({ "state": state, "view": view })))
}

async fn get_ai_presets(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
) -> Result<Json<Vec<adapters::AiPresetInfo>>, AdapterError> {
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
    let result = tokio::task::spawn_blocking(move || search_adapter.ai_move(&req.state, &req.preset))
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
) -> Result<Json<adapters::Analysis>, AdapterError> {
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
    let app_state = Arc::new(AppState { games: registry() });

    // `ui/`'s Vite build (`pnpm build`, or `pnpm dev`'s proxy in
    // development -- see ui/README.md) is the only frontend now; the old
    // hand-rolled `server/static/app.js` was retired in PLAN-UI.md session 4
    // once it stopped matching session 2's stateless API.
    let static_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("server/static/dist");

    let app = api_router(app_state).fallback_service(ServeDir::new(static_dir));

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
    use mcts::games::druid::HashedState;
    use std::time::Instant;
    use tower::ServiceExt;

    fn test_app() -> Router {
        api_router(Arc::new(AppState { games: registry() }))
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
            app,
            "/api/games/druid/new",
            json!({ "config": { "size": { "w": w, "h": h } } }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        body_json(&body)["state"].clone()
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
        assert_eq!(kinds, vec!["druid", "ttt"]);
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

        // PLAN-UI.md session 9: every error response is a structured
        // `{error, code}` JSON body now, not a bare string.
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
        let (status, _) =
            http_post_raw(test_app(), "/api/games/druid/new", b"{not valid json".to_vec()).await;
        assert_eq!(status, HttpStatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_analyze_clamps_out_of_range_budget_ms_instead_of_rejecting() {
        // A `budget_ms` far outside `DruidAdapter`'s clamp range must not
        // fail the request -- it's silently bounded to a sane value instead
        // (`adapters::druid::clamp_budget_ms`, unit-tested directly in that
        // module). Using the forced-win position keeps this fast regardless
        // of how large a budget was actually honored, since MCTS-Solver
        // stops the moment the root is proven.
        let app = test_app();
        let state = forced_win_state();

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

        let (status, body) =
            http_post_json(app.clone(), "/api/games/druid/legal_moves", json!({ "state": state }))
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
    async fn test_state_round_trips_through_from_state() {
        // A state this server just emitted must deserialize back into an
        // equivalent `HashedState` (this is the whole point of
        // `HashedState::from_state` -- see its doc comment in
        // `src/games/druid.rs`), confirmed here via the public HTTP surface
        // rather than reaching into adapter internals.
        use mcts::game::Game;
        let app = test_app();
        let state = new_druid_state(app.clone(), 5, 5).await;

        let (status, body) =
            http_post_json(app, "/api/games/druid/legal_moves", json!({ "state": state })).await;
        assert_eq!(status, HttpStatusCode::OK);
        let moves = body_json(&body)["moves"].as_array().unwrap().len();

        let expected = HashedState::new(mcts::games::druid::Size { w: 5, h: 5 });
        let mut expected_moves: Vec<mcts::games::druid::Move> = Vec::new();
        mcts::games::druid::Druid::generate_actions(&expected, &mut expected_moves);
        assert_eq!(moves, expected_moves.len());
    }

    // Same forced-win position `server/main.rs`'s old test suite used:
    // Black owns 2 of 3 in each of two columns on a 3x3 board, giving Black
    // two winning threats after White is forced to block one.
    fn forced_win_state() -> serde_json::Value {
        use mcts::game::Game;
        use mcts::games::druid::{Druid, Move, Piece, Size};
        let size = Size { w: 3, h: 3 };
        let mut state = HashedState::new(size);
        let moves = [Piece::Sarsen; 7];
        let cells: [u8; 7] = [0, 1, 3, 4, 2, 7, 5];
        for (piece, cell) in moves.iter().zip(cells.iter()) {
            state = Druid::apply(state, &Move(*piece, *cell));
        }
        serde_json::to_value(state.state()).unwrap()
    }

    #[tokio::test]
    async fn test_ai_move_converts_forced_win() {
        let app = test_app();
        let state = forced_win_state();

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
        let state = forced_win_state();

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
        // Not asserting "most visits": MCTS-Solver stops the search the
        // moment the root is proven, which can fire right after the winning
        // move's first visit -- see `test_root_report_flags_the_proven_winning_move`
        // in src/strategies/tests.rs for the full reasoning. The suggested
        // move being both present and proven (checked above) is what's
        // actually guaranteed.
    }

    #[tokio::test]
    async fn test_ai_move_engine_cache_hits_on_repeated_state() {
        // Two `ai_move` calls on the *same* state (no move applied in
        // between) should reuse and grow the same cached engine's arena
        // rather than rebuilding from scratch each time -- see
        // `adapters::druid::EngineCache`'s doc comment for exactly what this
        // cache does and doesn't cover. Uses `analyze`'s `total_visits`
        // (which reports the live engine's `root_stats`) as the growth
        // signal, since `ai_move`'s response doesn't expose arena size.
        let app = test_app();
        let state = new_druid_state(app.clone(), 5, 5).await;

        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/druid/analyze",
            json!({ "state": state, "preset": "easy", "budget_ms": 50 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let visits_1 = body_json(&body)["total_visits"].as_u64().unwrap();

        let (status, body) = http_post_json(
            app,
            "/api/games/druid/analyze",
            json!({ "state": state, "preset": "easy", "budget_ms": 50 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let visits_2 = body_json(&body)["total_visits"].as_u64().unwrap();

        assert!(
            visits_2 > visits_1,
            "repeated analyze on the same state should keep growing the cached engine's \
             visits (1st: {visits_1}, 2nd: {visits_2}) -- looks like the cache missed and \
             rebuilt instead of reusing"
        );
    }

    #[tokio::test]
    async fn test_ai_move_engine_cache_misses_on_different_state() {
        let app = test_app();
        let state_a = new_druid_state(app.clone(), 5, 5).await;
        let state_b = new_druid_state(app.clone(), 7, 7).await;

        let (status, body) = http_post_json(
            app.clone(),
            "/api/games/druid/analyze",
            json!({ "state": state_a, "preset": "medium", "budget_ms": 50 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let visits_a = body_json(&body)["total_visits"].as_u64().unwrap();

        // A different (larger) board is a guaranteed cache miss: not only a
        // different hash, but a different arena entirely. If this somehow
        // reused `state_a`'s engine, `total_visits` here would already be
        // >= `visits_a` plus this run's own iterations instead of starting
        // fresh.
        let (status, body) = http_post_json(
            app,
            "/api/games/druid/analyze",
            json!({ "state": state_b, "preset": "medium", "budget_ms": 50 }),
        )
        .await;
        assert_eq!(status, HttpStatusCode::OK);
        let visits_b = body_json(&body)["total_visits"].as_u64().unwrap();

        assert!(
            visits_b < visits_a + 5,
            "a different state's analyze should start from a fresh engine, not build on \
             state_a's {visits_a} visits (got {visits_b})"
        );
    }

    // The AI's thinking budget runs on a `spawn_blocking` thread with
    // `num_tree_threads` `thread::scope` workers underneath it. This
    // confirms that pattern keeps the async executor free -- other requests
    // should complete quickly while an AI move is in flight, not queue up
    // behind it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_ai_move_does_not_stall_other_requests() {
        let app = test_app();
        let state = new_druid_state(app.clone(), 5, 5).await;

        let ai_app = app.clone();
        let ai_state = state.clone();
        let ai_task = tokio::spawn(async move {
            let start = Instant::now();
            let (status, _) = http_post_json(
                ai_app,
                "/api/games/druid/ai_move",
                json!({ "state": ai_state, "preset": "strong" }),
            )
            .await;
            (status, start.elapsed())
        });

        // Give the AI request time to get into its thinking budget before
        // probing.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let mut probe_latencies = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let (status, _) = http_post_json(
                app.clone(),
                "/api/games/druid/legal_moves",
                json!({ "state": state }),
            )
            .await;
            assert_eq!(status, HttpStatusCode::OK);
            probe_latencies.push(start.elapsed());
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        let (ai_status, ai_elapsed) = ai_task.await.unwrap();
        assert_eq!(ai_status, HttpStatusCode::OK);
        assert!(
            ai_elapsed >= std::time::Duration::from_secs(2),
            "AI move returned in {ai_elapsed:?}, expected it to use ~3s of its Strong budget"
        );

        for latency in probe_latencies {
            assert!(
                latency < std::time::Duration::from_millis(500),
                "a legal_moves request took {latency:?} while an AI move was in flight -- \
                 looks like it stalled behind the AI request instead of running concurrently"
            );
        }
    }

    // Tic-tac-toe (PLAN-UI.md session 8): the second game proving the
    // `GameAdapter` contract generalizes. Deliberately lighter than Druid's
    // suite above -- no engine-cache or concurrency tests, since
    // `adapters::ttt::TttAdapter` has neither.

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
        assert_eq!(body["move"], 7, "expected the forced block at cell 7: {body}");
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

    #[tokio::test]
    async fn test_ai_move_rebuilds_on_preset_switch() {
        // A client can request a different preset on the next `ai_move`
        // call for the same state. There's no persisted single engine to
        // "switch" anymore (each cache entry is already keyed by preset),
        // but this confirms it doesn't panic or deadlock, and that a real
        // search still runs for the new preset.
        let app = test_app();
        let state = forced_win_state();

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
