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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mcts_bench::duckdb_composition::BenchAdapters;

use axum::{
    extract::DefaultBodyLimit,
    extract::Path,
    extract::State as AxumState,
    http::{HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::{cors::CorsLayer, services::ServeDir, timeout::TimeoutLayer};

use adapter::{AdapterError, AiMoveResult, AiPresetInfo, Analysis, GameAdapter};
use game_host::{SearchReport, TunerInfo};

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

async fn get_strategy_families(
    AxumState(app): AxumState<Arc<AppState>>,
    Path(kind): Path<String>,
) -> Result<Json<Option<TunerInfo>>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    Ok(Json(adapter.tuner()))
}

#[derive(Deserialize)]
struct AiMoveRequest {
    state: Value,
    preset: String,
    #[serde(default)]
    custom: Option<Value>,
}

/// `ai_move` preserves the host's final-search report while adding the view
/// the HTTP client needs to render the returned state.
#[derive(Serialize)]
struct AiMoveResponse {
    #[serde(rename = "move")]
    mv: Value,
    state: Value,
    view: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    search: Option<SearchReport>,
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
) -> Result<Json<AiMoveResponse>, AdapterError> {
    let adapter = find_adapter(&app, &kind)?;
    let search_adapter = adapter.clone();
    let result = tokio::task::spawn_blocking(move || {
        search_adapter.ai_move(&req.state, &req.preset, req.custom.as_ref())
    })
    .await
    .map_err(|e| AdapterError::internal(e.to_string()))??;
    let AiMoveResult { mv, state, search } = result;
    let view = adapter.view(&state)?;
    Ok(Json(AiMoveResponse {
        mv,
        state,
        view,
        search,
    }))
}

#[derive(Deserialize)]
struct AnalyzeRequest {
    state: Value,
    preset: String,
    #[serde(default)]
    custom: Option<Value>,
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
        adapter.analyze(&req.state, &req.preset, req.custom.as_ref(), req.budget_ms)
    })
    .await
    .map_err(|e| AdapterError::internal(e.to_string()))??;
    Ok(Json(analysis))
}

async fn get_strategy_schema() -> Json<Value> {
    Json(mcts_tune::config_ir_schema::axis_schema())
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
        .route("/api/strategy-schema", get(get_strategy_schema))
        .route("/api/games/{kind}/new", post(post_new))
        .route("/api/games/{kind}/legal_moves", post(post_legal_moves))
        .route("/api/games/{kind}/view", post(post_view))
        .route("/api/games/{kind}/apply", post(post_apply))
        .route("/api/games/{kind}/ai_presets", get(get_ai_presets))
        .route(
            "/api/games/{kind}/strategy-families",
            get(get_strategy_families),
        );

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

/// Log every HTTP request (method, path, status, duration) to stdout.
async fn log_request(req: axum::http::Request<axum::body::Body>, next: Next) -> Response {
    let start = Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();

    let response = next.run(req).await;

    let duration = start.elapsed();
    let status = response.status().as_u16();

    println!("  {:>6}  {}  {}  ({duration:?})", method, uri, status);

    response
}

#[tokio::main]
async fn main() {
    let app_state = Arc::new(AppState {
        games: Arc::new(adapter::registry()),
    });

    // Open (or create) the benchmark database.  Only the server process ever
    // opens `bench.duckdb` read-write; `bin/bench` and the Python tuner
    // harness communicate via JSONL files and the registry log instead.
    let bench_runs_dir = PathBuf::from(mcts_bench::launch::BENCH_RUNS_DIR);
    let bench_db_path = bench_runs_dir.join("bench.duckdb");
    // Read-only SQLite projection of version-4 tuner runs. Defaults under the
    // runs root; `MCTS_TUNER_PROJECTION_DB` overrides it.
    let tuner_projection_db = std::env::var_os("MCTS_TUNER_PROJECTION_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|| bench_runs_dir.join("tuner-projection.sqlite"));
    let bench_adapters =
        BenchAdapters::open(&bench_db_path).expect("failed to open benchmark database");
    // Frozen-objective JSON files the tuner launch form offers. Defaults to
    // the checked-in `tuner/objectives`; `MCTS_TUNER_OBJECTIVES_DIR` overrides.
    let tuner_objectives_dir = std::env::var_os("MCTS_TUNER_OBJECTIVES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tuner/objectives"));
    let bench_state = Arc::new(bench::BenchState {
        #[cfg(test)]
        db: bench::TestDatabase::unavailable(),
        projects_repository: bench_adapters.projects_repository,
        run_repository: bench_adapters.run_repository,
        run_command_repository: bench_adapters.run_command_repository,
        bench_runs_dir,
        tuner_objectives_dir,
        process_group_signaller: Arc::new(bench::signal_process_group),
        tuner_projection_db,
        tuner_projection_refresh: Arc::new(bench::shell_refresh),
    });

    // Start the background ingest loop.  Every 5 seconds it reads
    // registry.log and running runs' log.jsonl files, upserts into the
    // DuckDB, and runs PID liveness reconciliation — so runs launched
    // via the API have their match results and terminal status appear
    // within one polling cycle of the child process exiting.
    {
        let ingest = bench_adapters.ingest.clone();
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
                if let Err(e) = ingest.ingest_once(&bench_runs) {
                    eprintln!("bench ingest error: {e}");
                }
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
        .fallback_service(ServeDir::new(static_dir))
        .layer(axum::middleware::from_fn(log_request));

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
    use game_host::{
        AnalysisAction, SearchGraphMode, SearchReportReason, SearchReportStatus, SearchTermination,
        SearchWarning,
    };
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

    fn transport_report(status: SearchReportStatus) -> SearchReport {
        let (reason, warnings) = match status {
            SearchReportStatus::Available => (None, vec![]),
            SearchReportStatus::Partial => (
                Some(SearchReportReason::RootParallelPvSingleTree),
                vec![SearchWarning::RootParallelPvSingleTree],
            ),
            SearchReportStatus::Unavailable => {
                (Some(SearchReportReason::StrategyUnsupported), vec![])
            }
        };
        SearchReport {
            schema_version: 1,
            status,
            reason,
            elapsed_seconds: None,
            iteration_limit: Some(10),
            time_limit_seconds: None,
            completed_iterations: 10,
            termination: Some(SearchTermination::Iterations),
            selected_action: Some(json!("chosen")),
            actions: vec![],
            principal_variation: vec![json!("chosen")],
            root_visits: 10,
            tree_nodes: 11,
            mean_depth: None,
            max_depth: None,
            graph_mode: Some(SearchGraphMode::Tree),
            tt_reads: 0,
            tt_writes: 0,
            tt_hits: 0,
            tt_hit_ratio: None,
            iterations_per_second: None,
            warnings,
        }
    }

    fn transport_search(preset: &str) -> Option<SearchReport> {
        match preset {
            "available" => Some(transport_report(SearchReportStatus::Available)),
            "partial" => Some(transport_report(SearchReportStatus::Partial)),
            "unavailable" => Some(transport_report(SearchReportStatus::Unavailable)),
            "legacy" => None,
            other => panic!("unexpected transport test preset: {other}"),
        }
    }

    struct TransportAdapter;

    impl GameAdapter for TransportAdapter {
        fn kind(&self) -> &'static str {
            "transport"
        }

        fn label(&self) -> &'static str {
            "Transport"
        }

        fn description(&self) -> &'static str {
            "Search-report transport test adapter"
        }

        fn default_config(&self) -> Value {
            json!({})
        }

        fn new_state(&self, _config: Value) -> Result<Value, AdapterError> {
            Ok(json!({ "position": "initial" }))
        }

        fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, AdapterError> {
            Ok(vec![json!("chosen")])
        }

        fn apply(&self, _state: &Value, _mv: &Value) -> Result<Value, AdapterError> {
            Ok(json!({ "position": "after" }))
        }

        fn view(&self, state: &Value) -> Result<Value, AdapterError> {
            Ok(json!({ "view_of": state["position"] }))
        }

        fn ai_presets(&self) -> Vec<AiPresetInfo> {
            vec![]
        }

        fn ai_move(
            &self,
            _state: &Value,
            preset: &str,
            _custom: Option<&Value>,
        ) -> Result<AiMoveResult, AdapterError> {
            Ok(AiMoveResult {
                mv: json!("chosen"),
                state: json!({ "position": "after" }),
                search: transport_search(preset),
            })
        }

        fn analyze(
            &self,
            _state: &Value,
            preset: &str,
            _custom: Option<&Value>,
            _budget_ms: Option<u64>,
        ) -> Result<Analysis, AdapterError> {
            Ok(Analysis {
                actions: vec![AnalysisAction {
                    action: json!("chosen"),
                    visits: 10,
                    mean_value: 0.5,
                    is_proven: false,
                }],
                principal_variation: vec![json!("chosen")],
                total_visits: 10,
                suggested_move: Some(json!("chosen")),
                search: transport_search(preset),
            })
        }

        fn tuner(&self) -> Option<TunerInfo> {
            None
        }
    }

    fn transport_test_app() -> Router {
        let games: Arc<HashMap<&'static str, Arc<dyn GameAdapter>>> = Arc::new(HashMap::from([(
            "transport",
            Arc::new(TransportAdapter) as Arc<dyn GameAdapter>,
        )]));
        api_router(Arc::new(AppState { games }))
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

    #[tokio::test]
    async fn test_search_reports_survive_ai_move_and_analyze_transport() {
        let app = transport_test_app();
        for (preset, expected_status) in [
            ("available", Some("available")),
            ("partial", Some("partial")),
            ("unavailable", Some("unavailable")),
            ("legacy", None),
        ] {
            let request = json!({ "state": { "position": "before" }, "preset": preset });
            let (status, body) =
                http_post_json(app.clone(), "/api/games/transport/ai_move", request.clone()).await;
            assert_eq!(status, HttpStatusCode::OK);
            let ai_move = body_json(&body);
            assert_eq!(ai_move["move"], "chosen");
            assert_eq!(ai_move["state"]["position"], "after");
            assert_eq!(ai_move["view"], json!({ "view_of": "after" }));

            let (status, body) =
                http_post_json(app.clone(), "/api/games/transport/analyze", request).await;
            assert_eq!(status, HttpStatusCode::OK);
            let analysis = body_json(&body);
            assert_eq!(analysis["suggested_move"], "chosen");
            assert_eq!(analysis["actions"][0]["action"], "chosen");

            for response in [&ai_move, &analysis] {
                match expected_status {
                    Some(expected_status) => {
                        assert_eq!(response["search"]["status"], expected_status);
                        assert!(response["search"]["elapsed_seconds"].is_null());
                        assert!(response["search"]["time_limit_seconds"].is_null());
                    }
                    None => assert!(response.get("search").is_none()),
                }
            }
        }
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
                "akron",
                "atarigo",
                "breakthrough",
                "congo",
                "druid",
                "focus-2p",
                "focus-3p",
                "focus-4p",
                "gonnect",
                "hex-gen",
                "ingenious",
                "knightthrough",
                "margo",
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
    async fn test_ttt_strategy_families_lists_family_choices() {
        let (status, body) = http_get(test_app(), "/api/games/ttt/strategy-families").await;
        assert_eq!(status, HttpStatusCode::OK);
        let body = body_json(&body);
        let family_choices = body["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == "family")
            .expect("tuner info has a family parameter")["choices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(family_choices.contains(&"ucb1"));
        assert!(family_choices.contains(&"random"));
        assert!(family_choices.contains(&"negamax"));
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
