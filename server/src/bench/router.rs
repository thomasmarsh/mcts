#![allow(unused_imports)]
use std::collections::HashMap;
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
use mcts_bench::supervised_launch::LaunchDescriptor;

use super::{
    commands::*,
    runs::*,
    traces::*,
    tuning::{
        add_tuning_session_budget, get_tuning_analysis_overview, get_tuning_session,
        get_tuning_sessions, get_tuning_trial_detail, get_tuning_trials, resume_tuning_session,
        stop_tuning_session,
    },
    types::*,
};
// Router constructor
// ---------------------------------------------------------------------------

/// Build the `/api/bench/*` sub-router, ready to be merged into the
/// server's main router.
pub fn bench_router(state: Arc<BenchState>) -> Router {
    // The launch route can run long if the child is slow to start (fork +
    // exec), but normally should complete in milliseconds.  Give it 30s
    // headroom anyway.
    let launch_timeout = TimeoutLayer::with_status_code(
        StatusCode::GATEWAY_TIMEOUT,
        std::time::Duration::from_secs(30),
    );

    let cors = CorsLayer::new()
        .allow_origin([
            "http://127.0.0.1:7878".parse::<HeaderValue>().unwrap(),
            "http://localhost:5173".parse::<HeaderValue>().unwrap(),
            "http://127.0.0.1:5173".parse::<HeaderValue>().unwrap(),
        ])
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PATCH,
            Method::PUT,
            Method::DELETE,
        ])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    Router::new()
        .route("/api/bench/tuner/kinds", get(list_tuner_kinds))
        .route("/api/bench/tuner/sessions", get(get_tuning_sessions))
        .route(
            "/api/bench/tuner/sessions/{session_id}",
            get(get_tuning_session),
        )
        .route(
            "/api/bench/tuner/sessions/{session_id}/stop",
            post(stop_tuning_session),
        )
        .route(
            "/api/bench/tuner/sessions/{session_id}/resume",
            post(resume_tuning_session).layer(launch_timeout),
        )
        .route(
            "/api/bench/tuner/sessions/{session_id}/budget",
            post(add_tuning_session_budget).layer(launch_timeout),
        )
        .route(
            "/api/bench/tuner/sessions/{session_id}/analysis",
            get(get_tuning_analysis_overview),
        )
        .route(
            "/api/bench/tuner/sessions/{session_id}/trials",
            get(get_tuning_trials),
        )
        .route(
            "/api/bench/tuner/sessions/{session_id}/trials/{trial_id}",
            get(get_tuning_trial_detail),
        )
        .route("/api/bench/runs", get(list_runs))
        .route("/api/bench/runs/{run_id}", get(get_run))
        .route("/api/bench/runs/{run_id}/log", get(get_run_log))
        .route("/api/bench/runs/{run_id}/stdout", get(get_run_stdout))
        .route("/api/bench/runs/{run_id}/trials", get(get_run_trials))
        .route("/api/bench/runs/{run_id}/games", get(get_run_games))
        .route(
            "/api/bench/runs/{run_id}/games/{game_seq}/moves",
            get(get_run_game_moves),
        )
        .route("/api/bench/runs/{run_id}/live", get(live_run_moves))
        .route("/api/bench/launch", post(launch_run).layer(launch_timeout))
        .route("/api/bench/runs/{run_id}/stop", post(stop_run))
        .route("/api/bench/runs/{run_id}", delete(delete_run))
        .layer(cors)
        .with_state(state)
}

// ---------------------------------------------------------------------------
/// `GET /api/bench/tuner/kinds`
///
/// Per-game tuner metadata (search space, baselines, eval rounds), queried
/// by spawning each of `mcts_bench`'s registered game binaries once with
/// `tune describe` (see `mcts_bench::games::describe_tuners`) rather than
/// through `server::adapter::registry()`'s live gameplay sessions -- that
/// registry only covers the games with a UI renderer, which used to leave
/// tunable-but-UI-less games (e.g. `nim`) unable to appear here even though
/// `POST /api/bench/launch` never needed a live session for them either
/// (the tuner CLI subprocess it spawns locates the game binary itself).
/// Only games that implement `tuner()` appear -- tuning support is opt-in
/// per game.
pub(crate) async fn list_tuner_kinds() -> Json<Vec<TunerGameInfo>> {
    let mut games: Vec<TunerGameInfo> = mcts_bench::games::describe_tuners()
        .into_iter()
        .map(|(kind, tuner)| TunerGameInfo {
            game: kind.to_string(),
            tuner,
        })
        .collect();
    games.sort_by(|a, b| a.game.cmp(&b.game));
    Json(games)
}
