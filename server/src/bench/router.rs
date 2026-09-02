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
    runs::*,
    traces::*,
    tuner_api,
    tuner_runs::{
        extend_tuner_run, get_tuner_run, get_tuner_run_log, launch_tuner_run,
        list_tuner_objectives, list_tuner_runs, stop_tuner_run,
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
        .route("/api/bench/tuner/objectives", get(list_tuner_objectives))
        .route(
            "/api/bench/tuner/runs",
            post(launch_tuner_run)
                .get(list_tuner_runs)
                .layer(launch_timeout),
        )
        .route("/api/bench/tuner/runs/{run_id}", get(get_tuner_run))
        .route("/api/bench/tuner/runs/{run_id}/stop", post(stop_tuner_run))
        .route("/api/bench/tuner/runs/{run_id}/log", get(get_tuner_run_log))
        .route(
            "/api/bench/tuner/runs/{run_id}/extend",
            post(extend_tuner_run).layer(launch_timeout),
        )
        // Read-only projection API. The operational journal routes above
        // answer "is it running / stop it"; these answer "what did it find",
        // served entirely from the SQLite read model.
        .route(
            "/api/bench/tuner/projection/refresh",
            post(tuner_api::refresh).layer(launch_timeout),
        )
        .route(
            "/api/bench/tuner/projection/runs",
            get(tuner_api::list_runs),
        )
        .route(
            "/api/bench/tuner/projection/runs/{run_id}",
            get(tuner_api::run_detail),
        )
        .route(
            "/api/bench/tuner/projection/runs/{run_id}/cohorts",
            get(tuner_api::cohorts),
        )
        .route(
            "/api/bench/tuner/projection/runs/{run_id}/candidates",
            get(tuner_api::candidates),
        )
        .route(
            "/api/bench/tuner/projection/runs/{run_id}/candidates/{candidate_id}",
            get(tuner_api::candidate),
        )
        .route(
            "/api/bench/tuner/projection/runs/{run_id}/pairs",
            get(tuner_api::pairs),
        )
        .route(
            "/api/bench/tuner/projection/runs/{run_id}/pairs/{pair_id}/games",
            get(tuner_api::pair_games),
        )
        .route(
            "/api/bench/tuner/projection/runs/{run_id}/validation",
            get(tuner_api::validation),
        )
        .route(
            "/api/bench/tuner/projection/runs/{run_id}/report",
            get(tuner_api::report),
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
/// launching a tuner run never needed a live session for them either (the
/// tuner CLI subprocess it spawns locates the game binary itself).
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
