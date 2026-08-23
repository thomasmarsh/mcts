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
use mcts_bench::experiment::ExperimentSpecV1;
use mcts_bench::identity;
use mcts_bench::launch::{self, LaunchedRun};
use mcts_bench::log::RegistryEvent;
use mcts_bench::projects_attempt::{CellRequest, ProjectsError, StartRequest};
use mcts_bench::supervised_launch::LaunchDescriptor;
use mcts_bench::tournament::wilson_interval;
use mcts_bench::StrategyInfo;

use super::{
    commands::*,
    ladder::*,
    projects::*,
    runs::*,
    traces::*,
    tuning::{get_tuning_session, get_tuning_sessions},
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
        .route("/api/bench/kinds", get(list_kinds))
        .route("/api/bench/tuner/kinds", get(list_tuner_kinds))
        .route("/api/bench/tuner/sessions", get(get_tuning_sessions))
        .route(
            "/api/bench/tuner/sessions/{session_id}",
            get(get_tuning_session),
        )
        .route(
            "/api/bench/projects",
            get(list_projects).post(create_project),
        )
        .route(
            "/api/bench/projects/{project_id}",
            get(get_project).patch(update_project),
        )
        .route(
            "/api/bench/projects/{project_id}/experiments",
            get(list_experiments).post(create_experiment),
        )
        .route(
            "/api/bench/experiments/{experiment_id}",
            get(get_experiment).put(update_experiment),
        )
        .route(
            "/api/bench/experiments/{experiment_id}/runs",
            post(launch_experiment).layer(launch_timeout),
        )
        .route("/api/bench/runs", get(list_runs))
        .route("/api/bench/runs/{run_id}", get(get_run))
        .route("/api/bench/runs/{run_id}/log", get(get_run_log))
        .route("/api/bench/runs/{run_id}/stdout", get(get_run_stdout))
        .route("/api/bench/runs/{run_id}/trials", get(get_run_trials))
        .route("/api/bench/runs/{run_id}/chain", get(get_run_chain))
        .route("/api/bench/runs/{run_id}/cells", get(get_run_cells))
        .route("/api/bench/runs/{run_id}/games", get(get_run_games))
        .route(
            "/api/bench/runs/{run_id}/games/{game_seq}/moves",
            get(get_run_game_moves),
        )
        .route("/api/bench/runs/{run_id}/live", get(live_run_moves))
        .route("/api/bench/leaderboard", get(get_leaderboard))
        .route("/api/bench/launch", post(launch_run).layer(launch_timeout))
        .route("/api/bench/runs/{run_id}/stop", post(stop_run))
        .route("/api/bench/runs/{run_id}", delete(delete_run))
        .route(
            "/api/bench/runs/{run_id}/resume",
            post(resume_run).layer(launch_timeout),
        )
        .route(
            "/api/bench/runs/{run_id}/advance-baseline",
            post(advance_baseline).layer(launch_timeout),
        )
        .layer(cors)
        .with_state(state)
}

// ---------------------------------------------------------------------------
pub(crate) async fn list_kinds() -> Json<Vec<BenchKindInfo>> {
    let game_registry = mcts_bench::registry();

    let mut games: Vec<BenchGameInfo> = game_registry
        .iter()
        .map(|(game_kind, bg)| BenchGameInfo {
            game: game_kind.to_string(),
            strategies: bg.strategies(),
        })
        .collect();
    games.sort_by(|a, b| a.game.cmp(&b.game));

    let kinds = vec![
        BenchKindInfo {
            kind: "round_robin".to_string(),
            label: "Round Robin".to_string(),
            description: "Every strategy plays every other strategy an equal number of times, both as first and second player.  Results are streamed as match_result JSONL lines, aggregated into a win-rate leaderboard with Wilson confidence intervals."
                .to_string(),
            games,
        },
        BenchKindInfo {
            kind: "tuner".to_string(),
            label: "Tuner Tuning".to_string(),
            description: "Runs a tuner hyperparameter-optimization sweep over a game's tunable strategy search space, playing rounds of a params-built candidate against one or more baseline instances per trial.  Results are streamed as trial JSONL lines.  See GET /api/bench/tuner/kinds for per-game tuner metadata (search space, baselines, eval rounds) instead of a strategies list."
                .to_string(),
            games: vec![],
        },
    ];

    Json(kinds)
}

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
