// Local web server for playing Druid in a browser with a 3D board.
//
// Serves a static three.js frontend and a small JSON API over an in-memory,
// single-session game state. There is no auth, no persistence, and no
// concurrency story beyond a mutex -- this is meant for local hot-seat play,
// not deployment.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use axum::{
    extract::State as AxumState,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::Serialize;
use tower_http::services::ServeDir;

use mcts::game::Game;
use mcts::games::druid::{Druid, HashedState, Move, Player, SIZE};
use mcts::strategies::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

// How long the AI is allowed to think per move. Druid's move generation and
// terminal checks are expensive (see the header comment in
// src/games/druid.rs), so this is a wall-clock budget, not an iteration
// count -- keeps the UI responsive regardless of board size.
const AI_TIME_BUDGET: Duration = Duration::from_secs(3);

struct AppState {
    game: Mutex<HashedState>,
}

#[derive(Serialize)]
struct GameView<'a> {
    size: mcts::games::druid::Size,
    player: Player,
    board: &'a [mcts::games::druid::Square],
    hand_black: &'a mcts::games::druid::Hand,
    hand_white: &'a mcts::games::druid::Hand,
    winner: Option<Player>,
    terminal: bool,
    // The move that produced this state, if any -- lets the frontend replay
    // moves incrementally to reconstruct the physical stack (including gaps
    // under bridging lintels) without the server needing to track history,
    // since `Square` only stores the current top owner/height.
    last_move: Option<Move>,
}

fn view(state: &HashedState, last_move: Option<Move>) -> GameView<'_> {
    let s = state.state();
    GameView {
        size: SIZE,
        player: s.player,
        board: &s.board,
        hand_black: &s.hand_black,
        hand_white: &s.hand_white,
        winner: Druid::winner(state),
        terminal: Druid::is_terminal(state),
        last_move,
    }
}

async fn get_state(AxumState(app): AxumState<Arc<AppState>>) -> Json<serde_json::Value> {
    let state = app.game.lock().unwrap();
    Json(serde_json::to_value(view(&state, None)).unwrap())
}

async fn get_legal_moves(AxumState(app): AxumState<Arc<AppState>>) -> Json<Vec<Move>> {
    let state = app.game.lock().unwrap();
    let mut moves = Vec::new();
    if !Druid::is_terminal(&state) {
        Druid::generate_actions(&state, &mut moves);
    }
    Json(moves)
}

async fn post_new(AxumState(app): AxumState<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut state = app.game.lock().unwrap();
    *state = HashedState::default();
    Json(serde_json::to_value(view(&state, None)).unwrap())
}

async fn post_move(
    AxumState(app): AxumState<Arc<AppState>>,
    Json(mv): Json<Move>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut state = app.game.lock().unwrap();

    if Druid::is_terminal(&state) {
        return Err((StatusCode::BAD_REQUEST, "game is over".into()));
    }

    let mut legal = Vec::new();
    Druid::generate_actions(&state, &mut legal);
    if !legal.contains(&mv) {
        return Err((StatusCode::BAD_REQUEST, "illegal move".into()));
    }

    *state = Druid::apply(state.clone(), &mv);
    Ok(Json(serde_json::to_value(view(&state, Some(mv))).unwrap()))
}

// Config lifted from the tuned `rave_mast_ucd` setup in demo/druid.rs, which
// SMAC3 hyperparameter search found effective for this game specifically.
fn build_ai() -> TreeSearch<Druid, strategy::RaveMastDm> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("mcts[rave]+mast+ucd")
            .expand_threshold(1)
            .use_transpositions(true)
            .q_init(QInit::Infinity)
            .max_time(AI_TIME_BUDGET)
            .select(
                select::Rave::default()
                    .ucb(select::RaveUcb::Ucb1Tuned {
                        exploration_constant: 0.2894182,
                    })
                    .threshold(204)
                    .schedule(select::RaveSchedule::MinMSE { bias: 5.2866714 }),
            )
            .simulate(
                simulate::DecisiveMove::new()
                    .inner(simulate::EpsilonGreedy::with_epsilon(0.7775134)),
            ),
    )
}

async fn post_ai_move(
    AxumState(app): AxumState<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let snapshot = {
        let state = app.game.lock().unwrap();
        if Druid::is_terminal(&state) {
            return Err((StatusCode::BAD_REQUEST, "game is over".into()));
        }
        state.clone()
    };

    // Run the search on a blocking thread -- it's CPU-bound for the full
    // AI_TIME_BUDGET and would otherwise stall the async executor.
    let action = tokio::task::spawn_blocking(move || {
        let mut ai = build_ai();
        ai.choose_action(&snapshot)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut state = app.game.lock().unwrap();
    let mut legal = Vec::new();
    Druid::generate_actions(&state, &mut legal);
    if !legal.contains(&action) {
        // The board changed (e.g. a human move landed) while the AI was
        // thinking on a now-stale snapshot -- drop the move rather than
        // apply something no longer legal.
        return Err((
            StatusCode::CONFLICT,
            "board changed while AI was thinking".into(),
        ));
    }
    *state = Druid::apply(state.clone(), &action);
    Ok(Json(serde_json::to_value(view(&state, Some(action))).unwrap()))
}

#[tokio::main]
async fn main() {
    let app_state = Arc::new(AppState {
        game: Mutex::new(HashedState::default()),
    });

    let static_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("server/static");

    let app = Router::new()
        .route("/api/state", get(get_state))
        .route("/api/legal_moves", get(get_legal_moves))
        .route("/api/move", post(post_move))
        .route("/api/ai_move", post(post_ai_move))
        .route("/api/new", post(post_new))
        .with_state(app_state)
        .fallback_service(ServeDir::new(static_dir));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878")
        .await
        .expect("failed to bind 127.0.0.1:7878");
    println!("Druid server listening on http://127.0.0.1:7878");
    axum::serve(listener, app).await.unwrap();
}
