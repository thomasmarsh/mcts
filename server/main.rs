// Local web server for playing Druid in a browser with a 3D board.
//
// Serves a static three.js frontend and a small JSON API over an in-memory,
// single-session game state. There is no auth, no persistence, and no
// concurrency story beyond a mutex -- this is meant for local hot-seat play,
// not deployment.

use std::sync::Mutex;

use axum::{
    extract::State as AxumState,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::Serialize;
use std::sync::Arc;
use tower_http::services::ServeDir;

use mcts::game::Game;
use mcts::games::druid::{Druid, HashedState, Move, Player, SIZE};

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
}

fn view(state: &HashedState) -> GameView<'_> {
    let s = state.state();
    GameView {
        size: SIZE,
        player: s.player,
        board: &s.board,
        hand_black: &s.hand_black,
        hand_white: &s.hand_white,
        winner: Druid::winner(state),
        terminal: Druid::is_terminal(state),
    }
}

async fn get_state(AxumState(app): AxumState<Arc<AppState>>) -> Json<serde_json::Value> {
    let state = app.game.lock().unwrap();
    Json(serde_json::to_value(view(&state)).unwrap())
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
    Json(serde_json::to_value(view(&state)).unwrap())
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
    Ok(Json(serde_json::to_value(view(&state)).unwrap()))
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
        .route("/api/new", post(post_new))
        .with_state(app_state)
        .fallback_service(ServeDir::new(static_dir));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878")
        .await
        .expect("failed to bind 127.0.0.1:7878");
    println!("Druid server listening on http://127.0.0.1:7878");
    axum::serve(listener, app).await.unwrap();
}
