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
use serde::{Deserialize, Serialize};
use tower_http::services::ServeDir;

use mcts::game::Game;
use mcts::games::druid::{Druid, HashedState, Move, Player, Size};
use mcts::strategies::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

struct AppState {
    game: Mutex<HashedState>,
}

// AI opponents, from weakest to strongest. Each preset pairs a search
// strategy with a wall-clock thinking budget -- Druid's move generation and
// terminal checks are expensive (see the header comment in
// src/games/druid.rs), so budgets are time-based rather than iteration
// counts, which keeps the UI responsive regardless of board size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AiPreset {
    Easy,
    Medium,
    Strong,
    Master,
}

impl AiPreset {
    const ALL: [AiPreset; 4] = [
        AiPreset::Easy,
        AiPreset::Medium,
        AiPreset::Strong,
        AiPreset::Master,
    ];

    fn label(self) -> &'static str {
        match self {
            AiPreset::Easy => "Easy",
            AiPreset::Medium => "Medium",
            AiPreset::Strong => "Strong",
            AiPreset::Master => "Master",
        }
    }

    fn description(self) -> &'static str {
        match self {
            AiPreset::Easy => "Plain UCB1 with random playouts, ~1s per move. Makes tactical mistakes.",
            AiPreset::Medium => "UCB1 with MAST-biased playouts, ~2s per move.",
            AiPreset::Strong => {
                "Tuned RAVE + MAST + decisive-move search, ~3s per move (SMAC3-tuned)."
            }
            AiPreset::Master => "Same search as Strong with a longer ~8s thinking budget.",
        }
    }

    fn time_budget(self) -> Duration {
        match self {
            AiPreset::Easy => Duration::from_secs(1),
            AiPreset::Medium => Duration::from_secs(2),
            AiPreset::Strong => Duration::from_secs(3),
            AiPreset::Master => Duration::from_secs(8),
        }
    }
}

#[derive(Serialize)]
struct AiPresetInfo {
    id: AiPreset,
    label: &'static str,
    description: &'static str,
}

async fn get_ai_presets() -> Json<Vec<AiPresetInfo>> {
    Json(
        AiPreset::ALL
            .iter()
            .map(|&id| AiPresetInfo {
                id,
                label: id.label(),
                description: id.description(),
            })
            .collect(),
    )
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
        size: s.size,
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

#[derive(Deserialize)]
struct NewGameRequest {
    size: Size,
}

async fn post_new(
    AxumState(app): AxumState<Arc<AppState>>,
    Json(req): Json<NewGameRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !req.size.is_supported() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported board size {}x{}: each side must be at least 3, \
                 and the board can't be so large it overflows the Zobrist hash table",
                req.size.w, req.size.h
            ),
        ));
    }

    let mut state = app.game.lock().unwrap();
    *state = HashedState::new(req.size);
    Ok(Json(serde_json::to_value(view(&state, None)).unwrap()))
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

// `Strong` reuses the tuned `rave_mast_ucd` setup from demo/druid.rs, which
// SMAC3 hyperparameter search found effective for this game specifically.
// The other presets reuse strategy types exercised there too (`Ucb1`,
// `Ucb1Mast`), just with shorter time budgets, giving a real strength
// gradient rather than only a time-budget knob.
fn build_ai(preset: AiPreset) -> Box<dyn Search<G = Druid>> {
    let budget = preset.time_budget();
    match preset {
        AiPreset::Easy => Box::new(TreeSearch::<Druid, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("ai/easy")
                .expand_threshold(1)
                .use_transpositions(true)
                .q_init(QInit::Infinity)
                .max_time(budget)
                .select(select::Ucb1::with_c(1.414)),
        )),
        AiPreset::Medium => Box::new(TreeSearch::<Druid, strategy::Ucb1Mast>::new().config(
            SearchConfig::new()
                .name("ai/medium")
                .expand_threshold(1)
                .use_transpositions(true)
                .q_init(QInit::Infinity)
                .max_time(budget)
                .select(select::Ucb1::with_c(1.625))
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
        )),
        AiPreset::Strong | AiPreset::Master => {
            Box::new(TreeSearch::<Druid, strategy::RaveMastDm>::new().config(
                SearchConfig::new()
                    .name(if preset == AiPreset::Strong {
                        "ai/strong"
                    } else {
                        "ai/master"
                    })
                    .expand_threshold(1)
                    .use_transpositions(true)
                    .q_init(QInit::Infinity)
                    .max_time(budget)
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
            ))
        }
    }
}

#[derive(Deserialize)]
struct AiMoveRequest {
    preset: AiPreset,
}

async fn post_ai_move(
    AxumState(app): AxumState<Arc<AppState>>,
    Json(req): Json<AiMoveRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let snapshot = {
        let state = app.game.lock().unwrap();
        if Druid::is_terminal(&state) {
            return Err((StatusCode::BAD_REQUEST, "game is over".into()));
        }
        state.clone()
    };

    // Run the search on a blocking thread -- it's CPU-bound for the full
    // thinking budget and would otherwise stall the async executor.
    let action = tokio::task::spawn_blocking(move || {
        let mut ai = build_ai(req.preset);
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
        .route("/api/ai_presets", get(get_ai_presets))
        .route("/api/new", post(post_new))
        .with_state(app_state)
        .fallback_service(ServeDir::new(static_dir));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878")
        .await
        .expect("failed to bind 127.0.0.1:7878");
    println!("Druid server listening on http://127.0.0.1:7878");
    axum::serve(listener, app).await.unwrap();
}
