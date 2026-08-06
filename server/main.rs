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
use mcts::games::druid::{
    Druid, DruidHeuristic, DruidHeuristicWeights, HashedState, Move, Player,
    RaveDecisiveHeuristic, Size,
};
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
            AiPreset::Easy => "Plain UCB1 with random playouts and MCTS-Solver for tactical sharpness, ~1s per move.",
            AiPreset::Medium => "UCB1 with MAST-biased playouts and MCTS-Solver for tactical sharpness, ~2s per move.",
            AiPreset::Strong => {
                "Tuned RAVE + heuristic-guided + decisive-move search with MCTS-Solver for \
                 tactical sharpness, ~3s per move (SMAC3-tuned), searching one shared tree \
                 across all available CPU cores."
            }
            AiPreset::Master => {
                "Same search as Strong, parallelized the same way, with a longer ~8s \
                 thinking budget."
            }
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

// Number of threads Strong/Master search across. It was found that
// single-threaded search is the weakest mode available at every board
// size tested, and pure tree-parallel search (one shared tree, N worker
// threads) won outright at 5x5 and tied every other mode at 9x9 -- unlike
// root parallelism (N independent trees), it never lost across either tested
// size and doesn't pay N times the tree memory, so it's used here as a
// single default rather than switching configs by board size. Derived from
// the actual machine's core count rather than hardcoding the 8 cores session
// 10's benchmarks happened to run on, so this stays sensible on whatever
// hardware the server runs on.
fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// `Strong`/`Master` keep the SMAC3-tuned `select::Rave` hyperparameters from
// demo/druid.rs's `rave_mast_ucd` setup, but the playout policy is
// `DruidHeuristic`-guided rather than `Mast` -- PLAN-DRUID.md Session 6's
// grid sweep (epsilon x weights, n=30+/point) found the heuristic beats a
// uniform-playout baseline by a statistically solid margin (aggregated
// ~62% vs. chance) at epsilon=0.5 with equal (1.0/1.0/1.0) heuristic
// weights, so this replaces `strategy::RaveMastDm`'s `Mast`-based simulate
// strategy with `druid::RaveDecisiveHeuristic`'s. The other presets reuse
// strategy types exercised in demo/druid.rs too (`Ucb1`, `Ucb1Mast`), just
// with shorter time budgets, giving a real strength gradient rather than
// only a time-budget knob. Easy/Medium stay single-threaded on purpose, so
// the difficulty gradient reflects search quality, not just core count.
//
// All four presets enable `use_mcts_solver(true)`: every one gets proven-win/
// loss selection bias, and every one also gets early termination once the
// root is proven, whether the search runs single-threaded (Easy/Medium) or
// tree-parallel (Strong/Master, via `num_tree_threads`).
fn build_ai(preset: AiPreset) -> Box<dyn Search<G = Druid>> {
    let budget = preset.time_budget();
    match preset {
        AiPreset::Easy => Box::new(
            TreeSearch::<Druid, strategy::Ucb1>::new().config(
                SearchConfig::new()
                    .name("ai/easy")
                    .expand_threshold(1)
                    .use_transpositions(true)
                    .use_mcts_solver(true)
                    .q_init(QInit::Infinity)
                    .max_time(budget)
                    .select(select::Ucb1::with_c(1.414)),
            ),
        ),
        AiPreset::Medium => Box::new(
            TreeSearch::<Druid, strategy::Ucb1Mast>::new().config(
                SearchConfig::new()
                    .name("ai/medium")
                    .expand_threshold(1)
                    .use_transpositions(true)
                    .use_mcts_solver(true)
                    .q_init(QInit::Infinity)
                    .max_time(budget)
                    .select(select::Ucb1::with_c(1.625))
                    .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
            ),
        ),
        AiPreset::Strong | AiPreset::Master => Box::new(
            TreeSearch::<Druid, RaveDecisiveHeuristic>::new().config(
                SearchConfig::new()
                    .name(if preset == AiPreset::Strong {
                        "ai/strong"
                    } else {
                        "ai/master"
                    })
                    .expand_threshold(1)
                    .use_transpositions(true)
                    .use_mcts_solver(true)
                    .q_init(QInit::Infinity)
                    .max_time(budget)
                    .num_tree_threads(ai_thread_count())
                    .select(
                        select::Rave::default()
                            .ucb(select::RaveUcb::Ucb1Tuned {
                                exploration_constant: 0.2894182,
                            })
                            .threshold(204)
                            .schedule(select::RaveSchedule::MinMSE { bias: 5.2866714 }),
                    )
                    .simulate(simulate::DecisiveMove::new().inner(
                        simulate::EpsilonGreedy::default().epsilon(0.5).inner(
                            DruidHeuristic::new(DruidHeuristicWeights {
                                block_threat: 1.0,
                                defend_fork: 1.0,
                                threaten_connection: 1.0,
                            }),
                        ),
                    )),
            ),
        ),
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
    Ok(Json(
        serde_json::to_value(view(&state, Some(action))).unwrap(),
    ))
}

// Split out from `main` so tests can exercise the API surface directly
// (`tower::ServiceExt::oneshot`) without binding a real socket or serving
// static files.
fn api_router(app_state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/state", get(get_state))
        .route("/api/legal_moves", get(get_legal_moves))
        .route("/api/move", post(post_move))
        .route("/api/ai_move", post(post_ai_move))
        .route("/api/ai_presets", get(get_ai_presets))
        .route("/api/new", post(post_new))
        .with_state(app_state)
}

#[tokio::main]
async fn main() {
    let app_state = Arc::new(AppState {
        game: Mutex::new(HashedState::default()),
    });

    let static_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("server/static");

    let app = api_router(app_state).fallback_service(ServeDir::new(static_dir));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878")
        .await
        .expect("failed to bind 127.0.0.1:7878");
    println!("Druid server listening on http://127.0.0.1:7878");
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode as HttpStatusCode};
    use std::time::Instant;
    use tower::ServiceExt;

    fn test_app() -> Router {
        let app_state = Arc::new(AppState {
            game: Mutex::new(HashedState::default()),
        });
        api_router(app_state)
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

    // The AI's thinking budget (up to Master's 8s) runs on a
    // `spawn_blocking` thread with `num_tree_threads` `thread::scope` workers
    // underneath it (see `post_ai_move`'s doc comment), after releasing the
    // game `Mutex`. This confirms that pattern actually keeps the async
    // executor free -- other requests should complete quickly while an AI
    // move is in flight, not queue up behind it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_ai_move_does_not_stall_other_requests() {
        let app = test_app();

        let ai_app = app.clone();
        let ai_task = tokio::spawn(async move {
            let body = serde_json::to_vec(&serde_json::json!({"preset": "strong"})).unwrap();
            let req = Request::builder()
                .method("POST")
                .uri("/api/ai_move")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            let start = Instant::now();
            let resp = ai_app.oneshot(req).await.unwrap();
            (resp.status(), start.elapsed())
        });

        // Give the AI request time to acquire+release the game lock and get
        // into its thinking budget before probing.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let mut probe_latencies = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let (status, _) = http_get(app.clone(), "/api/state").await;
            assert_eq!(status, HttpStatusCode::OK);
            probe_latencies.push(start.elapsed());
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        let (ai_status, ai_elapsed) = ai_task.await.unwrap();
        assert_eq!(ai_status, HttpStatusCode::OK);
        // Sanity check that the AI request actually spent real time
        // thinking (Strong's budget is ~3s) rather than erroring out early
        // and making the "other requests weren't blocked" result vacuous.
        assert!(
            ai_elapsed >= Duration::from_secs(2),
            "AI move returned in {ai_elapsed:?}, expected it to use ~3s of its Strong budget"
        );

        for latency in probe_latencies {
            assert!(
                latency < Duration::from_millis(500),
                "GET /api/state took {latency:?} while an AI move (tree-parallel across \
                 {} threads) was in flight -- looks like it stalled behind the AI request \
                 instead of running concurrently",
                ai_thread_count(),
            );
        }
    }

    // Easy's old description admitted "Makes tactical mistakes" --
    // forced wins/losses a few plies deep is exactly what MCTS-Solver fixes.
    // This plays a forced-win position (variant of `druid.rs`'s
    // `test_mcts_solver_finds_forced_win`) through the *real* `build_ai`
    // preset, not a bespoke TreeSearch, and confirms it still finds the win
    // with the real 1s production budget now that solver is on.
    //
    // Construction uses only public `Game::apply` -- no private HashedState
    // field poking, so this can live in the server binary's own tests (which
    // don't have access to `#[cfg(test)] resync_caches` in the lib).
    #[test]
    fn test_easy_preset_with_solver_finds_forced_win() {
        use mcts::games::druid::{Piece, Size};
        // 3x3 board where Black owns 2 of 3 in each of two columns (x=0 and x=2).
        // Build via legal moves so caches stay valid, with White filling the
        // middle column to keep move count odd -> White to move at end.
        let size = Size { w: 3, h: 3 };
        let mut state = HashedState::new(size);
        // Sequence: B(0,0) W(1,0) B(0,1) W(1,1) B(2,0) W(1,2) B(2,1) -> White to move.
        // Indices: (x+y*w): (0,0)=0, (1,0)=1, (0,1)=3, (1,1)=4, (2,0)=2, (1,2)=7, (2,1)=5.
        // (0,2)=6 and (2,2)=8 remain empty -- Black's two winning threats.
        let moves = [
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
        ];
        let cells: [u8; 7] = [0, 1, 3, 4, 2, 7, 5];
        for (piece, cell) in moves.iter().zip(cells.iter()) {
            let mv = mcts::games::druid::Move(*piece, *cell);
            state = Druid::apply(state, &mv);
        }
        assert_eq!(
            Druid::player_to_move(&state),
            Player::White,
            "setup should end with White to move"
        );
        assert!(
            !Druid::is_terminal(&state),
            "setup should not already be terminal"
        );

        // Production Easy preset: Ucb1, 1s budget, now with solver on.
        // Exercises the actual preset plumbing rather than a test-only TreeSearch.
        let mut ai = build_ai(AiPreset::Easy);
        let white_move = ai.choose_action(&state);
        let after_white = Druid::apply(state.clone(), &white_move);

        let mut ai2 = build_ai(AiPreset::Easy);
        let black_move = ai2.choose_action(&after_white);
        let after_black = Druid::apply(after_white, &black_move);
        assert_eq!(
            Druid::winner(&after_black),
            Some(Player::Black),
            "Easy (with solver, via build_ai) should convert the forced win: \
             after White block + Black reply, Black must have won"
        );
    }

    #[test]
    fn test_all_presets_construct() {
        let _ = build_ai(AiPreset::Easy);
        let _ = build_ai(AiPreset::Medium);
        let _ = build_ai(AiPreset::Strong);
        let _ = build_ai(AiPreset::Master);
    }

    // Same forced-win position as `test_easy_preset_with_solver_finds_forced_win`,
    // but through the tree-parallel `Strong` preset, whose worker threads each
    // check the shared root's proven status directly rather than through a
    // per-thread-local read. Also confirms the early-termination win pays off
    // in wall-clock terms: on a 3-ply-deep forced win, proving the root should
    // take a small fraction of the full budget rather than grinding it out,
    // unlike `test_ai_move_does_not_stall_other_requests`'s undecided position
    // (which legitimately uses the whole budget).
    #[test]
    fn test_strong_preset_with_solver_finds_forced_win() {
        use mcts::games::druid::{Piece, Size};
        let size = Size { w: 3, h: 3 };
        let mut state = HashedState::new(size);
        let moves = [
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
            Piece::Sarsen,
        ];
        let cells: [u8; 7] = [0, 1, 3, 4, 2, 7, 5];
        for (piece, cell) in moves.iter().zip(cells.iter()) {
            let mv = mcts::games::druid::Move(*piece, *cell);
            state = Druid::apply(state, &mv);
        }
        assert_eq!(Druid::player_to_move(&state), Player::White);
        assert!(!Druid::is_terminal(&state));

        let start = Instant::now();
        let mut ai = build_ai(AiPreset::Strong);
        let white_move = ai.choose_action(&state);
        let white_elapsed = start.elapsed();
        let after_white = Druid::apply(state.clone(), &white_move);

        let start = Instant::now();
        let mut ai2 = build_ai(AiPreset::Strong);
        let black_move = ai2.choose_action(&after_white);
        let black_elapsed = start.elapsed();
        let after_black = Druid::apply(after_white, &black_move);

        assert_eq!(
            Druid::winner(&after_black),
            Some(Player::Black),
            "Strong (tree-parallel, with solver) should convert the forced win"
        );
        let budget = AiPreset::Strong.time_budget();
        assert!(
            white_elapsed < budget / 2 && black_elapsed < budget / 2,
            "expected the proven root to short-circuit well before the {budget:?} budget, \
             got white={white_elapsed:?} black={black_elapsed:?}"
        );
    }
}
