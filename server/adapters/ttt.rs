//! `SimpleGameCodec` impl for tic-tac-toe, reducing what was a 275-line
//! hand-written `GameAdapter` to ~65 lines of genuinely per-game code.
//! The blanket `GameAdapter` impl comes from `SimpleAdapter<TicTacToe>`.

use serde::{Deserialize, Serialize};

use mcts::game::Game;
use mcts::games::ttt::{HashedPosition, Move, Piece, Position, TicTacToe};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

use crate::adapters::simple::{PresetSpec, SimpleGameCodec};

// -- Wire format types -------------------------------------------------------

/// The wire shape for a position: a plain 9-cell array plus whose turn it
/// is, deliberately not `Position`'s internal packed-`u32` encoding (an
/// implementation detail of the engine's move generator/hasher, not
/// something a client should need to understand).
#[derive(Serialize, Deserialize)]
pub struct WireState {
    pub turn: Piece,
    pub cells: [Option<Piece>; 9],
}

#[derive(Serialize)]
pub struct GameView {
    pub turn: Piece,
    pub cells: [Option<Piece>; 9],
    pub winner: Option<Piece>,
    pub terminal: bool,
}

// -- Helpers -----------------------------------------------------------------

fn cells_of(position: &Position) -> [Option<Piece>; 9] {
    std::array::from_fn(|i| position.get(i))
}

// -- Preset builders ---------------------------------------------------------

fn build_easy() -> Box<dyn Search<G = TicTacToe>> {
    Box::new(
        TreeSearch::<TicTacToe, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("ttt/easy")
                .expand_threshold(1)
                .max_iterations(30)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = TicTacToe>> {
    Box::new(
        TreeSearch::<TicTacToe, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("ttt/strong")
                .expand_threshold(0)
                .max_iterations(5000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

// -- SimpleGameCodec impl ----------------------------------------------------

impl SimpleGameCodec for TicTacToe {
    type WireState = WireState;
    type WireMove = u8;
    type WireView = GameView;

    const KIND: &'static str = "ttt";
    const LABEL: &'static str = "Tic-Tac-Toe";
    const DESCRIPTION: &'static str = "Classic 3x3 tic-tac-toe -- the trivial second game exercising the game-agnostic \
         UI contract.";

    const PRESETS: &'static [PresetSpec<Self>] = &[
        PresetSpec {
            id: "easy",
            label: "Easy",
            description: "Plain UCB1 with a shallow iteration budget -- makes mistakes.",
            build: build_easy,
        },
        PresetSpec {
            id: "strong",
            label: "Strong",
            description: "UCB1 with MCTS-Solver, deep enough to solve the tree from most positions -- \
                 plays perfectly (win or draw).",
            build: build_strong,
        },
    ];

    fn to_wire_state(state: &Self::S) -> Self::WireState {
        WireState {
            turn: state.position.turn,
            cells: cells_of(&state.position),
        }
    }

    fn from_wire_state(state: Self::WireState) -> Self::S {
        let mut position = Position::new();
        position.turn = state.turn;
        for (i, cell) in state.cells.into_iter().enumerate() {
            if let Some(piece) = cell {
                position.set(i, piece);
            }
        }
        HashedPosition::from_position(position)
    }

    fn to_wire_move(mv: &Self::A) -> Self::WireMove {
        mv.0
    }

    fn from_wire_move(mv: Self::WireMove) -> Self::A {
        Move(mv)
    }

    fn game_view(state: &Self::S) -> Self::WireView {
        GameView {
            turn: state.position.turn,
            cells: cells_of(&state.position),
            winner: state.position.winner(),
            terminal: TicTacToe::is_terminal(state),
        }
    }
}