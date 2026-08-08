//! `SimpleGameCodec` impl for traffic lights, reducing what was a 310-line
//! hand-written `GameAdapter` to ~100 lines of genuinely per-game code.
//! The blanket `GameAdapter` impl comes from
//! `SimpleAdapter<TrafficLights>`.

use serde::{Deserialize, Serialize};

use mcts::game::Game;
use mcts::games::traffic_lights::{
    HashedPosition, Move, Piece, Player, Position, TrafficLights,
};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

use crate::adapters::simple::{PresetSpec, SimpleGameCodec};

// -- Wire format types -------------------------------------------------------

/// The wire shape for a position: a plain 9-element array of cell colors
/// plus whose turn it is ("A" or "B").
#[derive(Serialize, Deserialize)]
pub struct WireState {
    pub turn: String,
    pub cells: [Option<String>; 9],
}

#[derive(Serialize)]
pub struct GameView {
    pub turn: String,
    pub cells: [Option<String>; 9],
    pub winner: Option<String>,
    pub terminal: bool,
}

// -- Helpers -----------------------------------------------------------------

fn cell_name(piece: Option<Piece>) -> Option<String> {
    match piece {
        Some(Piece::R) => Some("R".into()),
        Some(Piece::Y) => Some("Y".into()),
        Some(Piece::G) => Some("G".into()),
        None => None,
    }
}

fn player_name(p: Player) -> &'static str {
    match p {
        Player::First => "A",
        Player::Second => "B",
    }
}

fn parse_player(name: &str) -> Player {
    match name {
        "A" => Player::First,
        "B" => Player::Second,
        other => panic!("invalid player from deserialized wire state: {other:?}"),
    }
}

fn parse_cell(name: &str) -> Piece {
    match name {
        "R" => Piece::R,
        "Y" => Piece::Y,
        "G" => Piece::G,
        other => panic!("invalid cell piece from deserialized wire state: {other:?}"),
    }
}

fn cells_of(position: &Position) -> [Option<String>; 9] {
    std::array::from_fn(|i| cell_name(position.get(i)))
}

// -- Preset builders ---------------------------------------------------------

fn build_easy() -> Box<dyn Search<G = TrafficLights>> {
    Box::new(
        TreeSearch::<TrafficLights, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tl/easy")
                .expand_threshold(1)
                .max_iterations(30)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = TrafficLights>> {
    Box::new(
        TreeSearch::<TrafficLights, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tl/strong")
                .expand_threshold(0)
                .max_iterations(10000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

// -- SimpleGameCodec impl ----------------------------------------------------

impl SimpleGameCodec for TrafficLights {
    type WireState = WireState;
    type WireMove = u8;
    type WireView = GameView;

    const KIND: &'static str = "traffic-lights";
    const LABEL: &'static str = "Traffic Lights";
    const DESCRIPTION: &'static str = "A 3×3 game where each cell cycles through Red → Yellow → Green. \
         Make three of the same colour in a row to win.";

    const PRESETS: &'static [PresetSpec<Self>] = &[
        PresetSpec {
            id: "easy",
            label: "Easy",
            description: "Plain UCB1 with a shallow iteration budget -- plays somewhat randomly.",
            build: build_easy,
        },
        PresetSpec {
            id: "strong",
            label: "Strong",
            description: "UCB1 with MCTS-Solver, deep enough to solve most positions -- plays near-perfectly.",
            build: build_strong,
        },
    ];

    fn to_wire_state(state: &Self::S) -> Self::WireState {
        WireState {
            turn: player_name(state.position.turn).into(),
            cells: cells_of(&state.position),
        }
    }

    fn from_wire_state(state: Self::WireState) -> Self::S {
        let turn = parse_player(&state.turn);
        let mut board = 0u32;
        for (i, cell) in state.cells.into_iter().enumerate() {
            if let Some(name) = cell {
                let piece = parse_cell(&name);
                // Board stores 0=empty, 1=R, 2=Y, 3=G, while
                // `Piece` discriminants are R=0, Y=1, G=2 — add 1 to
                // convert from discriminant to board-bit encoding.
                let bits = (piece as u32) + 1;
                board |= bits << (i * 2);
            }
        }
        let mut position = Position {
            turn,
            winner: false,
            board,
        };
        position.winner = position.has_winner();
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
            turn: player_name(state.position.turn).into(),
            cells: cells_of(&state.position),
            winner: if state.position.winner {
                Some(player_name(state.position.turn).into())
            } else {
                None
            },
            terminal: TrafficLights::is_terminal(state),
        }
    }
}