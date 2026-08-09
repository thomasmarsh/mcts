//! `SimpleGameCodec` impl for Othello. The blanket `GameAdapter` impl comes
//! from `SimpleAdapter<Othello>`.

use serde::{Deserialize, Serialize};

use mcts::game::Game;
use mcts::games::bitboard::BitBoard;
use mcts::games::othello::{Move, Othello, Player, State};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

use crate::adapters::simple::{PresetSpec, SimpleGameCodec};

// -- Wire format types -------------------------------------------------------

/// Compact wire format: raw u64 bitboards for black and white discs, plus
/// turn / last-pass flag. The UI decodes bitboards into a display grid.
#[derive(Serialize, Deserialize)]
pub struct WireState {
    pub black: u64,
    pub white: u64,
    pub turn: String,
    pub last_pass: bool,
}

#[derive(Serialize)]
pub struct GameView {
    pub black: u64,
    pub white: u64,
    pub turn: String,
    pub last_pass: bool,
    pub winner: Option<String>,
    pub terminal: bool,
}

// -- Helpers -----------------------------------------------------------------

fn player_name(p: Player) -> &'static str {
    match p {
        Player::Black => "Black",
        Player::White => "White",
    }
}

fn parse_player(name: &str) -> Player {
    match name {
        "Black" => Player::Black,
        "White" => Player::White,
        other => panic!("invalid player from deserialized wire state: {other:?}"),
    }
}

// -- Preset builders ---------------------------------------------------------

fn build_easy() -> Box<dyn Search<G = Othello>> {
    Box::new(
        TreeSearch::<Othello, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("othello/easy")
                .expand_threshold(1)
                .max_iterations(30)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_medium() -> Box<dyn Search<G = Othello>> {
    Box::new(
        TreeSearch::<Othello, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("othello/medium")
                .expand_threshold(1)
                .max_iterations(1000)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = Othello>> {
    Box::new(
        TreeSearch::<Othello, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("othello/strong")
                .expand_threshold(0)
                .max_iterations(10000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

// -- SimpleGameCodec impl ----------------------------------------------------

impl SimpleGameCodec for Othello {
    type WireState = WireState;
    type WireMove = u8;
    type WireView = GameView;

    const KIND: &'static str = "othello";
    const LABEL: &'static str = "Othello";
    const DESCRIPTION: &'static str = "Classic 8×8 Reversi/Othello — outflank your opponent's discs by \
         sandwiching them between your own. Pass when you have no legal moves; \
         double-pass ends the game.";

    const PRESETS: &'static [PresetSpec<Self>] = &[
        PresetSpec {
            id: "easy",
            label: "Easy",
            description: "Plain UCB1 with a shallow iteration budget — makes obvious mistakes.",
            build: build_easy,
        },
        PresetSpec {
            id: "medium",
            label: "Medium",
            description: "UCB1 with a moderate iteration budget — plays competently.",
            build: build_medium,
        },
        PresetSpec {
            id: "strong",
            label: "Strong",
            description: "UCB1 with MCTS-Solver and deep iterations — plays strongly.",
            build: build_strong,
        },
    ];

    fn to_wire_state(state: &Self::S) -> Self::WireState {
        WireState {
            black: state.black.bits(),
            white: state.white.bits(),
            turn: player_name(state.turn).into(),
            last_pass: state.last_pass,
        }
    }

    fn from_wire_state(state: Self::WireState) -> Self::S {
        State {
            black: BitBoard::new(state.black),
            white: BitBoard::new(state.white),
            turn: parse_player(&state.turn),
            last_pass: state.last_pass,
            hashes: [0u64; 8],
        }
    }

    fn to_wire_move(mv: &Self::A) -> Self::WireMove {
        mv.0
    }

    fn from_wire_move(mv: Self::WireMove) -> Self::A {
        Move(mv)
    }

    fn game_view(state: &Self::S) -> Self::WireView {
        let winner = Othello::winner(state);
        GameView {
            black: state.black.bits(),
            white: state.white.bits(),
            turn: player_name(state.turn).into(),
            last_pass: state.last_pass,
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: Othello::is_terminal(state),
        }
    }
}