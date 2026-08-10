//! `SimpleGameCodec` impl for Knightthrough (chess-knight-move variant of
//! Breakthrough). The blanket `GameAdapter` impl comes from
//! `SimpleAdapter<Knightthrough<8, 8>>`.

use serde::{Deserialize, Serialize};

use mcts::game::Game;
use mcts::games::bitboard::BitBoard;
use mcts::games::knightthrough::{Knightthrough, Move, Player, State};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

use crate::adapters::simple::{PresetSpec, SimpleGameCodec};

// -- Wire format types -------------------------------------------------------

/// Wire format using hex strings for bitboards. JSON numbers can't represent
/// 64-bit values above 2^53 without precision loss (JavaScript's
/// `Number.MAX_SAFE_INTEGER`), which corrupts the bitboard pattern. Hex
/// strings survive the JSON round-trip losslessly.
#[derive(Serialize, Deserialize)]
pub struct WireState {
    pub black: String,
    pub white: String,
    pub turn: String,
    pub winner: bool,
}

#[derive(Serialize)]
pub struct GameView {
    pub black: String,
    pub white: String,
    pub turn: String,
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

fn build_easy() -> Box<dyn Search<G = Knightthrough<8, 8>>> {
    Box::new(
        TreeSearch::<Knightthrough<8, 8>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("knightthrough/easy")
                .expand_threshold(1)
                .max_iterations(100)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = Knightthrough<8, 8>>> {
    Box::new(
        TreeSearch::<Knightthrough<8, 8>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("knightthrough/strong")
                .expand_threshold(0)
                .max_iterations(5000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

// -- SimpleGameCodec impl ----------------------------------------------------

impl SimpleGameCodec for Knightthrough<8, 8> {
    type WireState = WireState;
    type WireMove = [u8; 2];
    type WireView = GameView;

    const KIND: &'static str = "knightthrough";
    const LABEL: &'static str = "Knightthrough";
    const DESCRIPTION: &'static str = "Breakthrough with knight moves — pieces move in L-shapes \
         (like chess knights) rather than forward/ diagonally. First to reach \
         the opponent's back rank wins.";

    const PRESETS: &'static [PresetSpec<Self>] = &[
        PresetSpec {
            id: "easy",
            label: "Easy",
            description: "Plain UCB1 with a moderate iteration budget — plays reasonably.",
            build: build_easy,
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
            black: format!("{:016x}", state.black().bits()),
            white: format!("{:016x}", state.white().bits()),
            turn: player_name(state.turn()).into(),
            winner: state.has_winner(),
        }
    }

    fn from_wire_state(state: Self::WireState) -> Self::S {
        let parse_hex = |s: &str| u64::from_str_radix(s, 16)
            .expect("hex bitboard string");
        State::new(
            BitBoard::new(parse_hex(&state.black)),
            BitBoard::new(parse_hex(&state.white)),
            parse_player(&state.turn),
            state.winner,
        )
    }

    fn to_wire_move(mv: &Self::A) -> Self::WireMove {
        [mv.0, mv.1]
    }

    fn from_wire_move(mv: Self::WireMove) -> Self::A {
        Move(mv[0], mv[1])
    }

    fn game_view(state: &Self::S) -> Self::WireView {
        let winner = Knightthrough::<8, 8>::winner(state);
        GameView {
            black: format!("{:016x}", state.black().bits()),
            white: format!("{:016x}", state.white().bits()),
            turn: player_name(state.turn()).into(),
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: Knightthrough::<8, 8>::is_terminal(state),
        }
    }
}