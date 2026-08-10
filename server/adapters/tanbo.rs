//! `SimpleGameCodec` impl for Tanbo (9×9).  The blanket `GameAdapter` impl
//! comes from `SimpleAdapter<Tanbo<9>>`.

use serde::{Deserialize, Serialize};

use mcts::game::Game;
use mcts::games::tanbo::{Move, Player, State, Tanbo};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

use crate::adapters::simple::{PresetSpec, SimpleGameCodec};

// -- Wire format types -------------------------------------------------------

/// Wire shape: a flat cell array plus whose turn it is.  The engine's
/// internal `State<9>` is `Serialize` but not `Deserialize`, and its serde
/// representation of `Option<Player>` is an ugly `{"Black": null}` /
/// `{"White": null}` / `null` — this explicit shape avoids that.
#[derive(Serialize, Deserialize)]
pub struct WireState {
    pub cells: Vec<Option<String>>,
    pub turn: String,
}

#[derive(Serialize)]
pub struct GameView {
    pub cells: Vec<Option<String>>,
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

fn build_easy() -> Box<dyn Search<G = Tanbo<9>>> {
    Box::new(
        TreeSearch::<Tanbo<9>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tanbo/easy")
                .expand_threshold(1)
                .max_iterations(100)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = Tanbo<9>>> {
    Box::new(
        TreeSearch::<Tanbo<9>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tanbo/strong")
                .expand_threshold(0)
                .max_iterations(5000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

// -- SimpleGameCodec impl ----------------------------------------------------

impl SimpleGameCodec for Tanbo<9> {
    type WireState = WireState;
    type WireMove = u16;
    type WireView = GameView;

    const KIND: &'static str = "tanbo";
    const LABEL: &'static str = "Tanbo";
    const DESCRIPTION: &'static str = "A 9×9 abstract strategy game where players place stones adjacent to \
         exactly one of their existing stones, then remove every bounded group \
         (of either colour) — a group with no legal extension vanishes immediately.";

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
            turn: player_name(state.turn).into(),
            cells: state
                .board
                .iter()
                .map(|c| c.map(|p| player_name(p).to_string()))
                .collect(),
        }
    }

    fn from_wire_state(state: Self::WireState) -> Self::S {
        let mut board = Vec::with_capacity(81);
        for cell in &state.cells {
            board.push(match cell.as_deref() {
                Some("Black") => Some(Player::Black),
                Some("White") => Some(Player::White),
                _ => None,
            });
        }
        State {
            board,
            turn: parse_player(&state.turn),
            winner: None,
        }
    }

    fn to_wire_move(mv: &Self::A) -> Self::WireMove {
        mv.0
    }

    fn from_wire_move(mv: Self::WireMove) -> Self::A {
        Move(mv)
    }

    fn game_view(state: &Self::S) -> Self::WireView {
        let winner = Tanbo::<9>::winner(state);
        GameView {
            turn: player_name(state.turn).into(),
            cells: state
                .board
                .iter()
                .map(|c| c.map(|p| player_name(p).to_string()))
                .collect(),
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: Tanbo::<9>::is_terminal(state),
        }
    }
}