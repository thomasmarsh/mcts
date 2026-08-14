//! Standalone Hex game binary that speaks the JSON-line subprocess protocol
//! on stdin/stdout.
//!
//! Built by `cargo build -p game-hex-gen` and used by the server/bench
//! crates via `game_host::SubprocessAdapter`.

use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_hex_gen::{HashedPosition, Hex, Move, Player, Position};
use mcts::game::{Game, PlayerIndex};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

const SIDE: usize = 11;

// ---------------------------------------------------------------------------
// Wire format types
// ---------------------------------------------------------------------------

/// The wire shape for a position: a plain `SIDE * SIDE`-cell array (row-major,
/// `row * SIDE + col`; a `Vec` rather than a fixed-size array since serde's
/// derive only implements `Serialize`/`Deserialize` for array lengths up to
/// 32) plus whose turn it is, deliberately not `Position`'s internal
/// per-player bitboard encoding.
#[derive(Serialize, Deserialize)]
struct WireState {
    turn: Player,
    cells: Vec<Option<Player>>,
}

#[derive(Serialize)]
struct GameView {
    turn: Player,
    cells: Vec<Option<Player>>,
    winner: Option<Player>,
    terminal: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cells_of(position: &Position) -> Vec<Option<Player>> {
    (0..SIDE * SIDE)
        .map(|i| {
            if position.occupied[0].get(i) {
                Some(Player::P0)
            } else if position.occupied[1].get(i) {
                Some(Player::P1)
            } else {
                None
            }
        })
        .collect()
}

fn state_to_value(state: &HashedPosition) -> Value {
    serde_json::to_value(WireState {
        turn: state.position.turn,
        cells: cells_of(&state.position),
    })
    .expect("WireState always serializes")
}

fn value_to_state(v: &Value) -> Result<HashedPosition, HostError> {
    let wire: WireState = serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
    // `cells` is a `Vec`, not a fixed-size array (see `WireState`'s doc comment), so serde can't
    // reject a wrong-length one on its own the way it would for a `[T; N]` field.
    if wire.cells.len() != SIDE * SIDE {
        return Err(HostError::bad_request(format!(
            "invalid state: cells has {} entries, expected {}",
            wire.cells.len(),
            SIDE * SIDE
        )));
    }
    let mut state = HashedPosition::new();
    for (i, cell) in wire.cells.into_iter().enumerate() {
        if let Some(player) = cell {
            state.position.occupied[player.to_index()].set(i);
        }
    }
    state.position.turn = wire.turn;
    Ok(state)
}

// ---------------------------------------------------------------------------
// AI presets
// ---------------------------------------------------------------------------

fn build_easy() -> Box<dyn Search<G = Hex>> {
    Box::new(
        TreeSearch::<Hex, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("hex-gen/easy")
                .expand_threshold(1)
                .max_iterations(30)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = Hex>> {
    Box::new(
        TreeSearch::<Hex, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("hex-gen/strong")
                .expand_threshold(0)
                .max_iterations(3000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

struct PresetEntry {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    build: fn() -> Box<dyn Search<G = Hex>>,
}

const PRESETS: &[PresetEntry] = &[
    PresetEntry {
        id: "easy",
        label: "Easy",
        description: "Plain UCB1 with a shallow iteration budget -- makes mistakes.",
        build: build_easy,
    },
    PresetEntry {
        id: "strong",
        label: "Strong",
        description: "UCB1 with MCTS-Solver and a deeper iteration budget -- meaningfully \
             stronger than Easy, though an 11x11 board is far too large for MCTS-Solver to prove \
             (let alone solve) from the opening.",
        build: build_strong,
    },
];

fn find_preset(id: &str) -> Result<&'static PresetEntry, HostError> {
    PRESETS
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| HostError::not_found(format!("unknown preset: {id}")))
}

// ---------------------------------------------------------------------------
// GameAdapter implementation
// ---------------------------------------------------------------------------

struct HexGenAdapter;

impl GameAdapter for HexGenAdapter {
    fn kind(&self) -> &'static str {
        "hex-gen"
    }

    fn label(&self) -> &'static str {
        "Hex"
    }

    fn description(&self) -> &'static str {
        "Regulation 11x11 Hex -- connect your two opposite edges. Generated by gdl's \
         Core-IR-to-Rust codegen (see gdl/ROADMAP.md phase 6), the first hexagonal-board game \
         wired into the UI."
    }

    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&HashedPosition::new()))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut moves = Vec::new();
        if !Hex::is_terminal(&s) {
            Hex::generate_actions(&s, &mut moves);
        }
        Ok(moves.into_iter().map(|m| Value::from(m.0 as u64)).collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let idx = mv
            .as_u64()
            .ok_or_else(|| HostError::bad_request("move must be a cell index"))?;
        let action = Move(idx as u8);

        if Hex::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Hex::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }

        Ok(state_to_value(&Hex::apply(s, &action)))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let view = GameView {
            turn: s.position.turn,
            cells: cells_of(&s.position),
            winner: s.position.winner(),
            terminal: Hex::is_terminal(&s),
        };
        serde_json::to_value(view).map_err(|e| HostError::internal(format!("serialize view: {e}")))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        PRESETS
            .iter()
            .map(|p| AiPresetInfo {
                id: p.id.to_string(),
                label: p.label.to_string(),
                description: p.description.to_string(),
            })
            .collect()
    }

    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?;
        let spec = find_preset(preset)?;

        if Hex::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }

        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = Hex::apply(s, &action);

        Ok(AiMoveResult {
            mv: Value::from(action.0 as u64),
            state: state_to_value(&next),
        })
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        let spec = find_preset(preset)?;

        if Hex::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }

        let mut ai = (spec.build)();
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);

        let suggested_move = report
            .principal_variation
            .first()
            .map(|a| Value::from(a.0 as u64));

        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: Value::from(a.action.0 as u64),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| Value::from(a.0 as u64))
                .collect(),
            total_visits: report.total_visits,
            suggested_move,
        })
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    run_cli(HexGenAdapter);
}
