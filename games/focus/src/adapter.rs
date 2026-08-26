//! Generic `GameAdapter` for `Focus<P>`, speaking the JSON-line subprocess
//! protocol on stdin/stdout.
//!
//! Written once here, generic over the player count `P`, then instantiated
//! by three thin binaries under `src/bin/` (`focus_2p.rs`/`focus_3p.rs`/
//! `focus_4p.rs`) -- each player count is its own wire-protocol "kind" since
//! neither `mcts-bench`'s registry nor `ui/app/src/games.ts` has a notion of
//! a player count that varies within one kind (see `games/ingenious/src/
//! main.rs`'s doc comment, which hit the same constraint first).

use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{cell_color_at, cell_height, Focus, Move, Player, State};
use mcts::game::{Game, PlayerIndex};
use mcts_tune::presets::PresetTable;

/// Fixed seed for every `ai_move`/`analyze` search built through [`presets`]
/// -- `GameAdapter::ai_move`/`analyze` take no seed argument, so this is the
/// only seed available to `mcts_tune::presets::PresetTable::build`.
const PRESET_SEED: u64 = 0;

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/focus/presets.json` (or the file named by `FOCUS_PRESETS_PATH`),
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment). Shared by all three player counts: the preset table is just
/// generic search-strategy parameters, not board- or player-count-specific.
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("FOCUS_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path).expect("games/focus/presets.json must parse")
    })
}

// ---------------------------------------------------------------------------
// Wire format types
// ---------------------------------------------------------------------------

/// The wire shape for a position. `cells` is a `Vec` (not `State`'s own
/// `[u16; 64]`) since serde's derive only implements `Serialize`/
/// `Deserialize` for array lengths up to 32; `reserves` is a `Vec` for the
/// same reason it needs its length checked explicitly -- one wire struct
/// serves all three player counts, so its length can't be pinned to a const
/// generic the way `State<P>`'s own field is.
#[derive(Serialize, Deserialize)]
struct WireState {
    cells: Vec<u16>,
    reserves: Vec<u8>,
    turn: usize,
    hash: u64,
}

#[derive(Serialize)]
struct GameView {
    /// One entry per board cell (64, row-major), bottom-to-top piece colors;
    /// empty for both an empty cell and a notched-off invalid corner (the
    /// board's fixed 8x8-minus-corners shape is a UI-side constant, not
    /// transmitted here -- same convention as `games/ingenious`'s hex board).
    board: Vec<Vec<u8>>,
    reserves: Vec<u8>,
    current_player: usize,
    winner: Option<usize>,
    terminal: bool,
}

fn cell_stack(w: u16) -> Vec<u8> {
    if w == 0 {
        return Vec::new();
    }
    (0..cell_height(w)).map(|j| cell_color_at(w, j)).collect()
}

fn state_to_value<const P: usize>(state: &State<P>) -> Value {
    serde_json::to_value(WireState {
        cells: state.cells.to_vec(),
        reserves: state.reserves.to_vec(),
        turn: state.turn.to_index(),
        hash: state.hash,
    })
    .expect("WireState always serializes")
}

fn value_to_state<const P: usize>(v: &Value) -> Result<State<P>, HostError> {
    let wire: WireState = serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
    if wire.cells.len() != 64 {
        return Err(HostError::bad_request(format!(
            "invalid state: cells has {} entries, expected 64",
            wire.cells.len()
        )));
    }
    if wire.reserves.len() != P {
        return Err(HostError::bad_request(format!(
            "invalid state: reserves has {} entries, expected {P}",
            wire.reserves.len()
        )));
    }
    let mut cells = [0u16; 64];
    cells.copy_from_slice(&wire.cells);
    let mut reserves = [0u8; P];
    reserves.copy_from_slice(&wire.reserves);
    Ok(State {
        cells,
        reserves,
        turn: Player(wire.turn as u8),
        hash: wire.hash,
    })
}

fn action_to_value(action: &Move) -> Value {
    serde_json::to_value(action).expect("Move always serializes")
}

fn value_to_action(v: &Value) -> Result<Move, HostError> {
    serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid move: {e}")))
}

// ---------------------------------------------------------------------------
// GameAdapter implementation
// ---------------------------------------------------------------------------

pub struct FocusAdapter<const P: usize>;

impl<const P: usize> FocusAdapter<P> {
    const fn kind_str() -> &'static str {
        match P {
            2 => "focus-2p",
            3 => "focus-3p",
            4 => "focus-4p",
            _ => panic!("Focus supports 2, 3, or 4 players"),
        }
    }

    const fn label_str() -> &'static str {
        match P {
            2 => "Focus (2p)",
            3 => "Focus (3p)",
            4 => "Focus (4p)",
            _ => panic!("Focus supports 2, 3, or 4 players"),
        }
    }
}

impl<const P: usize> GameAdapter for FocusAdapter<P> {
    fn kind(&self) -> &'static str {
        Self::kind_str()
    }

    fn label(&self) -> &'static str {
        Self::label_str()
    }

    fn description(&self) -> &'static str {
        "Focus (Domination) -- slide or split a stack orthogonally exactly as many squares as \
         it is tall, merging onto whatever you land on; a stack over five high buries its \
         bottom pieces, returning your own colour to hand and capturing everyone else's. \
         Whoever still has a legal move when everyone else is stuck wins."
    }

    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&State::<P>::default()))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state::<P>(state)?;
        let mut moves = Vec::new();
        if !Focus::<P>::is_terminal(&s) {
            Focus::<P>::generate_actions(&s, &mut moves);
        }
        Ok(moves.iter().map(action_to_value).collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state::<P>(state)?;
        let action = value_to_action(mv)?;

        if Focus::<P>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Focus::<P>::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }

        Ok(state_to_value(&Focus::<P>::apply(s, &action)))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state::<P>(state)?;
        let view = GameView {
            board: s.cells.iter().map(|&w| cell_stack(w)).collect(),
            reserves: s.reserves.to_vec(),
            current_player: s.turn.to_index(),
            winner: Focus::<P>::winner(&s).map(|p| p.to_index()),
            terminal: Focus::<P>::is_terminal(&s),
        };
        serde_json::to_value(view).map_err(|e| HostError::internal(format!("serialize view: {e}")))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        presets().ai_presets()
    }

    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError> {
        let custom_spec = custom
            .map(|v| serde_json::from_value::<mcts_tune::presets::CustomStrategySpec>(v.clone()))
            .transpose()
            .map_err(|e| HostError::bad_request(format!("invalid custom strategy: {e}")))?;
        let s = value_to_state::<P>(state)?;
        if Focus::<P>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Focus<P>>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (action, search) = mcts_tune::choose_action_with_report(&mut *ai, &s, action_to_value);
        let next = Focus::<P>::apply(s, &action);
        Ok(AiMoveResult {
            mv: action_to_value(&action),
            state: state_to_value(&next),
            search: Some(search),
        })
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let custom_spec = custom
            .map(|v| serde_json::from_value::<mcts_tune::presets::CustomStrategySpec>(v.clone()))
            .transpose()
            .map_err(|e| HostError::bad_request(format!("invalid custom strategy: {e}")))?;
        let s = value_to_state::<P>(state)?;
        if Focus::<P>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Focus<P>>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (selected_action, search) =
            mcts_tune::choose_action_with_report(&mut *ai, &s, action_to_value);
        Ok(mcts_tune::legacy_analysis_with_report(
            &*ai,
            &s,
            &selected_action,
            search,
            action_to_value,
        ))
    }
}

/// Entry point shared by all three per-player-count binaries.
pub fn main<const P: usize>() {
    run_cli(FocusAdapter::<P>);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trips_for<const P: usize>() {
        let v = FocusAdapter::<P>.new_state(serde_json::json!({})).unwrap();
        assert_eq!(v["cells"].as_array().unwrap().len(), 64);
        assert_eq!(v["reserves"].as_array().unwrap().len(), P);

        let moves = FocusAdapter::<P>.legal_moves(&v).unwrap();
        assert!(!moves.is_empty());
        let next = FocusAdapter::<P>.apply(&v, &moves[0]).unwrap();
        assert_eq!(next["cells"].as_array().unwrap().len(), 64);

        let view = FocusAdapter::<P>.view(&v).unwrap();
        assert_eq!(view["current_player"], 0);
        assert_eq!(view["terminal"], false);
        assert_eq!(view["winner"], Value::Null);

        let presets = FocusAdapter::<P>.ai_presets();
        assert!(!presets.is_empty());

        let legal = FocusAdapter::<P>.legal_moves(&v).unwrap();
        let result = FocusAdapter::<P>.ai_move(&v, "easy", None).unwrap();
        assert!(legal.contains(&result.mv));
        assert_ne!(result.state, v);
    }

    #[test]
    fn new_state_legal_moves_apply_view_and_ai_move_round_trip_2p() {
        round_trips_for::<2>();
    }

    #[test]
    fn new_state_legal_moves_apply_view_and_ai_move_round_trip_3p() {
        round_trips_for::<3>();
    }

    #[test]
    fn new_state_legal_moves_apply_view_and_ai_move_round_trip_4p() {
        round_trips_for::<4>();
    }

    #[test]
    fn apply_rejects_an_illegal_move() {
        let state = FocusAdapter::<2>.new_state(serde_json::json!({})).unwrap();
        let bogus = serde_json::json!(65535u16);
        assert!(FocusAdapter::<2>.apply(&state, &bogus).is_err());
    }

    #[test]
    fn kind_and_label_are_distinct_per_player_count() {
        assert_eq!(FocusAdapter::<2>.kind(), "focus-2p");
        assert_eq!(FocusAdapter::<3>.kind(), "focus-3p");
        assert_eq!(FocusAdapter::<4>.kind(), "focus-4p");
    }
}
