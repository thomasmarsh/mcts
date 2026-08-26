//! Standalone Ingenious game binary that speaks the JSON-line subprocess
//! protocol on stdin/stdout.
//!
//! Built by `cargo build -p game-ingenious` and used by the server/bench
//! crates via `game_host::SubprocessAdapter`. Serves the 2-player board
//! (`Ingenious2`/`State<2>`) only -- a 3-player binary needs its own
//! adapter kind (a separate `GameInfo.kind`/seat list; `ui/app/src/games.ts`
//! has no notion of a player count that varies within one kind), which is a
//! separate, later addition.

use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_ingenious::{Action, Color, Ingenious2, Phase, State, NUM_CELLS, NUM_COLORS, RACK_SIZE};
use mcts::game::{Game, PlayerIndex};
use mcts_tune::presets::PresetTable;

const NUM_PLAYERS: usize = 2;

/// Fixed seed for every `ai_move`/`analyze` search built through [`presets`]
/// -- `GameAdapter::ai_move`/`analyze` take no seed argument, so this is the
/// only seed available to `mcts_tune::presets::PresetTable::build`.
const PRESET_SEED: u64 = 0;

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/ingenious/presets.json` (or the file named by
/// `INGENIOUS_PRESETS_PATH`), read fresh from disk at every startup -- not
/// embedded via `include_str!`, so editing it never triggers a rebuild (see
/// `PresetTable::load_from_path`'s doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("INGENIOUS_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path).expect("games/ingenious/presets.json must parse")
    })
}

// ---------------------------------------------------------------------------
// Wire format types
// ---------------------------------------------------------------------------

type Rack = [Option<(Color, Color)>; RACK_SIZE];

/// Mirrors `Phase`'s two variants under `serde`'s default externally-tagged
/// unit-variant representation (a bare string) -- `Phase` itself doesn't
/// derive `Serialize`/`Deserialize` since it's an internal search-relevant
/// field, not part of the game crate's public wire contract.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WirePhase {
    Place,
    SwapDecision,
}

impl From<Phase> for WirePhase {
    fn from(p: Phase) -> Self {
        match p {
            Phase::Place => WirePhase::Place,
            Phase::SwapDecision => WirePhase::SwapDecision,
        }
    }
}

impl From<WirePhase> for Phase {
    fn from(p: WirePhase) -> Self {
        match p {
            WirePhase::Place => Phase::Place,
            WirePhase::SwapDecision => Phase::SwapDecision,
        }
    }
}

/// The wire shape for a position. `board` is a `Vec` (not `State`'s own
/// `[Option<Color>; NUM_CELLS]`) since serde's derive only implements
/// `Serialize`/`Deserialize` for array lengths up to 32 and `NUM_CELLS` is
/// 169; every other field is small enough to derive directly.
/// `board_tile_counts` can't be recovered from `board` alone (the board
/// records per-cell colors, not which pairs of cells came from the same
/// physical tile), so it has to round-trip explicitly, same as every other
/// `State` field that determines legal moves or scoring.
#[derive(Serialize, Deserialize)]
struct WireState {
    board: Vec<Option<Color>>,
    board_tile_counts: [[u8; NUM_COLORS]; NUM_COLORS],
    racks: [Rack; NUM_PLAYERS],
    score: [[u8; NUM_COLORS]; NUM_PLAYERS],
    bonus_used: [[bool; NUM_COLORS]; NUM_PLAYERS],
    has_moved: [bool; NUM_PLAYERS],
    claimed_symbols: [bool; NUM_COLORS],
    current_player: usize,
    phase: WirePhase,
    pending_bonus: u8,
    winner_immediate: Option<usize>,
    rng: u64,
}

#[derive(Serialize)]
struct GameView {
    board: Vec<Option<Color>>,
    racks: [Rack; NUM_PLAYERS],
    score: [[u8; NUM_COLORS]; NUM_PLAYERS],
    bonus_used: [[bool; NUM_COLORS]; NUM_PLAYERS],
    has_moved: [bool; NUM_PLAYERS],
    claimed_symbols: [bool; NUM_COLORS],
    current_player: usize,
    phase: WirePhase,
    pending_bonus: u8,
    winner: Option<usize>,
    terminal: bool,
}

fn state_to_value(state: &State<NUM_PLAYERS>) -> Value {
    serde_json::to_value(WireState {
        board: state.board.to_vec(),
        board_tile_counts: state.board_tile_counts,
        racks: state.racks,
        score: state.score,
        bonus_used: state.bonus_used,
        has_moved: state.has_moved,
        claimed_symbols: state.claimed_symbols,
        current_player: state.current_player,
        phase: state.phase.into(),
        pending_bonus: state.pending_bonus,
        winner_immediate: state.winner_immediate,
        rng: state.rng,
    })
    .expect("WireState always serializes")
}

fn value_to_state(v: &Value) -> Result<State<NUM_PLAYERS>, HostError> {
    let wire: WireState = serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
    if wire.board.len() != NUM_CELLS {
        return Err(HostError::bad_request(format!(
            "invalid state: board has {} entries, expected {NUM_CELLS}",
            wire.board.len()
        )));
    }
    let mut board = [None; NUM_CELLS];
    board.copy_from_slice(&wire.board);
    Ok(State {
        board,
        board_tile_counts: wire.board_tile_counts,
        racks: wire.racks,
        score: wire.score,
        bonus_used: wire.bonus_used,
        has_moved: wire.has_moved,
        claimed_symbols: wire.claimed_symbols,
        current_player: wire.current_player,
        phase: wire.phase.into(),
        pending_bonus: wire.pending_bonus,
        winner_immediate: wire.winner_immediate,
        rng: wire.rng,
    })
}

fn action_to_value(action: &Action) -> Value {
    serde_json::to_value(action).expect("Action always serializes")
}

fn value_to_action(v: &Value) -> Result<Action, HostError> {
    serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid move: {e}")))
}

// ---------------------------------------------------------------------------
// GameAdapter implementation
// ---------------------------------------------------------------------------

struct IngeniousAdapter;

impl GameAdapter for IngeniousAdapter {
    fn kind(&self) -> &'static str {
        "ingenious"
    }

    fn label(&self) -> &'static str {
        "Ingenious"
    }

    fn description(&self) -> &'static str {
        "Ingenious -- place double-ended hex tiles to build same-colour runs, scoring every \
         colour you touch; your total is your *lowest* colour, so a well-rounded board beats a \
         single long run."
    }

    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&State::<NUM_PLAYERS>::default()))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut moves = Vec::new();
        if !Ingenious2::is_terminal(&s) {
            Ingenious2::generate_actions(&s, &mut moves);
        }
        Ok(moves.iter().map(action_to_value).collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let action = value_to_action(mv)?;

        if Ingenious2::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Ingenious2::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }

        Ok(state_to_value(&Ingenious2::apply(s, &action)))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let view = GameView {
            board: s.board.to_vec(),
            racks: s.racks,
            score: s.score,
            bonus_used: s.bonus_used,
            has_moved: s.has_moved,
            claimed_symbols: s.claimed_symbols,
            current_player: s.current_player,
            phase: s.phase.into(),
            pending_bonus: s.pending_bonus,
            winner: Ingenious2::winner(&s).map(|p| p.to_index()),
            terminal: Ingenious2::is_terminal(&s),
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
        let s = value_to_state(state)?;
        if Ingenious2::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Ingenious2>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (action, search) = mcts_tune::choose_action_with_report(&mut *ai, &s, action_to_value);
        let next = Ingenious2::apply(s, &action);
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
        let s = value_to_state(state)?;
        if Ingenious2::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Ingenious2>(
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

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    run_cli(IngeniousAdapter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_round_trips_through_the_wire_format() {
        let v = IngeniousAdapter.new_state(serde_json::json!({})).unwrap();
        assert_eq!(v["board"].as_array().unwrap().len(), NUM_CELLS);
        assert_eq!(v["racks"].as_array().unwrap().len(), NUM_PLAYERS);
    }

    #[test]
    fn legal_moves_and_apply_round_trip() {
        let state = IngeniousAdapter.new_state(serde_json::json!({})).unwrap();
        let moves = IngeniousAdapter.legal_moves(&state).unwrap();
        assert!(!moves.is_empty());
        let next = IngeniousAdapter.apply(&state, &moves[0]).unwrap();
        assert_eq!(next["board"].as_array().unwrap().len(), NUM_CELLS);
    }

    #[test]
    fn apply_rejects_an_illegal_move() {
        let state = IngeniousAdapter.new_state(serde_json::json!({})).unwrap();
        let bogus = serde_json::json!("Swap");
        assert!(IngeniousAdapter.apply(&state, &bogus).is_err());
    }

    #[test]
    fn view_reports_current_player_and_non_terminal_start() {
        let state = IngeniousAdapter.new_state(serde_json::json!({})).unwrap();
        let view = IngeniousAdapter.view(&state).unwrap();
        assert_eq!(view["current_player"], 0);
        assert_eq!(view["terminal"], false);
        assert_eq!(view["winner"], Value::Null);
    }

    #[test]
    fn ai_presets_are_loaded_from_disk() {
        let presets = IngeniousAdapter.ai_presets();
        assert!(!presets.is_empty());
    }

    #[test]
    fn ai_move_returns_a_legal_move_and_advances_the_state() {
        let state = IngeniousAdapter.new_state(serde_json::json!({})).unwrap();
        let legal = IngeniousAdapter.legal_moves(&state).unwrap();
        let result = IngeniousAdapter.ai_move(&state, "easy", None).unwrap();
        assert!(legal.contains(&result.mv));
        assert_ne!(result.state, state);
    }
}
