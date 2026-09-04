//! Standalone Tic-Tac-Toe game binary that speaks the JSON-line subprocess
//! protocol on stdin/stdout.
//!
//! Built by `cargo build -p game-ttt` and used by the server/bench crates
//! via `game_host::SubprocessAdapter`.

use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError, TunerInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_ttt::{HashedPosition, Move, Piece, Position, TicTacToe};
use mcts::game::Game;
use mcts_tune::presets::PresetTable;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

/// Fixed seed for every `ai_move`/`analyze`/fallback-baseline search built
/// through [`presets`] -- `GameAdapter::ai_move`/`analyze` take no seed
/// argument, so this is the only seed available to
/// `mcts_tune::presets::PresetTable::build`.
const PRESET_SEED: u64 = 0;

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/ttt/presets.json` (or the file named by `TTT_PRESETS_PATH`),
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("TTT_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path).expect("games/ttt/presets.json must parse")
    })
}

// ---------------------------------------------------------------------------
// Wire format types
// ---------------------------------------------------------------------------

/// The wire shape for a position: a plain 9-cell array plus whose turn it
/// is, deliberately not `Position`'s internal packed-`u32` encoding.
#[derive(Serialize, Deserialize)]
struct WireState {
    turn: Piece,
    cells: [Option<Piece>; 9],
}

#[derive(Serialize)]
struct GameView {
    turn: Piece,
    cells: [Option<Piece>; 9],
    winner: Option<Piece>,
    terminal: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cells_of(position: &Position) -> [Option<Piece>; 9] {
    std::array::from_fn(|i| position.get(i))
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
    let mut position = Position::new();
    position.turn = wire.turn;
    for (i, cell) in wire.cells.into_iter().enumerate() {
        if let Some(piece) = cell {
            position.set(i, piece);
        }
    }
    Ok(HashedPosition::from_position(position))
}

// ---------------------------------------------------------------------------
// AI presets
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GameAdapter implementation
// ---------------------------------------------------------------------------

struct TttAdapter;

impl GameAdapter for TttAdapter {
    fn kind(&self) -> &'static str {
        "ttt"
    }

    fn label(&self) -> &'static str {
        "Tic-Tac-Toe"
    }

    fn description(&self) -> &'static str {
        "Classic 3x3 tic-tac-toe -- the trivial second game exercising the game-agnostic UI contract."
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
        if !TicTacToe::is_terminal(&s) {
            TicTacToe::generate_actions(&s, &mut moves);
        }
        Ok(moves.into_iter().map(|m| Value::from(m.0 as u64)).collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let idx = mv
            .as_u64()
            .ok_or_else(|| HostError::bad_request("move must be a cell index"))?;
        let action = Move(idx as u8);

        if TicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        TicTacToe::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }

        Ok(state_to_value(&TicTacToe::apply(s, &action)))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let view = GameView {
            turn: s.position.turn,
            cells: cells_of(&s.position),
            winner: s.position.winner(),
            terminal: TicTacToe::is_terminal(&s),
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

        if TicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }

        let mut ai = mcts_tune::presets::build_strategy::<TicTacToe>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (action, search) = mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
            Value::from(action.0 as u64)
        });
        let next = TicTacToe::apply(s, &action);

        Ok(AiMoveResult {
            mv: Value::from(action.0 as u64),
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

        if TicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }

        let mut ai = mcts_tune::presets::build_strategy::<TicTacToe>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (selected_action, search) =
            mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
                Value::from(action.0 as u64)
            });
        Ok(mcts_tune::legacy_analysis_with_report(
            &*ai,
            &s,
            &selected_action,
            search,
            |action| Value::from(action.0 as u64),
        ))
    }

    fn tuner(&self) -> Option<TunerInfo> {
        let baselines = presets().ai_preset_ids();
        Some(TunerInfo {
            game_config: self.default_config(),
            ..mcts_tune::strategy_tuner_info_with_mcgs(&baselines, TUNE_EVAL_ROUNDS, true)
        })
    }

    fn tune_eval(
        &self,
        params: Value,
        rounds: u32,
        seed: Option<u64>,
        baseline: Option<String>,
        baseline_config: Option<Value>,
        _game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        trace_game_sequence_start: Option<u64>,
        on_game: &mut dyn FnMut(game_host::ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        // TicTacToe has a real `Game::zobrist_hash` override, so
        // transpositions (the `true` below) are safe -- see
        // `generic_tune_eval`'s doc comment.
        mcts_tune::generic_tune_eval::<TicTacToe>(
            presets(),
            "games/ttt/presets.json",
            true,
            PRESET_SEED,
            baseline,
            params,
            rounds,
            seed,
            baseline_config,
            max_iterations,
            max_time_ms,
            state_to_value,
            |_, action| Some(Value::from(action.0 as u64)),
            trace_path,
            trace_game_sequence_start,
            on_game,
        )
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    run_cli(TttAdapter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "slow: plays real self-play games through mcts-tune at production iteration counts (seconds for small games, tens of minutes for large boards like druid) -- mcts-tune's own crate has a fast per-variant unit suite covering dispatch; this only additionally proves this game's own Game impl round-trips end to end. Run explicitly with `cargo test --bins -- --ignored`."]
    #[test]
    fn tune_eval_round_trips() {
        let params = serde_json::json!({
            "algorithm": "mcts",
            "select": "rave",
            "simulate": "decisive_move_mast",
            "decisive_move_mode": "win_loss",
            "threshold": 700,
            "c": 0.3,
            "epsilon": 0.1,
            "q_init": "Infinity",
            "final_action": "robust_child",
            "schedule": "threshold",
            "rave": 700,
            "rave_ucb": "tuned",
        });
        let result = TttAdapter
            .tune_eval(
                params,
                1,
                Some(0),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &mut |_| Ok(()),
            )
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
