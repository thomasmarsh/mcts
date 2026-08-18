//! Standalone Tic-Tac-Toe game binary that speaks the JSON-line subprocess
//! protocol on stdin/stdout.
//!
//! Built by `cargo build -p game-ttt` and used by the server/bench crates
//! via `game_host::SubprocessAdapter`.

use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
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

/// The parsed `easy`/`strong` preset table -- `games/ttt/presets.json`'s
/// embedded defaults, or an operator-supplied override file named by
/// `TTT_PRESETS_PATH` (see `PresetTable::load`'s doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let override_path = env::var("TTT_PRESETS_PATH").ok().map(PathBuf::from);
        PresetTable::load(include_str!("../presets.json"), override_path.as_deref())
            .expect("games/ttt/presets.json must parse")
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
        let action = ai.choose_action(&s);
        let next = TicTacToe::apply(s, &action);

        Ok(AiMoveResult {
            mv: Value::from(action.0 as u64),
            state: state_to_value(&next),
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

    fn tuner(&self) -> Option<TunerInfo> {
        Some(TunerInfo {
            game_config: self.default_config(),
            ..mcts_tune::strategy_tuner_info_with_mcgs(&["strong"], TUNE_EVAL_ROUNDS, true)
        })
    }

    fn tune_eval(
        &self,
        params: Value,
        rounds: u32,
        seed: Option<u64>,
        _baseline: Option<String>,
        baseline_config: Option<Value>,
        _game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        on_game: &mut dyn FnMut(game_host::ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        // `use_transpositions: true` requires a real `Game::zobrist_hash`
        // override -- TicTacToe has one, so merging transposed nodes during
        // the candidate's search is safe here.
        let outcome = if let Some(cfg) = baseline_config {
            let baseline_seed = seed.unwrap_or(0);
            // This opponent is itself a `build_search`-built config, on
            // the same iteration-based footing as the candidate -- both
            // sides get the *same* budget (an operator's `max_iterations`
            // override included) so there's nothing to match asymmetrically
            // (see `SearchBudget`'s and `build_search`'s doc comments).
            let budget = mcts_tune::SearchBudget {
                max_iterations,
                max_time: max_time_ms.map(std::time::Duration::from_millis),
                ..Default::default()
            };
            // Fail fast on an invalid baseline config, before any games are
            // played -- mirrors how a bad candidate `params` is already
            // rejected during `TrialParams` deserialization inside
            // `strategy_tune_eval` itself.
            mcts_tune::build_search::<TicTacToe>(&cfg, baseline_seed, true, &budget)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                true,
                budget,
                move || {
                    mcts_tune::build_search::<TicTacToe>(&cfg, baseline_seed, true, &budget)
                        .expect("baseline_config already validated above")
                },
                Default::default(),
                trace_path.as_deref(),
                on_game,
            )?
        } else {
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                true,
                mcts_tune::SearchBudget {
                    max_iterations,
                    max_time: max_time_ms.map(std::time::Duration::from_millis),
                    ..Default::default()
                },
                move || {
                    presets()
                        .build::<TicTacToe>("strong", PRESET_SEED)
                        .expect("games/ttt/presets.json's \"strong\" preset must build")
                },
                Default::default(),
                trace_path.as_deref(),
                on_game,
            )?
        };
        Ok(serde_json::json!({
            "cost": outcome.cost,
            "wins": outcome.wins,
            "losses": outcome.losses,
            "draws": outcome.draws,
        }))
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

    #[ignore = "slow: plays real self-play games through mcts-tune at production iteration counts (seconds for small games, tens of minutes for large boards like druid) -- mcts-tune's own crate has a fast per-family unit suite covering dispatch; this only additionally proves this game's own Game impl round-trips end to end. Run explicitly with `cargo test --bins -- --ignored`."]
    #[test]
    fn tune_eval_round_trips() {
        let params = serde_json::json!({
            "family": "rave",
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
                &mut |_| Ok(()),
            )
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
