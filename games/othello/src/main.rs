use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_core::bitboard::BitBoard;
use game_othello::{Move, Othello, Player, State};
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

/// The parsed `easy`/`medium`/`strong` preset table -- loaded at runtime
/// from `games/othello/presets.json` (or the file named by
/// `OTHELLO_PRESETS_PATH`), falling back to the compiled-in defaults only
/// if that path is missing (see `PresetTable::load`'s doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("OTHELLO_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load(include_str!("../presets.json"), Some(&presets_path))
            .expect("games/othello/presets.json must parse")
    })
}

#[derive(Serialize, Deserialize)]
struct WireState {
    black: u64,
    white: u64,
    turn: String,
    last_pass: bool,
}
#[derive(Serialize)]
struct GameView {
    black: u64,
    white: u64,
    turn: String,
    last_pass: bool,
    winner: Option<String>,
    terminal: bool,
}

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
        _ => panic!("invalid player"),
    }
}

fn state_to_value(s: &State) -> Value {
    serde_json::to_value(WireState {
        black: s.black.bits(),
        white: s.white.bits(),
        turn: player_name(s.turn).into(),
        last_pass: s.last_pass,
    })
    .expect("")
}
fn value_to_state(v: &Value) -> Result<State, HostError> {
    let w: WireState =
        serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    Ok(State {
        black: BitBoard::new(w.black),
        white: BitBoard::new(w.white),
        turn: parse_player(&w.turn),
        last_pass: w.last_pass,
        hashes: [0u64; 8],
    })
}

struct OthAdapter;
impl GameAdapter for OthAdapter {
    fn kind(&self) -> &'static str {
        "othello"
    }
    fn label(&self) -> &'static str {
        "Othello"
    }
    fn description(&self) -> &'static str {
        "Classic 8×8 Reversi/Othello — outflank your opponent's discs."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        let mut s = State {
            black: BitBoard::new(0),
            white: BitBoard::new(0),
            turn: Player::Black,
            last_pass: false,
            hashes: [0u64; 8],
        };
        s.black = BitBoard::new(0x0000000810000000);
        s.white = BitBoard::new(0x0000001008000000);
        Ok(state_to_value(&s))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !Othello::is_terminal(&s) {
            Othello::generate_actions(&s, &mut mv);
        }
        Ok(mv.into_iter().map(|m| Value::from(m.0 as u64)).collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let idx = mv
            .as_u64()
            .ok_or_else(|| HostError::bad_request("move must be a cell index"))?
            as u8;
        if idx == 64 {
            return Ok(state_to_value(&State {
                last_pass: true,
                ..s
            }));
        } // pass
        let action = Move(idx);
        if Othello::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Othello::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Othello::apply(s, &action)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let winner = Othello::winner(&s);
        serde_json::to_value(GameView {
            black: s.black.bits(),
            white: s.white.bits(),
            turn: player_name(s.turn).into(),
            last_pass: s.last_pass,
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: Othello::is_terminal(&s),
        })
        .map_err(|e| HostError::internal(e.to_string()))
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
        if Othello::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Othello>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let action = ai.choose_action(&s);
        let next = Othello::apply(s, &action);
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
        _: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let custom_spec = custom
            .map(|v| serde_json::from_value::<mcts_tune::presets::CustomStrategySpec>(v.clone()))
            .transpose()
            .map_err(|e| HostError::bad_request(format!("invalid custom strategy: {e}")))?;
        let s = value_to_state(state)?;
        if Othello::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Othello>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);
        let suggested = report
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
            suggested_move: suggested,
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
        // override -- Othello has one, so merging transposed nodes during
        // the candidate's search is safe here (see `generic_tune_eval`'s
        // doc comment).
        mcts_tune::generic_tune_eval::<Othello>(
            presets(),
            "strong",
            "games/othello/presets.json",
            true,
            PRESET_SEED,
            params,
            rounds,
            seed,
            baseline_config,
            max_iterations,
            max_time_ms,
            trace_path,
            on_game,
        )
    }
}

fn main() {
    run_cli(OthAdapter);
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
        let result = OthAdapter
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
