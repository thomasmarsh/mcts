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
use game_knightthrough::{Knightthrough, Move, Player, State};
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

/// The parsed `easy`/`strong` preset table --
/// `games/knightthrough/presets.json`'s embedded defaults, or an
/// operator-supplied override file named by `KNIGHTTHROUGH_PRESETS_PATH`
/// (see `PresetTable::load`'s doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let override_path = env::var("KNIGHTTHROUGH_PRESETS_PATH")
            .ok()
            .map(PathBuf::from);
        PresetTable::load(include_str!("../presets.json"), override_path.as_deref())
            .expect("games/knightthrough/presets.json must parse")
    })
}

#[derive(Serialize, Deserialize)]
struct WireState {
    black: String,
    white: String,
    turn: String,
    winner: bool,
}
#[derive(Serialize)]
struct GameView {
    black: String,
    white: String,
    turn: String,
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

fn state_to_value(s: &State<8, 8>) -> Value {
    serde_json::to_value(WireState {
        black: format!("{:016x}", s.black().bits()),
        white: format!("{:016x}", s.white().bits()),
        turn: player_name(s.turn()).into(),
        winner: s.has_winner(),
    })
    .expect("")
}
fn value_to_state(v: &Value) -> Result<State<8, 8>, HostError> {
    let w: WireState =
        serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    let parse_hex = |s: &str| {
        u64::from_str_radix(s, 16).map_err(|e| HostError::bad_request(format!("invalid hex: {e}")))
    };
    Ok(State::new(
        BitBoard::new(parse_hex(&w.black)?),
        BitBoard::new(parse_hex(&w.white)?),
        parse_player(&w.turn),
        w.winner,
    ))
}

struct KtAdapter;
impl GameAdapter for KtAdapter {
    fn kind(&self) -> &'static str {
        "knightthrough"
    }
    fn label(&self) -> &'static str {
        "Knightthrough"
    }
    fn description(&self) -> &'static str {
        "Breakthrough with knight moves — pieces move in L-shapes rather than forward/diagonally."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&State::new(
            BitBoard::new(0xffff000000000000),
            BitBoard::new(0x000000000000ffff),
            Player::Black,
            false,
        )))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !Knightthrough::<8, 8>::is_terminal(&s) {
            Knightthrough::<8, 8>::generate_actions(&s, &mut mv);
        }
        Ok(mv
            .into_iter()
            .map(|m| Value::Array(vec![Value::from(m.0 as u64), Value::from(m.1 as u64)]))
            .collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let arr = mv
            .as_array()
            .ok_or_else(|| HostError::bad_request("move must be [from, to]"))?;
        let action = Move(
            arr[0].as_u64().unwrap() as u8,
            arr[1].as_u64().unwrap() as u8,
        );
        if Knightthrough::<8, 8>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Knightthrough::<8, 8>::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Knightthrough::<8, 8>::apply(s, &action)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let winner = Knightthrough::<8, 8>::winner(&s);
        serde_json::to_value(GameView {
            black: format!("{:016x}", s.black().bits()),
            white: format!("{:016x}", s.white().bits()),
            turn: player_name(s.turn()).into(),
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: Knightthrough::<8, 8>::is_terminal(&s),
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
        if Knightthrough::<8, 8>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Knightthrough<8, 8>>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let action = ai.choose_action(&s);
        let next = Knightthrough::<8, 8>::apply(s, &action);
        Ok(AiMoveResult {
            mv: Value::Array(vec![
                Value::from(action.0 as u64),
                Value::from(action.1 as u64),
            ]),
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
        if Knightthrough::<8, 8>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Knightthrough<8, 8>>(
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
            .map(|a| Value::Array(vec![Value::from(a.0 as u64), Value::from(a.1 as u64)]));
        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: Value::Array(vec![
                        Value::from(a.action.0 as u64),
                        Value::from(a.action.1 as u64),
                    ]),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| Value::Array(vec![Value::from(a.0 as u64), Value::from(a.1 as u64)]))
                .collect(),
            total_visits: report.total_visits,
            suggested_move: suggested,
        })
    }

    fn tuner(&self) -> Option<TunerInfo> {
        Some(TunerInfo {
            game_config: self.default_config(),
            ..mcts_tune::strategy_tuner_info(&["strong"], TUNE_EVAL_ROUNDS)
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
        // Knightthrough's `Game::zobrist_hash` is the default constant `0`,
        // so transpositions must stay off -- see `generic_tune_eval`'s doc
        // comment.
        mcts_tune::generic_tune_eval::<Knightthrough<8, 8>>(
            presets(),
            "strong",
            "games/knightthrough/presets.json",
            false,
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
    run_cli(KtAdapter);
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
        let result = KtAdapter
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
