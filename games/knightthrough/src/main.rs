use game_host::{run_stdin_stdout, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use mcts::game::Game;
use mcts::bitboard::BitBoard;
use game_knightthrough::{Knightthrough, Move, Player, State};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

#[derive(Serialize, Deserialize)]
struct WireState { black: String, white: String, turn: String, winner: bool }
#[derive(Serialize)]
struct GameView { black: String, white: String, turn: String, winner: Option<String>, terminal: bool }

fn player_name(p: Player) -> &'static str { match p { Player::Black => "Black", Player::White => "White" } }
fn parse_player(name: &str) -> Player { match name { "Black" => Player::Black, "White" => Player::White, _ => panic!("invalid player") } }

fn state_to_value(s: &State<8, 8>) -> Value {
    serde_json::to_value(WireState { black: format!("{:016x}", s.black().bits()), white: format!("{:016x}", s.white().bits()), turn: player_name(s.turn()).into(), winner: s.has_winner() }).expect("")
}
fn value_to_state(v: &Value) -> Result<State<8, 8>, HostError> {
    let w: WireState = serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    let parse_hex = |s: &str| u64::from_str_radix(s, 16).map_err(|e| HostError::bad_request(format!("invalid hex: {e}")));
    Ok(State::new(BitBoard::new(parse_hex(&w.black)?), BitBoard::new(parse_hex(&w.white)?), parse_player(&w.turn), w.winner))
}

fn build_easy() -> Box<dyn Search<G = Knightthrough<8, 8>>> {
    Box::new(TreeSearch::<Knightthrough<8, 8>, strategy::Ucb1>::new().config(
        SearchConfig::new().name("knightthrough/easy").expand_threshold(1).max_iterations(100).q_init(QInit::Infinity)))
}
fn build_strong() -> Box<dyn Search<G = Knightthrough<8, 8>>> {
    Box::new(TreeSearch::<Knightthrough<8, 8>, strategy::Ucb1>::new().config(
        SearchConfig::new().name("knightthrough/strong").expand_threshold(0).max_iterations(5000).use_mcts_solver(true).q_init(QInit::Loss)))
}

const PRESETS: &[PresetEntry] = &[
    PresetEntry { id: "easy", label: "Easy", description: "Plain UCB1 with moderate budget.", build: build_easy },
    PresetEntry { id: "strong", label: "Strong", description: "UCB1 with MCTS-Solver, deep iterations.", build: build_strong },
];
struct PresetEntry { id: &'static str, label: &'static str, description: &'static str, build: fn() -> Box<dyn Search<G = Knightthrough<8, 8>>> }

struct KtAdapter;
impl GameAdapter for KtAdapter {
    fn kind(&self) -> &'static str { "knightthrough" }
    fn label(&self) -> &'static str { "Knightthrough" }
    fn description(&self) -> &'static str { "Breakthrough with knight moves — pieces move in L-shapes rather than forward/diagonally." }
    fn default_config(&self) -> Value { serde_json::json!({}) }
    fn new_state(&self, _: Value) -> Result<Value, HostError> { Ok(state_to_value(&State::new(BitBoard::new(0x000000000000ffff), BitBoard::new(0xffff000000000000), Player::Black, false))) }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?; let mut mv = Vec::new();
        if !Knightthrough::<8, 8>::is_terminal(&s) { Knightthrough::<8, 8>::generate_actions(&s, &mut mv); }
        Ok(mv.into_iter().map(|m| Value::Array(vec![Value::from(m.0 as u64), Value::from(m.1 as u64)])).collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let arr = mv.as_array().ok_or_else(|| HostError::bad_request("move must be [from, to]"))?;
        let action = Move(arr[0].as_u64().unwrap() as u8, arr[1].as_u64().unwrap() as u8);
        if Knightthrough::<8, 8>::is_terminal(&s) { return Err(HostError::bad_request("game is over")); }
        let mut legal = Vec::new(); Knightthrough::<8, 8>::generate_actions(&s, &mut legal);
        if !legal.contains(&action) { return Err(HostError::bad_request("illegal move")); }
        Ok(state_to_value(&Knightthrough::<8, 8>::apply(s, &action)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?; let winner = Knightthrough::<8, 8>::winner(&s);
        serde_json::to_value(GameView { black: format!("{:016x}", s.black().bits()), white: format!("{:016x}", s.white().bits()), turn: player_name(s.turn()).into(), winner: winner.map(|p| player_name(p).to_string()), terminal: Knightthrough::<8, 8>::is_terminal(&s) }).map_err(|e| HostError::internal(e.to_string()))
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> { PRESETS.iter().map(|p| AiPresetInfo { id: p.id.into(), label: p.label.into(), description: p.description.into() }).collect() }
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?; let spec = PRESETS.iter().find(|p| p.id == preset).ok_or_else(|| HostError::not_found("unknown preset"))?;
        if Knightthrough::<8, 8>::is_terminal(&s) { return Err(HostError::bad_request("game is over")); }
        let mut ai = (spec.build)(); let action = ai.choose_action(&s); let next = Knightthrough::<8, 8>::apply(s, &action);
        Ok(AiMoveResult { mv: Value::Array(vec![Value::from(action.0 as u64), Value::from(action.1 as u64)]), state: state_to_value(&next) })
    }
    fn analyze(&self, state: &Value, preset: &str, _: Option<u64>) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?; let spec = PRESETS.iter().find(|p| p.id == preset).ok_or_else(|| HostError::not_found("unknown preset"))?;
        if Knightthrough::<8, 8>::is_terminal(&s) { return Err(HostError::bad_request("game is over")); }
        let mut ai = (spec.build)(); let _ = ai.choose_action(&s); let report = ai.root_report(&s);
        let suggested = report.principal_variation.first().map(|a| Value::Array(vec![Value::from(a.0 as u64), Value::from(a.1 as u64)]));
        Ok(Analysis { actions: report.actions.into_iter().map(|a| AnalysisAction { action: Value::Array(vec![Value::from(a.action.0 as u64), Value::from(a.action.1 as u64)]), visits: a.visits, mean_value: a.mean_value, is_proven: a.is_proven }).collect(), principal_variation: report.principal_variation.into_iter().map(|a| Value::Array(vec![Value::from(a.0 as u64), Value::from(a.1 as u64)])).collect(), total_visits: report.total_visits, suggested_move: suggested })
    }
}

fn main() { run_stdin_stdout(KtAdapter); }