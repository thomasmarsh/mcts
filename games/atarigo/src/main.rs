use game_host::{run_stdin_stdout, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_atarigo::{AtariGo, Move, Player, State};
use game_core::bitboard::BitBoard;
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

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

fn player_name(p: Player) -> &'static str { match p { Player::Black => "Black", Player::White => "White" } }
fn parse_player(name: &str) -> Player { match name { "Black" => Player::Black, "White" => Player::White, _ => panic!("invalid player") } }

fn state_to_value(s: &State<8>) -> Value {
    serde_json::to_value(WireState {
        black: format!("{:016x}", s.black().bits()),
        white: format!("{:016x}", s.white().bits()),
        turn: player_name(s.turn()).into(),
        winner: s.has_winner(),
    }).expect("")
}
fn value_to_state(v: &Value) -> Result<State<8>, HostError> {
    let w: WireState = serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    let parse_hex = |s: &str| u64::from_str_radix(s, 16).map_err(|e| HostError::bad_request(format!("invalid hex: {e}")));
    Ok(State {
        black: BitBoard::new(parse_hex(&w.black)?),
        white: BitBoard::new(parse_hex(&w.white)?),
        turn: parse_player(&w.turn),
        winner: w.winner,
    })
}

fn build_easy() -> Box<dyn Search<G = AtariGo<8>>> {
    Box::new(TreeSearch::<AtariGo<8>, strategy::Ucb1>::new().config(
        SearchConfig::new().name("atarigo/easy").expand_threshold(1).max_iterations(100).q_init(QInit::Infinity)))
}
fn build_strong() -> Box<dyn Search<G = AtariGo<8>>> {
    Box::new(TreeSearch::<AtariGo<8>, strategy::Ucb1>::new().config(
        SearchConfig::new().name("atarigo/strong").expand_threshold(0).max_iterations(5000).use_mcts_solver(true).q_init(QInit::Loss)))
}

const PRESETS: &[PresetEntry] = &[
    PresetEntry { id: "easy", build: build_easy },
    PresetEntry { id: "strong", build: build_strong },
];
struct PresetEntry { id: &'static str, build: fn() -> Box<dyn Search<G = AtariGo<8>>> }

struct AtarigoAdapter;

impl GameAdapter for AtarigoAdapter {
    fn kind(&self) -> &'static str { "atarigo" }
    fn label(&self) -> &'static str { "AtariGo" }
    fn description(&self) -> &'static str { "A Go-like game where capturing a single stone wins." }
    fn default_config(&self) -> Value { serde_json::json!({}) }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&State::default()))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !AtariGo::<8>::is_terminal(&s) { AtariGo::<8>::generate_actions(&s, &mut mv); }
        Ok(mv.into_iter().map(|m| serde_json::to_value(m).unwrap()).collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let m: Move = serde_json::from_value(mv.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
        if AtariGo::<8>::is_terminal(&s) { return Err(HostError::bad_request("game is over")); }
        let mut legal = Vec::new();
        AtariGo::<8>::generate_actions(&s, &mut legal);
        if !legal.contains(&m) { return Err(HostError::bad_request("illegal move")); }
        Ok(state_to_value(&AtariGo::<8>::apply(s, &m)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let winner = AtariGo::<8>::winner(&s);
        serde_json::to_value(GameView {
            black: format!("{:016x}", s.black().bits()),
            white: format!("{:016x}", s.white().bits()),
            turn: player_name(s.turn()).into(),
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: AtariGo::<8>::is_terminal(&s),
        }).map_err(|e| HostError::internal(e.to_string()))
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        PRESETS.iter().map(|p| AiPresetInfo { id: p.id.into(), label: p.id.into(), description: "".into() }).collect()
    }
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS.iter().find(|p| p.id == preset).ok_or_else(|| HostError::not_found("unknown preset"))?;
        if AtariGo::<8>::is_terminal(&s) { return Err(HostError::bad_request("game is over")); }
        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = AtariGo::<8>::apply(s, &action);
        Ok(AiMoveResult { mv: serde_json::to_value(&action).unwrap(), state: state_to_value(&next) })
    }
    fn analyze(&self, state: &Value, preset: &str, _: Option<u64>) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS.iter().find(|p| p.id == preset).ok_or_else(|| HostError::not_found("unknown preset"))?;
        if AtariGo::<8>::is_terminal(&s) { return Err(HostError::bad_request("game is over")); }
        let mut ai = (spec.build)();
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);
        let suggested = report.principal_variation.first().map(|a| serde_json::to_value(a).unwrap());
        Ok(Analysis {
            actions: report.actions.into_iter().map(|a| AnalysisAction {
                action: serde_json::to_value(&a.action).unwrap(),
                visits: a.visits, mean_value: a.mean_value, is_proven: a.is_proven,
            }).collect(),
            principal_variation: report.principal_variation.into_iter().map(|a| serde_json::to_value(a).unwrap()).collect(),
            total_visits: report.total_visits, suggested_move: suggested,
        })
    }
}

fn main() { run_stdin_stdout(AtarigoAdapter); }