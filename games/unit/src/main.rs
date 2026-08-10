use game_host::{run_stdin_stdout, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError};
use serde_json::Value;

use game_unit::{Player, Unit, UnitGame};
use mcts::game::Game;

struct UnitAdapter;

impl GameAdapter for UnitAdapter {
    fn kind(&self) -> &'static str { "unit" }
    fn label(&self) -> &'static str { "Unit" }
    fn description(&self) -> &'static str { "A trivial game with one move" }
    fn default_config(&self) -> Value { serde_json::json!({}) }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({"done": false}))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let done = state["done"].as_bool().unwrap_or(true);
        if done { Ok(vec![]) } else { Ok(vec![Value::Null]) }
    }
    fn apply(&self, state: &Value, _mv: &Value) -> Result<Value, HostError> {
        let done = state["done"].as_bool().unwrap_or(true);
        if done { return Err(HostError::bad_request("game is over")); }
        Ok(serde_json::json!({"done": true}))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        Ok(state.clone())
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> { vec![] }
    fn ai_move(&self, _state: &Value, _preset: &str) -> Result<AiMoveResult, HostError> {
        Err(HostError::not_found("no ai presets"))
    }
    fn analyze(&self, _state: &Value, _preset: &str, _budget_ms: Option<u64>) -> Result<Analysis, HostError> {
        Err(HostError::not_found("no ai presets"))
    }
}

fn main() { run_stdin_stdout(UnitAdapter); }