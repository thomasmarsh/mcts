use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError};
use serde_json::Value;

struct NullAdapter;

impl GameAdapter for NullAdapter {
    fn kind(&self) -> &'static str {
        "null"
    }
    fn label(&self) -> &'static str {
        "Null"
    }
    fn description(&self) -> &'static str {
        "A trivial game with no moves — always terminal"
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({}))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        // Always terminal; no legal moves.
        let _ = state;
        Ok(vec![])
    }
    fn apply(&self, _state: &Value, _mv: &Value) -> Result<Value, HostError> {
        Err(HostError::bad_request("game is over"))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        Ok(state.clone())
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        vec![]
    }
    fn ai_move(
        &self,
        _state: &Value,
        _preset: &str,
        _custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError> {
        Err(HostError::not_found("no ai presets"))
    }
    fn analyze(
        &self,
        _state: &Value,
        _preset: &str,
        _custom: Option<&Value>,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        Err(HostError::not_found("no ai presets"))
    }
}

fn main() {
    run_cli(NullAdapter);
}
