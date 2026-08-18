use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError};
use serde_json::Value;

struct ShibumiAdapter;

impl GameAdapter for ShibumiAdapter {
    fn kind(&self) -> &'static str {
        "shibumi"
    }
    fn label(&self) -> &'static str {
        "Shibumi"
    }
    fn description(&self) -> &'static str {
        "A shibumi stacking game utility — not a playable Game trait."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({"shibumi": "not a real game"}))
    }
    fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, HostError> {
        Ok(vec![])
    }
    fn apply(&self, _state: &Value, _mv: &Value) -> Result<Value, HostError> {
        Err(HostError::bad_request("shibumi is not a playable game"))
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
    run_cli(ShibumiAdapter);
}
