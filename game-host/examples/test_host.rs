//! A minimal game host binary used to test `SubprocessAdapter`.
//!
//! Implements a simple tic-tac-toe-like game that speaks the JSON-line
//! subprocess protocol on stdin/stdout.  Built automatically by `cargo test`
//! and spawned by the `SubprocessAdapter` tests in `subprocess.rs`.

use game_host::{
    run_stdin_stdout, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
};
use serde_json::Value;

struct TestHost;

impl GameAdapter for TestHost {
    fn kind(&self) -> &'static str {
        "test"
    }

    fn label(&self) -> &'static str {
        "Test Game"
    }

    fn description(&self) -> &'static str {
        "Minimal game host for SubprocessAdapter tests"
    }

    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        let cells: Vec<Value> = (0..9).map(|_| Value::Null).collect();
        Ok(serde_json::json!({
            "cells": cells,
            "turn": "X",
        }))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let cells = state["cells"].as_array().ok_or_else(|| {
            HostError::bad_request("state missing cells")
        })?;
        let moves: Vec<Value> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_null())
            .map(|(i, _)| Value::from(i as u64))
            .collect();
        Ok(moves)
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let idx = mv.as_u64().ok_or_else(|| {
            HostError::bad_request("move must be a cell index")
        })? as usize;

        let mut cells: Vec<Value> = state["cells"]
            .as_array()
            .ok_or_else(|| HostError::bad_request("state missing cells"))?
            .clone();

        if idx >= cells.len() {
            return Err(HostError::bad_request(format!(
                "cell index {idx} out of bounds"
            )));
        }
        if !cells[idx].is_null() {
            return Err(HostError::bad_request(format!("cell {idx} already taken")));
        }

        let turn = state["turn"].as_str().unwrap_or("X");
        cells[idx] = Value::String(turn.to_owned());
        let next_turn = if turn == "X" { "O" } else { "X" };

        Ok(serde_json::json!({
            "cells": cells,
            "turn": next_turn,
        }))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({
            "terminal": false,
            "turn": state["turn"],
        }))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        vec![
            AiPresetInfo {
                id: "easy".into(),
                label: "Easy".into(),
                description: "Picks the first legal move".into(),
            },
            AiPresetInfo {
                id: "strong".into(),
                label: "Strong".into(),
                description: "Picks the last legal move".into(),
            },
        ]
    }

    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let moves = self.legal_moves(state)?;
        if moves.is_empty() {
            return Err(HostError::bad_request("no legal moves"));
        }
        let mv = match preset {
            "easy" => moves[0].clone(),
            "strong" => moves.last().unwrap().clone(),
            other => return Err(HostError::not_found(format!("unknown preset: {other}"))),
        };
        let new_state = self.apply(state, &mv)?;
        Ok(AiMoveResult {
            mv,
            state: new_state,
        })
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let moves = self.legal_moves(state)?;
        if moves.is_empty() {
            return Err(HostError::bad_request("no legal moves"));
        }
        let suggested = match preset {
            "easy" => Some(moves[0].clone()),
            "strong" => Some(moves.last().unwrap().clone()),
            other => return Err(HostError::not_found(format!("unknown preset: {other}"))),
        };

        let actions: Vec<AnalysisAction> = moves
            .iter()
            .enumerate()
            .map(|(i, m)| AnalysisAction {
                action: m.clone(),
                visits: (moves.len() - i) as u32,
                mean_value: if i == 0 { 0.7 } else { 0.3 },
                is_proven: false,
            })
            .collect();

        Ok(Analysis {
            total_visits: moves.len() as u32,
            actions,
            principal_variation: suggested.clone().into_iter().collect(),
            suggested_move: suggested,
        })
    }
}

fn main() {
    run_stdin_stdout(TestHost);
}