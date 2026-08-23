use crate::*;
use serde_json::Value;
use std::cell::RefCell;
use std::io::Cursor;

#[derive(Default)]
pub(super) struct ValidationCounts {
    pub(super) new_state: usize,
    pub(super) builds: usize,
    pub(super) plays: usize,
}

thread_local! {
    pub(super) static VALIDATION_COUNTS: RefCell<ValidationCounts> = RefCell::new(ValidationCounts::default());
}

/// A minimal fake adapter for testing the protocol dispatch loop.
/// Responds with just enough data to verify round-trip correctness.
pub(super) struct FakeAdapter;

impl GameAdapter for FakeAdapter {
    fn kind(&self) -> &'static str {
        "fake"
    }
    fn label(&self) -> &'static str {
        "Fake Game"
    }
    fn description(&self) -> &'static str {
        "A minimal fake adapter for testing"
    }

    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({"board": [], "turn": "X"}))
    }

    fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, HostError> {
        Ok(vec![serde_json::json!(0), serde_json::json!(1)])
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let turn = state.get("turn").and_then(|t| t.as_str()).unwrap_or("X");
        let next_turn = if turn == "X" { "O" } else { "X" };
        Ok(serde_json::json!({
            "board": [mv],
            "turn": next_turn,
        }))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({
            "terminal": false,
            "turn": state.get("turn"),
        }))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        vec![AiPresetInfo {
            id: "random".into(),
            label: "Random".into(),
            description: "Picks a random legal move".into(),
        }]
    }

    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        _custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError> {
        if preset == "random" {
            let next = self.apply(state, &serde_json::json!(0))?;
            Ok(AiMoveResult {
                mv: serde_json::json!(0),
                state: next,
                search: None,
            })
        } else {
            Err(HostError::not_found(format!("unknown preset: {preset}")))
        }
    }

    fn analyze(
        &self,
        _state: &Value,
        preset: &str,
        _custom: Option<&Value>,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        if preset != "random" {
            return Err(HostError::not_found(format!("unknown preset: {preset}")));
        }
        let mv = serde_json::json!(0);
        Ok(Analysis {
            actions: vec![AnalysisAction {
                action: mv.clone(),
                visits: 10,
                mean_value: 0.5,
                is_proven: false,
            }],
            principal_variation: vec![mv],
            total_visits: 10,
            suggested_move: Some(serde_json::json!(0)),
            search: None,
        })
    }
}

// -----------------------------------------------------------------------
// Helper: send JSON lines into run_host and collect responses
// -----------------------------------------------------------------------

fn send_requests(lines: &[&str]) -> Vec<String> {
    let input = Cursor::new(lines.join("\n"));
    let mut output = Cursor::new(Vec::new());
    run_host(input, &mut output, FakeAdapter);
    let raw = String::from_utf8(output.into_inner()).unwrap();
    raw.lines().map(|l| l.to_owned()).collect()
}

pub(super) fn parse_response(line: &str) -> Response {
    serde_json::from_str(line).unwrap()
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[test]
fn test_kind() {
    let lines = send_requests(&[r#"{"id":1,"method":"kind","params":{}}"#]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 1);
            assert_eq!(result, "fake");
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_label() {
    let lines = send_requests(&[r#"{"id":2,"method":"label","params":{}}"#]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 2);
            assert_eq!(result, "Fake Game");
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_description() {
    let lines = send_requests(&[r#"{"id":3,"method":"description","params":{}}"#]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 3);
            assert!(result.as_str().unwrap().contains("fake"));
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_default_config() {
    let lines = send_requests(&[r#"{"id":4,"method":"default_config","params":{}}"#]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 4);
            assert_eq!(result, serde_json::json!({}));
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_new_state() {
    let lines = send_requests(&[r#"{"id":5,"method":"new","params":{"config":{}}}"#]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 5);
            assert_eq!(result.get("turn").and_then(|t| t.as_str()), Some("X"));
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_legal_moves() {
    let lines = send_requests(&[
        r#"{"id":6,"method":"new","params":{"config":{}}}"#,
        r#"{"id":7,"method":"legal_moves","params":{"state":{"board":[],"turn":"X"}}}"#,
    ]);
    assert_eq!(lines.len(), 2);
    let resp = parse_response(&lines[1]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 7);
            let moves = result.as_array().unwrap();
            assert_eq!(moves.len(), 2);
            assert_eq!(moves[0], 0);
            assert_eq!(moves[1], 1);
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_apply() {
    let lines = send_requests(&[
        r#"{"id":8,"method":"new","params":{"config":{}}}"#,
        r#"{"id":9,"method":"apply","params":{"state":{"board":[],"turn":"X"},"move":0}}"#,
    ]);
    assert_eq!(lines.len(), 2);
    let resp = parse_response(&lines[1]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 9);
            assert_eq!(result.get("turn").and_then(|t| t.as_str()), Some("O"));
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_apply_missing_params() {
    let lines = send_requests(&[r#"{"id":10,"method":"apply","params":{}}"#]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Error { id, error } => {
            assert_eq!(id, 10);
            assert_eq!(error.code, 400);
            assert!(error.message.contains("missing parameter"));
        }
        _ => panic!("expected error response"),
    }
}

#[test]
fn test_view() {
    let lines = send_requests(&[
        r#"{"id":11,"method":"new","params":{"config":{}}}"#,
        r#"{"id":12,"method":"view","params":{"state":{"board":[],"turn":"X"}}}"#,
    ]);
    assert_eq!(lines.len(), 2);
    let resp = parse_response(&lines[1]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 12);
            assert_eq!(
                result.get("terminal").and_then(|t| t.as_bool()),
                Some(false)
            );
            assert_eq!(result.get("turn").and_then(|t| t.as_str()), Some("X"));
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_terminal() {
    let lines = send_requests(&[
        r#"{"id":13,"method":"terminal","params":{"state":{"board":[],"turn":"X"}}}"#,
    ]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 13);
            assert_eq!(
                result.get("terminal").and_then(|t| t.as_bool()),
                Some(false)
            );
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_ai_presets() {
    let lines = send_requests(&[r#"{"id":14,"method":"ai_presets","params":{}}"#]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 14);
            let presets = result.as_array().unwrap();
            assert_eq!(presets.len(), 1);
            assert_eq!(
                presets[0].get("id").and_then(|v| v.as_str()),
                Some("random")
            );
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_ai_move() {
    let lines = send_requests(&[
        r#"{"id":15,"method":"new","params":{"config":{}}}"#,
        r#"{"id":16,"method":"ai_move","params":{"state":{"board":[],"turn":"X"},"preset":"random"}}"#,
    ]);
    assert_eq!(lines.len(), 2);
    let resp = parse_response(&lines[1]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 16);
            assert_eq!(result.get("mv").and_then(|v| v.as_u64()), Some(0));
            assert!(result.get("state").is_some());
            assert!(result.get("search").is_none());
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_ai_move_unknown_preset() {
    let lines = send_requests(&[
        r#"{"id":17,"method":"new","params":{"config":{}}}"#,
        r#"{"id":18,"method":"ai_move","params":{"state":{"board":[],"turn":"X"},"preset":"nope"}}"#,
    ]);
    assert_eq!(lines.len(), 2);
    let resp = parse_response(&lines[1]);
    match resp {
        Response::Error { id, error } => {
            assert_eq!(id, 18);
            assert_eq!(error.code, 404);
        }
        _ => panic!("expected error response"),
    }
}

#[test]
fn test_analyze() {
    let lines = send_requests(&[
        r#"{"id":19,"method":"new","params":{"config":{}}}"#,
        r#"{"id":20,"method":"analyze","params":{"state":{"board":[],"turn":"X"},"preset":"random"}}"#,
    ]);
    assert_eq!(lines.len(), 2);
    let resp = parse_response(&lines[1]);
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 20);
            let actions = result.get("actions").and_then(|a| a.as_array()).unwrap();
            assert_eq!(actions.len(), 1);
            assert_eq!(actions[0].get("visits").and_then(|v| v.as_u64()), Some(10));
            assert!(result.get("search").is_none());
        }
        _ => panic!("expected success response"),
    }
}

#[test]
fn test_unknown_method() {
    let lines = send_requests(&[r#"{"id":99,"method":"nonexistent","params":{}}"#]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Error { id, error } => {
            assert_eq!(id, 99);
            assert_eq!(error.code, 404);
        }
        _ => panic!("expected error response"),
    }
}

#[test]
fn test_malformed_json() {
    let lines = send_requests(&["this is not json"]);
    assert_eq!(lines.len(), 1);
    let resp = parse_response(&lines[0]);
    match resp {
        Response::Error { id, error } => {
            assert_eq!(id, 0);
            assert_eq!(error.code, 400);
            assert!(error.message.contains("invalid request"));
        }
        _ => panic!("expected error response"),
    }
}

#[test]
fn test_multiple_requests() {
    let lines = send_requests(&[
        r#"{"id":1,"method":"kind","params":{}}"#,
        r#"{"id":2,"method":"label","params":{}}"#,
        r#"{"id":3,"method":"ai_presets","params":{}}"#,
    ]);
    assert_eq!(lines.len(), 3);
    for (i, line) in lines.iter().enumerate() {
        let resp = parse_response(line);
        match resp {
            Response::Success { id, .. } => assert_eq!(id, (i + 1) as u64),
            _ => panic!("expected success for request {}", i + 1),
        }
    }
}

#[test]
fn test_blank_lines_are_skipped() {
    let lines = send_requests(&[
        r#"{"id":1,"method":"kind","params":{}}"#,
        "", // blank line
        r#"{"id":2,"method":"label","params":{}}"#,
    ]);
    // blank line produces no output, so we get 2 responses
    assert_eq!(lines.len(), 2);
}

// -----------------------------------------------------------------------
// run_cli tests
// -----------------------------------------------------------------------
