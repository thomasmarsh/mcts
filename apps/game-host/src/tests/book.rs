use super::cli::{run_cli_capture, run_cli_capture_with};
use super::support::parse_response;
use crate::*;
use serde_json::Value;

struct BookableFakeAdapter;

impl GameAdapter for BookableFakeAdapter {
    fn kind(&self) -> &'static str {
        "bookable-fake"
    }
    fn label(&self) -> &'static str {
        "Bookable Fake Game"
    }
    fn description(&self) -> &'static str {
        "A fake adapter that supports book generation, for testing `book` subcommands"
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({}))
    }
    fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, HostError> {
        Ok(vec![])
    }
    fn apply(&self, state: &Value, _mv: &Value) -> Result<Value, HostError> {
        Ok(state.clone())
    }
    fn view(&self, _state: &Value) -> Result<Value, HostError> {
        Ok(serde_json::json!({"terminal": true}))
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
        Err(HostError::not_found("not implemented in test fake"))
    }
    fn analyze(
        &self,
        _state: &Value,
        _preset: &str,
        _custom: Option<&Value>,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        Err(HostError::not_found("not implemented in test fake"))
    }

    fn book(&self) -> Option<BookInfo> {
        Some(BookInfo {
            id: "test".into(),
            default_rounds: 20,
            game_config: serde_json::json!({}),
            game_config_schema: Default::default(),
        })
    }

    fn book_build(
        &self,
        rounds: u32,
        seed: Option<u64>,
        game_config: Option<Value>,
    ) -> Result<Value, HostError> {
        Ok(serde_json::json!({
            "rounds": rounds,
            "seed": seed,
            "game_config": game_config,
        }))
    }
}

#[test]
fn test_run_cli_book_describe_unsupported_when_book_none() {
    let (out, code) = run_cli_capture(&["book", "describe"], "");
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_book_describe_prints_book_info() {
    let (out, code) = run_cli_capture_with(BookableFakeAdapter, &["book", "describe"], "");
    assert_eq!(code, 0);
    let info: BookInfo = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(info.id, "test");
    assert_eq!(info.default_rounds, 20);
}

#[test]
fn test_run_cli_book_build_prints_result_verbatim() {
    let (out, code) = run_cli_capture_with(
        BookableFakeAdapter,
        &["book", "build", "--rounds", "5", "--seed", "7"],
        "",
    );
    assert_eq!(code, 0);
    let result: Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(result["rounds"], 5);
    assert_eq!(result["seed"], 7);
}

#[test]
fn test_run_cli_book_build_missing_rounds_errors() {
    let (out, code) = run_cli_capture_with(BookableFakeAdapter, &["book", "build"], "");
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_book_build_unsupported_by_default() {
    let (out, code) = run_cli_capture(&["book", "build", "--rounds", "5"], "");
    assert_eq!(code, 1);
    assert!(out.is_empty());
}

#[test]
fn test_run_cli_book_with_no_further_args_falls_back_to_stdin_stdout_loop() {
    let (out, code) = run_cli_capture(&["book"], "{\"id\":11,\"method\":\"kind\",\"params\":{}}\n");
    assert_eq!(code, 0);
    let resp = parse_response(out.lines().next().unwrap());
    match resp {
        Response::Success { id, result } => {
            assert_eq!(id, 11);
            assert_eq!(result, "fake");
        }
        _ => panic!("expected success response"),
    }
}
