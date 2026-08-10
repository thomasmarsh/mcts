//! Protocol helper for game subprocess binaries.
//!
//! Each game kind (druid, ttt, othello, …) builds a standalone binary that
//! speaks the JSON-line subprocess protocol over stdin/stdout using the
//! types and `run_host` function in this crate.
//!
//! The server/bench crates also depend on this crate for the `GameAdapter`
//! trait and the request/response types used by the `SubprocessAdapter`
//! (Step 3 of the workspace migration).

pub mod build_info;
pub mod subprocess;

use serde_json::Value;
use std::fmt;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

// ---------------------------------------------------------------------------
// Wire protocol types
// ---------------------------------------------------------------------------

/// One request read from stdin: a single JSON line.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Request {
    /// Unique request identifier, echoed back in the response.
    pub id: u64,
    /// Method name — maps to a `GameAdapter` method.
    pub method: String,
    /// Method-specific parameters.
    pub params: Value,
}

/// One response written to stdout: a single JSON line.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Response {
    /// Successful method call.
    Success {
        id: u64,
        result: Value,
    },
    /// Failed method call.
    Error {
        id: u64,
        error: ErrorBody,
    },
}

/// Structured error body within an error response.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ErrorBody {
    /// HTTP-style status code (400, 404, 500, …).
    pub code: u16,
    /// Human-readable error description.
    pub message: String,
}

// ---------------------------------------------------------------------------
// HostError
// ---------------------------------------------------------------------------

/// A simple, HTTP-style error type used by the `GameAdapter` trait methods.
///
/// Carries an integer code (matching HTTP status conventions) and a
/// human-readable message.  The `run_host` function converts these into
/// `Response::Error` when a method fails.  No external HTTP framework
/// dependency — the server crate wraps this in its own `AdapterError` if
/// axum integration is needed.
#[derive(Debug)]
pub struct HostError {
    pub code: u16,
    pub message: String,
}

impl HostError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self { code: 400, message: message.into() }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self { code: 404, message: message.into() }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self { code: 500, message: message.into() }
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for HostError {}

// ---------------------------------------------------------------------------
// Response types (mirror `server/adapters/` shapes)
// ---------------------------------------------------------------------------

/// Information about one AI preset exposed by a game.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AiPresetInfo {
    pub id: String,
    pub label: String,
    pub description: String,
}

/// The result of a completed `ai_move`: the chosen move and the resulting
/// state, so the caller can apply both without a second round-trip.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AiMoveResult {
    pub mv: Value,
    pub state: Value,
}

/// One candidate root action returned from `analyze`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AnalysisAction {
    pub action: Value,
    pub visits: u32,
    pub mean_value: f64,
    pub is_proven: bool,
}

/// Full analysis returned from `analyze`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Analysis {
    pub actions: Vec<AnalysisAction>,
    pub principal_variation: Vec<Value>,
    pub total_visits: u32,
    pub suggested_move: Option<Value>,
}

// ---------------------------------------------------------------------------
// GameAdapter trait
// ---------------------------------------------------------------------------

/// Type-erased, per-game-kind adapter over `mcts::game::Game` +
/// `mcts::strategies::Search`.
///
/// Every method is stateless: state flows in as a JSON `Value` and back out
/// as another.  Concrete adapters deserialize `Value` arguments into real
/// game types, call through to `Game`/`Search`, and re-serialize the result.
/// This is the same shape as `server/adapters/`'s `GameAdapter` trait but
/// with a simpler error type (no axum dependency).
pub trait GameAdapter: Send + Sync {
    fn kind(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn description(&self) -> &'static str;

    /// A default/example config value for `new_state`.  Also serves as a
    /// config schema hint for generic new-game forms.
    fn default_config(&self) -> Value;

    fn new_state(&self, config: Value) -> Result<Value, HostError>;
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError>;
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError>;
    fn view(&self, state: &Value) -> Result<Value, HostError>;

    fn ai_presets(&self) -> Vec<AiPresetInfo>;
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError>;
    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError>;
}

// ---------------------------------------------------------------------------
// run_host
// ---------------------------------------------------------------------------

/// Read JSON-line requests from `reader`, dispatch each to `adapter`, write
/// JSON-line responses to `writer`.
///
/// Terminates when `reader` reaches EOF (stdin closed or pipe broken).
/// Errors on individual lines (malformed JSON, missing params, adapter
/// failures) produce error responses and continue — a single bad request
/// never kills the host.
pub fn run_host<R: Read, W: Write, A: GameAdapter>(reader: R, writer: W, adapter: A) {
    let reader = BufReader::new(reader);
    let mut writer = BufWriter::new(writer);

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                // stdin error (broken pipe etc.) — stop
                eprintln!("game-host: stdin error: {e}");
                break;
            }
        };

        let trimmed = line.trim().to_owned();
        if trimmed.is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(&trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::Error {
                    id: 0,
                    error: ErrorBody {
                        code: 400,
                        message: format!("invalid request: {e}"),
                    },
                };
                let _ = writeln!(writer, "{}", serde_json::to_string(&resp).unwrap());
                let _ = writer.flush();
                continue;
            }
        };

        let result = dispatch(&adapter, &req);
        let resp = match result {
            Ok(v) => Response::Success { id: req.id, result: v },
            Err(e) => Response::Error {
                id: req.id,
                error: ErrorBody { code: e.code, message: e.message },
            },
        };
        let json = serde_json::to_string(&resp).expect("Response always serializes");
        let _ = writeln!(writer, "{json}");
        let _ = writer.flush();
    }
}

/// Convenience wrapper that reads from stdin and writes to stdout.
pub fn run_stdin_stdout<A: GameAdapter>(adapter: A) {
    run_host(io::stdin(), io::stdout(), adapter);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch<A: GameAdapter>(adapter: &A, req: &Request) -> Result<Value, HostError> {
    match req.method.as_str() {
        // --- Metadata (no params needed) ---
        "kind" => ok_value(adapter.kind()),
        "label" => ok_value(adapter.label()),
        "description" => ok_value(adapter.description()),
        "default_config" => Ok(adapter.default_config()),

        // --- State methods ---
        "new" => adapter.new_state(req.params["config"].clone()),

        "legal_moves" => {
            let state = param(&req.params, "state")?;
            adapter.legal_moves(state).and_then(ok_value)
        }

        "apply" => {
            let state = param(&req.params, "state")?;
            let mv = param(&req.params, "move")?;
            adapter.apply(state, mv)
        }

        "view" => {
            let state = param(&req.params, "state")?;
            adapter.view(state)
        }

        "terminal" => {
            let state = param(&req.params, "state")?;
            view_terminal(adapter, state)
        }

        // --- AI methods ---
        "ai_presets" => ok_value(adapter.ai_presets()),

        "ai_move" => {
            let state = param(&req.params, "state")?;
            let preset = param_str(&req.params, "preset")?;
            adapter.ai_move(state, preset).and_then(ok_value)
        }

        "analyze" => {
            let state = param(&req.params, "state")?;
            let preset = param_str(&req.params, "preset")?;
            let budget_ms = req.params.get("budget_ms").and_then(|v| v.as_u64());
            adapter.analyze(state, preset, budget_ms).and_then(ok_value)
        }

        other => Err(HostError::not_found(format!("unknown method: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Extract a named field from a JSON object.
fn param<'a>(params: &'a Value, name: &str) -> Result<&'a Value, HostError> {
    params
        .get(name)
        .ok_or_else(|| HostError::bad_request(format!("missing parameter: {name}")))
}

/// Extract a named string field from a JSON object.
fn param_str<'a>(params: &'a Value, name: &str) -> Result<&'a str, HostError> {
    let v = param(params, name)?;
    v.as_str()
        .ok_or_else(|| HostError::bad_request(format!("parameter {name} must be a string")))
}

/// Serialize a value to JSON Value.
fn ok_value<T: serde::Serialize>(t: T) -> Result<Value, HostError> {
    serde_json::to_value(t).map_err(|e| HostError::internal(format!("serialization: {e}")))
}

/// Extract terminal/winner info from a state via the `view` method.
fn view_terminal<A: GameAdapter>(adapter: &A, state: &Value) -> Result<Value, HostError> {
    let view = adapter.view(state)?;
    let terminal = view
        .get("terminal")
        .and_then(|t| t.as_bool())
        .unwrap_or(false);
    Ok(serde_json::json!({
        "terminal": terminal,
        "winner": view.get("winner"),
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A minimal fake adapter for testing the protocol dispatch loop.
    /// Responds with just enough data to verify round-trip correctness.
    struct FakeAdapter;

    impl GameAdapter for FakeAdapter {
        fn kind(&self) -> &'static str { "fake" }
        fn label(&self) -> &'static str { "Fake Game" }
        fn description(&self) -> &'static str { "A minimal fake adapter for testing" }

        fn default_config(&self) -> Value { serde_json::json!({}) }

        fn new_state(&self, _config: Value) -> Result<Value, HostError> {
            Ok(serde_json::json!({"board": [], "turn": "X"}))
        }

        fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, HostError> {
            Ok(vec![serde_json::json!(0), serde_json::json!(1)])
        }

        fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
            let turn = state
                .get("turn")
                .and_then(|t| t.as_str())
                .unwrap_or("X");
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

        fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
            if preset == "random" {
                let next = self.apply(state, &serde_json::json!(0))?;
                Ok(AiMoveResult {
                    mv: serde_json::json!(0),
                    state: next,
                })
            } else {
                Err(HostError::not_found(format!("unknown preset: {preset}")))
            }
        }

        fn analyze(
            &self,
            _state: &Value,
            preset: &str,
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

    fn parse_response(line: &str) -> Response {
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
                assert_eq!(result.get("terminal").and_then(|t| t.as_bool()), Some(false));
                assert_eq!(result.get("turn").and_then(|t| t.as_str()), Some("X"));
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_terminal() {
        let lines = send_requests(&[r#"{"id":13,"method":"terminal","params":{"state":{"board":[],"turn":"X"}}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 13);
                assert_eq!(result.get("terminal").and_then(|t| t.as_bool()), Some(false));
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
                assert_eq!(presets[0].get("id").and_then(|v| v.as_str()), Some("random"));
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
                assert_eq!(
                    actions[0].get("visits").and_then(|v| v.as_u64()),
                    Some(10)
                );
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
            "",  // blank line
            r#"{"id":2,"method":"label","params":{}}"#,
        ]);
        // blank line produces no output, so we get 2 responses
        assert_eq!(lines.len(), 2);
    }
}