//! Subprocess-based `GameAdapter` implementation.
//!
//! Spawns a game binary (a standalone process that speaks the JSON-line
//! subprocess protocol on stdin/stdout) and routes `GameAdapter` method
//! calls over its pipes.  The server and bench crates use this to talk to
//! per-game binaries without compiling any game-specific code.

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde_json::Value;

use crate::{
    AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError, Request, Response, TunerInfo,
};

// ---------------------------------------------------------------------------
// SubprocessProcess
// ---------------------------------------------------------------------------

/// A running game subprocess with its pipe handles.
struct SubprocessProcess {
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    /// Keep `Child` alive so pipes don't close.
    _child: Child,
}

// `Child`, `ChildStdin`, `ChildStdout` are all `Send`.
// The `Send` bound below ensures the whole thing is `Send`.

// ---------------------------------------------------------------------------
// Helper types for pipe handles
// ---------------------------------------------------------------------------

use std::process::ChildStdin;
use std::process::ChildStdout;

// ---------------------------------------------------------------------------
// SubprocessAdapter
// ---------------------------------------------------------------------------

/// A `GameAdapter` that delegates to a game subprocess over stdin/stdout
/// JSON-line protocol.
///
/// Spawns the game binary eagerly in `new()` and fetches metadata (kind,
/// label, description, default_config) immediately.  The process stays
/// alive for the lifetime of the adapter; if it dies during a call, the
/// adapter restarts it and retries once.
pub struct SubprocessAdapter {
    binary_path: PathBuf,
    /// The subprocess handle (or `None` if not yet spawned, or restarting).
    inner: Mutex<Option<SubprocessProcess>>,
    /// Cached metadata — fetched once in `new()`, then stored as leaked
    /// `&'static str` so `GameAdapter`'s `&'static str` contract is met.
    kind: &'static str,
    label: &'static str,
    description: &'static str,
    default_config: Value,
}

impl SubprocessAdapter {
    /// Create a new adapter that spawns `binary_path` as a subprocess.
    ///
    /// The binary must speak the JSON-line protocol on stdin/stdout (see
    /// `run_host`/`run_stdin_stdout` in this crate).  Panics if the binary
    /// cannot be spawned or doesn't respond to metadata queries.
    pub fn new(binary_path: impl Into<PathBuf>) -> Self {
        let binary_path = binary_path.into();

        // Spawn the process and fetch metadata eagerly.  If this fails
        // there's no point continuing — the adapter can't function without
        // a working subprocess.
        let mut proc =
            spawn(&binary_path).expect("failed to spawn game binary for SubprocessAdapter");

        let kind = fetch_string(&mut proc, "kind");
        let label = fetch_string(&mut proc, "label");
        let description = fetch_string(&mut proc, "description");
        let default_config = fetch_value(&mut proc, "default_config");

        Self {
            binary_path,
            inner: Mutex::new(Some(proc)),
            kind: Box::leak(kind.into_boxed_str()),
            label: Box::leak(label.into_boxed_str()),
            description: Box::leak(description.into_boxed_str()),
            default_config,
        }
    }

    /// Send a one-shot request to the subprocess and return the result
    /// `Value`.  On any communication error, restart the process and retry
    /// once.
    fn request(&self, method: &str, params: Value) -> Result<Value, HostError> {
        fn try_send(
            adapter: &SubprocessAdapter,
            method: &str,
            params: &Value,
        ) -> Result<Value, HostError> {
            let mut guard = adapter.inner.lock().unwrap();
            let proc = guard
                .as_mut()
                .ok_or_else(|| HostError::internal("subprocess not available (restart needed)"))?;

            let req = Request {
                id: 1,
                method: method.to_owned(),
                params: params.clone(),
            };
            let line = serde_json::to_string(&req)
                .map_err(|e| HostError::internal(format!("serialize request: {e}")))?;

            // Write request.
            proc.stdin
                .write_all(line.as_bytes())
                .and_then(|_| proc.stdin.write_all(b"\n"))
                .and_then(|_| proc.stdin.flush())
                .map_err(|e| HostError::internal(format!("write to subprocess: {e}")))?;

            // Read response.
            let mut response = String::new();
            proc.stdout
                .read_line(&mut response)
                .map_err(|e| HostError::internal(format!("read from subprocess: {e}")))?;

            if response.is_empty() {
                return Err(HostError::internal("subprocess closed stdout"));
            }

            let resp: Response = serde_json::from_str(response.trim())
                .map_err(|e| HostError::internal(format!("parse response: {e}")))?;

            match resp {
                Response::Success { result, .. } => Ok(result),
                Response::Error { error, .. } => Err(HostError {
                    code: error.code,
                    message: error.message,
                }),
            }
        }

        // Try once.  On failure, restart and retry once.
        match try_send(self, method, &params) {
            Ok(v) => Ok(v),
            Err(first_err) => {
                // Mark the process as dead, restart, retry once.
                *self.inner.lock().unwrap() = None;
                *self.inner.lock().unwrap() = Some(
                    spawn(&self.binary_path)
                        .map_err(|e| HostError::internal(format!("restart subprocess: {e}")))?,
                );
                try_send(self, method, &params).map_err(|_| {
                    // Return the original error if retry also fails.
                    first_err
                })
            }
        }
    }
}

impl GameAdapter for SubprocessAdapter {
    fn kind(&self) -> &'static str {
        self.kind
    }

    fn label(&self) -> &'static str {
        self.label
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn default_config(&self) -> Value {
        self.default_config.clone()
    }

    fn new_state(&self, config: Value) -> Result<Value, HostError> {
        self.request("new", serde_json::json!({"config": config}))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let result = self.request("legal_moves", serde_json::json!({"state": state}))?;
        serde_json::from_value(result)
            .map_err(|e| HostError::internal(format!("parse legal_moves: {e}")))
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        self.request("apply", serde_json::json!({"state": state, "move": mv}))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        self.request("view", serde_json::json!({"state": state}))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        // If the subprocess fails, return an empty list.
        self.request("ai_presets", serde_json::json!({}))
            .and_then(|v| {
                serde_json::from_value(v)
                    .map_err(|e| HostError::internal(format!("parse ai_presets: {e}")))
            })
            .unwrap_or_default()
    }

    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError> {
        let mut params = serde_json::json!({"state": state, "preset": preset});
        if let Some(custom) = custom {
            params["custom"] = custom.clone();
        }
        let result = self.request("ai_move", params)?;
        serde_json::from_value(result)
            .map_err(|e| HostError::internal(format!("parse ai_move: {e}")))
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let mut params = serde_json::json!({"state": state, "preset": preset});
        if let Some(custom) = custom {
            params["custom"] = custom.clone();
        }
        if let Some(ms) = budget_ms {
            params["budget_ms"] = serde_json::json!(ms);
        }
        let result = self.request("analyze", params)?;
        serde_json::from_value(result)
            .map_err(|e| HostError::internal(format!("parse analyze: {e}")))
    }

    fn tuner(&self) -> Option<TunerInfo> {
        // Unlike `ai_presets`, a game not supporting tuning is the normal
        // case (only traffic-lights does today) -- any failure, including a
        // clean "tuning not supported" error response, just means `None`.
        self.request("tuner", serde_json::json!({}))
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
    }
}

impl Drop for SubprocessAdapter {
    fn drop(&mut self) {
        // Kill the subprocess if still alive.
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(mut proc) = guard.take() {
                let _ = proc._child.kill();
                let _ = proc._child.wait();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Spawn the game binary.
fn spawn(binary_path: &Path) -> io::Result<SubprocessProcess> {
    let mut child = Command::new(binary_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("failed to capture subprocess stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("failed to capture subprocess stdout"))?;

    Ok(SubprocessProcess {
        stdin: BufWriter::new(stdin),
        stdout: BufReader::new(stdout),
        _child: child,
    })
}

/// Send a metadata request (method takes no params) and return the result
/// as a deserialized `String`.
fn fetch_string(proc: &mut SubprocessProcess, method: &str) -> String {
    let req = Request {
        id: 1,
        method: method.to_owned(),
        params: serde_json::json!({}),
    };
    let line = serde_json::to_string(&req).expect("Request serializes");
    proc.stdin
        .write_all(line.as_bytes())
        .and_then(|_| proc.stdin.write_all(b"\n"))
        .and_then(|_| proc.stdin.flush())
        .expect("write metadata request");

    let mut response = String::new();
    proc.stdout
        .read_line(&mut response)
        .expect("read metadata response");

    let resp: Response = serde_json::from_str(response.trim()).expect("parse metadata response");
    match resp {
        Response::Success { result, .. } => result
            .as_str()
            .expect("metadata field is a string")
            .to_owned(),
        Response::Error { error, .. } => {
            panic!(
                "metadata request {method} failed: {} ({})",
                error.message, error.code
            );
        }
    }
}

/// Send a metadata request and return the result as a deserialized `Value`.
fn fetch_value(proc: &mut SubprocessProcess, method: &str) -> Value {
    let req = Request {
        id: 1,
        method: method.to_owned(),
        params: serde_json::json!({}),
    };
    let line = serde_json::to_string(&req).expect("Request serializes");
    proc.stdin
        .write_all(line.as_bytes())
        .and_then(|_| proc.stdin.write_all(b"\n"))
        .and_then(|_| proc.stdin.flush())
        .expect("write metadata request");

    let mut response = String::new();
    proc.stdout
        .read_line(&mut response)
        .expect("read metadata response");

    let resp: Response = serde_json::from_str(response.trim()).expect("parse metadata response");
    match resp {
        Response::Success { result, .. } => result,
        Response::Error { error, .. } => {
            panic!(
                "metadata request {method} failed: {} ({})",
                error.message, error.code
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Path to the test host example binary, compiled as part of the crate.
    fn test_host_binary() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // Navigate up to the workspace root, then into target/debug/examples/
        let workspace = manifest.parent().expect("game-host is a workspace member");
        let mut path = workspace.join("target");
        path.push("debug");
        path.push("examples");

        #[cfg(target_os = "windows")]
        let exe = "test_host.exe";
        #[cfg(not(target_os = "windows"))]
        let exe = "test_host";

        path.push(exe);
        path
    }

    #[test]
    fn test_kind_label_description() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        assert_eq!(adapter.kind(), "test");
        assert_eq!(adapter.label(), "Test Game");
        assert!(adapter.description().contains("SubprocessAdapter"));
    }

    #[test]
    fn test_default_config() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let config = adapter.default_config();
        assert_eq!(config, serde_json::json!({}));
    }

    #[test]
    fn test_new_state() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        assert_eq!(state.get("turn").and_then(|t| t.as_str()), Some("X"));
    }

    #[test]
    fn test_legal_moves() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let moves = adapter.legal_moves(&state).unwrap();
        assert_eq!(moves.len(), 9);
        assert_eq!(moves[0], serde_json::json!(0));
    }

    #[test]
    fn test_apply_legal_move() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let next = adapter.apply(&state, &serde_json::json!(0)).unwrap();
        assert_eq!(next.get("turn").and_then(|t| t.as_str()), Some("O"));
    }

    #[test]
    fn test_apply_illegal_move() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let err = adapter.apply(&state, &serde_json::json!(99)).unwrap_err();
        // The test host treats move 99 as illegal
        assert_eq!(err.code, 400);
    }

    #[test]
    fn test_view() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let view = adapter.view(&state).unwrap();
        assert_eq!(view.get("terminal").and_then(|t| t.as_bool()), Some(false));
        assert_eq!(view.get("turn").and_then(|t| t.as_str()), Some("X"));
    }

    #[test]
    fn test_ai_presets() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let presets = adapter.ai_presets();
        assert_eq!(presets.len(), 2);
        assert_eq!(presets[0].id, "easy");
        assert_eq!(presets[1].id, "strong");
    }

    #[test]
    fn test_ai_move() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let result = adapter.ai_move(&state, "easy", None).unwrap();
        assert!(result.mv.as_u64().is_some_and(|i| i < 9));
        assert_eq!(result.state.get("turn").and_then(|t| t.as_str()), Some("O"));
    }

    #[test]
    fn test_ai_move_unknown_preset() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let err = adapter.ai_move(&state, "nonexistent", None).unwrap_err();
        assert_eq!(err.code, 404);
    }

    #[test]
    fn test_analyze() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let analysis = adapter.analyze(&state, "easy", None, None).unwrap();
        assert_eq!(analysis.actions.len(), 9);
        assert!(analysis.suggested_move.is_some());
    }

    #[test]
    fn test_analyze_unknown_preset() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let err = adapter
            .analyze(&state, "nonexistent", None, None)
            .unwrap_err();
        assert_eq!(err.code, 404);
    }

    #[test]
    fn test_tuner_round_trips_over_jsonl() {
        let adapter = SubprocessAdapter::new(test_host_binary());
        let info = adapter.tuner().expect("test host declares a tuner");
        assert_eq!(info.id, "test");
        assert_eq!(info.baselines, vec!["strong".to_string()]);
        assert_eq!(info.eval_rounds, 5);
        assert_eq!(info.parameters.len(), 1);
        assert_eq!(info.parameters[0].name, "c");
    }

    #[test]
    fn test_multiple_requests_reuse_process() {
        let adapter = SubprocessAdapter::new(test_host_binary());

        // Several requests on the same process.
        assert_eq!(adapter.kind(), "test");
        assert_eq!(adapter.label(), "Test Game");
        let state = adapter.new_state(serde_json::json!({})).unwrap();
        let moves = adapter.legal_moves(&state).unwrap();
        assert_eq!(moves.len(), 9);
        let view = adapter.view(&state).unwrap();
        assert!(!view.get("terminal").and_then(|t| t.as_bool()).unwrap());
    }
}
