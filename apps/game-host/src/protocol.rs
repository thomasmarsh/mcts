use crate::{ErrorBody, GameAdapter, HostError, Request, Response};
use serde_json::Value;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

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
            Ok(v) => Response::Success {
                id: req.id,
                result: v,
            },
            Err(e) => Response::Error {
                id: req.id,
                error: ErrorBody {
                    code: e.code,
                    message: e.message,
                },
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
        "tuner" => ok_value(adapter.tuner()),

        "ai_move" => {
            let state = param(&req.params, "state")?;
            let preset = param_str(&req.params, "preset")?;
            let custom = req.params.get("custom");
            adapter.ai_move(state, preset, custom).and_then(ok_value)
        }

        "analyze" => {
            let state = param(&req.params, "state")?;
            let preset = param_str(&req.params, "preset")?;
            let custom = req.params.get("custom");
            let budget_ms = req.params.get("budget_ms").and_then(|v| v.as_u64());
            adapter
                .analyze(state, preset, custom, budget_ms)
                .and_then(ok_value)
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
