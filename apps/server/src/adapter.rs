//! Type-erased, JSON-in/JSON-out contract one `GameAdapter` per game kind
//! implements, so `main.rs`'s router depends on `Arc<dyn GameAdapter>`
//! instead of hard-coding any single game's concrete types. Every method is
//! stateless: state flows in as a JSON `Value` (round-tripped from a prior
//! response) and back out again, never read from or written to server-side
//! session storage.
//!
//! All game kinds share a single generic wrapper (`SubprocessGameAdapter`)
//! that delegates to a `SubprocessAdapter` per subprocess binary — no
//! per-game adapter structs needed.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use game_host::subprocess::SubprocessAdapter;
use game_host::GameAdapter as _; // trait with methods SubprocessAdapter implements
use serde::Serialize;
use serde_json::Value;

pub use game_host::{AiMoveResult, AiPresetInfo, Analysis, TunerInfo};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// An adapter-level error, carrying the HTTP status it should map to.
/// Implements `IntoResponse` directly as a structured
/// `{error, code}` JSON body -- `code` is just the numeric status, not a
/// separate machine-readable enum, since no caller today needs to
/// distinguish errors any finer than the status already does. Route
/// handlers return `Result<_, AdapterError>` and `?` straight through a
/// `GameAdapter` call.
#[derive(Debug)]
pub struct AdapterError {
    pub status: StatusCode,
    pub message: String,
}

impl AdapterError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
    code: u16,
}

impl IntoResponse for AdapterError {
    fn into_response(self) -> Response {
        let code = self.status.as_u16();
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
                code,
            }),
        )
            .into_response()
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A type-erased, per-game-kind adapter over `mcts::game::Game` +
/// `mcts::algorithms::Search`. Each concrete adapter deserializes its
/// `Value` arguments into the real `G::S`/`G::A`, calls straight through to
/// `Game`/`Search`, and re-serializes the result -- all per-game
/// specificity lives inside the game binary; nothing outside this trait
/// (`main.rs`'s router, in particular) ever names a concrete game type.
pub trait GameAdapter: Send + Sync {
    fn kind(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn description(&self) -> &'static str;

    /// A default/example config value for `new_state`. Also serves as
    /// `/api/games`' `config_schema` field -- not full JSON Schema (nothing
    /// in this codebase generates one, and building that generator is out of
    /// scope for what a local single-user tool's new-game form needs), just
    /// a value shape a generic form can pre-fill and let the user edit.
    fn default_config(&self) -> Value;

    fn new_state(&self, config: Value) -> Result<Value, AdapterError>;
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, AdapterError>;
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, AdapterError>;
    /// Player-to-move/terminal/winner/board, everything a renderer needs to
    /// display `state` -- the JSON shape today's `GameView` produces for
    /// Druid, generalized.
    fn view(&self, state: &Value) -> Result<Value, AdapterError>;

    fn ai_presets(&self) -> Vec<AiPresetInfo>;
    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
    ) -> Result<AiMoveResult, AdapterError>;
    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        budget_ms: Option<u64>,
    ) -> Result<Analysis, AdapterError>;

    /// Tunable strategy search-space metadata (the `algorithm` and policy
    /// axes, their parameters, and the conditions gating which parameters
    /// apply to which variant).
    /// `None` for a game with no tuner support.
    fn tuner(&self) -> Option<TunerInfo>;
}

// ---------------------------------------------------------------------------
// Generic subprocess-backed adapter (replaces all per-game adapter structs)
// ---------------------------------------------------------------------------

struct SubprocessGameAdapter {
    inner: SubprocessAdapter,
}

impl SubprocessGameAdapter {
    fn new(binary: PathBuf) -> Self {
        Self {
            inner: SubprocessAdapter::new(binary),
        }
    }
}

impl GameAdapter for SubprocessGameAdapter {
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    fn label(&self) -> &'static str {
        self.inner.label()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn default_config(&self) -> Value {
        self.inner.default_config()
    }

    fn new_state(&self, config: Value) -> Result<Value, AdapterError> {
        self.inner.new_state(config).map_err(host_to_adapter)
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, AdapterError> {
        self.inner.legal_moves(state).map_err(host_to_adapter)
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, AdapterError> {
        self.inner.apply(state, mv).map_err(host_to_adapter)
    }

    fn view(&self, state: &Value) -> Result<Value, AdapterError> {
        self.inner.view(state).map_err(host_to_adapter)
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        self.inner.ai_presets()
    }

    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
    ) -> Result<AiMoveResult, AdapterError> {
        self.inner
            .ai_move(state, preset, custom)
            .map_err(host_to_adapter)
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        budget_ms: Option<u64>,
    ) -> Result<Analysis, AdapterError> {
        self.inner
            .analyze(state, preset, custom, budget_ms)
            .map_err(host_to_adapter)
    }

    fn tuner(&self) -> Option<TunerInfo> {
        self.inner.tuner()
    }
}

// ---------------------------------------------------------------------------
// Binary path resolution
// ---------------------------------------------------------------------------

/// Resolve the path to a game binary (`game-{pkg_name}`) in the workspace's
/// target directory, matching the current build profile (debug or release).
fn binary_path(pkg_name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.parent().expect("server is a workspace member");
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let mut path = workspace.join("target").join(profile).join(pkg_name);
    path.set_extension(std::env::consts::EXE_SUFFIX);
    path
}

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn host_to_adapter(e: game_host::HostError) -> AdapterError {
    match e.code {
        400 => AdapterError::bad_request(e.message),
        404 => AdapterError::not_found(e.message),
        _ => AdapterError::internal(e.message),
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Build the map of all known game kinds, each backed by a subprocess
/// binary. Panics at startup if any binary is missing (the server is
/// unusable without them).
pub fn registry() -> HashMap<&'static str, Arc<dyn GameAdapter>> {
    let entries: Vec<(&str, &str)> = vec![
        ("akron", "game-akron"),
        ("atarigo", "game-atarigo"),
        ("breakthrough", "game-breakthrough"),
        ("congo", "game-congo"),
        ("druid", "game-druid"),
        ("focus-2p", "game-focus-2p"),
        ("focus-3p", "game-focus-3p"),
        ("focus-4p", "game-focus-4p"),
        ("gonnect", "game-gonnect"),
        ("hex-gen", "game-hex-gen"),
        ("ingenious", "game-ingenious"),
        ("knightthrough", "game-knightthrough"),
        ("margo", "game-margo"),
        ("othello", "game-othello"),
        ("tak", "game-tak"),
        ("tanbo", "game-tanbo"),
        ("traffic-lights", "game-traffic-lights"),
        ("ttt", "game-ttt"),
    ];

    entries
        .into_iter()
        .map(|(kind, pkg)| {
            let path = binary_path(pkg);
            let adapter: Arc<dyn GameAdapter> = Arc::new(SubprocessGameAdapter::new(path));
            (kind, adapter)
        })
        .collect()
}
