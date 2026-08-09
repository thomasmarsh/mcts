// Type-erased, JSON-in/JSON-out contract one `GameAdapter` per game kind
// implements, so `main.rs`'s router depends on `Arc<dyn GameAdapter>`
// instead of hard-coding any single game's concrete types. Every method is
// stateless: state flows in as a JSON `Value` (round-tripped from a prior
// response) and back out again, never read from or written to server-side
// session storage. One submodule per game kind (`druid`, ...), each holding
// its own concrete adapter type.

pub mod druid;
pub mod othello;
pub mod simple;
pub mod traffic_lights;
pub mod ttt;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::Value;

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

#[derive(serde::Serialize)]
struct ErrorBody {
    error: String,
    code: u16,
}

impl IntoResponse for AdapterError {
    fn into_response(self) -> Response {
        let code = self.status.as_u16();
        (self.status, Json(ErrorBody { error: self.message, code })).into_response()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AiPresetInfo {
    pub id: String,
    pub label: String,
    pub description: String,
}

/// The result of a completed `ai_move`: the move chosen and the resulting
/// state, so a stateless client can apply both without a second round trip.
pub struct AiMoveResult {
    pub mv: Value,
    pub state: Value,
}

/// One candidate root action from `analyze`, mirroring
/// `mcts::strategies::ActionReport` but with the action encoded as JSON
/// instead of a generic `G::A` -- see that type's doc comment for field
/// meaning.
#[derive(serde::Serialize)]
pub struct AnalysisAction {
    pub action: Value,
    pub visits: u32,
    pub mean_value: f64,
    pub is_proven: bool,
}

#[derive(serde::Serialize)]
pub struct Analysis {
    pub actions: Vec<AnalysisAction>,
    pub principal_variation: Vec<Value>,
    pub total_visits: u32,
    /// The top candidate: a proven win if one exists among `actions`,
    /// otherwise the most-visited action -- matching the engine's own
    /// MCTS-Solver priority for `select_final_action`/`RobustChild`.
    pub suggested_move: Option<Value>,
}

/// A type-erased, per-game-kind adapter over `mcts::game::Game` +
/// `mcts::strategies::Search`. Each concrete adapter (`DruidAdapter` today)
/// deserializes its `Value` arguments into the real `G::S`/`G::A`, calls
/// straight through to `Game`/`Search`, and re-serializes the result -- all
/// per-game specificity lives inside that one small impl; nothing outside
/// this trait (`main.rs`'s router, in particular) ever names a concrete game
/// type.
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
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, AdapterError>;
    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        budget_ms: Option<u64>,
    ) -> Result<Analysis, AdapterError>;
}
