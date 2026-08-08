//! Generic "simple game" adapter path for iteration-bounded, cache-free,
//! fixed-config games (ttt, traffic-lights). A `SimpleGameCodec` impl +
//! `PresetSpec` table replaces a hand-written ~276-line `GameAdapter`
//! impl with ~60-90 lines of genuinely per-game code.
//!
//! # Design
//!
//! `SimpleAdapter<C>` (a `PhantomData<C>` wrapper) provides a blanket
//! `impl<C: SimpleGameCodec> GameAdapter for SimpleAdapter<C>`, writing
//! every `GameAdapter` trait method exactly once in terms of the `Game`/
//! `Search` traits and `C`'s codec methods:
//!
//!   - `new_state` / `legal_moves` / `apply` / `view` / `ai_presets` /
//!     `ai_move` / `analyze` -- each has the same shape across both
//!     simple games (terminal-check → build-AI → choose-action →
//!     report/serialize), with only the concrete `Game`/`G::S`/`G::A`
//!     types and the wire-serialization logic differing between them.
//!   - `SimpleGameCodec` captures exactly that per-game surface: the wire
//!     format types, kind/label/description constants, state/move/view
//!     translations, the preset table, and a default-state constructor.
//!
//! Druid keeps its own hand-written `GameAdapter` impl (it needs
//! `EngineCache`, `NewGameConfig`, time-budgeted presets, and
//! thread-count logic that this generic path doesn't support).

use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use mcts::game::Game;
use mcts::strategies::Search;

use crate::adapters::{
    AdapterError, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter,
};

// ---------------------------------------------------------------------------
// PresetSpec
// ---------------------------------------------------------------------------

/// One AI preset in a simple game's preset table. `G` is the concrete `Game`
/// type the preset's `build` function returns a `Search<G = G>` for.
pub struct PresetSpec<G: Game> {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    /// Builds a fresh search engine for this preset. Called on every
    /// `ai_move`/`analyze` (no cross-request caching for simple games --
    /// the search is cheap enough that rebuilding from scratch is
    /// simpler than bookkeeping).
    pub build: fn() -> Box<dyn Search<G = G>>,
}

// `fn()` pointers are `Sync + Send` (they capture nothing), so
// `PresetSpec<G>` is too -- required by the `GameAdapter` blanket impl.
unsafe impl<G: Game> Sync for PresetSpec<G> {}
unsafe impl<G: Game> Send for PresetSpec<G> {}

// ---------------------------------------------------------------------------
// SimpleGameCodec trait
// ---------------------------------------------------------------------------

/// Per-game wire-format and metadata, enough to drive `SimpleAdapter`'s
/// blanket `GameAdapter` impl.
///
/// Every `SimpleGameCodec: Game`, so the blanket impl can call
/// `C::is_terminal`, `C::generate_actions`, `C::apply`, etc. -- the
/// `Game` trait methods -- without knowing which concrete game type `C` is.
pub trait SimpleGameCodec: Game where Self: 'static {
    /// Wire-format state: the JSON shape clients send and receive.
    /// Deliberately distinct from `Self::S` (the engine's internal state)
    /// so the wire format can be a friendly JSON shape even when `Self::S`
    /// uses a packed binary encoding (see `ttt.rs`'s `WireState` comment).
    type WireState: Serialize + DeserializeOwned;

    /// Wire-format move: the JSON shape for a single action.
    type WireMove: Serialize + DeserializeOwned;

    /// Wire-format view: the JSON shape for `GameAdapter::view`.
    type WireView: Serialize;

    const KIND: &'static str;
    const LABEL: &'static str;
    const DESCRIPTION: &'static str;

    /// The preset table. Each preset's `build` function returns a fresh
    /// `Box<dyn Search<G = Self>>` for that preset's search config.
    const PRESETS: &'static [PresetSpec<Self>];

    // -- Wire translation ----------------------------------------------------

    fn to_wire_state(state: &Self::S) -> Self::WireState;
    fn from_wire_state(state: Self::WireState) -> Self::S;
    fn to_wire_move(mv: &Self::A) -> Self::WireMove;
    fn from_wire_move(mv: Self::WireMove) -> Self::A;
    fn game_view(state: &Self::S) -> Self::WireView;

    /// A fresh default state (empty board, first player to move).
    /// Defaults to `Self::S::default()`, which is correct for games where
    /// `HashedPosition` impls `Default` (both ttt and traffic-lights do).
    /// Override if a game's default state needs custom construction.
    fn default_state() -> Self::S {
        Self::S::default()
    }
}

// ---------------------------------------------------------------------------
// SimpleAdapter
// ---------------------------------------------------------------------------

/// A `GameAdapter` that delegates every method to a `SimpleGameCodec`.
pub struct SimpleAdapter<C>(PhantomData<C>);

impl<C: SimpleGameCodec> Default for SimpleAdapter<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: SimpleGameCodec> SimpleAdapter<C> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_value<T: Serialize>(t: T) -> Result<Value, AdapterError> {
    serde_json::to_value(t).map_err(|e| AdapterError::internal(format!("serialization: {e}")))
}

fn deser_state<C: SimpleGameCodec>(v: &Value) -> Result<C::S, AdapterError> {
    let wire: C::WireState =
        serde_json::from_value(v.clone()).map_err(|e| AdapterError::bad_request(format!("invalid state: {e}")))?;
    Ok(C::from_wire_state(wire))
}

fn find_preset<C: SimpleGameCodec>(id: &str) -> Result<&'static PresetSpec<C>, AdapterError> {
    C::PRESETS
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| AdapterError::bad_request(format!("unknown preset {id:?}")))
}

// ---------------------------------------------------------------------------
// Blanket GameAdapter impl
// ---------------------------------------------------------------------------

impl<C: SimpleGameCodec + 'static> GameAdapter for SimpleAdapter<C> {
    fn kind(&self) -> &'static str {
        C::KIND
    }

    fn label(&self) -> &'static str {
        C::LABEL
    }

    fn description(&self) -> &'static str {
        C::DESCRIPTION
    }

    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, AdapterError> {
        to_value(C::to_wire_state(&C::default_state()))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, AdapterError> {
        let s = deser_state::<C>(state)?;
        let mut moves = Vec::new();
        if !C::is_terminal(&s) {
            C::generate_actions(&s, &mut moves);
        }
        Ok(moves
            .into_iter()
            .map(|m| serde_json::to_value(C::to_wire_move(&m)).expect("WireMove always serializes"))
            .collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, AdapterError> {
        let s = deser_state::<C>(state)?;
        let wire_mv: C::WireMove = serde_json::from_value(mv.clone())
            .map_err(|e| AdapterError::bad_request(format!("invalid move: {e}")))?;
        let action = C::from_wire_move(wire_mv);

        if C::is_terminal(&s) {
            return Err(AdapterError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        C::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(AdapterError::bad_request("illegal move"));
        }
        to_value(C::to_wire_state(&C::apply(s, &action)))
    }

    fn view(&self, state: &Value) -> Result<Value, AdapterError> {
        let s = deser_state::<C>(state)?;
        to_value(C::game_view(&s))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        C::PRESETS
            .iter()
            .map(|p| AiPresetInfo {
                id: p.id.to_string(),
                label: p.label.to_string(),
                description: p.description.to_string(),
            })
            .collect()
    }

    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, AdapterError> {
        let s = deser_state::<C>(state)?;
        let spec = find_preset::<C>(preset)?;
        if C::is_terminal(&s) {
            return Err(AdapterError::bad_request("game is over"));
        }

        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = C::apply(s, &action);

        Ok(AiMoveResult {
            mv: serde_json::to_value(C::to_wire_move(&action)).expect("WireMove always serializes"),
            state: serde_json::to_value(C::to_wire_state(&next)).expect("WireState always serializes"),
        })
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, AdapterError> {
        let s = deser_state::<C>(state)?;
        let spec = find_preset::<C>(preset)?;
        if C::is_terminal(&s) {
            return Err(AdapterError::bad_request("game is over"));
        }

        let mut ai = (spec.build)();
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);

        let suggested_move = report
            .principal_variation
            .first()
            .map(|a| serde_json::to_value(C::to_wire_move(a)).expect("WireMove always serializes"));

        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: serde_json::to_value(C::to_wire_move(&a.action))
                        .expect("WireMove always serializes"),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| serde_json::to_value(C::to_wire_move(&a)).expect("WireMove always serializes"))
                .collect(),
            total_visits: report.total_visits,
            suggested_move,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::adapters::GameAdapter;
    use super::*;

    type Adapter = SimpleAdapter<mcts::games::ttt::TicTacToe>;

    #[test]
    fn test_kind_label_description() {
        let a = Adapter::new();
        assert_eq!(a.kind(), "ttt");
        assert_eq!(a.label(), "Tic-Tac-Toe");
        assert!(a.description().contains("tic-tac-toe"));
    }

    #[test]
    fn test_new_state_creates_empty_board() {
        let a = Adapter::new();
        let state = a.new_state(serde_json::json!({})).unwrap();
        let cells = state.get("cells").and_then(|c| c.as_array());
        assert_eq!(cells.map(|c| c.len()), Some(9));
        assert!(cells.unwrap().iter().all(|x| x.is_null()));
        assert_eq!(state.get("turn").and_then(|t| t.as_str()), Some("X"));
    }

    #[test]
    fn test_legal_moves_on_fresh_board() {
        let a = Adapter::new();
        let state = a.new_state(serde_json::json!({})).unwrap();
        let moves = a.legal_moves(&state).unwrap();
        assert_eq!(moves.len(), 9);
    }

    #[test]
    fn test_apply_legal_move() {
        let a = Adapter::new();
        let state = a.new_state(serde_json::json!({})).unwrap();
        let next = a.apply(&state, &serde_json::json!(0)).unwrap();
        let cells0 = next.pointer("/cells/0").and_then(|c| c.as_str());
        assert_eq!(cells0, Some("X"));
        assert_eq!(next.get("turn").and_then(|t| t.as_str()), Some("O"));
    }

    #[test]
    fn test_apply_illegal_move() {
        let a = Adapter::new();
        let state = a.new_state(serde_json::json!({})).unwrap();
        let s1 = a.apply(&state, &serde_json::json!(4)).unwrap();
        let result = a.apply(&s1, &serde_json::json!(4));
        assert!(result.is_err());
    }

    #[test]
    fn test_view_on_fresh_board() {
        let a = Adapter::new();
        let state = a.new_state(serde_json::json!({})).unwrap();
        let view = a.view(&state).unwrap();
        assert_eq!(view.get("terminal").and_then(|t| t.as_bool()), Some(false));
        assert_eq!(view.get("turn").and_then(|t| t.as_str()), Some("X"));
        assert!(view.get("winner").is_none() || view.get("winner").and_then(|w| w.as_str()).is_none());
    }

    #[test]
    fn test_ai_presets_returns_correct_ids() {
        let a = Adapter::new();
        let presets = a.ai_presets();
        assert_eq!(presets.len(), 2);
        assert_eq!(presets[0].id, "easy");
        assert_eq!(presets[1].id, "strong");
    }

    #[test]
    fn test_ai_move_returns_valid_move() {
        let a = Adapter::new();
        let state = a.new_state(serde_json::json!({})).unwrap();
        let result = a.ai_move(&state, "easy").unwrap();
        assert!(result.mv.as_u64().is_some_and(|i| i < 9));
        assert_eq!(result.state.get("turn").and_then(|t| t.as_str()), Some("O"));
    }

    #[test]
    fn test_ai_move_rejects_unknown_preset() {
        let a = Adapter::new();
        let state = a.new_state(serde_json::json!({})).unwrap();
        let result = a.ai_move(&state, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_returns_actions() {
        let a = Adapter::new();
        let state = a.new_state(serde_json::json!({})).unwrap();
        let analysis = a.analyze(&state, "easy", None).unwrap();
        assert_eq!(analysis.actions.len(), 9);
        assert!(analysis.suggested_move.is_some());
    }

    #[test]
    fn test_analyze_rejects_terminal_state() {
        let a = Adapter::new();
        let s0 = a.new_state(serde_json::json!({})).unwrap();
        let s1 = a.apply(&s0, &serde_json::json!(0)).unwrap(); // X
        let s2 = a.apply(&s1, &serde_json::json!(3)).unwrap(); // O
        let s3 = a.apply(&s2, &serde_json::json!(1)).unwrap(); // X
        let s4 = a.apply(&s3, &serde_json::json!(4)).unwrap(); // O
        let s5 = a.apply(&s4, &serde_json::json!(2)).unwrap(); // X wins

        let view = a.view(&s5).unwrap();
        assert_eq!(view.get("terminal").and_then(|t| t.as_bool()), Some(true));

        let result = a.analyze(&s5, "easy", None);
        assert!(result.is_err());
    }
}