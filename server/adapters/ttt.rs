// `GameAdapter` impl for tic-tac-toe (PLAN-UI.md session 8) -- the second
// game proving Session 2's contract generalizes beyond Druid. Deliberately
// far smaller than `adapters::druid`: no engine-reuse cache (a tic-tac-toe
// search is cheap enough -- a few thousand iterations at most -- that
// rebuilding an engine from scratch on every call is not worth the extra
// bookkeeping Druid's `EngineCache` carries for its multi-second budgets),
// and only two AI presets instead of four.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use mcts::game::Game;
use mcts::games::ttt::{HashedPosition, Move, Piece, Position, TicTacToe};
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

use crate::adapters::{
    AdapterError, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiPreset {
    Easy,
    Strong,
}

impl AiPreset {
    const ALL: [AiPreset; 2] = [AiPreset::Easy, AiPreset::Strong];

    fn id(self) -> &'static str {
        match self {
            AiPreset::Easy => "easy",
            AiPreset::Strong => "strong",
        }
    }

    fn parse(id: &str) -> Option<AiPreset> {
        AiPreset::ALL.into_iter().find(|p| p.id() == id)
    }

    fn label(self) -> &'static str {
        match self {
            AiPreset::Easy => "Easy",
            AiPreset::Strong => "Strong",
        }
    }

    fn description(self) -> &'static str {
        match self {
            AiPreset::Easy => "Plain UCB1 with a shallow iteration budget -- makes mistakes.",
            AiPreset::Strong => {
                "UCB1 with MCTS-Solver, deep enough to solve the tree from most positions -- \
                 plays perfectly (win or draw)."
            }
        }
    }
}

// Iteration-based, not time-based like Druid's presets: tic-tac-toe's whole
// game tree (5478 distinct positions) is small enough that even `Strong`'s
// budget finishes in well under a millisecond, so there's no thinking-time
// UX to protect the way Druid's server-side time budgets are.
fn build_ai(preset: AiPreset) -> Box<dyn Search<G = TicTacToe>> {
    match preset {
        AiPreset::Easy => Box::new(
            TreeSearch::<TicTacToe, strategy::Ucb1>::new().config(
                SearchConfig::new()
                    .name("ttt/easy")
                    .expand_threshold(1)
                    .max_iterations(30)
                    .q_init(QInit::Infinity),
            ),
        ),
        AiPreset::Strong => Box::new(
            TreeSearch::<TicTacToe, strategy::Ucb1>::new().config(
                SearchConfig::new()
                    .name("ttt/strong")
                    .expand_threshold(0)
                    .max_iterations(5000)
                    .use_mcts_solver(true)
                    .q_init(QInit::Loss),
            ),
        ),
    }
}

#[derive(Default)]
pub struct TttAdapter;

/// The wire shape for a position: a plain 9-cell array plus whose turn it
/// is, deliberately not `Position`'s internal packed-`u32` encoding (an
/// implementation detail of the engine's move generator/hasher, not
/// something a client should need to understand) -- mirrors
/// `adapters::druid`'s `State`-is-the-wire-format/`HashedState`-is-internal
/// split, just without a separate named `State` type on the engine side
/// since `Position` already plays that role for tic-tac-toe.
#[derive(Serialize, Deserialize)]
struct WireState {
    turn: Piece,
    cells: [Option<Piece>; 9],
}

#[derive(Serialize)]
struct GameView {
    turn: Piece,
    cells: [Option<Piece>; 9],
    winner: Option<Piece>,
    terminal: bool,
}

fn cells_of(position: &Position) -> [Option<Piece>; 9] {
    std::array::from_fn(|i| position.get(i))
}

fn state_to_value(state: &HashedPosition) -> Value {
    serde_json::to_value(WireState {
        turn: state.position.turn,
        cells: cells_of(&state.position),
    })
    .expect("WireState always serializes")
}

/// Deserializes a client-supplied state back into a `HashedPosition`. Only
/// the `Piece`/array shapes are validated by `serde` here (a wrong-length
/// `cells` array is rejected for free by the fixed-size `[Option<Piece>; 9]`
/// type) -- deeper consistency checks (e.g. a `turn` that doesn't match the
/// piece counts on the board) are deliberately left to PLAN-UI.md session 9's
/// hardening pass, matching `adapters::druid::value_to_state`'s own
/// discipline.
fn value_to_state(state: &Value) -> Result<HashedPosition, AdapterError> {
    let wire: WireState = serde_json::from_value(state.clone())
        .map_err(|e| AdapterError::bad_request(format!("invalid state: {e}")))?;
    let mut position = Position::new();
    position.turn = wire.turn;
    for (i, cell) in wire.cells.into_iter().enumerate() {
        if let Some(piece) = cell {
            position.set(i, piece);
        }
    }
    Ok(HashedPosition::from_position(position))
}

fn parse_preset(preset: &str) -> Result<AiPreset, AdapterError> {
    AiPreset::parse(preset).ok_or_else(|| AdapterError::bad_request(format!("unknown preset {preset:?}")))
}

impl GameAdapter for TttAdapter {
    fn kind(&self) -> &'static str {
        "ttt"
    }

    fn label(&self) -> &'static str {
        "Tic-Tac-Toe"
    }

    fn description(&self) -> &'static str {
        "Classic 3x3 tic-tac-toe -- the trivial second game exercising the game-agnostic \
         UI contract (PLAN-UI.md session 8)."
    }

    fn default_config(&self) -> Value {
        json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, AdapterError> {
        Ok(state_to_value(&HashedPosition::new()))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, AdapterError> {
        let state = value_to_state(state)?;
        let mut moves = Vec::new();
        if !TicTacToe::is_terminal(&state) {
            TicTacToe::generate_actions(&state, &mut moves);
        }
        Ok(moves.into_iter().map(|m| json!(m.0)).collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, AdapterError> {
        let state = value_to_state(state)?;
        let index: u8 = serde_json::from_value(mv.clone())
            .map_err(|e| AdapterError::bad_request(format!("invalid move: {e}")))?;
        let mv = Move(index);

        if TicTacToe::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        TicTacToe::generate_actions(&state, &mut legal);
        if !legal.contains(&mv) {
            return Err(AdapterError::bad_request("illegal move"));
        }
        Ok(state_to_value(&TicTacToe::apply(state, &mv)))
    }

    fn view(&self, state: &Value) -> Result<Value, AdapterError> {
        let state = value_to_state(state)?;
        Ok(serde_json::to_value(GameView {
            turn: state.position.turn,
            cells: cells_of(&state.position),
            winner: state.position.winner(),
            terminal: TicTacToe::is_terminal(&state),
        })
        .expect("GameView always serializes"))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        AiPreset::ALL
            .iter()
            .map(|&id| AiPresetInfo {
                id: id.id(),
                label: id.label(),
                description: id.description(),
            })
            .collect()
    }

    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, AdapterError> {
        let state = value_to_state(state)?;
        let preset = parse_preset(preset)?;
        if TicTacToe::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }

        let mut ai = build_ai(preset);
        let action = ai.choose_action(&state);
        let next = TicTacToe::apply(state, &action);

        Ok(AiMoveResult {
            mv: json!(action.0),
            state: state_to_value(&next),
        })
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, AdapterError> {
        let state = value_to_state(state)?;
        let preset = parse_preset(preset)?;
        if TicTacToe::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }

        // No time-budget override (unlike Druid's `analyze`): every preset
        // here is already iteration-bounded and finishes in microseconds, so
        // there's nothing for a `budget_ms` override to meaningfully do.
        let mut ai = build_ai(preset);
        let _ = ai.choose_action(&state);
        let report = ai.root_report(&state);

        let suggested_move = report.principal_variation.first().map(|a| json!(a.0));

        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: json!(a.action.0),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| json!(a.0))
                .collect(),
            total_visits: report.total_visits,
            suggested_move,
        })
    }
}
