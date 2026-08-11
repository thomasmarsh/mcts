//! Standalone Tic-Tac-Toe game binary that speaks the JSON-line subprocess
//! protocol on stdin/stdout.
//!
//! Built by `cargo build -p game-ttt` and used by the server/bench crates
//! via `game_host::SubprocessAdapter`.

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_ttt::{HashedPosition, Move, Piece, Position, TicTacToe};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

// ---------------------------------------------------------------------------
// Wire format types
// ---------------------------------------------------------------------------

/// The wire shape for a position: a plain 9-cell array plus whose turn it
/// is, deliberately not `Position`'s internal packed-`u32` encoding.
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn value_to_state(v: &Value) -> Result<HashedPosition, HostError> {
    let wire: WireState = serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
    let mut position = Position::new();
    position.turn = wire.turn;
    for (i, cell) in wire.cells.into_iter().enumerate() {
        if let Some(piece) = cell {
            position.set(i, piece);
        }
    }
    Ok(HashedPosition::from_position(position))
}

// ---------------------------------------------------------------------------
// AI presets
// ---------------------------------------------------------------------------

fn build_easy() -> Box<dyn Search<G = TicTacToe>> {
    Box::new(
        TreeSearch::<TicTacToe, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("ttt/easy")
                .expand_threshold(1)
                .max_iterations(30)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = TicTacToe>> {
    Box::new(
        TreeSearch::<TicTacToe, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("ttt/strong")
                .expand_threshold(0)
                .max_iterations(5000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

struct PresetEntry {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    build: fn() -> Box<dyn Search<G = TicTacToe>>,
}

const PRESETS: &[PresetEntry] = &[
    PresetEntry {
        id: "easy",
        label: "Easy",
        description: "Plain UCB1 with a shallow iteration budget -- makes mistakes.",
        build: build_easy,
    },
    PresetEntry {
        id: "strong",
        label: "Strong",
        description: "UCB1 with MCTS-Solver, deep enough to solve the tree from most positions -- \
             plays perfectly (win or draw).",
        build: build_strong,
    },
];

// ---------------------------------------------------------------------------
// GameAdapter implementation
// ---------------------------------------------------------------------------

struct TttAdapter;

impl GameAdapter for TttAdapter {
    fn kind(&self) -> &'static str {
        "ttt"
    }

    fn label(&self) -> &'static str {
        "Tic-Tac-Toe"
    }

    fn description(&self) -> &'static str {
        "Classic 3x3 tic-tac-toe -- the trivial second game exercising the game-agnostic UI contract."
    }

    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&HashedPosition::new()))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut moves = Vec::new();
        if !TicTacToe::is_terminal(&s) {
            TicTacToe::generate_actions(&s, &mut moves);
        }
        Ok(moves.into_iter().map(|m| Value::from(m.0 as u64)).collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let idx = mv
            .as_u64()
            .ok_or_else(|| HostError::bad_request("move must be a cell index"))?;
        let action = Move(idx as u8);

        if TicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        TicTacToe::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }

        Ok(state_to_value(&TicTacToe::apply(s, &action)))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let view = GameView {
            turn: s.position.turn,
            cells: cells_of(&s.position),
            winner: s.position.winner(),
            terminal: TicTacToe::is_terminal(&s),
        };
        serde_json::to_value(view).map_err(|e| HostError::internal(format!("serialize view: {e}")))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        PRESETS
            .iter()
            .map(|p| AiPresetInfo {
                id: p.id.to_string(),
                label: p.label.to_string(),
                description: p.description.to_string(),
            })
            .collect()
    }

    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found(format!("unknown preset: {preset}")))?;

        if TicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }

        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = TicTacToe::apply(s, &action);

        Ok(AiMoveResult {
            mv: Value::from(action.0 as u64),
            state: state_to_value(&next),
        })
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found(format!("unknown preset: {preset}")))?;

        if TicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }

        let mut ai = (spec.build)();
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);

        let suggested_move = report
            .principal_variation
            .first()
            .map(|a| Value::from(a.0 as u64));

        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: Value::from(a.action.0 as u64),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| Value::from(a.0 as u64))
                .collect(),
            total_visits: report.total_visits,
            suggested_move,
        })
    }

    fn tuner(&self) -> Option<TunerInfo> {
        Some(mcts_tune::rave_tuner_info("strong", TUNE_EVAL_ROUNDS))
    }

    fn tune_eval(&self, params: Value, rounds: u32, seed: Option<u64>) -> Result<Value, HostError> {
        // `use_transpositions: true` requires a real `Game::zobrist_hash`
        // override -- TicTacToe has one, so merging transposed nodes during
        // the candidate's search is safe here.
        let outcome = mcts_tune::rave_tune_eval(&params, rounds, seed, true, build_strong)?;
        Ok(serde_json::json!({
            "cost": outcome.cost,
            "wins": outcome.wins,
            "losses": outcome.losses,
            "draws": outcome.draws,
        }))
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    run_cli(TttAdapter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tune_eval_round_trips() {
        let params = serde_json::json!({
            "threshold": 700,
            "c": 0.3,
            "epsilon": 0.1,
            "q_init": "Infinity",
            "final_action": "robust_child",
            "schedule": "threshold",
            "rave": 700,
            "rave_ucb": "tuned",
        });
        let result = TttAdapter
            .tune_eval(params, 1, Some(0))
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
