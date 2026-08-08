// `GameAdapter` impl for traffic lights -- the third game
// proving the contract generalizes beyond Druid and tic-tac-toe.
// Builds on the same pattern as `adapters::ttt`: no engine-reuse cache
// (traffic lights' game tree is similarly small), two AI presets, and
// a board state wire format that mirrors the engine's `Position` (not
// its internal packed encoding).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use mcts::game::Game;
use mcts::games::traffic_lights::{
    HashedPosition, Move, Piece, Player, Position, TrafficLights,
};
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
            AiPreset::Easy => "Plain UCB1 with a shallow iteration budget -- plays somewhat randomly.",
            AiPreset::Strong => {
                "UCB1 with MCTS-Solver, deep enough to solve most positions -- plays near-perfectly."
            }
        }
    }
}

fn build_ai(preset: AiPreset) -> Box<dyn Search<G = TrafficLights>> {
    match preset {
        AiPreset::Easy => Box::new(
            TreeSearch::<TrafficLights, strategy::Ucb1>::new().config(
                SearchConfig::new()
                    .name("tl/easy")
                    .expand_threshold(1)
                    .max_iterations(30)
                    .q_init(QInit::Infinity),
            ),
        ),
        AiPreset::Strong => Box::new(
            TreeSearch::<TrafficLights, strategy::Ucb1>::new().config(
                SearchConfig::new()
                    .name("tl/strong")
                    .expand_threshold(0)
                    .max_iterations(10000)
                    .use_mcts_solver(true)
                    .q_init(QInit::Loss),
            ),
        ),
    }
}

#[derive(Default)]
pub struct TrafficLightsAdapter;

/// The wire shape for a position: a plain 9-element array of cell colors
/// plus whose turn it is, matching the same structure as TTT's
/// `WireState` but with cell colors (`"R"`/`"Y"`/`"G"`) instead of
/// player marks (`"X"`/`"O"`).
#[derive(Serialize, Deserialize)]
struct WireState {
    turn: String,
    cells: [Option<String>; 9],
}

#[derive(Serialize)]
struct GameView {
    turn: String,
    cells: [Option<String>; 9],
    winner: Option<String>,
    terminal: bool,
}

fn cell_name(piece: Option<Piece>) -> Option<String> {
    match piece {
        Some(Piece::R) => Some("R".into()),
        Some(Piece::Y) => Some("Y".into()),
        Some(Piece::G) => Some("G".into()),
        None => None,
    }
}

fn player_name(p: Player) -> &'static str {
    match p {
        Player::First => "A",
        Player::Second => "B",
    }
}

fn parse_player(name: &str) -> Result<Player, AdapterError> {
    match name {
        "A" => Ok(Player::First),
        "B" => Ok(Player::Second),
        other => Err(AdapterError::bad_request(format!("unknown player {other:?}"))),
    }
}

fn parse_cell(name: &str) -> Option<Piece> {
    match name {
        "R" => Some(Piece::R),
        "Y" => Some(Piece::Y),
        "G" => Some(Piece::G),
        _ => None,
    }
}

fn cells_of(position: &Position) -> [Option<String>; 9] {
    std::array::from_fn(|i| cell_name(position.get(i)))
}

fn state_to_value(state: &HashedPosition) -> Value {
    serde_json::to_value(WireState {
        turn: player_name(state.position.turn).into(),
        cells: cells_of(&state.position),
    })
    .expect("WireState always serializes")
}

fn value_to_state(v: &Value) -> Result<HashedPosition, AdapterError> {
    let wire: WireState = serde_json::from_value(v.clone())
        .map_err(|e| AdapterError::bad_request(format!("invalid state: {e}")))?;
    let turn = parse_player(&wire.turn)?;
    let mut board = 0u32;
    for (i, cell) in wire.cells.into_iter().enumerate() {
        if let Some(name) = cell {
            let piece = parse_cell(&name).ok_or_else(|| {
                AdapterError::bad_request(format!("invalid cell piece {name:?}"))
            })?;
            // Board stores 0=empty, 1=R, 2=Y, 3=G, while
            // `Piece` discriminants are R=0, Y=1, G=2 — add 1 to
            // convert from discriminant to board-bit encoding.
            let bits = (piece as u32) + 1;
            board |= bits << (i * 2);
        }
    }
    let mut position = Position { turn, winner: false, board };
    // Re-derive the winner flag — the wire state may have been
    // produced before a winning line was completed, but if there
    // happens to be one on the reconstructed board (e.g. a stale
    // client state that still carries a won position) the server
    // must see it as terminal, otherwise play could continue after
    // the game is over.
    position.winner = position.has_winner();
    Ok(HashedPosition::from_position(position))
}

fn parse_preset(preset: &str) -> Result<AiPreset, AdapterError> {
    AiPreset::parse(preset)
        .ok_or_else(|| AdapterError::bad_request(format!("unknown preset {preset:?}")))
}

impl GameAdapter for TrafficLightsAdapter {
    fn kind(&self) -> &'static str {
        "traffic-lights"
    }

    fn label(&self) -> &'static str {
        "Traffic Lights"
    }

    fn description(&self) -> &'static str {
        "A 3×3 game where each cell cycles through Red → Yellow → Green. \
         Make three of the same colour in a row to win."
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
        if !TrafficLights::is_terminal(&state) {
            TrafficLights::generate_actions(&state, &mut moves);
        }
        Ok(moves.into_iter().map(|m| json!(m.0)).collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, AdapterError> {
        let state = value_to_state(state)?;
        let raw: u8 = serde_json::from_value(mv.clone())
            .map_err(|e| AdapterError::bad_request(format!("invalid move: {e}")))?;
        let mv = Move(raw);

        if TrafficLights::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        TrafficLights::generate_actions(&state, &mut legal);
        if !legal.contains(&mv) {
            return Err(AdapterError::bad_request("illegal move"));
        }
        Ok(state_to_value(&TrafficLights::apply(state, &mv)))
    }

    fn view(&self, state: &Value) -> Result<Value, AdapterError> {
        let state = value_to_state(state)?;
        Ok(serde_json::to_value(GameView {
            turn: player_name(state.position.turn).into(),
            cells: cells_of(&state.position),
            winner: if state.position.winner {
                Some(player_name(state.position.turn).into())
            } else {
                None
            },
            terminal: TrafficLights::is_terminal(&state),
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
        if TrafficLights::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }

        let mut ai = build_ai(preset);
        let action = ai.choose_action(&state);
        let next = TrafficLights::apply(state, &action);

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
        if TrafficLights::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }

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