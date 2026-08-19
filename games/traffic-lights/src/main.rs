use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_traffic_lights::{HashedPosition, Move, Piece, Player, Position, TrafficLights};
use mcts::game::Game;
use mcts_tune::presets::PresetTable;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

/// Fixed seed for every `ai_move`/`analyze`/fallback-baseline search built
/// through [`presets`] -- `GameAdapter::ai_move`/`analyze` take no seed
/// argument, so this is the only seed available to
/// `mcts_tune::presets::PresetTable::build`.
const PRESET_SEED: u64 = 0;

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/traffic-lights/presets.json` (or the file named by `TRAFFIC_LIGHTS_PRESETS_PATH`),
/// falling back to the compiled-in defaults only if that path is missing
/// (see `PresetTable::load`'s doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("TRAFFIC_LIGHTS_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load(include_str!("../presets.json"), Some(&presets_path))
            .expect("games/traffic-lights/presets.json must parse")
    })
}

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
fn parse_player(name: &str) -> Player {
    match name {
        "A" => Player::First,
        "B" => Player::Second,
        _ => panic!("invalid player"),
    }
}
fn parse_cell(name: &str) -> Piece {
    match name {
        "R" => Piece::R,
        "Y" => Piece::Y,
        "G" => Piece::G,
        _ => panic!("invalid cell"),
    }
}

fn cells_of(position: &Position) -> [Option<String>; 9] {
    std::array::from_fn(|i| cell_name(position.get(i)))
}

fn state_to_value(s: &HashedPosition) -> Value {
    serde_json::to_value(WireState {
        turn: player_name(s.position.turn).into(),
        cells: cells_of(&s.position),
    })
    .expect("")
}
fn value_to_state(v: &Value) -> Result<HashedPosition, HostError> {
    let w: WireState =
        serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    let turn = parse_player(&w.turn);
    let mut board = 0u32;
    for (i, cell) in w.cells.into_iter().enumerate() {
        if let Some(name) = cell {
            let piece = parse_cell(&name);
            board |= ((piece as u32) + 1) << (i * 2);
        }
    }
    let mut pos = Position {
        turn,
        winner: false,
        board,
    };
    pos.winner = pos.has_winner();
    Ok(HashedPosition::from_position(pos))
}

struct TlAdapter;
impl GameAdapter for TlAdapter {
    fn kind(&self) -> &'static str {
        "traffic-lights"
    }
    fn label(&self) -> &'static str {
        "Traffic Lights"
    }
    fn description(&self) -> &'static str {
        "A 3×3 game where each cell cycles through Red → Yellow → Green. Make three of the same colour in a row to win."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&HashedPosition::new()))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !TrafficLights::is_terminal(&s) {
            TrafficLights::generate_actions(&s, &mut mv);
        }
        Ok(mv.into_iter().map(|m| Value::from(m.0 as u64)).collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let idx = mv
            .as_u64()
            .ok_or_else(|| HostError::bad_request("must be u64"))? as u8;
        let action = Move(idx);
        if TrafficLights::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        TrafficLights::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&TrafficLights::apply(s, &action)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let winner = if s.position.winner {
            Some(player_name(s.position.turn).into())
        } else {
            None
        };
        serde_json::to_value(GameView {
            turn: player_name(s.position.turn).into(),
            cells: cells_of(&s.position),
            winner,
            terminal: TrafficLights::is_terminal(&s),
        })
        .map_err(|e| HostError::internal(e.to_string()))
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        presets().ai_presets()
    }
    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError> {
        let custom_spec = custom
            .map(|v| serde_json::from_value::<mcts_tune::presets::CustomStrategySpec>(v.clone()))
            .transpose()
            .map_err(|e| HostError::bad_request(format!("invalid custom strategy: {e}")))?;
        let s = value_to_state(state)?;
        if TrafficLights::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<TrafficLights>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let action = ai.choose_action(&s);
        let next = TrafficLights::apply(s, &action);
        Ok(AiMoveResult {
            mv: Value::from(action.0 as u64),
            state: state_to_value(&next),
        })
    }
    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        _: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let custom_spec = custom
            .map(|v| serde_json::from_value::<mcts_tune::presets::CustomStrategySpec>(v.clone()))
            .transpose()
            .map_err(|e| HostError::bad_request(format!("invalid custom strategy: {e}")))?;
        let s = value_to_state(state)?;
        if TrafficLights::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<TrafficLights>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);
        let suggested = report
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
            suggested_move: suggested,
        })
    }

    fn tuner(&self) -> Option<TunerInfo> {
        Some(TunerInfo {
            game_config: self.default_config(),
            ..mcts_tune::strategy_tuner_info_with_mcgs(&["strong"], TUNE_EVAL_ROUNDS, true)
        })
    }

    fn tune_eval(
        &self,
        params: Value,
        rounds: u32,
        seed: Option<u64>,
        _baseline: Option<String>,
        baseline_config: Option<Value>,
        _game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        on_game: &mut dyn FnMut(game_host::ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        // `use_transpositions: true` requires a real `Game::zobrist_hash`
        // override -- TrafficLights has one (see `lib.rs`), so merging
        // transposed nodes during the candidate's search is safe here (see
        // `generic_tune_eval`'s doc comment).
        mcts_tune::generic_tune_eval::<TrafficLights>(
            presets(),
            "strong",
            "games/traffic-lights/presets.json",
            true,
            PRESET_SEED,
            params,
            rounds,
            seed,
            baseline_config,
            max_iterations,
            max_time_ms,
            trace_path,
            on_game,
        )
    }
}

fn main() {
    run_cli(TlAdapter);
}
