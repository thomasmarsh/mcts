use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_traffic_lights::{HashedPosition, Move, Piece, Player, Position, TrafficLights};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

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

fn build_easy() -> Box<dyn Search<G = TrafficLights>> {
    Box::new(
        TreeSearch::<TrafficLights, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tl/easy")
                .expand_threshold(1)
                .max_iterations(30)
                .q_init(QInit::Infinity),
        ),
    )
}
fn build_strong() -> Box<dyn Search<G = TrafficLights>> {
    Box::new(
        TreeSearch::<TrafficLights, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tl/strong")
                .expand_threshold(0)
                .max_iterations(10000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

const PRESETS: &[PresetEntry] = &[
    PresetEntry {
        id: "easy",
        label: "Easy",
        description: "Shallow budget — plays somewhat randomly.",
        build: build_easy,
    },
    PresetEntry {
        id: "strong",
        label: "Strong",
        description: "Deep MCTS-Solver — plays near-perfectly.",
        build: build_strong,
    },
];
struct PresetEntry {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    build: fn() -> Box<dyn Search<G = TrafficLights>>,
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
        PRESETS
            .iter()
            .map(|p| AiPresetInfo {
                id: p.id.into(),
                label: p.label.into(),
                description: p.description.into(),
            })
            .collect()
    }
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found("unknown preset"))?;
        if TrafficLights::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = TrafficLights::apply(s, &action);
        Ok(AiMoveResult {
            mv: Value::from(action.0 as u64),
            state: state_to_value(&next),
        })
    }
    fn analyze(&self, state: &Value, preset: &str, _: Option<u64>) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found("unknown preset"))?;
        if TrafficLights::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
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
            ..mcts_tune::strategy_tuner_info(&["strong"], TUNE_EVAL_ROUNDS)
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
        trace_path: Option<std::path::PathBuf>,
    ) -> Result<Value, HostError> {
        // `use_transpositions: true` requires a real `Game::zobrist_hash`
        // override -- TrafficLights has one (see `lib.rs`), so merging
        // transposed nodes during the candidate's search is safe here.
        let outcome = if let Some(cfg) = baseline_config {
            let baseline_seed = seed.unwrap_or(0);
            // This opponent is itself a `build_search`-built config, on
            // the same iteration-based footing as the candidate -- both
            // sides get the *same* budget (an operator's `max_iterations`
            // override included) so there's nothing to match asymmetrically
            // (see `SearchBudget`'s and `build_search`'s doc comments).
            let budget = mcts_tune::SearchBudget {
                max_iterations,
                ..Default::default()
            };
            // Fail fast on an invalid baseline config, before any games are
            // played -- mirrors how a bad candidate `params` is already
            // rejected during `TrialParams` deserialization inside
            // `strategy_tune_eval` itself.
            mcts_tune::build_search::<TrafficLights>(&cfg, baseline_seed, true, &budget)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                true,
                budget,
                move || {
                    mcts_tune::build_search::<TrafficLights>(&cfg, baseline_seed, true, &budget)
                        .expect("baseline_config already validated above")
                },
                Default::default(),
                trace_path.as_deref(),
            )?
        } else {
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                true,
                mcts_tune::SearchBudget {
                    max_iterations,
                    ..Default::default()
                },
                build_strong,
                Default::default(),
                trace_path.as_deref(),
            )?
        };
        Ok(serde_json::json!({
            "cost": outcome.cost,
            "wins": outcome.wins,
            "losses": outcome.losses,
            "draws": outcome.draws,
        }))
    }
}

fn main() {
    run_cli(TlAdapter);
}
