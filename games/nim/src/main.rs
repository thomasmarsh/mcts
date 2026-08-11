use game_host::{
    run_stdin_stdout, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_nim::{Nim, NimState, Player};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

#[derive(Serialize, Deserialize)]
struct WireState {
    stacks: Vec<u64>,
    turn: String,
}

fn player_name(p: Player) -> &'static str {
    match p {
        Player::Black => "Black",
        Player::White => "White",
    }
}
fn parse_player(name: &str) -> Player {
    match name {
        "Black" => Player::Black,
        "White" => Player::White,
        _ => panic!("invalid player"),
    }
}

fn state_to_value(s: &NimState) -> Value {
    serde_json::to_value(WireState {
        stacks: s.game.get_stacks().iter().map(|x| x.0).collect(),
        turn: player_name(s.turn).into(),
    })
    .expect("")
}
fn value_to_state(v: &Value) -> Result<NimState, HostError> {
    let w: WireState =
        serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    let stacks: Vec<_> = w.stacks.iter().map(|&x| nimlib::Stack(x)).collect();
    let rules = vec![nimlib::NimRule {
        take: nimlib::TakeSize::Any,
        split: nimlib::Split::Optional,
    }];
    Ok(NimState {
        game: nimlib::NimGame::new(rules.clone(), stacks),
        rules,
        turn: parse_player(&w.turn),
    })
}

fn build_easy() -> Box<dyn Search<G = Nim>> {
    Box::new(
        TreeSearch::<Nim, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("nim/easy")
                .expand_threshold(1)
                .max_iterations(100)
                .q_init(QInit::Infinity),
        ),
    )
}
fn build_strong() -> Box<dyn Search<G = Nim>> {
    Box::new(
        TreeSearch::<Nim, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("nim/strong")
                .expand_threshold(0)
                .max_iterations(5000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

const PRESETS: &[PresetEntry] = &[
    PresetEntry {
        id: "easy",
        build: build_easy,
    },
    PresetEntry {
        id: "strong",
        build: build_strong,
    },
];
struct PresetEntry {
    id: &'static str,
    build: fn() -> Box<dyn Search<G = Nim>>,
}

struct NimAdapter;

impl GameAdapter for NimAdapter {
    fn kind(&self) -> &'static str {
        "nim"
    }
    fn label(&self) -> &'static str {
        "Nim"
    }
    fn description(&self) -> &'static str {
        "A classic impartial game where players take tokens from stacks."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&NimState::new()))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !Nim::is_terminal(&s) {
            Nim::generate_actions(&s, &mut mv);
        }
        Ok(mv
            .into_iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let m: nimlib::NimAction = serde_json::from_value(mv.clone())
            .map_err(|e| HostError::bad_request(e.to_string()))?;
        if Nim::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        Ok(state_to_value(&Nim::apply(s, &m)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        Ok(state.clone())
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        PRESETS
            .iter()
            .map(|p| AiPresetInfo {
                id: p.id.into(),
                label: p.id.into(),
                description: "".into(),
            })
            .collect()
    }
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found("unknown preset"))?;
        if Nim::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = Nim::apply(s, &action);
        Ok(AiMoveResult {
            mv: serde_json::to_value(&action).unwrap(),
            state: state_to_value(&next),
        })
    }
    fn analyze(&self, state: &Value, preset: &str, _: Option<u64>) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found("unknown preset"))?;
        if Nim::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);
        let suggested = report
            .principal_variation
            .first()
            .map(|a| serde_json::to_value(a).unwrap());
        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: serde_json::to_value(&a.action).unwrap(),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| serde_json::to_value(a).unwrap())
                .collect(),
            total_visits: report.total_visits,
            suggested_move: suggested,
        })
    }
}

fn main() {
    run_stdin_stdout(NimAdapter);
}
