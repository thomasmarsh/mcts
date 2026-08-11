use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_tak::{Move, Player, State, Tak};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

fn player_name(p: Player) -> &'static str {
    match p {
        Player::White => "White",
        Player::Black => "Black",
    }
}
fn parse_player(name: &str) -> Player {
    match name {
        "White" => Player::White,
        "Black" => Player::Black,
        _ => panic!("invalid player"),
    }
}

/// Serialisable snapshot of a Tak state.
#[derive(Serialize, Deserialize)]
struct WireState {
    cells: Vec<String>, // hex cell words, N*N elements
    stones: [u8; 2],
    caps: [u8; 2],
    turn: String,
    opening: bool,
}

fn state_to_value(s: &State<5>) -> Value {
    let n = 5;
    let cells: Vec<String> = s.cells[..n * n]
        .iter()
        .map(|w| format!("{:016x}", w))
        .collect();
    serde_json::to_value(WireState {
        cells,
        stones: s.stones,
        caps: s.caps,
        turn: player_name(s.turn).into(),
        opening: s.opening,
    })
    .expect("WireState serializes")
}

fn value_to_state(v: &Value) -> Result<State<5>, HostError> {
    let w: WireState =
        serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    let parse_hex = |s: &str| {
        u64::from_str_radix(s, 16).map_err(|e| HostError::bad_request(format!("invalid hex: {e}")))
    };
    let mut s = State::<5> {
        opening: w.opening,
        turn: parse_player(&w.turn),
        stones: w.stones,
        caps: w.caps,
        ..Default::default()
    };
    for (i, hex) in w.cells.iter().enumerate() {
        if i < 5 * 5 {
            s.set_cell(i, parse_hex(hex)?);
        }
    }
    s.hash = s.recompute_hash();
    Ok(s)
}

fn build_easy() -> Box<dyn Search<G = Tak<5>>> {
    Box::new(
        TreeSearch::<Tak<5>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tak/easy")
                .expand_threshold(1)
                .max_iterations(100)
                .q_init(QInit::Infinity),
        ),
    )
}
fn build_strong() -> Box<dyn Search<G = Tak<5>>> {
    Box::new(
        TreeSearch::<Tak<5>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tak/strong")
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
    build: fn() -> Box<dyn Search<G = Tak<5>>>,
}

struct TakAdapter;

impl GameAdapter for TakAdapter {
    fn kind(&self) -> &'static str {
        "tak"
    }
    fn label(&self) -> &'static str {
        "Tak"
    }
    fn description(&self) -> &'static str {
        "An abstract strategy game played on a 5x5 board. First to build a road connecting opposite edges wins."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&State::<5>::default()))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !Tak::<5>::is_terminal(&s) {
            Tak::<5>::generate_actions(&s, &mut mv);
        }
        Ok(mv
            .into_iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let m: Move = serde_json::from_value(mv.clone())
            .map_err(|e| HostError::bad_request(e.to_string()))?;
        if Tak::<5>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Tak::<5>::generate_actions(&s, &mut legal);
        if !legal.contains(&m) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Tak::<5>::apply(s, &m)))
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
        if Tak::<5>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = Tak::<5>::apply(s, &action);
        Ok(AiMoveResult {
            mv: serde_json::to_value(action).unwrap(),
            state: state_to_value(&next),
        })
    }
    fn analyze(&self, state: &Value, preset: &str, _: Option<u64>) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found("unknown preset"))?;
        if Tak::<5>::is_terminal(&s) {
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
                    action: serde_json::to_value(a.action).unwrap(),
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

    fn tuner(&self) -> Option<TunerInfo> {
        Some(mcts_tune::rave_tuner_info("strong", TUNE_EVAL_ROUNDS))
    }

    fn tune_eval(&self, params: Value, rounds: u32, seed: Option<u64>) -> Result<Value, HostError> {
        // `use_transpositions: true` requires a real `Game::zobrist_hash`
        // override -- Tak has one, so merging transposed nodes during the
        // candidate's search is safe here.
        let outcome = mcts_tune::rave_tune_eval(&params, rounds, seed, true, build_strong)?;
        Ok(serde_json::json!({
            "cost": outcome.cost,
            "wins": outcome.wins,
            "losses": outcome.losses,
            "draws": outcome.draws,
        }))
    }
}

fn main() {
    run_cli(TakAdapter);
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
        let result = TakAdapter
            .tune_eval(params, 1, Some(0))
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
