use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_core::bitboard::BitBoard;
use game_othello::{Move, Othello, Player, State};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

#[derive(Serialize, Deserialize)]
struct WireState {
    black: u64,
    white: u64,
    turn: String,
    last_pass: bool,
}
#[derive(Serialize)]
struct GameView {
    black: u64,
    white: u64,
    turn: String,
    last_pass: bool,
    winner: Option<String>,
    terminal: bool,
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

fn state_to_value(s: &State) -> Value {
    serde_json::to_value(WireState {
        black: s.black.bits(),
        white: s.white.bits(),
        turn: player_name(s.turn).into(),
        last_pass: s.last_pass,
    })
    .expect("")
}
fn value_to_state(v: &Value) -> Result<State, HostError> {
    let w: WireState =
        serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    Ok(State {
        black: BitBoard::new(w.black),
        white: BitBoard::new(w.white),
        turn: parse_player(&w.turn),
        last_pass: w.last_pass,
        hashes: [0u64; 8],
    })
}

fn build_easy() -> Box<dyn Search<G = Othello>> {
    Box::new(
        TreeSearch::<Othello, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("othello/easy")
                .expand_threshold(1)
                .max_iterations(30)
                .q_init(QInit::Infinity),
        ),
    )
}
fn build_medium() -> Box<dyn Search<G = Othello>> {
    Box::new(
        TreeSearch::<Othello, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("othello/medium")
                .expand_threshold(1)
                .max_iterations(1000)
                .q_init(QInit::Infinity),
        ),
    )
}
fn build_strong() -> Box<dyn Search<G = Othello>> {
    Box::new(
        TreeSearch::<Othello, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("othello/strong")
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
        description: "Shallow budget — obvious mistakes.",
        build: build_easy,
    },
    PresetEntry {
        id: "medium",
        label: "Medium",
        description: "Moderate budget — plays competently.",
        build: build_medium,
    },
    PresetEntry {
        id: "strong",
        label: "Strong",
        description: "Deep MCTS-Solver — plays strongly.",
        build: build_strong,
    },
];
struct PresetEntry {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    build: fn() -> Box<dyn Search<G = Othello>>,
}

struct OthAdapter;
impl GameAdapter for OthAdapter {
    fn kind(&self) -> &'static str {
        "othello"
    }
    fn label(&self) -> &'static str {
        "Othello"
    }
    fn description(&self) -> &'static str {
        "Classic 8×8 Reversi/Othello — outflank your opponent's discs."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        let mut s = State {
            black: BitBoard::new(0),
            white: BitBoard::new(0),
            turn: Player::Black,
            last_pass: false,
            hashes: [0u64; 8],
        };
        s.black = BitBoard::new(0x0000000810000000);
        s.white = BitBoard::new(0x0000001008000000);
        Ok(state_to_value(&s))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !Othello::is_terminal(&s) {
            Othello::generate_actions(&s, &mut mv);
        }
        Ok(mv.into_iter().map(|m| Value::from(m.0 as u64)).collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let idx = mv
            .as_u64()
            .ok_or_else(|| HostError::bad_request("move must be a cell index"))?
            as u8;
        if idx == 64 {
            return Ok(state_to_value(&State {
                last_pass: true,
                ..s
            }));
        } // pass
        let action = Move(idx);
        if Othello::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Othello::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Othello::apply(s, &action)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let winner = Othello::winner(&s);
        serde_json::to_value(GameView {
            black: s.black.bits(),
            white: s.white.bits(),
            turn: player_name(s.turn).into(),
            last_pass: s.last_pass,
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: Othello::is_terminal(&s),
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
        if Othello::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = Othello::apply(s, &action);
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
        if Othello::is_terminal(&s) {
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
        // override -- Othello has one, so merging transposed nodes during
        // the candidate's search is safe here.
        let outcome = if let Some(cfg) = baseline_config {
            let baseline_seed = seed.unwrap_or(0);
            // This opponent is itself a `build_search`-built config, on
            // the same iteration-based footing as the candidate -- both
            // sides get the *same* budget (an operator's `max_iterations`
            // override included) so there's nothing to match asymmetrically
            // (see `SearchBudget`'s and `build_search`'s doc comments).
            let budget = mcts_tune::SearchBudget {
                max_iterations,
                max_time: max_time_ms.map(std::time::Duration::from_millis),
                ..Default::default()
            };
            // Fail fast on an invalid baseline config, before any games are
            // played -- mirrors how a bad candidate `params` is already
            // rejected during `TrialParams` deserialization inside
            // `strategy_tune_eval` itself.
            mcts_tune::build_search::<Othello>(&cfg, baseline_seed, true, &budget)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                true,
                budget,
                move || {
                    mcts_tune::build_search::<Othello>(&cfg, baseline_seed, true, &budget)
                        .expect("baseline_config already validated above")
                },
                Default::default(),
                trace_path.as_deref(),
                on_game,
            )?
        } else {
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                true,
                mcts_tune::SearchBudget {
                    max_iterations,
                    max_time: max_time_ms.map(std::time::Duration::from_millis),
                    ..Default::default()
                },
                build_strong,
                Default::default(),
                trace_path.as_deref(),
                on_game,
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
    run_cli(OthAdapter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "slow: plays real self-play games through mcts-tune at production iteration counts (seconds for small games, tens of minutes for large boards like druid) -- mcts-tune's own crate has a fast per-family unit suite covering dispatch; this only additionally proves this game's own Game impl round-trips end to end. Run explicitly with `cargo test --bins -- --ignored`."]
    #[test]
    fn tune_eval_round_trips() {
        let params = serde_json::json!({
            "family": "rave",
            "threshold": 700,
            "c": 0.3,
            "epsilon": 0.1,
            "q_init": "Infinity",
            "final_action": "robust_child",
            "schedule": "threshold",
            "rave": 700,
            "rave_ucb": "tuned",
        });
        let result = OthAdapter
            .tune_eval(
                params,
                1,
                Some(0),
                None,
                None,
                None,
                None,
                None,
                None,
                &mut |_| Ok(()),
            )
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
