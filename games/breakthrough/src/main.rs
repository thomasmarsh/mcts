use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_breakthrough::{Breakthrough, Move, Player, State};
use game_core::bitboard::BitBoard;
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

#[derive(Serialize, Deserialize)]
struct WireState {
    black: String,
    white: String,
    turn: String,
    winner: bool,
}

#[derive(Serialize)]
struct GameView {
    black: String,
    white: String,
    turn: String,
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
        _ => panic!("invalid player: {name:?}"),
    }
}

fn state_to_value(s: &State<8, 8>) -> Value {
    serde_json::to_value(WireState {
        black: format!("{:016x}", s.black().bits()),
        white: format!("{:016x}", s.white().bits()),
        turn: player_name(s.turn()).into(),
        winner: s.has_winner(),
    })
    .expect("WireState serializes")
}

fn value_to_state(v: &Value) -> Result<State<8, 8>, HostError> {
    let w: WireState = serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
    let parse_hex = |s: &str| {
        u64::from_str_radix(s, 16).map_err(|e| HostError::bad_request(format!("invalid hex: {e}")))
    };
    Ok(State::new(
        BitBoard::new(parse_hex(&w.black)?),
        BitBoard::new(parse_hex(&w.white)?),
        parse_player(&w.turn),
        w.winner,
    ))
}

fn build_easy() -> Box<dyn Search<G = Breakthrough<8, 8>>> {
    Box::new(
        TreeSearch::<Breakthrough<8, 8>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("breakthrough/easy")
                .expand_threshold(1)
                .max_iterations(100)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = Breakthrough<8, 8>>> {
    Box::new(
        TreeSearch::<Breakthrough<8, 8>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("breakthrough/strong")
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
        label: "Easy",
        description: "Plain UCB1 with moderate budget.",
        build: build_easy,
    },
    PresetEntry {
        id: "strong",
        label: "Strong",
        description: "UCB1 with MCTS-Solver, deep iterations.",
        build: build_strong,
    },
];

struct PresetEntry {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    build: fn() -> Box<dyn Search<G = Breakthrough<8, 8>>>,
}

struct BtAdapter;

impl GameAdapter for BtAdapter {
    fn kind(&self) -> &'static str {
        "breakthrough"
    }
    fn label(&self) -> &'static str {
        "Breakthrough"
    }
    fn description(&self) -> &'static str {
        "A fast 8×8 board game where pieces move forward like pawns. First to reach the opponent's back rank wins."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&State::new(
            BitBoard::new(0xffff000000000000),
            BitBoard::new(0x000000000000ffff),
            Player::Black,
            false,
        )))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut moves = Vec::new();
        if !Breakthrough::<8, 8>::is_terminal(&s) {
            Breakthrough::<8, 8>::generate_actions(&s, &mut moves);
        }
        Ok(moves
            .into_iter()
            .map(|m| Value::Array(vec![Value::from(m.0 as u64), Value::from(m.1 as u64)]))
            .collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let arr = mv
            .as_array()
            .ok_or_else(|| HostError::bad_request("move must be [from, to]"))?;
        let action = Move(
            arr[0]
                .as_u64()
                .ok_or_else(|| HostError::bad_request("invalid from"))? as u8,
            arr[1]
                .as_u64()
                .ok_or_else(|| HostError::bad_request("invalid to"))? as u8,
        );
        if Breakthrough::<8, 8>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Breakthrough::<8, 8>::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Breakthrough::<8, 8>::apply(s, &action)))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let winner = Breakthrough::<8, 8>::winner(&s);
        serde_json::to_value(GameView {
            black: format!("{:016x}", s.black().bits()),
            white: format!("{:016x}", s.white().bits()),
            turn: player_name(s.turn()).into(),
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: Breakthrough::<8, 8>::is_terminal(&s),
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
            .ok_or_else(|| HostError::not_found(format!("unknown preset: {preset}")))?;
        if Breakthrough::<8, 8>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = Breakthrough::<8, 8>::apply(s, &action);
        Ok(AiMoveResult {
            mv: Value::Array(vec![
                Value::from(action.0 as u64),
                Value::from(action.1 as u64),
            ]),
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
        if Breakthrough::<8, 8>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);
        let suggested = report
            .principal_variation
            .first()
            .map(|a| Value::Array(vec![Value::from(a.0 as u64), Value::from(a.1 as u64)]));
        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: Value::Array(vec![
                        Value::from(a.action.0 as u64),
                        Value::from(a.action.1 as u64),
                    ]),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| Value::Array(vec![Value::from(a.0 as u64), Value::from(a.1 as u64)]))
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
    ) -> Result<Value, HostError> {
        // Breakthrough's `Game::zobrist_hash` is the default constant `0`,
        // so transpositions must stay off -- see `mcts-tune`'s
        // `strategy_tune_eval` doc comment.
        let outcome = if let Some(cfg) = baseline_config {
            let baseline_seed = seed.unwrap_or(0);
            // Fail fast on an invalid baseline config, before any games are
            // played -- mirrors how a bad candidate `params` is already
            // rejected during `TrialParams` deserialization inside
            // `strategy_tune_eval` itself.
            mcts_tune::build_search::<Breakthrough<8, 8>>(&cfg, baseline_seed, false)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                false,
                mcts_tune::SearchBudget::default(),
                move || {
                    mcts_tune::build_search::<Breakthrough<8, 8>>(&cfg, baseline_seed, false)
                        .expect("baseline_config already validated above")
                },
                Default::default(),
            )?
        } else {
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                false,
                mcts_tune::SearchBudget::default(),
                build_strong,
                Default::default(),
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
    run_cli(BtAdapter);
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
        let result = BtAdapter
            .tune_eval(params, 1, Some(0), None, None, None)
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
