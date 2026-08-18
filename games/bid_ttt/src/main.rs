use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_bid_ttt::{BiddingTicTacToe, Move, Piece};
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

/// The parsed `easy`/`strong` preset table -- `games/bid_ttt/presets.json`'s
/// embedded defaults, or an operator-supplied override file named by
/// `BID_TTT_PRESETS_PATH` (see `PresetTable::load`'s doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let override_path = env::var("BID_TTT_PRESETS_PATH").ok().map(PathBuf::from);
        PresetTable::load(include_str!("../presets.json"), override_path.as_deref())
            .expect("games/bid_ttt/presets.json must parse")
    })
}

fn apply_move(mut s: BiddingTicTacToe, m: &Move) -> BiddingTicTacToe {
    s.apply(*m);
    s
}
#[derive(Serialize, Deserialize)]
struct WireState {
    board: Vec<Option<String>>,
    chips_x: u16,
    bid_x: u16,
    chips_o: u16,
    bid_o: u16,
    tiebreaker: String,
    phase: String,
}

fn piece_name(p: Piece) -> &'static str {
    match p {
        Piece::X => "X",
        Piece::O => "O",
    }
}
fn parse_piece(s: &str) -> Piece {
    match s {
        "X" => Piece::X,
        "O" => Piece::O,
        _ => panic!("invalid piece"),
    }
}

fn state_to_value(s: &BiddingTicTacToe) -> Value {
    serde_json::to_value(WireState {
        board: s
            .board
            .iter()
            .map(|p| p.map(|p| piece_name(p).to_string()))
            .collect(),
        chips_x: s.x.chips,
        bid_x: s.x.bid,
        chips_o: s.o.chips,
        bid_o: s.o.bid,
        tiebreaker: piece_name(s.tiebreaker).into(),
        phase: format!("{:?}", s.phase),
    })
    .expect("")
}
fn value_to_state(v: &Value) -> Result<BiddingTicTacToe, HostError> {
    let w: WireState =
        serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    let mut b = BiddingTicTacToe::new();
    for (i, cell) in w.board.iter().enumerate() {
        b.board[i] = cell.as_deref().map(parse_piece);
    }
    b.x.chips = w.chips_x;
    b.x.bid = w.bid_x;
    b.o.chips = w.chips_o;
    b.o.bid = w.bid_o;
    b.tiebreaker = parse_piece(&w.tiebreaker);
    // Phase reconstruction from phase field (format round-trip)
    b.phase = match w.phase.as_str() {
        "BidX" => game_bid_ttt::Phase::BidX,
        "BidO" => game_bid_ttt::Phase::BidO,
        "Tie" => game_bid_ttt::Phase::Tie,
        "PlayX" => game_bid_ttt::Phase::PlayX,
        "PlayO" => game_bid_ttt::Phase::PlayO,
        _ => {
            return Err(HostError::bad_request(format!(
                "unknown phase: {}",
                w.phase
            )))
        }
    };
    Ok(b)
}

struct BttAdapter;

impl GameAdapter for BttAdapter {
    fn kind(&self) -> &'static str {
        "bid-ttt"
    }
    fn label(&self) -> &'static str {
        "Bidding TicTacToe"
    }
    fn description(&self) -> &'static str {
        "Tic-Tac-Toe with bidding for the right to move."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }
    fn new_state(&self, _: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&BiddingTicTacToe::new()))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut mv = Vec::new();
        if !BiddingTicTacToe::is_terminal(&s) {
            BiddingTicTacToe::generate_actions(&s, &mut mv);
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
        if BiddingTicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        BiddingTicTacToe::generate_actions(&s, &mut legal);
        if !legal.contains(&m) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&apply_move(s, &m)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        Ok(state.clone())
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        presets().ai_presets()
    }
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?;
        if BiddingTicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = presets().build::<BiddingTicTacToe>(preset, PRESET_SEED)?;
        let action = ai.choose_action(&s);
        let next = apply_move(s, &action);
        Ok(AiMoveResult {
            mv: serde_json::to_value(action).unwrap(),
            state: state_to_value(&next),
        })
    }
    fn analyze(&self, state: &Value, preset: &str, _: Option<u64>) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        if BiddingTicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = presets().build::<BiddingTicTacToe>(preset, PRESET_SEED)?;
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
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        on_game: &mut dyn FnMut(game_host::ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        // BiddingTicTacToe's `Game::zobrist_hash` is the default constant
        // `0`, so transpositions must stay off -- see `mcts-tune`'s
        // `strategy_tune_eval` doc comment.
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
            mcts_tune::build_search::<BiddingTicTacToe>(&cfg, baseline_seed, false, &budget)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                false,
                budget,
                move || {
                    mcts_tune::build_search::<BiddingTicTacToe>(&cfg, baseline_seed, false, &budget)
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
                false,
                mcts_tune::SearchBudget {
                    max_iterations,
                    max_time: max_time_ms.map(std::time::Duration::from_millis),
                    ..Default::default()
                },
                move || {
                    presets()
                        .build::<BiddingTicTacToe>("strong", PRESET_SEED)
                        .expect("games/bid_ttt/presets.json's \"strong\" preset must build")
                },
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
    run_cli(BttAdapter);
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
        let result = BttAdapter
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
