use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError, TunerInfo};
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

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/bid_ttt/presets.json` (or the file named by `BID_TTT_PRESETS_PATH`),
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("BID_TTT_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path).expect("games/bid_ttt/presets.json must parse")
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
        if BiddingTicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<BiddingTicTacToe>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (action, search) = mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
            serde_json::to_value(action).expect("BiddingTicTacToe action always serializes")
        });
        let next = apply_move(s, &action);
        Ok(AiMoveResult {
            mv: serde_json::to_value(action).unwrap(),
            state: state_to_value(&next),
            search: Some(search),
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
        if BiddingTicTacToe::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<BiddingTicTacToe>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (selected_action, search) =
            mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
                serde_json::to_value(action).expect("BiddingTicTacToe action always serializes")
            });
        Ok(mcts_tune::legacy_analysis_with_report(
            &*ai,
            &s,
            &selected_action,
            search,
            |action| {
                serde_json::to_value(action).expect("BiddingTicTacToe action always serializes")
            },
        ))
    }

    fn tuner(&self) -> Option<TunerInfo> {
        let baselines = presets().ai_preset_ids();
        Some(TunerInfo {
            game_config: self.default_config(),
            ..mcts_tune::strategy_tuner_info(&baselines, TUNE_EVAL_ROUNDS)
        })
    }

    fn tune_eval(
        &self,
        params: Value,
        rounds: u32,
        seed: Option<u64>,
        baseline: Option<String>,
        baseline_config: Option<Value>,
        _game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        trace_game_sequence_start: Option<u64>,
        on_game: &mut dyn FnMut(game_host::ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        // BiddingTicTacToe's `Game::zobrist_hash` is the default constant
        // `0`, so transpositions must stay off -- see `generic_tune_eval`'s
        // doc comment.
        mcts_tune::generic_tune_eval::<BiddingTicTacToe>(
            presets(),
            "games/bid_ttt/presets.json",
            false,
            PRESET_SEED,
            baseline,
            params,
            rounds,
            seed,
            baseline_config,
            max_iterations,
            max_time_ms,
            state_to_value,
            |_, action| {
                Some(
                    serde_json::to_value(action)
                        .expect("BiddingTicTacToe action always serializes"),
                )
            },
            trace_path,
            trace_game_sequence_start,
            on_game,
        )
    }
}

fn main() {
    run_cli(BttAdapter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "slow: plays real self-play games through mcts-tune at production iteration counts (seconds for small games, tens of minutes for large boards like druid) -- mcts-tune's own crate has a fast per-variant unit suite covering dispatch; this only additionally proves this game's own Game impl round-trips end to end. Run explicitly with `cargo test --bins -- --ignored`."]
    #[test]
    fn tune_eval_round_trips() {
        let params = serde_json::json!({
            "algorithm": "mcts",
            "select": "rave",
            "simulate": "decisive_move_mast",
            "decisive_move_mode": "win_loss",
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
                None,
                &mut |_| Ok(()),
            )
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
