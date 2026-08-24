use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError, TunerInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_tak::{Move, Player, State, Tak};
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
/// `games/tak/presets.json` (or the file named by `TAK_PRESETS_PATH`),
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment).
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("TAK_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path).expect("games/tak/presets.json must parse")
    })
}

fn player_name(p: Player) -> &'static str {
    match p {
        Player::White => "White",
        Player::Black => "Black",
    }
}

/// Wire state: a TPS (Tak Positional System) string for the board layout,
/// plus pre-computed metadata fields a client would otherwise need to derive
/// from the TPS (reserves, opening-phase flag). Moves are PTN (Portable Tak
/// Notation) strings -- not a custom JSON shape -- throughout the protocol,
/// so any PTN-speaking tool can consume this game's outputs directly.
#[derive(Serialize, Deserialize)]
struct WireState {
    tps: String,
    stones: [u8; 2],
    caps: [u8; 2],
    turn: String,
    opening: bool,
}

/// `GameView` adds the display-only `terminal`/`winner` fields a renderer
/// needs but a round-tripped `WireState` doesn't carry.
#[derive(Serialize)]
struct GameView {
    tps: String,
    stones: [u8; 2],
    caps: [u8; 2],
    turn: String,
    opening: bool,
    winner: Option<String>,
    terminal: bool,
}

fn state_to_value(s: &State<5>) -> Value {
    serde_json::to_value(WireState {
        tps: s.to_tps(),
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
    // TPS is the canonical record; the explicit fields are pre-computed
    // convenience for the client and are not validated here (the server
    // always produces consistent output).
    State::<5>::from_tps(&w.tps).map_err(HostError::bad_request)
}

/// Convert a `Move` to its PTN string. The board width (5 for the current
/// adapter) is needed to turn square indices into coordinates.
fn move_to_ptn(m: Move) -> String {
    Tak::<5>::notation(&State::<5>::default(), &m)
}

/// Parse a PTN move string into a `Move`.
fn ptn_to_move(ptn: &str) -> Result<Move, HostError> {
    Move::from_ptn(ptn, 5).map_err(HostError::bad_request)
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
            .map(|m| Value::String(move_to_ptn(m)))
            .collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let ptn = mv
            .as_str()
            .ok_or_else(|| HostError::bad_request("move must be a PTN string"))?;
        let m = ptn_to_move(ptn)?;
        if Tak::<5>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Tak::<5>::generate_actions(&s, &mut legal);
        if !legal.contains(&m) {
            return Err(HostError::bad_request(format!("illegal move: {}", ptn)));
        }
        Ok(state_to_value(&Tak::<5>::apply(s, &m)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        Ok(serde_json::to_value(GameView {
            tps: s.to_tps(),
            stones: s.stones,
            caps: s.caps,
            turn: player_name(s.turn).into(),
            opening: s.opening,
            winner: Tak::<5>::winner(&s).map(player_name).map(String::from),
            terminal: Tak::<5>::is_terminal(&s),
        })
        .expect("GameView serializes"))
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
        if Tak::<5>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Tak<5>>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (action, search) = mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
            Value::String(move_to_ptn(*action))
        });
        let next = Tak::<5>::apply(s, &action);
        Ok(AiMoveResult {
            mv: Value::String(move_to_ptn(action)),
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
        if Tak::<5>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Tak<5>>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let encode = |action: &_| Value::String(move_to_ptn(*action));
        let (selected_action, search) = mcts_tune::choose_action_with_report(&mut *ai, &s, encode);
        Ok(mcts_tune::legacy_analysis_with_report(
            &*ai,
            &s,
            &selected_action,
            search,
            encode,
        ))
    }

    fn tuner(&self) -> Option<TunerInfo> {
        let baselines = presets().ai_preset_ids();
        Some(TunerInfo {
            game_config: self.default_config(),
            ..mcts_tune::strategy_tuner_info_with_mcgs(&baselines, TUNE_EVAL_ROUNDS, true)
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
        mcts_tune::generic_tune_eval::<Tak<5>>(
            presets(),
            "games/tak/presets.json",
            true,
            PRESET_SEED,
            baseline,
            params,
            rounds,
            seed,
            baseline_config,
            max_iterations,
            max_time_ms,
            state_to_value,
            |_, action| Some(Value::String(move_to_ptn(*action))),
            trace_path,
            trace_game_sequence_start,
            on_game,
        )
    }
}

fn main() {
    run_cli(TakAdapter);
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use game_tak::{make_cell, CAP, FLAT, WALL};

    #[test]
    fn state_round_trips_through_wire_shape() {
        let s = State::<5>::default();
        let v = state_to_value(&s);
        let back = value_to_state(&v).expect("default state round-trips");
        assert_eq!(back, s);
    }

    #[test]
    fn state_round_trips_after_opening() {
        let mut s = State::<5>::default();
        s = s.apply(&Move::place(0, FLAT)); // White places Black's flat at a1
        s = s.apply(&Move::place(12, FLAT)); // Black places White's flat at c3
        let v = state_to_value(&s);
        let back = value_to_state(&v).expect("post-opening state round-trips");
        assert_eq!(back, s);
    }

    #[test]
    fn state_round_trips_with_stacks() {
        let mut s = State::<5>::default();
        s.opening = false;
        // A 3-high stack at square 6: White bottom, Black mid, White top (flat).
        s.set_cell(6, make_cell(0b010, 3, FLAT));
        // A standing stone at square 0.
        s.set_cell(0, make_cell(1, 1, WALL));
        // A capstone at square 24.
        s.set_cell(24, make_cell(0, 1, CAP));
        // Decrement reserves to match: 2 White flats, 1 White cap, 1 Black flat,
        // 1 Black wall on the board.
        s.stones = [19, 19]; // 21 - 2 White flats, 21 - 2 Black pieces
        s.caps = [0, 1]; // 1 - 1 White cap, 1 - 0 Black caps
        s.hash = s.recompute_hash();
        let v = state_to_value(&s);
        let back = value_to_state(&v).expect("state with stacks round-trips");
        assert_eq!(back, s);
    }

    #[test]
    fn move_ptn_round_trips() {
        let place = Move::place(0, WALL);
        let ptn = move_to_ptn(place);
        assert_eq!(ptn, "Sa1");
        let decoded = ptn_to_move(&ptn).expect("placement round-trips");
        assert_eq!(decoded, place);

        let spread = Move::spread(12, 1, 3, 0b110); // take 3 from c3 east, drop (2, 1)
        let ptn = move_to_ptn(spread);
        assert_eq!(ptn, "3c3>21");
        let decoded = ptn_to_move(&ptn).expect("spread round-trips");
        assert_eq!(decoded, spread);
    }

    #[test]
    fn move_ptn_single_spread() {
        let spread = Move::spread(0, 1, 1, 0b1); // take 1 from a1 east
        let ptn = move_to_ptn(spread);
        assert_eq!(ptn, "a1>");
        let decoded = ptn_to_move(&ptn).expect("single spread round-trips");
        assert_eq!(decoded, spread);
    }

    #[test]
    fn view_reports_terminal_and_winner() {
        let mut s = State::<5>::default();
        s.opening = false;
        // Vertical white road down column 0 (north-south road).
        for row in 0..5 {
            s.set_cell(row * 5, make_cell(0, 1, FLAT));
        }
        s.stones[0] -= 5; // 5 White flats placed
        s.hash = s.recompute_hash();
        let v = TakAdapter.view(&state_to_value(&s)).expect("view succeeds");
        assert_eq!(v["terminal"], true);
        assert_eq!(v["winner"], "White");
    }

    #[test]
    fn legal_moves_are_ptn_strings() {
        let s = State::<5>::default();
        let v = state_to_value(&s);
        let moves = TakAdapter.legal_moves(&v).expect("legal_moves succeeds");
        assert!(!moves.is_empty());
        // All moves are PTN strings (flat placements in the opening).
        for mv in &moves {
            let ptn = mv.as_str().expect("move is a string");
            assert!(ptn.len() >= 2, "PTN move '{}' is too short", ptn);
        }
    }

    #[test]
    fn apply_accepts_ptn_string() {
        let s = State::<5>::default();
        let v = state_to_value(&s);
        // Play the opening: White places opponent's flat at a1 (just "a1").
        let next = TakAdapter
            .apply(&v, &Value::String("a1".into()))
            .expect("apply succeeds");
        let back = value_to_state(&next).expect("result parses");
        assert_eq!(back.turn, Player::Black);
        assert!(back.opening);
    }

    #[test]
    fn apply_rejects_non_ptn_move() {
        let s = State::<5>::default();
        let v = state_to_value(&s);
        let err = TakAdapter
            .apply(
                &v,
                &serde_json::json!({"tag": "Place", "square": 0, "kind": "Flat"}),
            )
            .unwrap_err();
        assert_eq!(err.code, 400);
    }

    #[test]
    fn tps_in_wire_state_is_parseable() {
        let s = State::<5>::default();
        let v = state_to_value(&s);
        let tps = v["tps"].as_str().expect("tps field is a string");
        assert!(tps.starts_with("x5/"));
        assert!(tps.ends_with(" 1 1"));
    }

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
        let result = TakAdapter
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
