use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{run_cli, AiMoveResult, AiPresetInfo, Analysis, GameAdapter, HostError, TunerInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use bitboard::Dyn;
use game_atarigo::{AtariGo, Bits, Move, Player, State};
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
/// `games/atarigo/presets.json` (or the file named by `ATARIGO_PRESETS_PATH`),
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment). Presets are size-invariant,
/// varying only by the starting `State`'s own runtime dims.
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("ATARIGO_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path).expect("games/atarigo/presets.json must parse")
    })
}

/// Board sizes this binary serves, 3x3 through 19x19 -- a runtime `size`
/// field on `State` (see `game_atarigo::Bits`, `Board<[u64; 6], Dyn, Dyn>`)
/// rather than a distinct compiled type per size, so this is just a bounds
/// check now, not a dispatch table.
const MIN_SIZE: usize = 3;
const MAX_SIZE: usize = 19;
const DEFAULT_SIZE: usize = game_atarigo::DEFAULT_SIZE;

fn check_size(size: usize) -> Result<usize, HostError> {
    if (MIN_SIZE..=MAX_SIZE).contains(&size) {
        Ok(size)
    } else {
        Err(HostError::bad_request(format!(
            "unsupported board size {size} (supported: {MIN_SIZE}..={MAX_SIZE})"
        )))
    }
}

#[derive(Serialize, Deserialize)]
struct WireState {
    cells: Vec<Option<String>>,
    turn: String,
    winner: bool,
}

#[derive(Serialize)]
struct GameView {
    cells: Vec<Option<String>>,
    turn: String,
    winner: Option<String>,
    terminal: bool,
}

#[derive(Deserialize)]
struct NewGameConfig {
    size: usize,
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

fn color_at(s: &State, index: usize) -> Option<Player> {
    if s.black().get_index(index) {
        Some(Player::Black)
    } else if s.white().get_index(index) {
        Some(Player::White)
    } else {
        None
    }
}

fn state_to_value(s: &State) -> Value {
    let n = s.black().rows();
    serde_json::to_value(WireState {
        turn: player_name(s.turn()).into(),
        winner: s.has_winner(),
        cells: (0..n * n)
            .map(|i| color_at(s, i).map(|p| player_name(p).to_string()))
            .collect(),
    })
    .expect("")
}

fn parse_wire_state(v: &Value) -> Result<WireState, HostError> {
    serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))
}

/// Recovers the board size from a wire state's cell count -- no separate
/// `size` field is needed on the state wire format because `cells.len() ==
/// size * size` already determines it, and it must land in
/// `MIN_SIZE..=MAX_SIZE` to be a size this binary ever produced.
fn size_from_cell_count(len: usize) -> Result<usize, HostError> {
    (MIN_SIZE..=MAX_SIZE)
        .find(|&n| n * n == len)
        .ok_or_else(|| HostError::bad_request(format!("unexpected cell count {len}")))
}

fn state_from_wire(w: &WireState) -> Result<State, HostError> {
    let size = size_from_cell_count(w.cells.len())?;
    let mut black = Bits::new(Dyn(size), Dyn(size));
    let mut white = Bits::new(Dyn(size), Dyn(size));
    for (i, cell) in w.cells.iter().enumerate() {
        match cell.as_deref() {
            Some("Black") => black.set_index(i),
            Some("White") => white.set_index(i),
            _ => {}
        }
    }
    Ok(State::from_boards(
        black,
        white,
        parse_player(&w.turn),
        w.winner,
    ))
}

struct AtarigoAdapter;

impl GameAdapter for AtarigoAdapter {
    fn kind(&self) -> &'static str {
        "atarigo"
    }
    fn label(&self) -> &'static str {
        "AtariGo"
    }
    fn description(&self) -> &'static str {
        "A Go-like game where capturing a single stone wins."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({ "size": DEFAULT_SIZE })
    }
    fn new_state(&self, config: Value) -> Result<Value, HostError> {
        let config: NewGameConfig = serde_json::from_value(config)
            .map_err(|e| HostError::bad_request(format!("invalid config: {e}")))?;
        let size = check_size(config.size)?;
        Ok(state_to_value(&State::new(size)))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let w = parse_wire_state(state)?;
        let s = state_from_wire(&w)?;
        let mut mv = Vec::new();
        if !AtariGo::is_terminal(&s) {
            AtariGo::generate_actions(&s, &mut mv);
        }
        Ok(mv
            .into_iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let w = parse_wire_state(state)?;
        let s = state_from_wire(&w)?;
        let m: Move = serde_json::from_value(mv.clone())
            .map_err(|e| HostError::bad_request(e.to_string()))?;
        if AtariGo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        AtariGo::generate_actions(&s, &mut legal);
        if !legal.contains(&m) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&AtariGo::apply(s, &m)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let w = parse_wire_state(state)?;
        let s = state_from_wire(&w)?;
        let winner = AtariGo::winner(&s);
        serde_json::to_value(GameView {
            turn: player_name(s.turn()).into(),
            cells: (0..s.black().len())
                .map(|i| color_at(&s, i).map(|p| player_name(p).to_string()))
                .collect(),
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: AtariGo::is_terminal(&s),
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
        let w = parse_wire_state(state)?;
        let s = state_from_wire(&w)?;
        if AtariGo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<AtariGo>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (action, search) = mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
            serde_json::to_value(action).expect("AtariGo action always serializes")
        });
        let next = AtariGo::apply(s, &action);
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
        let w = parse_wire_state(state)?;
        let s = state_from_wire(&w)?;
        if AtariGo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<AtariGo>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let (selected_action, search) =
            mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
                serde_json::to_value(action).expect("AtariGo action always serializes")
            });
        Ok(mcts_tune::legacy_analysis_with_report(
            &*ai,
            &s,
            &selected_action,
            search,
            |action| serde_json::to_value(action).expect("AtariGo action always serializes"),
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
        game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        trace_game_sequence_start: Option<u64>,
        on_game: &mut dyn FnMut(game_host::ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        let size = match game_config {
            Some(cfg) => {
                let cfg: NewGameConfig = serde_json::from_value(cfg)
                    .map_err(|e| HostError::bad_request(format!("invalid game_config: {e}")))?;
                check_size(cfg.size)?
            }
            None => DEFAULT_SIZE,
        };
        let initial_state = State::new(size);
        // AtariGo's `Game::zobrist_hash` is the default constant `0`, so
        // transpositions must stay off -- see `mcts-tune`'s `strategy_tune_eval`
        // doc comment.
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
            mcts_tune::build_search::<AtariGo>(&cfg, baseline_seed, false, &budget)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                false,
                budget,
                move || {
                    mcts_tune::build_search::<AtariGo>(&cfg, baseline_seed, false, &budget)
                        .expect("baseline_config already validated above")
                },
                initial_state,
                state_to_value,
                |_, action| {
                    Some(serde_json::to_value(action).expect("AtariGo action always serializes"))
                },
                trace_path.as_deref(),
                trace_game_sequence_start,
                on_game,
            )?
        } else {
            let baseline_id = baseline
                .or_else(|| presets().ai_preset_ids().first().map(|s| s.to_string()))
                .expect("games/atarigo/presets.json must declare at least one preset");
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
                    presets().build::<AtariGo>(&baseline_id, PRESET_SEED).unwrap_or_else(|e| {
                        panic!("games/atarigo/presets.json's {baseline_id:?} preset must build: {e}")
                    })
                },
                initial_state,
                state_to_value,
                |_, action| {
                    Some(serde_json::to_value(action).expect("AtariGo action always serializes"))
                },
                trace_path.as_deref(),
                trace_game_sequence_start,
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
    run_cli(AtarigoAdapter);
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
        let result = AtarigoAdapter
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

    #[test]
    fn new_state_supports_every_advertised_size() {
        for n in MIN_SIZE..=MAX_SIZE {
            let v = AtarigoAdapter
                .new_state(serde_json::json!({ "size": n }))
                .unwrap_or_else(|e| panic!("new_state({n}) failed: {e}"));
            assert_eq!(v["cells"].as_array().unwrap().len(), n * n);
        }
    }

    #[test]
    fn new_state_rejects_unsupported_size() {
        assert!(AtarigoAdapter
            .new_state(serde_json::json!({ "size": 20 }))
            .is_err());
    }

    #[test]
    fn legal_moves_and_apply_round_trip_at_every_size() {
        for n in MIN_SIZE..=MAX_SIZE {
            let state = AtarigoAdapter
                .new_state(serde_json::json!({ "size": n }))
                .unwrap();
            let moves = AtarigoAdapter.legal_moves(&state).unwrap();
            assert!(
                !moves.is_empty(),
                "size {n} should have legal moves from the empty board"
            );
            let next = AtarigoAdapter.apply(&state, &moves[0]).unwrap();
            assert_eq!(next["cells"].as_array().unwrap().len(), n * n);
        }
    }
}
