use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_atarigo::{AtariGo, Move, Player, State};
use game_core::bigbitboard::BigBitBoard;
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
/// falling back to the compiled-in defaults only if that path is missing
/// (see `PresetTable::load`'s doc comment). Presets
/// are size-invariant: `build_easy`/`build_strong` never varied by `N`/
/// `WORDS`, only by which `Game<N, WORDS, CELLS>` `PresetTable::build` is
/// monomorphized for at each call site.
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("ATARIGO_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load(include_str!("../presets.json"), Some(&presets_path))
            .expect("games/atarigo/presets.json must parse")
    })
}

/// Board sizes this binary serves, 3x3 through 19x19. `WORDS`/`CELLS` are
/// derived from `N` (see `dispatch_size!`) rather than hand-listed per size --
/// board size is chosen at request time (via `new_state`'s `{"size": N}`
/// config, or inferred from an existing state's cell count) rather than
/// fixed at compile time.
const SUPPORTED_SIZES: &[usize] = &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
const DEFAULT_SIZE: usize = 9;

/// Runs `$body` with `$n`/`$words`/`$cells` bound as the matching `usize`
/// consts for board size `$size` (a runtime value). The match arms double as
/// validation: `$size` must be one of `SUPPORTED_SIZES` or the default arm
/// returns a `HostError::bad_request` -- so every caller of this macro
/// implicitly rejects an unsupported size before touching a `State`. Extending
/// the served range is a matter of adding `N` literals to the list below --
/// `CELLS`/`WORDS` are computed from `N` (the same `CELLS.div_ceil(64)`
/// relationship `BigBitBoard::CHECK_WORDS` asserts), not hand-transcribed.
macro_rules! dispatch_size {
    ($size:expr, $n:ident, $words:ident, $cells:ident, $body:block) => {
        dispatch_size!(@match $size, $n, $words, $cells, $body,
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
    };
    (@match $size:expr, $n:ident, $words:ident, $cells:ident, $body:block, $($lit:literal),+ $(,)?) => {
        match $size {
            $(
                $lit => {
                    const $n: usize = $lit;
                    const $cells: usize = $n * $n;
                    const $words: usize = $cells.div_ceil(64);
                    $body
                }
            )+
            other => {
                return Err(HostError::bad_request(format!(
                    "unsupported board size {other} (supported: 3..=19)"
                )))
            }
        }
    };
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

fn color_at<const N: usize, const WORDS: usize, const CELLS: usize>(
    s: &State<N, WORDS, CELLS>,
    index: usize,
) -> Option<Player> {
    if s.black().get(index) {
        Some(Player::Black)
    } else if s.white().get(index) {
        Some(Player::White)
    } else {
        None
    }
}

fn state_to_value<const N: usize, const WORDS: usize, const CELLS: usize>(
    s: &State<N, WORDS, CELLS>,
) -> Value {
    serde_json::to_value(WireState {
        turn: player_name(s.turn()).into(),
        winner: s.has_winner(),
        cells: (0..N * N)
            .map(|i| color_at(s, i).map(|p| player_name(p).to_string()))
            .collect(),
    })
    .expect("")
}

fn parse_wire_state(v: &Value) -> Result<WireState, HostError> {
    serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))
}

/// Recovers `N` from a wire state's cell count by matching it against
/// `SUPPORTED_SIZES` -- no separate `size` field is needed on the state
/// wire format because `cells.len() == N * N` already determines `N`
/// uniquely (unlike `WORDS` alone, which is ambiguous: `N=5` and `N=7` both
/// pack into a single word).
fn size_from_cell_count(len: usize) -> Result<usize, HostError> {
    SUPPORTED_SIZES
        .iter()
        .copied()
        .find(|&n| n * n == len)
        .ok_or_else(|| HostError::bad_request(format!("unexpected cell count {len}")))
}

fn state_from_wire<const N: usize, const WORDS: usize, const CELLS: usize>(
    w: &WireState,
) -> State<N, WORDS, CELLS> {
    let mut black = BigBitBoard::EMPTY;
    let mut white = BigBitBoard::EMPTY;
    for (i, cell) in w.cells.iter().enumerate() {
        match cell.as_deref() {
            Some("Black") => black.set(i),
            Some("White") => white.set(i),
            _ => {}
        }
    }
    State::from_boards(black, white, parse_player(&w.turn), w.winner)
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
        dispatch_size!(config.size, N, WORDS, CELLS, {
            Ok(state_to_value(&State::<N, WORDS, CELLS>::default()))
        })
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let w = parse_wire_state(state)?;
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, CELLS, {
            let s: State<N, WORDS, CELLS> = state_from_wire(&w);
            let mut mv = Vec::new();
            if !AtariGo::<N, WORDS, CELLS>::is_terminal(&s) {
                AtariGo::<N, WORDS, CELLS>::generate_actions(&s, &mut mv);
            }
            Ok(mv
                .into_iter()
                .map(|m| serde_json::to_value(m).unwrap())
                .collect())
        })
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let w = parse_wire_state(state)?;
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, CELLS, {
            let s: State<N, WORDS, CELLS> = state_from_wire(&w);
            let m: Move<N, WORDS> = serde_json::from_value(mv.clone())
                .map_err(|e| HostError::bad_request(e.to_string()))?;
            if AtariGo::<N, WORDS, CELLS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut legal = Vec::new();
            AtariGo::<N, WORDS, CELLS>::generate_actions(&s, &mut legal);
            if !legal.contains(&m) {
                return Err(HostError::bad_request("illegal move"));
            }
            Ok(state_to_value(&AtariGo::<N, WORDS, CELLS>::apply(s, &m)))
        })
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let w = parse_wire_state(state)?;
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, CELLS, {
            let s: State<N, WORDS, CELLS> = state_from_wire(&w);
            let winner = AtariGo::<N, WORDS, CELLS>::winner(&s);
            serde_json::to_value(GameView {
                turn: player_name(s.turn()).into(),
                cells: (0..N * N)
                    .map(|i| color_at(&s, i).map(|p| player_name(p).to_string()))
                    .collect(),
                winner: winner.map(|p| player_name(p).to_string()),
                terminal: AtariGo::<N, WORDS, CELLS>::is_terminal(&s),
            })
            .map_err(|e| HostError::internal(e.to_string()))
        })
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
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, CELLS, {
            let s: State<N, WORDS, CELLS> = state_from_wire(&w);
            if AtariGo::<N, WORDS, CELLS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut ai = mcts_tune::presets::build_strategy::<AtariGo<N, WORDS, CELLS>>(
                presets(),
                preset,
                custom_spec.as_ref(),
                PRESET_SEED,
            )?;
            let action = ai.choose_action(&s);
            let next = AtariGo::<N, WORDS, CELLS>::apply(s, &action);
            Ok(AiMoveResult {
                mv: serde_json::to_value(action).unwrap(),
                state: state_to_value(&next),
            })
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
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, CELLS, {
            let s: State<N, WORDS, CELLS> = state_from_wire(&w);
            if AtariGo::<N, WORDS, CELLS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut ai = mcts_tune::presets::build_strategy::<AtariGo<N, WORDS, CELLS>>(
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
        game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        on_game: &mut dyn FnMut(game_host::ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        let size = match game_config {
            Some(cfg) => {
                let cfg: NewGameConfig = serde_json::from_value(cfg)
                    .map_err(|e| HostError::bad_request(format!("invalid game_config: {e}")))?;
                cfg.size
            }
            None => DEFAULT_SIZE,
        };
        // AtariGo's `Game::zobrist_hash` is the default constant `0`, so
        // transpositions must stay off -- see `mcts-tune`'s `strategy_tune_eval`
        // doc comment.
        dispatch_size!(size, N, WORDS, CELLS, {
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
                mcts_tune::build_search::<AtariGo<N, WORDS, CELLS>>(
                    &cfg,
                    baseline_seed,
                    false,
                    &budget,
                )?;
                mcts_tune::strategy_tune_eval(
                    &params,
                    rounds,
                    seed,
                    false,
                    budget,
                    move || {
                        mcts_tune::build_search::<AtariGo<N, WORDS, CELLS>>(
                            &cfg,
                            baseline_seed,
                            false,
                            &budget,
                        )
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
                            .build::<AtariGo<N, WORDS, CELLS>>("strong", PRESET_SEED)
                            .expect("games/atarigo/presets.json's \"strong\" preset must build")
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
        })
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
        for &n in SUPPORTED_SIZES {
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
        for &n in SUPPORTED_SIZES {
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
