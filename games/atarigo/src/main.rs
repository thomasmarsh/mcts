use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_atarigo::{AtariGo, Move, Player, State};
use game_core::bigbitboard::BigBitBoard;
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

/// `(N, WORDS)` pairs this binary serves. Each is a distinct
/// `State<N, WORDS>` monomorphization -- see `dispatch_size!` below -- so
/// board size is chosen at request time (via `new_state`'s `{"size": N}`
/// config, or inferred from an existing state's cell count) rather than
/// fixed at compile time.
const SUPPORTED_SIZES: &[(usize, usize)] = &[(5, 1), (7, 1), (9, 2)];
const DEFAULT_SIZE: usize = 9;

/// Runs `$body` with `$n`/`$words` bound as the matching `usize` consts for
/// board size `$size` (a runtime value). The match arms double as
/// validation: `$size` must be one of `SUPPORTED_SIZES` or the default arm
/// returns a `HostError::bad_request` -- so every caller of this macro
/// implicitly rejects an unsupported size before touching a `State`.
macro_rules! dispatch_size {
    ($size:expr, $n:ident, $words:ident, $body:block) => {
        match $size {
            5 => {
                const $n: usize = 5;
                const $words: usize = 1;
                $body
            }
            7 => {
                const $n: usize = 7;
                const $words: usize = 1;
                $body
            }
            9 => {
                const $n: usize = 9;
                const $words: usize = 2;
                $body
            }
            other => {
                return Err(HostError::bad_request(format!(
                    "unsupported board size {other} (supported: 5, 7, 9)"
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

fn color_at<const N: usize, const WORDS: usize>(
    s: &State<N, WORDS>,
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

fn state_to_value<const N: usize, const WORDS: usize>(s: &State<N, WORDS>) -> Value {
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
        .map(|&(n, _)| n)
        .find(|&n| n * n == len)
        .ok_or_else(|| HostError::bad_request(format!("unexpected cell count {len}")))
}

fn state_from_wire<const N: usize, const WORDS: usize>(w: &WireState) -> State<N, WORDS> {
    let mut black = BigBitBoard::EMPTY;
    let mut white = BigBitBoard::EMPTY;
    for (i, cell) in w.cells.iter().enumerate() {
        match cell.as_deref() {
            Some("Black") => black.set(i),
            Some("White") => white.set(i),
            _ => {}
        }
    }
    State {
        black,
        white,
        turn: parse_player(&w.turn),
        winner: w.winner,
    }
}

fn build_easy<const N: usize, const WORDS: usize>() -> Box<dyn Search<G = AtariGo<N, WORDS>>> {
    Box::new(
        TreeSearch::<AtariGo<N, WORDS>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("atarigo/easy")
                .expand_threshold(1)
                .max_iterations(100)
                .q_init(QInit::Infinity),
        ),
    )
}
fn build_strong<const N: usize, const WORDS: usize>() -> Box<dyn Search<G = AtariGo<N, WORDS>>> {
    Box::new(
        TreeSearch::<AtariGo<N, WORDS>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("atarigo/strong")
                .expand_threshold(0)
                .max_iterations(5000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}
fn build_preset<const N: usize, const WORDS: usize>(
    id: &str,
) -> Result<Box<dyn Search<G = AtariGo<N, WORDS>>>, HostError> {
    match id {
        "easy" => Ok(build_easy::<N, WORDS>()),
        "strong" => Ok(build_strong::<N, WORDS>()),
        _ => Err(HostError::not_found("unknown preset")),
    }
}

const PRESET_IDS: &[&str] = &["easy", "strong"];

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
        dispatch_size!(config.size, N, WORDS, {
            Ok(state_to_value(&State::<N, WORDS>::default()))
        })
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let w = parse_wire_state(state)?;
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: State<N, WORDS> = state_from_wire(&w);
            let mut mv = Vec::new();
            if !AtariGo::<N, WORDS>::is_terminal(&s) {
                AtariGo::<N, WORDS>::generate_actions(&s, &mut mv);
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
        dispatch_size!(size, N, WORDS, {
            let s: State<N, WORDS> = state_from_wire(&w);
            let m: Move<N, WORDS> = serde_json::from_value(mv.clone())
                .map_err(|e| HostError::bad_request(e.to_string()))?;
            if AtariGo::<N, WORDS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut legal = Vec::new();
            AtariGo::<N, WORDS>::generate_actions(&s, &mut legal);
            if !legal.contains(&m) {
                return Err(HostError::bad_request("illegal move"));
            }
            Ok(state_to_value(&AtariGo::<N, WORDS>::apply(s, &m)))
        })
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let w = parse_wire_state(state)?;
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: State<N, WORDS> = state_from_wire(&w);
            let winner = AtariGo::<N, WORDS>::winner(&s);
            serde_json::to_value(GameView {
                turn: player_name(s.turn()).into(),
                cells: (0..N * N)
                    .map(|i| color_at(&s, i).map(|p| player_name(p).to_string()))
                    .collect(),
                winner: winner.map(|p| player_name(p).to_string()),
                terminal: AtariGo::<N, WORDS>::is_terminal(&s),
            })
            .map_err(|e| HostError::internal(e.to_string()))
        })
    }
    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        PRESET_IDS
            .iter()
            .map(|id| AiPresetInfo {
                id: (*id).into(),
                label: (*id).into(),
                description: "".into(),
            })
            .collect()
    }
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let w = parse_wire_state(state)?;
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: State<N, WORDS> = state_from_wire(&w);
            if AtariGo::<N, WORDS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut ai = build_preset::<N, WORDS>(preset)?;
            let action = ai.choose_action(&s);
            let next = AtariGo::<N, WORDS>::apply(s, &action);
            Ok(AiMoveResult {
                mv: serde_json::to_value(action).unwrap(),
                state: state_to_value(&next),
            })
        })
    }
    fn analyze(&self, state: &Value, preset: &str, _: Option<u64>) -> Result<Analysis, HostError> {
        let w = parse_wire_state(state)?;
        let size = size_from_cell_count(w.cells.len())?;
        dispatch_size!(size, N, WORDS, {
            let s: State<N, WORDS> = state_from_wire(&w);
            if AtariGo::<N, WORDS>::is_terminal(&s) {
                return Err(HostError::bad_request("game is over"));
            }
            let mut ai = build_preset::<N, WORDS>(preset)?;
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
        dispatch_size!(size, N, WORDS, {
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
                mcts_tune::build_search::<AtariGo<N, WORDS>>(&cfg, baseline_seed, false, &budget)?;
                mcts_tune::strategy_tune_eval(
                    &params,
                    rounds,
                    seed,
                    false,
                    budget,
                    move || {
                        mcts_tune::build_search::<AtariGo<N, WORDS>>(
                            &cfg,
                            baseline_seed,
                            false,
                            &budget,
                        )
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
                    mcts_tune::SearchBudget {
                        max_iterations,
                        ..Default::default()
                    },
                    build_strong::<N, WORDS>,
                    Default::default(),
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
            .tune_eval(params, 1, Some(0), None, None, None, None)
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }

    #[test]
    fn new_state_supports_every_advertised_size() {
        for &(n, _) in SUPPORTED_SIZES {
            let v = AtarigoAdapter
                .new_state(serde_json::json!({ "size": n }))
                .unwrap_or_else(|e| panic!("new_state({n}) failed: {e}"));
            assert_eq!(v["cells"].as_array().unwrap().len(), n * n);
        }
    }

    #[test]
    fn new_state_rejects_unsupported_size() {
        assert!(AtarigoAdapter
            .new_state(serde_json::json!({ "size": 6 }))
            .is_err());
    }

    #[test]
    fn legal_moves_and_apply_round_trip_at_every_size() {
        for &(n, _) in SUPPORTED_SIZES {
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
