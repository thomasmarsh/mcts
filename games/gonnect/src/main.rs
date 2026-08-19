use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, BookInfo, GameAdapter,
    HostError, TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use bitboard::Dyn;
use game_gonnect::book::{self, BookBuildConfig};
use game_gonnect::{Bits, Gonnect, Move, Player, State};
use mcts::game::Game;
use mcts::strategies::Search;
use mcts_tune::presets::PresetTable;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

/// Fixed seed for every `ai_move`/`analyze` search built through
/// [`presets`] -- `GameAdapter::ai_move`/`analyze` take no seed argument, so
/// this is the only seed available to `mcts_tune::presets::PresetTable::build`.
const PRESET_SEED: u64 = 0;

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/gonnect/presets.json` (or the file named by `GONNECT_PRESETS_PATH`),
/// falling back to the compiled-in defaults only if that path is missing
/// (see `PresetTable::load`'s doc comment). Presets are size-invariant:
/// `build_easy`/`build_strong` never varied by board size, only by the
/// starting `State`'s own runtime dims.
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("GONNECT_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load(include_str!("../presets.json"), Some(&presets_path))
            .expect("games/gonnect/presets.json must parse")
    })
}

/// Board sizes this binary serves, 3x3 through 19x19 -- a runtime `size`
/// field on `State` (see `game_gonnect::Bits`, `Board<[u64; 6], Dyn, Dyn>`)
/// rather than a distinct compiled type per size, so this is just a bounds
/// check now, not a dispatch table. 13x13 is Gonnect's traditional board
/// size and stays the default (see `game_gonnect::DEFAULT_SIZE`).
const MIN_SIZE: usize = 3;
const MAX_SIZE: usize = 19;
const DEFAULT_SIZE: usize = game_gonnect::DEFAULT_SIZE;

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
    ko_cells: Vec<Option<String>>,
    turn: String,
    can_swap: bool,
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

fn color_at(black: Bits, white: Bits, index: usize) -> Option<Player> {
    if black.get_index(index) {
        Some(Player::Black)
    } else if white.get_index(index) {
        Some(Player::White)
    } else {
        None
    }
}

fn state_to_value(s: &State) -> Value {
    let n = s.black().rows();
    serde_json::to_value(WireState {
        turn: player_name(s.turn()).into(),
        can_swap: true,
        winner: s.has_winner(),
        cells: (0..n * n)
            .map(|i| color_at(s.black(), s.white(), i).map(|p| player_name(p).to_string()))
            .collect(),
        // The ko boards aren't exposed on `State`, so the wire format keeps
        // its historical shape (mirroring `black`/`white`) without actually
        // round-tripping ko state -- see the comment in `state_from_wire`.
        ko_cells: (0..n * n)
            .map(|i| color_at(s.black(), s.white(), i).map(|p| player_name(p).to_string()))
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
    // The wire format doesn't carry ko state (see `state_to_value`), so a
    // state round-tripped through the host adapter always looks
    // "just captured nothing" to the ko rule -- ko violations a move earlier
    // in the same client session won't be caught after a round trip. This
    // matches the previous (fixed-size) adapter's behaviour, which had the
    // same gap.
    let ones = !Bits::new(Dyn(size), Dyn(size));
    Ok(State::from_parts(
        black,
        white,
        ones,
        ones,
        parse_player(&w.turn),
        w.can_swap,
        w.winner,
    ))
}

/// Path convention for a size-`N` opening book, matching what `book build`
/// (`examples/build_book.rs`'s default `--out`) writes.
fn book_path(n: usize) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("books/gonnect-{n}.json"))
}

/// Holds each supported board size's opening book (if one has been built --
/// see `book_path`), loaded once at process startup rather than per
/// request. `run_host`'s JSONL loop reuses a single `GameAdapter` instance
/// for the life of the subprocess, so this is exactly one load per game
/// session, not one per move. Only the sizes an opening book was ever built
/// for (9, 13, 19) get an entry; any other size in `MIN_SIZE..=MAX_SIZE`
/// simply has no book to consult.
struct GonnectAdapter {
    book_9: Option<book::BookIndex>,
    book_13: Option<book::BookIndex>,
    book_19: Option<book::BookIndex>,
}

impl GonnectAdapter {
    fn load() -> Self {
        Self {
            book_9: book::BookIndex::load(&book_path(9), 9),
            book_13: book::BookIndex::load(&book_path(13), 13),
            book_19: book::BookIndex::load(&book_path(19), 19),
        }
    }

    fn book_for_size(&self, size: usize) -> Option<&book::BookIndex> {
        match size {
            9 => self.book_9.as_ref(),
            13 => self.book_13.as_ref(),
            19 => self.book_19.as_ref(),
            _ => None,
        }
    }

    /// `build_strategy`, wrapped with a `book::BookAugmented` layer when
    /// this size has a loaded opening book -- only for `"strong"`: the book
    /// was self-play-generated at production strength, so it's a fit for
    /// strengthening the strong preset, not for the easy one's purpose of
    /// being beatable.
    fn augmented_preset(
        &self,
        size: usize,
        preset: &str,
        custom: Option<&mcts_tune::presets::CustomStrategySpec>,
    ) -> Result<Box<dyn Search<G = Gonnect> + '_>, HostError> {
        let inner =
            mcts_tune::presets::build_strategy::<Gonnect>(presets(), preset, custom, PRESET_SEED)?;
        Ok(match (preset, self.book_for_size(size)) {
            ("strong", Some(book)) => Box::new(book::BookAugmented::new(inner, book)),
            _ => inner,
        })
    }
}

impl GameAdapter for GonnectAdapter {
    fn kind(&self) -> &'static str {
        "gonnect"
    }
    fn label(&self) -> &'static str {
        "Gonnect"
    }
    fn description(&self) -> &'static str {
        "A Go-like connection game where connecting opposite edges wins."
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
        if !Gonnect::is_terminal(&s) {
            Gonnect::generate_actions(&s, &mut mv);
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
        if Gonnect::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Gonnect::generate_actions(&s, &mut legal);
        if !legal.contains(&m) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Gonnect::apply(s, &m)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let w = parse_wire_state(state)?;
        let s = state_from_wire(&w)?;
        let winner = Gonnect::winner(&s);
        serde_json::to_value(GameView {
            turn: player_name(s.turn()).into(),
            cells: (0..s.black().len())
                .map(|i| color_at(s.black(), s.white(), i).map(|p| player_name(p).to_string()))
                .collect(),
            winner: winner.map(|p| player_name(p).to_string()),
            terminal: Gonnect::is_terminal(&s),
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
        if Gonnect::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let size = s.black().rows();
        let mut ai = self.augmented_preset(size, preset, custom_spec.as_ref())?;
        let action = ai.choose_action(&s);
        let next = Gonnect::apply(s, &action);
        Ok(AiMoveResult {
            mv: serde_json::to_value(action).unwrap(),
            state: state_to_value(&next),
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
        if Gonnect::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let size = s.black().rows();
        let mut ai = self.augmented_preset(size, preset, custom_spec.as_ref())?;
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
                check_size(cfg.size)?
            }
            None => DEFAULT_SIZE,
        };
        let initial_state = State::new(size);
        // Gonnect's `Game::zobrist_hash` is the default constant `0`, so
        // transpositions must stay off -- see `mcts-tune`'s
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
            mcts_tune::build_search::<Gonnect>(&cfg, baseline_seed, false, &budget)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                false,
                budget,
                move || {
                    mcts_tune::build_search::<Gonnect>(&cfg, baseline_seed, false, &budget)
                        .expect("baseline_config already validated above")
                },
                initial_state,
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
                        .build::<Gonnect>("strong", PRESET_SEED)
                        .expect("\"strong\" preset must be buildable")
                },
                initial_state,
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

    fn book(&self) -> Option<BookInfo> {
        Some(BookInfo {
            id: "gonnect/qbf".into(),
            default_rounds: BookBuildConfig::default().rounds,
            game_config: self.default_config(),
        })
    }

    fn book_build(
        &self,
        rounds: u32,
        seed: Option<u64>,
        game_config: Option<Value>,
    ) -> Result<Value, HostError> {
        let size = match game_config {
            Some(cfg) => {
                let cfg: NewGameConfig = serde_json::from_value(cfg)
                    .map_err(|e| HostError::bad_request(format!("invalid game_config: {e}")))?;
                check_size(cfg.size)?
            }
            None => DEFAULT_SIZE,
        };
        let config = BookBuildConfig {
            rounds,
            seed: seed.unwrap_or(0),
            ..Default::default()
        };
        let built = book::build(size, &config, None, |_round, _plies, _utilities| {});
        serde_json::to_value(built).map_err(|e| HostError::internal(e.to_string()))
    }
}

fn main() {
    run_cli(GonnectAdapter::load());
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
        let result = GonnectAdapter::load()
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
            let v = GonnectAdapter::load()
                .new_state(serde_json::json!({ "size": n }))
                .unwrap_or_else(|e| panic!("new_state({n}) failed: {e}"));
            assert_eq!(v["cells"].as_array().unwrap().len(), n * n);
        }
    }

    #[test]
    fn new_state_rejects_unsupported_size() {
        assert!(GonnectAdapter::load()
            .new_state(serde_json::json!({ "size": 20 }))
            .is_err());
    }

    #[test]
    fn legal_moves_and_apply_round_trip_at_every_size() {
        for n in MIN_SIZE..=MAX_SIZE {
            let state = GonnectAdapter::load()
                .new_state(serde_json::json!({ "size": n }))
                .unwrap();
            let moves = GonnectAdapter::load().legal_moves(&state).unwrap();
            assert!(
                !moves.is_empty(),
                "size {n} should have legal moves from the empty board"
            );
            let next = GonnectAdapter::load().apply(&state, &moves[0]).unwrap();
            assert_eq!(next["cells"].as_array().unwrap().len(), n * n);
        }
    }
}
