use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, BookInfo, GameAdapter, HostError, TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use bitboard::Dyn;
use game_gonnect::book::{self, BookBuildConfig};
use game_gonnect::{Bits, Gonnect, Move, Player, State};
use mcts::game::Game;
use mcts::algorithms::Search;
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
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment). Presets are size-invariant:
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
        PresetTable::load_from_path(&presets_path).expect("games/gonnect/presets.json must parse")
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
    /// Raw `[u64; 6]` words (hex, like `Move`'s capture mask -- see that
    /// type's doc comment for why not plain numbers) of `State::ko_black`/
    /// `ko_white`, not a `cells`-shaped overlay: unlike `black`/`white`,
    /// which can never both cover the same cell, the sentinel "no ko active"
    /// value (`State::new`'s `ones`) sets every cell of *both* boards at
    /// once, so an `Option<Player>`-per-cell encoding (picking one color
    /// when both are set) can't round-trip it.
    ko_black_hex: Vec<String>,
    ko_white_hex: Vec<String>,
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

/// Hex-encodes a board's backing words, low word first -- mirrors `Move`'s
/// own `Serialize` impl (see that type's doc comment for why hex strings,
/// not raw `u64`s: `JSON.parse` on the client silently loses precision past
/// JS's 2^53 safe-integer range for a word with several scattered bits set).
fn bits_to_hex(b: Bits) -> Vec<String> {
    b.words().map(|w| format!("{w:016x}")).collect()
}

fn bits_from_hex(hex: &[String], size: usize) -> Result<Bits, HostError> {
    let mut b = Bits::new(Dyn(size), Dyn(size));
    for (w, s) in hex.iter().enumerate() {
        let mut word =
            u64::from_str_radix(s, 16).map_err(|e| HostError::bad_request(e.to_string()))?;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            word &= word - 1;
            b.set_index(w * 64 + bit);
        }
    }
    Ok(b)
}

fn state_to_value(s: &State) -> Value {
    let n = s.black().rows();
    serde_json::to_value(WireState {
        turn: player_name(s.turn()).into(),
        can_swap: s.can_swap(),
        winner: s.has_winner(),
        cells: (0..n * n)
            .map(|i| color_at(s.black(), s.white(), i).map(|p| player_name(p).to_string()))
            .collect(),
        ko_black_hex: bits_to_hex(s.ko_black()),
        ko_white_hex: bits_to_hex(s.ko_white()),
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
    let ko_black = bits_from_hex(&w.ko_black_hex, size)?;
    let ko_white = bits_from_hex(&w.ko_white_hex, size)?;
    Ok(State::from_parts(
        black,
        white,
        ko_black,
        ko_white,
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
        let (action, search) = mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
            serde_json::to_value(action).expect("Gonnect action always serializes")
        });
        let next = Gonnect::apply(s, &action);
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
        if Gonnect::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let size = s.black().rows();
        let mut ai = self.augmented_preset(size, preset, custom_spec.as_ref())?;
        let (selected_action, search) =
            mcts_tune::choose_action_with_report(&mut *ai, &s, |action| {
                serde_json::to_value(action).expect("Gonnect action always serializes")
            });
        Ok(mcts_tune::legacy_analysis_with_report(
            &*ai,
            &s,
            &selected_action,
            search,
            |action| serde_json::to_value(action).expect("Gonnect action always serializes"),
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
                state_to_value,
                |_, action| {
                    Some(serde_json::to_value(action).expect("Gonnect action always serializes"))
                },
                trace_path.as_deref(),
                trace_game_sequence_start,
                on_game,
            )?
        } else {
            let baseline_id = baseline
                .or_else(|| presets().ai_preset_ids().first().map(|s| s.to_string()))
                .expect("games/gonnect/presets.json must declare at least one preset");
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
                    presets().build::<Gonnect>(&baseline_id, PRESET_SEED).unwrap_or_else(|e| {
                        panic!("games/gonnect/presets.json's {baseline_id:?} preset must build: {e}")
                    })
                },
                initial_state,
                state_to_value,
                |_, action| {
                    Some(serde_json::to_value(action).expect("Gonnect action always serializes"))
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

    fn book(&self) -> Option<BookInfo> {
        Some(BookInfo {
            id: "gonnect/qbf".into(),
            default_rounds: BookBuildConfig::default().rounds,
            game_config: self.default_config(),
            game_config_schema: self.config_schema(),
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

    /// Reproduces the exact shape of a real game session: every move goes
    /// through `GonnectAdapter::apply`'s `Value`-in/`Value`-out contract, the
    /// same JSON round trip a stateless HTTP request makes (`adapter.rs`'s
    /// doc comment: "state flows in as a JSON `Value`... and back out
    /// again"). Plays a standard single-stone ko capture on a 5x5 board, then
    /// asserts the immediate recapture (`(2,2)`, index 12) is excluded from
    /// `legal_moves` on the position *after* that round trip -- catching a
    /// regression where the wire format silently drops `ko_black`/
    /// `ko_white` and the ko rule stops applying across requests, letting
    /// two players (or two AI presets) recapture back and forth forever.
    #[test]
    fn ko_rule_survives_the_json_state_round_trip() {
        let adapter = GonnectAdapter::load();
        let mut state = adapter.new_state(serde_json::json!({ "size": 5 })).unwrap();

        let apply_at = |state: &Value, target: usize| -> Value {
            let moves = adapter.legal_moves(state).unwrap();
            let mv = moves
                .iter()
                .find(|m| m[0].as_u64() == Some(target as u64))
                .unwrap_or_else(|| panic!("index {target} not legal in {moves:?}"));
            adapter.apply(state, mv).unwrap()
        };

        // Black (1,2)=7, White (2,2)=12, Black (2,1)=11, White (3,1)=16,
        // Black (2,3)=13, White (3,3)=18, Black (0,0)=0 (neutral), White
        // (4,2)=22 -- surrounds White's lone stone at 12 on 3 sides (7, 11,
        // 13) and pre-stages White stones on 3 sides of the point Black is
        // about to capture into (16, 18, 22), so that capture leaves Black's
        // new stone in atari -- the ko shape.
        for target in [7, 12, 11, 16, 13, 18, 0, 22] {
            state = apply_at(&state, target);
        }
        assert_eq!(state["winner"], Value::Bool(false));

        // Black plays (3,2)=17, capturing White's stone at 12.
        state = apply_at(&state, 17);
        assert_eq!(
            color_at_json(&state, 12),
            None,
            "White's stone at 12 should have been captured"
        );
        assert_eq!(
            color_at_json(&state, 17),
            Some("Black"),
            "Black's stone should now sit at 17"
        );

        // The immediate recapture at 12 would recreate the position from
        // just before Black's capturing move -- illegal under the ko rule.
        let moves = adapter.legal_moves(&state).unwrap();
        assert!(
            !moves.iter().any(|m| m[0].as_u64() == Some(12)),
            "ko-violating recapture at 12 should not be legal: {moves:?}"
        );
    }

    fn color_at_json(state: &Value, index: usize) -> Option<&str> {
        state["cells"][index].as_str()
    }
}
