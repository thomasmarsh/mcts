use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_tak::{
    cell_color_at, cell_height, cell_kind, make_cell, Move, Player, State, Tak, CAP, FLAT, WALL,
};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

fn player_name(p: Player) -> &'static str {
    match p {
        Player::White => "White",
        Player::Black => "Black",
    }
}
fn parse_player(name: &str) -> Result<Player, HostError> {
    match name {
        "White" => Ok(Player::White),
        "Black" => Ok(Player::Black),
        _ => Err(HostError::bad_request(format!("invalid player: {name}"))),
    }
}

fn kind_name(k: u8) -> &'static str {
    match k {
        FLAT => "Flat",
        WALL => "Wall",
        CAP => "Cap",
        _ => unreachable!("cell/move kind is always FLAT/WALL/CAP"),
    }
}
fn kind_from_name(s: &str) -> Result<u8, HostError> {
    match s {
        "Flat" => Ok(FLAT),
        "Wall" => Ok(WALL),
        "Cap" => Ok(CAP),
        _ => Err(HostError::bad_request(format!("invalid piece kind: {s}"))),
    }
}

/// Mirrors `Move::dir()`'s 0..4 index (see `games/tak/src/lib.rs`'s file
/// header: 0 = N, 1 = E, 2 = S, 3 = W).
const DIR_NAMES: [&str; 4] = ["North", "East", "South", "West"];
fn direction_name(dir_idx: usize) -> &'static str {
    DIR_NAMES[dir_idx]
}
fn direction_index(name: &str) -> Result<usize, HostError> {
    DIR_NAMES
        .iter()
        .position(|d| *d == name)
        .ok_or_else(|| HostError::bad_request(format!("invalid direction: {name}")))
}

/// One board cell's stack, decoded from the engine's packed `u64` cell word
/// (see `games/tak/src/lib.rs`'s file header for that internal encoding) into
/// a shape a client can use directly. `colors` is bottom-to-top; `None` means
/// an empty cell. Every piece below the top is always flat (walls/capstones
/// can never be covered), so only the top needs a `kind`.
#[derive(Serialize, Deserialize, Clone)]
struct WireStack {
    colors: Vec<String>, // "White" | "Black", bottom to top
    top_kind: String,    // "Flat" | "Wall" | "Cap"
}

fn cell_to_wire(w: u64) -> Option<WireStack> {
    if w == 0 {
        return None;
    }
    let h = cell_height(w);
    let colors = (0..h)
        .map(|j| player_name(player_from_bit(cell_color_at(w, j))).to_string())
        .collect();
    Some(WireStack {
        colors,
        top_kind: kind_name(cell_kind(w)).to_string(),
    })
}

fn wire_to_cell(stack: &Option<WireStack>) -> Result<u64, HostError> {
    let Some(stack) = stack else {
        return Ok(0);
    };
    let h = stack.colors.len() as u32;
    if h == 0 || h > 61 {
        return Err(HostError::bad_request("invalid stack height"));
    }
    let mut colors_bits = 0u64;
    for (j, c) in stack.colors.iter().enumerate() {
        colors_bits |= (parse_player(c)? as u64) << j;
    }
    let kind = kind_from_name(&stack.top_kind)?;
    Ok(make_cell(colors_bits, h, kind))
}

fn player_from_bit(bit: u8) -> Player {
    if bit == 0 {
        Player::White
    } else {
        Player::Black
    }
}

/// Wire shape for a `Move`, replacing the engine's packed `u32` (see
/// `games/tak/src/lib.rs`'s `Move`, which derives `Serialize`/`Deserialize`
/// with no attributes -- that's an internal MCTS-hot-path encoding, not a
/// deliberate wire contract) with a self-describing tagged JSON shape a
/// client can read/build without knowing the bit layout.
#[derive(Serialize, Deserialize)]
#[serde(tag = "tag")]
enum WireMove {
    Place {
        square: usize,
        kind: String, // "Flat" | "Wall" | "Cap"
    },
    Spread {
        square: usize,
        direction: String,    // "North" | "East" | "South" | "West"
        drop_sizes: Vec<u32>, // per-square drop counts, in walk order; sums to the take count
    },
}

fn move_to_wire(m: Move) -> WireMove {
    if m.is_spread() {
        WireMove::Spread {
            square: m.square(),
            direction: direction_name(m.dir()).to_string(),
            drop_sizes: m.drop_sizes(),
        }
    } else {
        WireMove::Place {
            square: m.square(),
            kind: kind_name(m.kind()).to_string(),
        }
    }
}

fn wire_to_move(w: WireMove) -> Result<Move, HostError> {
    match w {
        WireMove::Place { square, kind } => {
            if square >= 64 {
                return Err(HostError::bad_request("square out of range"));
            }
            Ok(Move::place(square, kind_from_name(&kind)?))
        }
        WireMove::Spread {
            square,
            direction,
            drop_sizes,
        } => {
            if square >= 64 {
                return Err(HostError::bad_request("square out of range"));
            }
            if drop_sizes.is_empty() {
                return Err(HostError::bad_request("empty drop_sizes"));
            }
            let dir_idx = direction_index(&direction)?;
            let mut mask = 0u32;
            let mut dropped = 0u32;
            for d in &drop_sizes {
                if *d == 0 {
                    return Err(HostError::bad_request("zero drop size"));
                }
                dropped += d;
                if dropped == 0 || dropped > 8 {
                    return Err(HostError::bad_request("take count out of range"));
                }
                mask |= 1 << (dropped - 1);
            }
            Ok(Move::spread(square, dir_idx, dropped, mask))
        }
    }
}

/// Serialisable snapshot of a Tak state.
#[derive(Serialize, Deserialize)]
struct WireState {
    cells: Vec<Option<WireStack>>, // N*N elements, row-major, row 0 = south edge
    stones: [u8; 2],
    caps: [u8; 2],
    turn: String,
    opening: bool,
}

/// `GameView`'s wire shape: `WireState`'s fields plus the display-only
/// `winner`/`terminal` a renderer needs but a round-tripped `GameState`
/// doesn't (mirrors `games/druid/src/main.rs`'s `GameView`).
#[derive(Serialize)]
struct GameView {
    cells: Vec<Option<WireStack>>,
    stones: [u8; 2],
    caps: [u8; 2],
    turn: String,
    opening: bool,
    winner: Option<String>,
    terminal: bool,
}

fn state_to_value(s: &State<5>) -> Value {
    let n = 5;
    let cells: Vec<Option<WireStack>> = s.cells[..n * n].iter().map(|&w| cell_to_wire(w)).collect();
    serde_json::to_value(WireState {
        cells,
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
    if w.cells.len() != 5 * 5 {
        return Err(HostError::bad_request("wrong cell count for a 5x5 board"));
    }
    let mut s = State::<5> {
        opening: w.opening,
        turn: parse_player(&w.turn)?,
        stones: w.stones,
        caps: w.caps,
        ..Default::default()
    };
    for (i, stack) in w.cells.iter().enumerate() {
        s.set_cell(i, wire_to_cell(stack)?);
    }
    s.hash = s.recompute_hash();
    Ok(s)
}

fn build_easy() -> Box<dyn Search<G = Tak<5>>> {
    Box::new(
        TreeSearch::<Tak<5>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tak/easy")
                .expand_threshold(1)
                .max_iterations(100)
                .q_init(QInit::Infinity),
        ),
    )
}
fn build_strong() -> Box<dyn Search<G = Tak<5>>> {
    Box::new(
        TreeSearch::<Tak<5>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("tak/strong")
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
        build: build_easy,
    },
    PresetEntry {
        id: "strong",
        build: build_strong,
    },
];
struct PresetEntry {
    id: &'static str,
    build: fn() -> Box<dyn Search<G = Tak<5>>>,
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
            .map(|m| serde_json::to_value(move_to_wire(m)).unwrap())
            .collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let wire_move: WireMove = serde_json::from_value(mv.clone())
            .map_err(|e| HostError::bad_request(e.to_string()))?;
        let m = wire_to_move(wire_move)?;
        if Tak::<5>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Tak::<5>::generate_actions(&s, &mut legal);
        if !legal.contains(&m) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Tak::<5>::apply(s, &m)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let n = 5;
        Ok(serde_json::to_value(GameView {
            cells: s.cells[..n * n].iter().map(|&w| cell_to_wire(w)).collect(),
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
        PRESETS
            .iter()
            .map(|p| AiPresetInfo {
                id: p.id.into(),
                label: p.id.into(),
                description: "".into(),
            })
            .collect()
    }
    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found("unknown preset"))?;
        if Tak::<5>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = Tak::<5>::apply(s, &action);
        Ok(AiMoveResult {
            mv: serde_json::to_value(move_to_wire(action)).unwrap(),
            state: state_to_value(&next),
        })
    }
    fn analyze(&self, state: &Value, preset: &str, _: Option<u64>) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found("unknown preset"))?;
        if Tak::<5>::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);
        let suggested = report
            .principal_variation
            .first()
            .map(|a| serde_json::to_value(move_to_wire(*a)).unwrap());
        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: serde_json::to_value(move_to_wire(a.action)).unwrap(),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| serde_json::to_value(move_to_wire(a)).unwrap())
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
        trace_path: Option<std::path::PathBuf>,
    ) -> Result<Value, HostError> {
        // `use_transpositions: true` requires a real `Game::zobrist_hash`
        // override -- Tak has one, so merging transposed nodes during the
        // candidate's search is safe here.
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
            mcts_tune::build_search::<Tak<5>>(&cfg, baseline_seed, true, &budget)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                true,
                budget,
                move || {
                    mcts_tune::build_search::<Tak<5>>(&cfg, baseline_seed, true, &budget)
                        .expect("baseline_config already validated above")
                },
                Default::default(),
                trace_path.as_deref(),
            )?
        } else {
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                true,
                mcts_tune::SearchBudget {
                    max_iterations,
                    ..Default::default()
                },
                build_strong,
                Default::default(),
                trace_path.as_deref(),
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
    run_cli(TakAdapter);
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trips_through_wire_shape() {
        let s = State::<5>::default();
        let v = state_to_value(&s);
        let back = value_to_state(&v).expect("default state round-trips");
        assert_eq!(back, s);
    }

    #[test]
    fn wire_state_decodes_stacks() {
        let mut s = State::<5>::default();
        s.opening = false;
        // A 3-high white-bottom / black-mid / white-top flat stack at square 6.
        // `colors`' bit j is the piece at height j (LSB = bottom); 1 = Black.
        s.set_cell(6, make_cell(0b010, 3, FLAT));
        let v = state_to_value(&s);
        let cells = v["cells"].as_array().expect("cells array");
        let stack = cells[6].as_object().expect("occupied cell is an object");
        let colors: Vec<&str> = stack["colors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap())
            .collect();
        assert_eq!(colors, vec!["White", "Black", "White"]);
        assert_eq!(stack["top_kind"], "Flat");
        assert!(cells[0].is_null());
    }

    #[test]
    fn move_wire_round_trips() {
        let place = Move::place(12, WALL);
        let wire = move_to_wire(place);
        let decoded = wire_to_move(wire).expect("place round-trips");
        assert_eq!(decoded, place);

        let spread = Move::spread(0, 1, 3, 0b110); // take 3, drop (2, 1)
        let wire = move_to_wire(spread);
        match &wire {
            WireMove::Spread {
                square,
                direction,
                drop_sizes,
            } => {
                assert_eq!(*square, 0);
                assert_eq!(direction, "East");
                assert_eq!(drop_sizes, &vec![2, 1]);
            }
            _ => panic!("expected a Spread"),
        }
        let decoded = wire_to_move(wire).expect("spread round-trips");
        assert_eq!(decoded, spread);
    }

    #[test]
    fn view_reports_terminal_and_winner() {
        let mut s = State::<5>::default();
        s.opening = false;
        // White road along the top row (row 0 is the north edge internally
        // since `idx = row * N + col`... just build a straightforward
        // vertical white road down column 0.
        for row in 0..5 {
            s.set_cell(row * 5, make_cell(0, 1, FLAT));
        }
        let v = TakAdapter.view(&state_to_value(&s)).expect("view succeeds");
        assert_eq!(v["terminal"], true);
        assert_eq!(v["winner"], "White");
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
            .tune_eval(params, 1, Some(0), None, None, None, None, None)
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
