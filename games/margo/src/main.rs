use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_margo::{Action, Margo, Player, State, DEFAULT_N, MAX_N, MIN_N};
use mcts::game::Game;
use mcts_tune::presets::PresetTable;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

/// Fixed seed for every `ai_move`/`analyze` search built through
/// [`presets`] -- `GameAdapter::ai_move`/`analyze` take no seed argument, so
/// this is the only seed available to `mcts_tune::presets::PresetTable::build`.
const PRESET_SEED: u64 = 0;

/// The parsed `easy`/`strong` preset table -- loaded at runtime from
/// `games/margo/presets.json` (or the file named by `MARGO_PRESETS_PATH`),
/// read fresh from disk at every startup -- not embedded via `include_str!`,
/// so editing it never triggers a rebuild (see `PresetTable::load_from_path`'s
/// doc comment). Presets are size-invariant, mirroring `games/gonnect`.
fn presets() -> &'static PresetTable {
    static PRESETS: OnceLock<PresetTable> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let presets_path = env::var("MARGO_PRESETS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"))
            });
        PresetTable::load_from_path(&presets_path).expect("games/margo/presets.json must parse")
    })
}

fn check_size(n: usize) -> Result<usize, HostError> {
    if (MIN_N..=MAX_N).contains(&n) {
        Ok(n)
    } else {
        Err(HostError::bad_request(format!(
            "unsupported board size {n} (supported: {MIN_N}..={MAX_N})"
        )))
    }
}

/// Wire encoding of a `State`, built entirely from `game_margo::State`'s
/// public flat-index accessors/`from_parts` -- see that type's doc comments
/// for why the crate carries no other public constructor: this adapter, and
/// only this adapter, is trusted to round-trip whatever it itself emitted.
#[derive(Serialize, Deserialize)]
struct WireState {
    n: u8,
    occupied: Vec<usize>,
    black: Vec<usize>,
    zombie: Vec<usize>,
    previous: Option<(Vec<usize>, Vec<usize>)>,
    turn: String,
    can_swap: bool,
}

/// One occupied cell's display data for [`GameView`] -- a renderer needs the
/// occupying colour and whether it's a zombie (rendered distinctly: still on
/// the board, permanently excluded from connectivity) but has no use for
/// `State`'s internal buried/visible bookkeeping, which is derived fresh
/// from `occupied`/`zombie` on every move rather than stored.
#[derive(Serialize)]
struct CellView {
    piece: &'static str,
    zombie: bool,
}

#[derive(Serialize)]
struct GameView {
    n: u8,
    /// One entry per flat index (`0..total_cells(n)`, see
    /// `pyramid::Pyramid::to_coord` for how a renderer turns an index back
    /// into `(col, row, level)`), `None` for an empty cell.
    cells: Vec<Option<CellView>>,
    turn: String,
    can_swap: bool,
    winner: Option<String>,
    terminal: bool,
}

#[derive(Deserialize)]
struct NewGameConfig {
    #[serde(default = "default_n")]
    n: usize,
}

fn default_n() -> usize {
    DEFAULT_N
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

fn state_to_value(s: &State) -> Value {
    serde_json::to_value(WireState {
        n: s.n() as u8,
        occupied: s.occupied_indices(),
        black: s.black_indices(),
        zombie: s.zombie_indices(),
        previous: s.previous_indices(),
        turn: player_name(s.turn()).into(),
        can_swap: s.swap_window_open(),
    })
    .expect("")
}

fn state_from_value(v: &Value) -> Result<State, HostError> {
    let w: WireState =
        serde_json::from_value(v.clone()).map_err(|e| HostError::bad_request(e.to_string()))?;
    let n = check_size(w.n as usize)?;
    let previous = w
        .previous
        .as_ref()
        .map(|(o, b)| (o.as_slice(), b.as_slice()));
    Ok(State::from_parts(
        n,
        &w.occupied,
        &w.black,
        &w.zombie,
        previous,
        parse_player(&w.turn),
        w.can_swap,
    ))
}

fn view_of(s: &State) -> GameView {
    let total = s.total_cells();
    let black: std::collections::HashSet<usize> = s.black_indices().into_iter().collect();
    let zombie: std::collections::HashSet<usize> = s.zombie_indices().into_iter().collect();
    let cells = (0..total)
        .map(|i| {
            if !s.is_occupied(i) {
                None
            } else {
                Some(CellView {
                    piece: player_name(if black.contains(&i) {
                        Player::Black
                    } else {
                        Player::White
                    }),
                    zombie: zombie.contains(&i),
                })
            }
        })
        .collect();
    let terminal = Margo::is_terminal(s);
    GameView {
        n: s.n() as u8,
        cells,
        turn: player_name(s.turn()).into(),
        can_swap: s.can_swap(),
        // `Margo::winner` is a piece-count comparison with no "game still in
        // progress" state of its own -- it always names whoever currently
        // has more pieces, which is meaningful only once the game has
        // actually ended. Gating on `terminal` here keeps `GameView`'s
        // `winner` field meaning "who won", not "who's currently ahead".
        winner: terminal
            .then(|| Margo::winner(s).map(|p| player_name(p).to_string()))
            .flatten(),
        terminal,
    }
}

struct MargoAdapter;

impl GameAdapter for MargoAdapter {
    fn kind(&self) -> &'static str {
        "margo"
    }
    fn label(&self) -> &'static str {
        "Margo"
    }
    fn description(&self) -> &'static str {
        "A Go-like connection/capture game played by stacking marbles into a pyramid; whoever has more pieces on the board when the mover is stuck wins."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({ "n": DEFAULT_N })
    }
    fn new_state(&self, config: Value) -> Result<Value, HostError> {
        let config: NewGameConfig = serde_json::from_value(config)
            .map_err(|e| HostError::bad_request(format!("invalid config: {e}")))?;
        let n = check_size(config.n)?;
        Ok(state_to_value(&State::new(n)))
    }
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = state_from_value(state)?;
        let mut actions = Vec::new();
        if !Margo::is_terminal(&s) {
            Margo::generate_actions(&s, &mut actions);
        }
        Ok(actions
            .into_iter()
            .map(|a| serde_json::to_value(a).unwrap())
            .collect())
    }
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = state_from_value(state)?;
        let action: Action = serde_json::from_value(mv.clone())
            .map_err(|e| HostError::bad_request(e.to_string()))?;
        if Margo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Margo::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Margo::apply(s, &action)))
    }
    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = state_from_value(state)?;
        serde_json::to_value(view_of(&s)).map_err(|e| HostError::internal(e.to_string()))
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
        let s = state_from_value(state)?;
        if Margo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Margo>(
            presets(),
            preset,
            custom_spec.as_ref(),
            PRESET_SEED,
        )?;
        let action = ai.choose_action(&s);
        let next = Margo::apply(s, &action);
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
        let s = state_from_value(state)?;
        if Margo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = mcts_tune::presets::build_strategy::<Margo>(
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
        on_game: &mut dyn FnMut(game_host::ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        let n = match game_config {
            Some(cfg) => {
                let cfg: NewGameConfig = serde_json::from_value(cfg)
                    .map_err(|e| HostError::bad_request(format!("invalid game_config: {e}")))?;
                check_size(cfg.n)?
            }
            None => DEFAULT_N,
        };
        let initial_state = State::new(n);
        let budget = mcts_tune::SearchBudget {
            max_iterations,
            max_time: max_time_ms.map(std::time::Duration::from_millis),
            ..Default::default()
        };
        let outcome =
            if let Some(cfg) = baseline_config {
                let baseline_seed = seed.unwrap_or(0);
                mcts_tune::build_search::<Margo>(&cfg, baseline_seed, false, &budget)?;
                mcts_tune::strategy_tune_eval(
                    &params,
                    rounds,
                    seed,
                    false,
                    budget,
                    move || {
                        mcts_tune::build_search::<Margo>(&cfg, baseline_seed, false, &budget)
                            .expect("baseline_config already validated above")
                    },
                    initial_state,
                    trace_path.as_deref(),
                    on_game,
                )?
            } else {
                let baseline_id = baseline
                    .or_else(|| presets().ai_preset_ids().first().map(|s| s.to_string()))
                    .expect("games/margo/presets.json must declare at least one preset");
                mcts_tune::strategy_tune_eval(
                    &params,
                    rounds,
                    seed,
                    false,
                    budget,
                    move || {
                        presets().build::<Margo>(&baseline_id, PRESET_SEED).unwrap_or_else(|e| {
                        panic!("games/margo/presets.json's {baseline_id:?} preset must build: {e}")
                    })
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
}

fn main() {
    run_cli(MargoAdapter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_supports_every_advertised_size() {
        for n in MIN_N..=MAX_N {
            let v = MargoAdapter
                .new_state(serde_json::json!({ "n": n }))
                .unwrap_or_else(|e| panic!("new_state({n}) failed: {e}"));
            assert_eq!(v["n"].as_u64(), Some(n as u64));
        }
    }

    #[test]
    fn new_state_rejects_unsupported_size() {
        assert!(MargoAdapter
            .new_state(serde_json::json!({ "n": MAX_N + 1 }))
            .is_err());
    }

    #[test]
    fn legal_moves_and_apply_round_trip() {
        let state = MargoAdapter
            .new_state(serde_json::json!({ "n": MIN_N }))
            .unwrap();
        let moves = MargoAdapter.legal_moves(&state).unwrap();
        assert!(!moves.is_empty());
        let next = MargoAdapter.apply(&state, &moves[0]).unwrap();
        assert_eq!(next["n"].as_u64(), Some(MIN_N as u64));
    }

    /// `Margo::winner` always names whoever currently has more pieces, with
    /// no "undecided" state of its own -- so `view`'s `winner` field must
    /// stay `null` before the game actually ends even though one side may
    /// already be ahead, or a client would read an ordinary mid-game lead
    /// as a finished result.
    #[test]
    fn view_winner_is_null_before_the_game_is_actually_over() {
        let state = MargoAdapter
            .new_state(serde_json::json!({ "n": MIN_N }))
            .unwrap();
        let moves = MargoAdapter.legal_moves(&state).unwrap();
        let next = MargoAdapter.apply(&state, &moves[0]).unwrap();
        let view = MargoAdapter.view(&next).unwrap();
        assert_eq!(view["terminal"], Value::Bool(false));
        assert_eq!(view["winner"], Value::Null);
    }

    /// Plays several plies (through the adapter's own JSON `Value`-in/-out
    /// contract, the same round trip a stateless HTTP request makes) and
    /// checks `view` stays internally consistent: exactly as many occupied
    /// cells as `occupied_indices`/`black_indices` produced, and swap only
    /// ever offered as Black's first reply -- catching a regression where
    /// the wire format silently drops `zombie`/`previous` state across
    /// requests.
    #[test]
    fn view_and_state_stay_consistent_across_json_round_trips() {
        let mut state = MargoAdapter
            .new_state(serde_json::json!({ "n": MIN_N }))
            .unwrap();
        for ply in 0..10 {
            let moves = MargoAdapter.legal_moves(&state).unwrap();
            if moves.is_empty() {
                break;
            }
            let is_swap = |m: &Value| m.as_str() == Some("Swap");
            if ply == 1 {
                assert!(
                    moves.iter().any(is_swap),
                    "swap must be offered as Black's first reply, got {moves:?}"
                );
            } else {
                assert!(!moves.iter().any(is_swap));
            }
            state = MargoAdapter.apply(&state, &moves[0]).unwrap();
            let view = MargoAdapter.view(&state).unwrap();
            let occupied_in_state = state["occupied"].as_array().unwrap().len();
            let occupied_in_view = view["cells"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| !c.is_null())
                .count();
            assert_eq!(occupied_in_state, occupied_in_view);
        }
    }
}
