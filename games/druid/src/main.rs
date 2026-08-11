//! Standalone Druid game binary that speaks the JSON-line subprocess
//! protocol on stdin/stdout.
//!
//! Built by `cargo build -p game-druid` and used by the server/bench crates
//! via `game_host::SubprocessAdapter`.
//!
//! This binary embeds the full Druid adapter tier -- the `EngineCache`,
//! the four time-budgeted AI presets (easy/medium/strong/master), and the
//! linear-sub-action `ai_move` loop -- so the server no longer needs to
//! compile any Druid-specific code. Preset scalar knobs are hardcoded here
//! (the server's `druid-presets.yaml` used to live at `server/config/`; the
//! binary keeps the same four presets with the same defaults).

use std::sync::Mutex;
use std::time::Duration;

use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

use game_druid::{
    apply_placed, Druid, HashedState, Move, Orientation, Piece, PieceKind, PlacedPiece, Player,
    Size, Square, State,
};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

// ---------------------------------------------------------------------------
// Preset knobs
// ---------------------------------------------------------------------------

/// Scalar knobs for one AI preset. `select_c`, `epsilon`, and
/// `backoff_threshold` may be unused by a preset's strategy shape (None).
#[derive(Debug, Clone, Copy)]
struct PresetConfig {
    label: &'static str,
    description: &'static str,
    time_budget_ms: u64,
    select_c: Option<f64>,
    num_threads: usize,
    epsilon: Option<f64>,
    backoff_threshold: Option<u32>,
}

/// Available CPU cores for the auto-thread-count presets.
fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

const PRESET_CONFIGS: [(&str, PresetConfig); 4] = [
    (
        "easy",
        PresetConfig {
            label: "Easy",
            description: "Plain UCB1 with random playouts and MCTS-Solver for tactical sharpness, ~1s per move.",
            time_budget_ms: 1000,
            select_c: Some(1.414),
            num_threads: 1,
            epsilon: None,
            backoff_threshold: None,
        },
    ),
    (
        "medium",
        PresetConfig {
            label: "Medium",
            description: "UCB1 with MAST-biased playouts and MCTS-Solver for tactical sharpness, ~2s per move.",
            time_budget_ms: 2000,
            select_c: Some(1.625),
            num_threads: 1,
            epsilon: Some(0.1),
            backoff_threshold: None,
        },
    ),
    (
        "strong",
        PresetConfig {
            label: "Strong",
            description: "N-gram-guided (NST) decisive-move search with MCTS-Solver for tactical \
                 sharpness, ~3s per move, searching one shared tree across all available CPU cores.",
            time_budget_ms: 3000,
            select_c: Some(1.414),
            num_threads: 0, // 0 = Auto (all cores)
            epsilon: Some(0.3),
            backoff_threshold: Some(5),
        },
    ),
    (
        "master",
        PresetConfig {
            label: "Master",
            description: "Same search as Strong, parallelized the same way, with a longer ~8s \
                 thinking budget.",
            time_budget_ms: 8000,
            select_c: Some(1.414),
            num_threads: 0, // 0 = Auto (all cores)
            epsilon: Some(0.3),
            backoff_threshold: Some(5),
        },
    ),
];

fn preset_cfg(id: &str) -> Option<&'static PresetConfig> {
    PRESET_CONFIGS
        .iter()
        .find(|(name, _)| *name == id)
        .map(|(_, c)| c)
}

// ---------------------------------------------------------------------------
// Strategy-shape type aliases
// ---------------------------------------------------------------------------

// `Strong`/`Master`'s strategy shape: `Ucb1` select (no RAVE/GRAVE) +
// `DecisiveMove<EpsilonGreedy<Nst>>` simulate.
type Ucb1DmNst = strategy::Compose<
    select::Ucb1,
    simulate::DecisiveMove<Druid, simulate::EpsilonGreedy<Druid, simulate::Nst>>,
>;

/// Build a fresh `TreeSearch` for `preset` with the given time budget.
fn build_ai(preset: &str, budget: Duration, cfg: &PresetConfig) -> Box<dyn Search<G = Druid>> {
    let threads = if cfg.num_threads == 0 {
        ai_thread_count()
    } else {
        cfg.num_threads
    };
    match preset {
        "easy" => Box::new(
            TreeSearch::<Druid, strategy::Ucb1>::new().config(
                SearchConfig::new()
                    .name("ai/easy")
                    .expand_threshold(1)
                    .use_transpositions(true)
                    .use_mcts_solver(true)
                    .reuse_tree(true)
                    .q_init(QInit::Infinity)
                    .max_time(budget)
                    .select(select::Ucb1::with_c(cfg.select_c.unwrap_or(1.414))),
            ),
        ),
        "medium" => Box::new(
            TreeSearch::<Druid, strategy::Ucb1Mast>::new().config(
                SearchConfig::new()
                    .name("ai/medium")
                    .expand_threshold(1)
                    .use_transpositions(true)
                    .use_mcts_solver(true)
                    .reuse_tree(true)
                    .q_init(QInit::Infinity)
                    .max_time(budget)
                    .select(select::Ucb1::with_c(cfg.select_c.unwrap_or(1.625)))
                    .simulate(simulate::EpsilonGreedy::with_epsilon(
                        cfg.epsilon.unwrap_or(0.1),
                    )),
            ),
        ),
        "strong" | "master" => Box::new(
            TreeSearch::<Druid, Ucb1DmNst>::new().config(
                SearchConfig::new()
                    .name(if preset == "strong" {
                        "ai/strong"
                    } else {
                        "ai/master"
                    })
                    .expand_threshold(1)
                    .use_transpositions(true)
                    .use_mcts_solver(true)
                    .reuse_tree(true)
                    .q_init(QInit::Infinity)
                    .max_time(budget)
                    .num_tree_threads(threads)
                    .simulate(
                        simulate::DecisiveMove::new().inner(
                            simulate::EpsilonGreedy::default()
                                .epsilon(cfg.epsilon.unwrap_or(0.3))
                                .inner(
                                    simulate::Nst::new()
                                        .backoff_threshold(cfg.backoff_threshold.unwrap_or(5)),
                                ),
                        ),
                    ),
            ),
        ),
        _ => unreachable!("validated preset id"),
    }
}

// ---------------------------------------------------------------------------
// Engine cache
// ---------------------------------------------------------------------------

type CacheEntry = (&'static str, u64, Box<dyn Search<G = Druid>>);

/// A small LRU cache of `(preset, state-hash) -> search`, so repeated
/// `ai_move`/`analyze` calls on the same position (e.g. a long UCT ponder
/// kept warm across requests) can carry forward the explored tree instead of
/// restarting. Keys are evicted oldest-first at `capacity`.
struct EngineCache {
    capacity: usize,
    entries: Mutex<Vec<CacheEntry>>,
}

impl EngineCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Mutex::new(Vec::with_capacity(capacity)),
        }
    }

    fn take(&self, preset: &'static str, hash: u64) -> Option<Box<dyn Search<G = Druid>>> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pos = entries
            .iter()
            .position(|(p, h, _)| *p == preset && *h == hash)?;
        Some(entries.remove(pos).2)
    }

    fn put(&self, preset: &'static str, hash: u64, engine: Box<dyn Search<G = Druid>>) {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.retain(|(p, h, _)| !(*p == preset && *h == hash));
        entries.insert(0, (preset, hash, engine));
        entries.truncate(self.capacity);
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct NewGameConfig {
    size: Size,
}

#[derive(Serialize)]
struct GameView<'a> {
    size: Size,
    player: Player,
    board: &'a [Square],
    hand_black: &'a game_druid::Hand,
    hand_white: &'a game_druid::Hand,
    winner: Option<Player>,
    terminal: bool,
}

// ---------------------------------------------------------------------------
// State helpers
// ---------------------------------------------------------------------------

fn state_to_value(state: &HashedState) -> Value {
    serde_json::to_value(state.state()).expect("State always serializes")
}

/// Deserializes a client-supplied state `Value` back into a `HashedState`.
/// Only `State` round-trips over the wire (the Zobrist hash/Connectivity/
/// MoveCache in `HashedState` are pure caches derived from it), so this
/// rebuilds them via `HashedState::from_state`.
fn value_to_state(state: &Value) -> Result<HashedState, HostError> {
    let state: State = serde_json::from_value(state.clone())
        .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
    if !state.size.is_supported() {
        return Err(HostError::bad_request(format!(
            "unsupported board size {}x{}",
            state.size.w, state.size.h
        )));
    }
    if state.board.len() != (state.size.w as usize) * (state.size.h as usize) {
        return Err(HostError::bad_request(
            "state board length doesn't match its size",
        ));
    }
    Ok(HashedState::from_state(state))
}

// ---------------------------------------------------------------------------
// Budget clamping
// ---------------------------------------------------------------------------

const MIN_ANALYZE_BUDGET_MS: u64 = 50;
const MAX_ANALYZE_BUDGET_MS: u64 = 20_000;

fn clamp_budget_ms(budget_ms: u64) -> u64 {
    budget_ms.clamp(MIN_ANALYZE_BUDGET_MS, MAX_ANALYZE_BUDGET_MS)
}

// ---------------------------------------------------------------------------
// DruidAdapter
// ---------------------------------------------------------------------------

struct DruidAdapter {
    cache: EngineCache,
}

impl Default for DruidAdapter {
    fn default() -> Self {
        Self {
            cache: EngineCache::new(8),
        }
    }
}

impl GameAdapter for DruidAdapter {
    fn kind(&self) -> &'static str {
        "druid"
    }

    fn label(&self) -> &'static str {
        "Druid"
    }

    fn description(&self) -> &'static str {
        "A connection game played with stackable sarsen and lintel pieces, designed by Cameron Browne."
    }

    fn default_config(&self) -> Value {
        serde_json::json!({ "size": game_druid::DEFAULT_SIZE })
    }

    fn new_state(&self, config: Value) -> Result<Value, HostError> {
        let config: NewGameConfig = serde_json::from_value(config)
            .map_err(|e| HostError::bad_request(format!("invalid config: {e}")))?;
        if !config.size.is_supported() {
            return Err(HostError::bad_request(format!(
                "unsupported board size {}x{}",
                config.size.w, config.size.h
            )));
        }
        Ok(state_to_value(&HashedState::new(config.size)))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let state = value_to_state(state)?;
        let mut moves = Vec::new();
        if !Druid::is_terminal(&state) {
            state.state().moves(&mut moves);
        }
        Ok(moves
            .into_iter()
            .map(|m| serde_json::to_value(m).expect("PlacedPiece always serializes"))
            .collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let state = value_to_state(state)?;
        let mv: PlacedPiece = serde_json::from_value(mv.clone())
            .map_err(|e| HostError::bad_request(format!("invalid move: {e}")))?;

        if Druid::is_terminal(&state) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        state.state().moves(&mut legal);
        if !legal.contains(&mv) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&apply_placed(state, mv)))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let state = value_to_state(state)?;
        let s = state.state();
        Ok(serde_json::to_value(GameView {
            size: s.size,
            player: s.player,
            board: &s.board,
            hand_black: &s.hand_black,
            hand_white: &s.hand_white,
            winner: Druid::winner(&state),
            terminal: Druid::is_terminal(&state),
        })
        .expect("GameView always serializes"))
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        PRESET_CONFIGS
            .iter()
            .map(|(id, c)| AiPresetInfo {
                id: (*id).to_string(),
                label: c.label.to_string(),
                description: c.description.to_string(),
            })
            .collect()
    }

    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let state = value_to_state(state)?;
        let cfg = preset_cfg(preset)
            .ok_or_else(|| HostError::bad_request(format!("unknown preset {preset:?}")))?;
        if Druid::is_terminal(&state) {
            return Err(HostError::bad_request("game is over"));
        }
        let static_preset: &'static str = PRESET_CONFIGS
            .iter()
            .find(|(id, _)| *id == preset)
            .unwrap()
            .0;
        let hash = Druid::zobrist_hash(&state);
        let mut ai = self.cache.take(static_preset, hash).unwrap_or_else(|| {
            build_ai(
                static_preset,
                Duration::from_millis(cfg.time_budget_ms),
                cfg,
            )
        });

        let mut chosen_kind: Option<PieceKind> = None;
        let mut chosen_orientation: Option<Orientation> = None;
        let mut ai_state = state.clone();

        loop {
            let mv = ai.choose_action(&ai_state);
            ai_state = Druid::apply(ai_state, &mv);
            match mv {
                Move::Piece(kind) => chosen_kind = Some(kind),
                Move::Orientation(o) => chosen_orientation = Some(o),
                Move::Cell(idx) => {
                    let piece = match chosen_kind {
                        Some(PieceKind::Sarsen) => Piece::Sarsen,
                        Some(PieceKind::Lintel) => {
                            Piece::Lintel(chosen_orientation.unwrap_or(Orientation::Horizontal))
                        }
                        _ => unreachable!("Cell action without prior Piece"),
                    };
                    let result = PlacedPiece(piece, idx);
                    self.cache.put(static_preset, hash, ai);
                    return Ok(AiMoveResult {
                        mv: serde_json::to_value(result).expect("PlacedPiece always serializes"),
                        state: state_to_value(&ai_state),
                    });
                }
            }
        }
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let state = value_to_state(state)?;
        let cfg = preset_cfg(preset)
            .ok_or_else(|| HostError::bad_request(format!("unknown preset {preset:?}")))?;
        if Druid::is_terminal(&state) {
            return Err(HostError::bad_request("game is over"));
        }
        let static_preset: &'static str = PRESET_CONFIGS
            .iter()
            .find(|(id, _)| *id == preset)
            .unwrap()
            .0;
        let hash = Druid::zobrist_hash(&state);

        let mut ai = match budget_ms {
            Some(ms) => build_ai(
                static_preset,
                Duration::from_millis(clamp_budget_ms(ms)),
                cfg,
            ),
            None => self.cache.take(static_preset, hash).unwrap_or_else(|| {
                build_ai(
                    static_preset,
                    Duration::from_millis(cfg.time_budget_ms),
                    cfg,
                )
            }),
        };

        let _ = ai.choose_action(&state);
        let report = ai.root_report(&state);

        if budget_ms.is_none() {
            self.cache.put(static_preset, hash, ai);
        }

        let suggested_move = report
            .principal_variation
            .first()
            .map(|a| serde_json::to_value(*a).expect("Move always serializes"));

        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: serde_json::to_value(a.action).expect("Move always serializes"),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .into_iter()
                .map(|a| serde_json::to_value(a).expect("Move always serializes"))
                .collect(),
            total_visits: report.total_visits,
            suggested_move,
        })
    }

    fn tuner(&self) -> Option<TunerInfo> {
        Some(mcts_tune::strategy_tuner_info("strong", TUNE_EVAL_ROUNDS))
    }

    fn tune_eval(&self, params: Value, rounds: u32, seed: Option<u64>) -> Result<Value, HostError> {
        let cfg = preset_cfg("strong").expect("strong preset is always registered");
        // `use_transpositions: true` requires a real `Game::zobrist_hash`
        // override -- Druid has one, so merging transposed nodes during the
        // candidate's search is safe here.
        let outcome = mcts_tune::strategy_tune_eval(&params, rounds, seed, true, || {
            build_ai("strong", Duration::from_millis(cfg.time_budget_ms), cfg)
        })?;
        Ok(serde_json::json!({
            "cost": outcome.cost,
            "wins": outcome.wins,
            "losses": outcome.losses,
            "draws": outcome.draws,
        }))
    }
}

fn main() {
    run_cli(DruidAdapter::default());
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
        let result = DruidAdapter::default()
            .tune_eval(params, 1, Some(0))
            .expect("tune_eval should round-trip with a minimal RAVE config");
        assert!(result["cost"].as_f64().is_some());
    }
}
