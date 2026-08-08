// `GameAdapter` impl for Druid -- the complex-adapter tier.
// Keeps its own hand-written impl (unlike `ttt`/`traffic-lights` which
// share the `SimpleGameCodec` blanket path): Druid needs `EngineCache`,
// `NewGameConfig`, time-budgeted presets, thread-count logic, and
// linear-sub-action loop for `ai_move` -- none of which the
// `SimpleAdapter` generic path supports.
//
// Scalar preset knobs (time budgets, thread counts, `select_c`, epsilon,
// backoff threshold) now live in `server/config/druid-presets.yaml` loaded
// at construction -- see that file's header comment for the schema.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use mcts::game::Game;
use mcts::games::druid::{
    apply_placed, Druid, HashedState, Move, Orientation, Piece, PieceKind, PlacedPiece, Player,
    Size, Square, State,
};
use mcts::strategies::mcts::{
    node::QInit, select, simulate, strategy, SearchConfig, TreeSearch,
};
use mcts::strategies::Search;

use crate::adapters::{
    AdapterError, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter,
};

// ---------------------------------------------------------------------------
// Preset config (loaded from YAML at startup)
// ---------------------------------------------------------------------------

/// Per-preset scalar knobs from `server/config/druid-presets.yaml`.
/// `select_c` and `epsilon`/`backoff_threshold` may be absent for presets
/// whose strategy shape doesn't use them (serde's `Option`-wrapping handles
/// the `null`/absent case for free).
#[derive(Debug, Clone, Deserialize)]
struct PresetConfig {
    label: String,
    description: String,
    time_budget_ms: u64,
    select_c: Option<f64>,
    num_threads: ThreadCountSpec,
    epsilon: Option<f64>,
    backoff_threshold: Option<u32>,
}

/// `"auto"` (use all available cores) or a fixed `usize`.
///
/// Custom `Deserialize` so YAML can express `"auto"` as a string
/// and `4` as an integer, rather than requiring `#[serde(untagged)]`
/// whose unit-variant try won't accept a non-null string like `"auto"`.
#[derive(Debug, Clone)]
enum ThreadCountSpec {
    Auto,
    Fixed(usize),
}

impl<'de> serde::Deserialize<'de> for ThreadCountSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de;

        struct ThreadCountVisitor;

        impl<'de> de::Visitor<'de> for ThreadCountVisitor {
            type Value = ThreadCountSpec;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("integer or \"auto\"")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<ThreadCountSpec, E> {
                Ok(ThreadCountSpec::Fixed(v as usize))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<ThreadCountSpec, E> {
                Ok(ThreadCountSpec::Fixed(v as usize))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<ThreadCountSpec, E> {
                if v == "auto" {
                    Ok(ThreadCountSpec::Auto)
                } else {
                    Err(de::Error::unknown_variant(v, &["auto"]))
                }
            }
        }

        deserializer.deserialize_any(ThreadCountVisitor)
    }
}

impl ThreadCountSpec {
    fn resolve(&self) -> usize {
        match self {
            ThreadCountSpec::Auto => ai_thread_count(),
            ThreadCountSpec::Fixed(n) => *n,
        }
    }
}

fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Top-level YAML shape.
#[derive(Debug, Deserialize)]
struct PresetsFile {
    presets: HashMap<String, PresetConfig>,
}

const CONFIG_PATH: &str = "server/config/druid-presets.yaml";

fn load_presets() -> HashMap<String, PresetConfig> {
    let content = match std::fs::read_to_string(CONFIG_PATH) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("warn: could not read {CONFIG_PATH} ({e}), using hardcoded fallback presets");
            return hardcoded_fallback();
        }
    };
    let parsed: PresetsFile = match serde_yaml::from_str(&content) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("warn: could not parse {CONFIG_PATH} ({e}), using hardcoded fallback presets");
            return hardcoded_fallback();
        }
    };
    parsed.presets
}

fn hardcoded_fallback() -> HashMap<String, PresetConfig> {
    let mut m = HashMap::new();
    m.insert(
        "easy".into(),
        PresetConfig {
            label: "Easy".into(),
            description: "Plain UCB1 with random playouts and MCTS-Solver for tactical sharpness, ~1s per move."
                .into(),
            time_budget_ms: 1000,
            select_c: Some(1.414),
            num_threads: ThreadCountSpec::Fixed(1),
            epsilon: None,
            backoff_threshold: None,
        },
    );
    m.insert(
        "medium".into(),
        PresetConfig {
            label: "Medium".into(),
            description: "UCB1 with MAST-biased playouts and MCTS-Solver for tactical sharpness, ~2s per move."
                .into(),
            time_budget_ms: 2000,
            select_c: Some(1.625),
            num_threads: ThreadCountSpec::Fixed(1),
            epsilon: Some(0.1),
            backoff_threshold: None,
        },
    );
    m.insert(
        "strong".into(),
        PresetConfig {
            label: "Strong".into(),
            description: "N-gram-guided (NST) decisive-move search with MCTS-Solver for tactical \
                 sharpness, ~3s per move, searching one shared tree across all available CPU cores."
                .into(),
            time_budget_ms: 3000,
            select_c: Some(1.414),
            num_threads: ThreadCountSpec::Auto,
            epsilon: Some(0.3),
            backoff_threshold: Some(5),
        },
    );
    m.insert(
        "master".into(),
        PresetConfig {
            label: "Master".into(),
            description: "Same search as Strong, parallelized the same way, with a longer ~8s \
                 thinking budget."
                .into(),
            time_budget_ms: 8000,
            select_c: Some(1.414),
            num_threads: ThreadCountSpec::Auto,
            epsilon: Some(0.3),
            backoff_threshold: Some(5),
        },
    );
    m
}

// ---------------------------------------------------------------------------
// AiPreset enum (strategy-shape dispatch only)
// ---------------------------------------------------------------------------

/// Which strategy type composition a preset uses. The `id()`/`parse()`
/// methods stay on the enum; all display strings and scalar knobs come from
/// the loaded config (see `PresetConfig` above).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiPreset {
    Easy,
    Medium,
    Strong,
    Master,
}

impl AiPreset {
    const ALL: [AiPreset; 4] = [
        AiPreset::Easy,
        AiPreset::Medium,
        AiPreset::Strong,
        AiPreset::Master,
    ];

    fn id(self) -> &'static str {
        match self {
            AiPreset::Easy => "easy",
            AiPreset::Medium => "medium",
            AiPreset::Strong => "strong",
            AiPreset::Master => "master",
        }
    }

    fn parse(id: &str) -> Option<AiPreset> {
        AiPreset::ALL.into_iter().find(|p| p.id() == id)
    }
}

// ---------------------------------------------------------------------------
// Strategy-shape type aliases
// ---------------------------------------------------------------------------

// `Strong`/`Master`'s strategy shape: `Ucb1` select (no RAVE/GRAVE) +
// `DecisiveMove<EpsilonGreedy<Nst>>` simulate, epsilon=0.3,
// backoff_threshold=5. This is the result of a recalibration that replaced
// the previously-shipped `select::Rave` + `DruidHeuristic` shape with this
// simpler one.
type Ucb1DmNst =
    strategy::Compose<select::Ucb1, simulate::DecisiveMove<Druid, simulate::EpsilonGreedy<Druid, simulate::Nst>>>;

// All four presets enable `use_mcts_solver(true)` (proven-win/loss selection
// bias and early termination) and `reuse_tree(true)` (carries forward stats
// when the same cached engine is handed a state it's already partly
// explored; see `EngineCache`'s doc comment for how that
// interacts with this server's now-stateless request model). Easy/Medium
// stay single-threaded on purpose, so the difficulty gradient reflects
// search quality, not just core count.
fn build_ai(
    preset: AiPreset,
    budget: Duration,
    cfg: &PresetConfig,
) -> Box<dyn Search<G = Druid>> {
    match preset {
        AiPreset::Easy => Box::new(
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
        AiPreset::Medium => Box::new(
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
                    .simulate(simulate::EpsilonGreedy::with_epsilon(cfg.epsilon.unwrap_or(0.1))),
            ),
        ),
        AiPreset::Strong | AiPreset::Master => Box::new(
            TreeSearch::<Druid, Ucb1DmNst>::new().config(
                SearchConfig::new()
                    .name(if preset == AiPreset::Strong {
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
                    .num_tree_threads(cfg.num_threads.resolve())
                    .simulate(simulate::DecisiveMove::new().inner(
                        simulate::EpsilonGreedy::default()
                            .epsilon(cfg.epsilon.unwrap_or(0.3))
                            .inner(simulate::Nst::new().backoff_threshold(
                                cfg.backoff_threshold.unwrap_or(5),
                            )),
                    )),
            ),
        ),
    }
}

// ---------------------------------------------------------------------------
// Engine cache (unchanged from prior design)
// ---------------------------------------------------------------------------

type CacheEntry = (AiPreset, u64, Box<dyn Search<G = Druid>>);

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

    fn take(&self, preset: AiPreset, hash: u64) -> Option<Box<dyn Search<G = Druid>>> {
        let mut entries =
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pos = entries
            .iter()
            .position(|(p, h, _)| *p == preset && *h == hash)?;
        Some(entries.remove(pos).2)
    }

    fn put(&self, preset: AiPreset, hash: u64, engine: Box<dyn Search<G = Druid>>) {
        let mut entries =
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.retain(|(p, h, _)| !(*p == preset && *h == hash));
        entries.insert(0, (preset, hash, engine));
        entries.truncate(self.capacity);
    }
}

// ---------------------------------------------------------------------------
// DruidAdapter
// ---------------------------------------------------------------------------

pub struct DruidAdapter {
    cache: EngineCache,
    presets: HashMap<String, PresetConfig>,
}

impl Default for DruidAdapter {
    fn default() -> Self {
        Self {
            cache: EngineCache::new(8),
            presets: load_presets(),
        }
    }
}

impl DruidAdapter {
    /// Returns the config for a preset (looked up by its id string), or
    /// `None` if the id isn't in the loaded config.
    fn preset_cfg(&self, p: AiPreset) -> &PresetConfig {
        // Every valid `AiPreset` should have an entry in the config
        // (the fallback map covers all four), so an unwrap-or-else
        // panic here is a genuine programmer error, not a runtime
        // fallback.
        self.presets
            .get(p.id())
            .expect("AiPreset variant missing from loaded presets")
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
    hand_black: &'a mcts::games::druid::Hand,
    hand_white: &'a mcts::games::druid::Hand,
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
/// Only `State` round-trips over the wire (the Zobrist hash/`Connectivity`/
/// `MoveCache` in `HashedState` are pure caches derived from it, see that
/// type's doc comment), so this rebuilds them from scratch via
/// `HashedState::from_state` rather than deserializing `HashedState`
/// directly.
fn value_to_state(state: &Value) -> Result<HashedState, AdapterError> {
    let state: State = serde_json::from_value(state.clone())
        .map_err(|e| AdapterError::bad_request(format!("invalid state: {e}")))?;
    if !state.size.is_supported() {
        return Err(AdapterError::bad_request(format!(
            "unsupported board size {}x{}",
            state.size.w, state.size.h
        )));
    }
    if state.board.len() != (state.size.w as usize) * (state.size.h as usize) {
        return Err(AdapterError::bad_request(
            "state board length doesn't match its size",
        ));
    }
    Ok(HashedState::from_state(state))
}

fn parse_preset(preset: &str) -> Result<AiPreset, AdapterError> {
    AiPreset::parse(preset)
        .ok_or_else(|| AdapterError::bad_request(format!("unknown preset {preset:?}")))
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
// GameAdapter impl
// ---------------------------------------------------------------------------

impl GameAdapter for DruidAdapter {
    fn kind(&self) -> &'static str {
        "druid"
    }

    fn label(&self) -> &'static str {
        "Druid"
    }

    fn description(&self) -> &'static str {
        "A connection game played with stackable sarsen and lintel pieces, \
         designed by Cameron Browne."
    }

    fn default_config(&self) -> Value {
        serde_json::json!({ "size": mcts::games::druid::DEFAULT_SIZE })
    }

    fn new_state(&self, config: Value) -> Result<Value, AdapterError> {
        let config: NewGameConfig = serde_json::from_value(config)
            .map_err(|e| AdapterError::bad_request(format!("invalid config: {e}")))?;
        if !config.size.is_supported() {
            return Err(AdapterError::bad_request(format!(
                "unsupported board size {}x{}: each side must be at least 3, \
                 and the board can't be so large it overflows the Zobrist hash table",
                config.size.w, config.size.h
            )));
        }
        Ok(state_to_value(&HashedState::new(config.size)))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, AdapterError> {
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

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, AdapterError> {
        let state = value_to_state(state)?;
        let mv: PlacedPiece = serde_json::from_value(mv.clone())
            .map_err(|e| AdapterError::bad_request(format!("invalid move: {e}")))?;

        if Druid::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        state.state().moves(&mut legal);
        if !legal.contains(&mv) {
            return Err(AdapterError::bad_request("illegal move"));
        }
        Ok(state_to_value(&apply_placed(state, mv)))
    }

    fn view(&self, state: &Value) -> Result<Value, AdapterError> {
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
        AiPreset::ALL
            .iter()
            .map(|&p| {
                let cfg = self.preset_cfg(p);
                AiPresetInfo {
                    id: p.id().to_string(),
                    label: cfg.label.clone(),
                    description: cfg.description.clone(),
                }
            })
            .collect()
    }

    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, AdapterError> {
        let state = value_to_state(state)?;
        let preset = parse_preset(preset)?;
        if Druid::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }

        let hash = Druid::zobrist_hash(&state);
        let cfg = self.preset_cfg(preset);
        let mut ai = self.cache.take(preset, hash).unwrap_or_else(|| {
            build_ai(preset, Duration::from_millis(cfg.time_budget_ms), cfg)
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
                    self.cache.put(preset, hash, ai);
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
    ) -> Result<Analysis, AdapterError> {
        let state = value_to_state(state)?;
        let preset = parse_preset(preset)?;
        if Druid::is_terminal(&state) {
            return Err(AdapterError::bad_request("game is over"));
        }

        let hash = Druid::zobrist_hash(&state);
        let cfg = self.preset_cfg(preset);

        let mut ai = match budget_ms {
            Some(ms) => build_ai(preset, Duration::from_millis(clamp_budget_ms(ms)), cfg),
            None => self
                .cache
                .take(preset, hash)
                .unwrap_or_else(|| build_ai(preset, Duration::from_millis(cfg.time_budget_ms), cfg)),
        };

        let _ = ai.choose_action(&state);
        let report = ai.root_report(&state);

        if budget_ms.is_none() {
            self.cache.put(preset, hash, ai);
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_budget_ms_bounds_out_of_range_requests() {
        assert_eq!(clamp_budget_ms(0), MIN_ANALYZE_BUDGET_MS);
        assert_eq!(clamp_budget_ms(1), MIN_ANALYZE_BUDGET_MS);
        assert_eq!(clamp_budget_ms(u64::MAX), MAX_ANALYZE_BUDGET_MS);
        assert_eq!(
            clamp_budget_ms(MAX_ANALYZE_BUDGET_MS + 1),
            MAX_ANALYZE_BUDGET_MS
        );
    }

    #[test]
    fn test_clamp_budget_ms_leaves_in_range_requests_unchanged() {
        let mid = (MIN_ANALYZE_BUDGET_MS + MAX_ANALYZE_BUDGET_MS) / 2;
        assert_eq!(clamp_budget_ms(mid), mid);
    }

    #[test]
    fn test_fallback_map_contains_all_four_presets() {
        let m = hardcoded_fallback();
        assert_eq!(m.len(), 4);
        assert!(m.contains_key("easy"));
        assert!(m.contains_key("medium"));
        assert!(m.contains_key("strong"));
        assert!(m.contains_key("master"));
    }

    #[test]
    fn test_fallback_easy_threads_is_fixed() {
        let m = hardcoded_fallback();
        assert!(matches!(
            m.get("easy").unwrap().num_threads,
            ThreadCountSpec::Fixed(1)
        ));
    }

    #[test]
    fn test_fallback_master_threads_is_auto() {
        let m = hardcoded_fallback();
        assert!(matches!(
            m.get("master").unwrap().num_threads,
            ThreadCountSpec::Auto
        ));
    }

    #[test]
    fn test_thread_count_spec_auto_resolves() {
        // Doesn't crash, returns a positive usize
        let resolved = ThreadCountSpec::Auto.resolve();
        assert!(resolved >= 1);
    }

    #[test]
    fn test_preset_cfg_loads_from_default_adapter() {
        let a = DruidAdapter::default();
        // Should have loaded the YAML file (or fallen back) successfully
        assert_eq!(a.presets.len(), 4);
        let easy = a.preset_cfg(AiPreset::Easy);
        assert_eq!(easy.time_budget_ms, 1000);
        let master = a.preset_cfg(AiPreset::Master);
        assert_eq!(master.time_budget_ms, 8000);
    }
}