// `GameAdapter` impl for Druid -- ports `main.rs`'s former Druid-specific,
// session-stateful handlers onto the stateless per-game contract.
// `build_ai`/`AiPreset`/`ai_thread_count` are
// unchanged from the prior server, just relocated here.

use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use mcts::game::Game;
use mcts::games::druid::{apply_placed, Druid, HashedState, Move, Orientation, Piece, PieceKind, PlacedPiece, Player, Size, Square, State};
use mcts::strategies::mcts::{
    node::QInit, select, simulate, strategy, SearchConfig, TreeSearch,
};
use mcts::strategies::Search;

use crate::adapters::{
    AdapterError, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter,
};

// AI opponents, from weakest to strongest. Each preset pairs a search
// strategy with a wall-clock thinking budget -- Druid's move generation and
// terminal checks are expensive (see the header comment in
// src/games/druid.rs), so budgets are time-based rather than iteration
// counts, which keeps the UI responsive regardless of board size.
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

    /// The wire id used in `/api/games/druid/ai_move` etc. request bodies
    /// and `/api/games/druid/ai_presets` response ids -- kept as plain
    /// strings at the `GameAdapter` boundary (see that trait's doc comment)
    /// rather than round-tripping the enum through serde, so this is the
    /// one place that name has to agree with itself.
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

    fn label(self) -> &'static str {
        match self {
            AiPreset::Easy => "Easy",
            AiPreset::Medium => "Medium",
            AiPreset::Strong => "Strong",
            AiPreset::Master => "Master",
        }
    }

    fn description(self) -> &'static str {
        match self {
            AiPreset::Easy => "Plain UCB1 with random playouts and MCTS-Solver for tactical sharpness, ~1s per move.",
            AiPreset::Medium => "UCB1 with MAST-biased playouts and MCTS-Solver for tactical sharpness, ~2s per move.",
            AiPreset::Strong => {
                "N-gram-guided (NST) decisive-move search with MCTS-Solver for tactical \
                 sharpness, ~3s per move, searching one shared tree across all available \
                 CPU cores."
            }
            AiPreset::Master => {
                "Same search as Strong, parallelized the same way, with a longer ~8s \
                 thinking budget."
            }
        }
    }

    fn default_time_budget(self) -> Duration {
        match self {
            AiPreset::Easy => Duration::from_secs(1),
            AiPreset::Medium => Duration::from_secs(2),
            AiPreset::Strong => Duration::from_secs(3),
            AiPreset::Master => Duration::from_secs(8),
        }
    }
}

// Number of threads Strong/Master search across. It was found that
// single-threaded search is the weakest mode available at every board
// size tested, and pure tree-parallel search (one shared tree, N worker
// threads) won outright at 5x5 and tied every other mode at 9x9 -- unlike
// root parallelism (N independent trees), it never lost across either tested
// size and doesn't pay N times the tree memory, so it's used here as a
// single default rather than switching configs by board size. Derived from
// the actual machine's core count rather than hardcoding the 8 cores session
// 10's benchmarks happened to run on, so this stays sensible on whatever
// hardware the server runs on.
fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// `Strong`/`Master`'s strategy shape: `Ucb1` select (no RAVE/GRAVE) +
// `DecisiveMove<EpsilonGreedy<Nst>>` simulate, epsilon=0.3,
// backoff_threshold=5. This is the result of a recalibration that replaced
// the previously-shipped `select::Rave` + `DruidHeuristic` shape with this
// simpler one.
type Ucb1DmNst = strategy::Compose<select::Ucb1, simulate::DecisiveMove<Druid, simulate::EpsilonGreedy<Druid, simulate::Nst>>>;

// All four presets enable `use_mcts_solver(true)` (proven-win/loss selection
// bias and early termination) and `reuse_tree(true)` (carries forward stats
// when the same cached engine is handed a state it's already partly
// explored; see `EngineCache`'s doc comment for how that
// interacts with this server's now-stateless request model). Easy/Medium
// stay single-threaded on purpose, so the difficulty gradient reflects
// search quality, not just core count.
fn build_ai(preset: AiPreset, budget: Duration) -> Box<dyn Search<G = Druid>> {
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
                    .select(select::Ucb1::with_c(1.414)),
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
                    .select(select::Ucb1::with_c(1.625))
                    .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
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
                    .num_tree_threads(ai_thread_count())
                    .simulate(simulate::DecisiveMove::new().inner(
                        simulate::EpsilonGreedy::default()
                            .epsilon(0.3)
                            .inner(simulate::Nst::new().backoff_threshold(5)),
                    )),
            ),
        ),
    }
}

/// A bounded, keyed cache of in-progress AI engines, replacing the old
/// server's single `Mutex<Option<PersistedAi>>` that persisted one engine
/// across an entire session's moves. That model doesn't fit a stateless
/// server: state now arrives fresh on every request (no session to hang a
/// single persisted engine off), and a client can hold several branches at
/// once (undo/redo, replay a different line) with no ordering guarantee
/// about which one the next request continues.
///
/// Keyed by `(preset, hash of the state the search runs from)`, capped at a
/// small number of entries (LRU-evicted). This deliberately only serves the
/// *same-state-again* case -- repeating `analyze` on an unchanged position,
/// or `analyze` immediately followed by `ai_move` on that same position --
/// not cross-move continuity (a real move changes the state's hash, which is
/// always a cache miss here). The old design's cross-move reuse depended on
/// exactly one engine surviving for the whole game, which is precisely what
/// a stateless, branch-capable client can no longer guarantee; each
/// `TreeSearch`'s own `reuse_tree`/`reuse_or_reset` still does its job
/// *within* a cache hit (repeated calls on the same state keep growing the
/// same arena), just not across the miss this
/// cache doesn't try to bridge.
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

    /// Removes and returns the cached engine for `(preset, hash)`, if any --
    /// taken rather than borrowed since running a search mutates it in
    /// place and `Box<dyn Search<G>>` isn't `Clone`. Callers re-insert via
    /// `put` after use.
    fn take(&self, preset: AiPreset, hash: u64) -> Option<Box<dyn Search<G = Druid>>> {
        // A panic mid-search on another request (this cache's only lock
        // holder besides `put`) would otherwise poison the mutex for every
        // future request forever -- this is a perf-only cache (see this
        // type's doc comment), never consulted for correctness, so a
        // recovered-but-possibly-inconsistent view of it is still strictly
        // better than every subsequent `ai_move`/`analyze` call 500ing.
        let mut entries = self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let pos = entries
            .iter()
            .position(|(p, h, _)| *p == preset && *h == hash)?;
        Some(entries.remove(pos).2)
    }

    /// Inserts `engine` as the most-recently-used entry for `(preset,
    /// hash)`, evicting the least-recently-used entry if now over capacity.
    fn put(&self, preset: AiPreset, hash: u64, engine: Box<dyn Search<G = Druid>>) {
        let mut entries = self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.retain(|(p, h, _)| !(*p == preset && *h == hash));
        entries.insert(0, (preset, hash, engine));
        entries.truncate(self.capacity);
    }
}

pub struct DruidAdapter {
    cache: EngineCache,
}

impl Default for DruidAdapter {
    fn default() -> Self {
        // 8 concurrent resident engines: generous for local hot-seat/one-
        // browser-tab use (today's actual usage), small enough that a
        // pathological client hammering `analyze` across many distinct
        // positions can't grow this unboundedly -- each `TreeSearch` arena
        // is real memory, unlike a typical "just cache a small value" LRU.
        Self {
            cache: EngineCache::new(8),
        }
    }
}

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

fn state_to_value(state: &HashedState) -> Value {
    serde_json::to_value(state.state()).expect("State always serializes")
}

/// Deserializes a client-supplied state `Value` back into a `HashedState`.
/// Only `State` round-trips over the wire (the Zobrist hash/`Connectivity`/
/// `MoveCache` in `HashedState` are pure caches derived from it, see that
/// type's doc comment), so this rebuilds them from scratch via
/// `HashedState::from_state` rather than deserializing `HashedState`
/// directly. Checks `size.is_supported()` and the board length before
/// calling `from_state` (which otherwise asserts/panics on either) --
/// deeper shape validation (e.g. a hand count that couldn't arise from real
/// play) is deliberately left to a future hardening pass, not
/// this one.
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
    AiPreset::parse(preset).ok_or_else(|| AdapterError::bad_request(format!("unknown preset {preset:?}")))
}

// A client-supplied `analyze` `budget_ms` override
// bypasses each preset's own `default_time_budget`, so nothing upstream of
// this adapter bounds it -- clamped, not rejected: a client asking for too
// little/too much is a UX mistake to correct, not a hostile request to
// reject. The upper bound
// leaves real headroom under `main.rs`'s `AI_ROUTE_TIMEOUT` (30s) for
// `spawn_blocking` scheduling and response-building overhead around the
// search itself.
const MIN_ANALYZE_BUDGET_MS: u64 = 50;
const MAX_ANALYZE_BUDGET_MS: u64 = 20_000;

fn clamp_budget_ms(budget_ms: u64) -> u64 {
    budget_ms.clamp(MIN_ANALYZE_BUDGET_MS, MAX_ANALYZE_BUDGET_MS)
}

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
            .map(|&id| AiPresetInfo {
                id: id.id(),
                label: id.label(),
                description: id.description(),
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
        let mut ai = self
            .cache
            .take(preset, hash)
            .unwrap_or_else(|| build_ai(preset, preset.default_time_budget()));

        // Loop through linearized sub-actions (Piece -> Orientation? -> Cell)
        // until the placement is complete, accumulating choices to reconstruct
        // a PlacedPiece for the wire response.
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
                        mv: serde_json::to_value(result)
                            .expect("PlacedPiece always serializes"),
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
        // An explicit budget override always gets a fresh engine rather
        // than participating in the cache: a cached engine's `max_time` was
        // fixed at construction (`SearchConfig` has no "change the budget
        // of an in-progress search" hook), so honoring a per-call override
        // and caching are mutually exclusive here. The default-budget path
        // (the common case -- `AnalysisPanel` doesn't expose a
        // budget control) still benefits from the cache normally.
        let mut ai = match budget_ms {
            Some(ms) => build_ai(preset, Duration::from_millis(clamp_budget_ms(ms))),
            None => self
                .cache
                .take(preset, hash)
                .unwrap_or_else(|| build_ai(preset, preset.default_time_budget())),
        };

        // Run search on the root position (one choose_action call populates
        // the tree for the root's first-level sub-decisions). Since the tree
        // is rerooted after each sub-action, we don't loop here: the report
        // reflects the current root's children, which are Move::Piece and
        // Move::Orientation sub-actions. The first PV action is typically
        // one of those sub-actions, not a full PlacedPiece -- the client
        // uses analysis for display only, so this is acceptable.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_budget_ms_bounds_out_of_range_requests() {
        assert_eq!(clamp_budget_ms(0), MIN_ANALYZE_BUDGET_MS);
        assert_eq!(clamp_budget_ms(1), MIN_ANALYZE_BUDGET_MS);
        assert_eq!(clamp_budget_ms(u64::MAX), MAX_ANALYZE_BUDGET_MS);
        assert_eq!(clamp_budget_ms(MAX_ANALYZE_BUDGET_MS + 1), MAX_ANALYZE_BUDGET_MS);
    }

    #[test]
    fn test_clamp_budget_ms_leaves_in_range_requests_unchanged() {
        let mid = (MIN_ANALYZE_BUDGET_MS + MAX_ANALYZE_BUDGET_MS) / 2;
        assert_eq!(clamp_budget_ms(mid), mid);
    }
}
