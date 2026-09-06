//! Iterative-deepening negamax (alpha-beta) as a standalone [`Search`]
//! strategy, independent of the MCTS tree/node machinery in
//! `algorithms::mcts`.
//!
//! Negamax only makes sense for deterministic, perfect-information,
//! two-player zero-sum games (the whole formulation relies on
//! `value(state) == -value(state_after_opponents_best_reply)`), so this is
//! a separate `Search` implementation rather than another `mcts::PolicyProfile`
//! -- it shares `Game` with the MCTS side, but nothing else.
//!
//! `Game` has no compile-time marker for "deterministic" or "perfect
//! information" -- there is no trait bound that could reject an unsuitable
//! `Game` at the `Negamax<G, E>` call site. What it does have is three
//! opt-in, defaulted capability accessors on `Game` itself
//! (`is_stochastic`/`has_hidden_information`/`alternating_moves`, all
//! defaulting to match every game in this repo today: deterministic,
//! perfect-information, alternating), which [`supports`] below checks at
//! runtime. `choose_action`'s `debug_assert!(negamax::supports::<G>())`
//! is the one automatic check -- a game that reports `true` for the accessors
//! it shouldn't (or just never overrides them) will still compile and run
//! against `Negamax` in release builds, just search the single
//! determinized/observed state in front of it as if it were the whole
//! truth, silently.
//!
//! # Opt-in requirements
//!
//! `Game` itself stays exactly as it is for every other strategy: no
//! negamax-specific method was added to it. What negamax needs beyond
//! `Game` -- a static evaluation of non-terminal states, used at the
//! search's depth cutoff -- is its own opt-in [`Evaluator`] trait,
//! implemented per-game (or not at all, for a game small enough to always
//! search to a terminal state). `Game::zobrist_hash` (already optional,
//! defaulting to `0`) is used opportunistically for the transposition
//! table when a game has implemented it; see `table.rs` for how an
//! unimplemented hash degrades to "useless" rather than "wrong".
//!
//! This terminal-vs-non-terminal split -- `Game::terminal_status`'s actual
//! win/loss/draw outcome always wins when it applies, `Evaluator` only
//! stands in for it at a depth cutoff. `terminal_score` (below) reads `Game::
//! terminal_status`/`PlayerIndex` directly rather than going through
//! `Evaluator`, and `Evaluator::evaluate` is never asked to score a state
//! `terminal_status` already resolved.
//!
//! # What's implemented
//!
//! - Iterative deepening (`NegamaxOptions::max_depth`/`max_time`), so a
//!   search can be time-budgeted like the MCTS strategies rather than only
//!   depth-budgeted, and always has a best move ready from the last fully
//!   completed depth if a deeper one times out mid-search.
//! - Alpha-beta pruning with principal-variation search (a null-window
//!   probe for every move after the first, re-searched with a full window
//!   only if it beats alpha) -- `NegamaxOptions::principal_variation_search`.
//! - A transposition table (`table.rs`) that both catches transposing move
//!   orders within one depth and seeds move ordering on the next, deeper
//!   iteration, with a choice of replacement policy
//!   (`NegamaxOptions::replacement` / `table::Replacement`: `Always`,
//!   `DepthPreferred` (the default), or `TwoTier`).
//! - Mate-distance scoring: a forced win/loss is scored `WIN_SCORE - ply`/
//!   `LOSS_SCORE + ply` rather than a flat sentinel, so search prefers the
//!   fastest win and the slowest loss, and iterative deepening stops once a
//!   mate is proven within the current depth rather than continuing to
//!   search past it.
//! - An optional narrow-window aspiration pass at the root
//!   (`NegamaxOptions::aspiration_window`) before each depth's definitive
//!   full-window search, purely to prime the transposition table's move
//!   ordering hint.
//! - History-heuristic move ordering (`NegamaxOptions::history_heuristic`):
//!   every action that causes a beta cutoff earns `depth * depth` in a
//!   table keyed by the action alone (not by the position it was played
//!   from), persisting across the whole `choose_action` call. Actions after
//!   the transposition-table move are then tried in descending history
//!   order, on the theory that a move which has been refuting other
//!   positions is likely to refute this one too.
//! - Singular extensions (`NegamaxOptions::singular_extension`): a node
//!   with exactly one legal move is searched one ply deeper than its
//!   parent asked for, since there's no branching to prune there anyway --
//!   capped so a single iterative-deepening pass can't more than double
//!   its nominal depth even against a long forced sequence.
//! - Countermove-history move ordering (`NegamaxOptions::countermove_heuristic`):
//!   like the history table above, but scored per `(previous action, this
//!   action)` pair instead of per-action alone -- "what refuted this
//!   specific reply elsewhere in the tree" is a sharper ordering signal
//!   than "what refutes things in general," on the theory that a move's
//!   best answer is often independent of the rest of the position. Reset
//!   per `choose_action` call, same as the plain history table, and
//!   consulted together with it (summed) when ordering non-transposition-
//!   table moves.
//! - Symmetry-aware transposition: the table is hashed/looked-up/stored
//!   keyed on `Game::canonical_representation(state)` rather than `state`
//!   itself, so two positions that are symmetry images of each other (a
//!   rotation/reflection reached via different move orders) share one
//!   entry instead of each being searched from scratch. `tt_move` and the
//!   stored `best_action` are translated between the literal board and
//!   whichever canonical orientation wrote/reads the entry via
//!   `Game::apply_to_action`/`invert_action`, mirroring how
//!   `algorithms::mcts::node::real_action`/`crate::symmetry::incoming_sym`
//!   do the same translation for MCTS's `ChildArray`s. A no-op for every
//!   game that hasn't overridden `canonical_representation`.
//! - Parallel search (`NegamaxOptions::num_threads`): Lazy-SMP-style root
//!   splitting rather than in-tree (YBWC) fan-out. Each worker thread runs
//!   its own complete iterative-deepening search of the same root against
//!   its own private history/countermove tables, sharing only the
//!   transposition table (wrapped in an `RwLock`, full-state-verified reads
//!   and writes exactly as in the single-threaded case). Threads' start
//!   depths are staggered (`1 + worker_index % num_threads`) so they don't
//!   all redo the same shallow work in lockstep; a thread that reaches a
//!   position first leaves every other thread a usable bound. The merge
//!   once every worker's search completes is "take the result from
//!   whichever thread reached the greatest completed depth, breaking ties
//!   on score" -- a negamax search's per-thread output is a single
//!   deterministic best move, not a visit distribution to combine the way
//!   MCTS's root parallelism does.
//!
//! # Room for more
//!
//! Deliberately not implemented, but each has a natural seam to land in
//! without disturbing this structure: null-move pruning and quiescence
//! search both need their own opt-in game-level hooks (a null move's
//! legality, and a "noisy" move subset) the way `Evaluator` is one.

mod table;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::game::{Canonical, Game, PlayerIndex, Real, TerminalStatus};
use crate::algorithms::{ActionReport, RootReport, Search};
pub use table::Replacement;
use table::{Bound, TranspositionTable};

/// The alpha-beta bounds passed to [`Negamax::negamax_search`], bundled into
/// one argument purely to keep that function's parameter count under
/// clippy's `too_many_arguments` threshold -- no meaning beyond "the current
/// search window."
#[derive(Clone, Copy)]
struct Window {
    alpha: Score,
    beta: Score,
}

/// `Score`/`WIN_SCORE`/`LOSS_SCORE`/`DRAW_SCORE`/`EVAL_MAGNITUDE_LIMIT`/
/// `Evaluator`/`MaterialBlind` moved to `crate::evaluator` so MCTS-side
/// minimax hybrids can depend on them without depending on this module --
/// re-exported here so existing `negamax::`-qualified call sites are
/// unaffected.
pub use crate::evaluator::{
    Evaluator, MaterialBlind, Score, DRAW_SCORE, EVAL_MAGNITUDE_LIMIT, LOSS_SCORE, WIN_SCORE,
};

/// Whether `G`'s declared capabilities (`Game::is_stochastic`/
/// `has_hidden_information`/`alternating_moves`/`num_players`) are ones
/// negamax's deterministic, perfect-information, two-player alternating-
/// move formulation can actually reason about.
pub fn supports<G: Game>() -> bool {
    G::num_players() == 2
        && !G::is_stochastic()
        && !G::has_hidden_information()
        && G::alternating_moves()
}

/// Configuration for [`Negamax`]. Every field has a builder method
/// (`with_*`) that consumes and returns `self`, matching this crate's
/// other strategy configs (see `mcts::SearchConfig`).
#[derive(Clone, Debug)]
pub struct NegamaxOptions {
    /// Iterative deepening never searches past this depth, regardless of
    /// `max_time`.
    pub max_depth: u32,
    /// If set, iterative deepening stops (returning the best move from the
    /// last fully completed depth) once this much wall-clock time has
    /// elapsed since `choose_action` was called.
    pub max_time: Option<Duration>,
    /// Transposition table size is `1 << table_bits` slots. `0` disables
    /// the table entirely.
    pub table_bits: u32,
    /// Which entry within a colliding slot is kept vs. overwritten. See
    /// `Replacement`.
    pub replacement: Replacement,
    /// If set, each depth past 2 first runs a narrow `[prev_score -
    /// window, prev_score + window]` search at the root purely to prime
    /// the transposition table's move-ordering hint before the definitive
    /// full-window pass. Only helps when the evaluation function is
    /// coarse-grained enough that consecutive depths' scores cluster.
    pub aspiration_window: Option<Score>,
    /// Principal-variation search: after the first (best-ordered) move at
    /// a node, probe the rest with a zero-width window and only re-search
    /// with the full window if a probe beats alpha. Cheap insurance against
    /// bad move ordering costing a wasted full-width search on the first
    /// try; essentially free when ordering is good.
    pub principal_variation_search: bool,
    /// Order each node's non-transposition-table moves by a history table
    /// (action -> accumulated `depth * depth` for every beta cutoff it has
    /// caused so far this search), instead of `Game::generate_actions`'
    /// raw order. See the module docs' "History-heuristic move ordering".
    pub history_heuristic: bool,
    /// Extend a forced line (a node with exactly one legal move) one ply
    /// deeper than the search would otherwise stop, capped per iteration so
    /// a long forced sequence can't more than double that iteration's
    /// nominal depth. See the module docs' "Singular extensions".
    pub singular_extension: bool,
    /// Order each node's non-transposition-table moves by a countermove-
    /// history table (keyed by `(action that led to this node, this
    /// node's action)`, scored the same `depth * depth` way as
    /// `history_heuristic`), in addition to the plain per-action history
    /// table. See the module docs' "Countermove-history move ordering".
    pub countermove_heuristic: bool,
    /// Number of root-splitting worker threads (Lazy SMP). `1` (the
    /// default) runs the existing single-threaded search with no locking
    /// overhead on the transposition table; anything greater spawns that
    /// many independent full searches of the root sharing one table. See
    /// the module docs' "Parallel search".
    pub num_threads: usize,
    pub verbose: bool,
}

impl Default for NegamaxOptions {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_time: None,
            table_bits: 20,
            replacement: Replacement::default(),
            aspiration_window: None,
            principal_variation_search: true,
            history_heuristic: true,
            singular_extension: true,
            countermove_heuristic: true,
            num_threads: 1,
            verbose: false,
        }
    }
}

impl NegamaxOptions {
    pub fn with_max_depth(mut self, max_depth: u32) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_max_time(mut self, max_time: Duration) -> Self {
        self.max_time = Some(max_time);
        self
    }

    pub fn with_table_bits(mut self, table_bits: u32) -> Self {
        self.table_bits = table_bits;
        self
    }

    pub fn with_replacement(mut self, replacement: Replacement) -> Self {
        self.replacement = replacement;
        self
    }

    pub fn with_aspiration_window(mut self, window: Score) -> Self {
        self.aspiration_window = Some(window);
        self
    }

    pub fn with_principal_variation_search(mut self, enabled: bool) -> Self {
        self.principal_variation_search = enabled;
        self
    }

    pub fn with_history_heuristic(mut self, enabled: bool) -> Self {
        self.history_heuristic = enabled;
        self
    }

    pub fn with_singular_extension(mut self, enabled: bool) -> Self {
        self.singular_extension = enabled;
        self
    }

    pub fn with_countermove_heuristic(mut self, enabled: bool) -> Self {
        self.countermove_heuristic = enabled;
        self
    }

    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// Iterative-deepening negamax. See the module docs for the feature list.
pub struct Negamax<G: Game, E: Evaluator<G>> {
    /// Shared via `Arc` rather than owned outright so parallel root
    /// splitting's worker searches (see `NegamaxOptions::num_threads`) can
    /// each get their own `Negamax` without requiring `E: Clone`.
    eval: Arc<E>,
    options: NegamaxOptions,
    /// Shared (and, once `num_threads > 1`, actually contended) via `Arc<
    /// RwLock<_>>` rather than owned outright, so every root-splitting
    /// worker thread reads/writes through the same table instead of each
    /// searching from a cold one -- see the module docs' "Parallel search".
    /// Single-threaded callers pay one uncontended lock acquisition per
    /// lookup/store, not a design compromise specific to this feature: it's
    /// the same full-state-verified table either way, just always reached
    /// through the lock.
    table: Option<Arc<RwLock<TranspositionTable<G>>>>,
    /// This worker's iterative-deepening start depth. `1` for a
    /// single-threaded search; staggered per worker
    /// (`1 + worker_index % num_threads`) under root splitting so threads
    /// don't all redo the same shallow iterations in lockstep. Not part of
    /// `NegamaxOptions` -- it's assigned per-worker by
    /// `choose_action_root_split`, not user-configurable.
    start_depth: u32,
    pv: Vec<G::A>,
    /// Every root move's score at the deepest completed iteration, sorted
    /// best-first -- both this depth's move-ordering seed for the next
    /// iteration and the source for `root_report`.
    root_scores: Vec<(G::A, Score)>,
    /// History-heuristic move-ordering scores, keyed by action alone (see
    /// `NegamaxOptions::history_heuristic`). Reset at the start of every
    /// `choose_action` call -- it's a within-search ordering hint, not
    /// meant to persist across different root positions.
    history: HashMap<G::A, i32>,
    /// Countermove-history move-ordering scores, keyed by `(action that led
    /// to this node, this node's action)` (see
    /// `NegamaxOptions::countermove_heuristic`). Reset alongside `history`
    /// at the start of every `choose_action` call.
    countermove: HashMap<(G::A, G::A), i32>,
    /// This iteration's singular-extension budget: `negamax_search` may
    /// extend a forced (single-legal-move) node one ply deeper only while
    /// `ply < singular_extension_cap`, so one iterative-deepening pass can
    /// never search more than double its nominal depth even against a long
    /// forced sequence.
    singular_extension_cap: u32,
    nodes_searched: u64,
    depth_reached: u32,
    name: String,
    game_type: PhantomData<G>,
}

impl<G: Game, E: Evaluator<G>> Negamax<G, E> {
    pub fn new(eval: E) -> Self {
        Self::new_with_options(eval, NegamaxOptions::default())
    }

    pub fn new_with_options(eval: E, options: NegamaxOptions) -> Self {
        let table = (options.table_bits > 0).then(|| {
            Arc::new(RwLock::new(TranspositionTable::new(
                options.table_bits,
                options.replacement,
            )))
        });
        Self {
            eval: Arc::new(eval),
            options,
            table,
            start_depth: 1,
            pv: Vec::new(),
            root_scores: Vec::new(),
            history: HashMap::new(),
            countermove: HashMap::new(),
            singular_extension_cap: 0,
            nodes_searched: 0,
            depth_reached: 0,
            name: "negamax".into(),
            game_type: PhantomData,
        }
    }

    pub fn nodes_searched(&self) -> u64 {
        self.nodes_searched
    }

    pub fn depth_reached(&self) -> u32 {
        self.depth_reached
    }

    /// The last completed iteration's root score, from the perspective of
    /// the player who was to move in the state `choose_action` was called
    /// with. `DRAW_SCORE` before any search has run.
    pub fn root_score(&self) -> Score {
        self.root_scores
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(DRAW_SCORE)
    }

    fn timed_out(&self, deadline: Option<Instant>) -> bool {
        match deadline {
            Some(d) => self.nodes_searched.is_multiple_of(1024) && Instant::now() >= d,
            None => false,
        }
    }

    /// `state`'s score from the perspective of `Game::player_to_move(state)`
    /// -- even when `state` is terminal, since that convention (see
    /// `TerminalStatus::utilities`, `Game::get_reward`) is what lets the
    /// recursive `-negamax_search(child, ...)` at the call site treat a
    /// terminal leaf exactly like any other returned score.
    fn terminal_score(state: &G::S) -> Option<Score> {
        let mover = G::player_to_move(state).to_index();
        match G::terminal_status(state) {
            TerminalStatus::NotTerminal => None,
            TerminalStatus::Draw => Some(DRAW_SCORE),
            TerminalStatus::Winner(w) => Some(if w.to_index() == mover {
                WIN_SCORE
            } else {
                LOSS_SCORE
            }),
        }
    }

    /// Recursive alpha-beta search of `state`'s subtree to `depth` plies,
    /// returning `state`'s score from `Game::player_to_move(state)`'s
    /// perspective, or `None` on timeout. `ply` is the distance from the
    /// root, used only to prefer faster wins / slower losses (see the
    /// module docs' "mate-distance scoring"). `window.alpha`/`window.beta`
    /// bundle the alpha-beta bounds into one argument purely to keep the
    /// parameter count clippy-sized -- see [`Window`].
    fn negamax_search(
        &mut self,
        state: &G::S,
        prev_action: Option<&G::A>,
        depth: u32,
        ply: u32,
        window: Window,
        deadline: Option<Instant>,
    ) -> Option<Score> {
        let Window { mut alpha, beta } = window;
        self.nodes_searched += 1;
        if self.timed_out(deadline) {
            return None;
        }

        if let Some(raw) = Self::terminal_score(state) {
            return Some(match raw {
                WIN_SCORE => WIN_SCORE - ply as Score,
                LOSS_SCORE => LOSS_SCORE + ply as Score,
                other => other,
            });
        }

        if depth == 0 {
            return Some(
                self.eval
                    .evaluate(state)
                    .clamp(-EVAL_MAGNITUDE_LIMIT, EVAL_MAGNITUDE_LIMIT),
            );
        }

        // Hashed/stored/looked-up in canonical orientation rather than
        // `state`'s literal one, so two positions that are symmetry images
        // of each other (reached via different move orders/orientations)
        // share the same slot instead of each being searched from scratch.
        // A no-op for any game that hasn't overridden
        // `Game::canonical_representation` (`sym` is always `IDENTITY`
        // there, per that method's doc comment).
        let (canonical, sym) = G::canonical_representation(Real(state.clone()));
        let canonical_state = canonical.into_inner();
        let hash = G::zobrist_hash(&canonical_state);
        let alpha_orig = alpha;
        let mut beta = beta;
        let mut tt_move = None;
        if let Some(table) = &self.table {
            let looked_up = table.read().unwrap().lookup(hash, &canonical_state);
            if let Some(entry) = looked_up {
                // The stored `best_action` was computed against whichever
                // orientation first wrote this entry, not necessarily
                // `sym` -- translate it back to `state`'s own real
                // orientation before it's usable as this call's move-
                // ordering hint.
                tt_move = entry
                    .best_action
                    .clone()
                    .map(|a| G::invert_action(Canonical(a), sym).into_inner());
                if entry.depth >= depth {
                    match entry.bound {
                        Bound::Exact => return Some(entry.score),
                        Bound::Lower => alpha = alpha.max(entry.score),
                        Bound::Upper => beta = beta.min(entry.score),
                    }
                    if alpha >= beta {
                        return Some(entry.score);
                    }
                }
            }
        }

        let mut actions = Vec::new();
        G::generate_actions(state, &mut actions);
        debug_assert!(
            !actions.is_empty(),
            "Game::terminal_status said NotTerminal but generate_actions is empty"
        );
        let mut ordered_start = 0;
        if let Some(tm) = &tt_move {
            if let Some(pos) = actions.iter().position(|a| a == tm) {
                actions.swap(0, pos);
                ordered_start = 1;
            }
        }
        if self.options.history_heuristic || self.options.countermove_heuristic {
            actions[ordered_start..].sort_by_key(|a| {
                let mut score = 0;
                if self.options.history_heuristic {
                    score += self.history.get(a).copied().unwrap_or(0);
                }
                if self.options.countermove_heuristic {
                    if let Some(prev) = prev_action {
                        score += self
                            .countermove
                            .get(&(prev.clone(), a.clone()))
                            .copied()
                            .unwrap_or(0);
                    }
                }
                std::cmp::Reverse(score)
            });
        }

        let child_depth = if self.options.singular_extension
            && actions.len() == 1
            && ply < self.singular_extension_cap
        {
            depth
        } else {
            depth - 1
        };

        let mut best_score = LOSS_SCORE - 1;
        let mut best_action = actions[0].clone();
        let mut searching_pv = false;

        for a in &actions {
            let child = G::apply(state.clone(), a);
            let score = if searching_pv && self.options.principal_variation_search {
                let probe = -self.negamax_search(
                    &child,
                    Some(a),
                    child_depth,
                    ply + 1,
                    Window {
                        alpha: -alpha - 1,
                        beta: -alpha,
                    },
                    deadline,
                )?;
                if probe > alpha && probe < beta {
                    -self.negamax_search(
                        &child,
                        Some(a),
                        child_depth,
                        ply + 1,
                        Window {
                            alpha: -beta,
                            beta: -probe,
                        },
                        deadline,
                    )?
                } else {
                    probe
                }
            } else {
                -self.negamax_search(
                    &child,
                    Some(a),
                    child_depth,
                    ply + 1,
                    Window {
                        alpha: -beta,
                        beta: -alpha,
                    },
                    deadline,
                )?
            };

            if score > best_score {
                best_score = score;
                best_action = a.clone();
            }
            if score > alpha {
                alpha = score;
                searching_pv = true;
            }
            if alpha >= beta {
                if self.options.history_heuristic {
                    *self.history.entry(a.clone()).or_insert(0) += (depth * depth) as i32;
                }
                if self.options.countermove_heuristic {
                    if let Some(prev) = prev_action {
                        *self
                            .countermove
                            .entry((prev.clone(), a.clone()))
                            .or_insert(0) += (depth * depth) as i32;
                    }
                }
                break;
            }
        }

        if let Some(table) = &self.table {
            let bound = if best_score <= alpha_orig {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            // Stored in the same canonical orientation the entry was
            // looked up in above, not `state`'s literal one -- see
            // `tt_move`'s translation on the read side.
            let canonical_best_action = G::apply_to_action(Real(best_action), sym).into_inner();
            table.write().unwrap().store(
                hash,
                &canonical_state,
                depth,
                best_score,
                bound,
                Some(canonical_best_action),
            );
        }

        Some(best_score)
    }

    /// Walk the transposition table's best-move chain from `state` to
    /// reconstruct the principal variation of the deepest completed
    /// iteration, for `principle_variation`/`root_report`. Bounded by
    /// `depth` since a TT chain can in principle cycle through a
    /// transposition back into a state already on the path.
    fn extract_pv(&self, state: &G::S, first: &G::A, depth: u32) -> Vec<G::A> {
        let mut pv = vec![first.clone()];
        let Some(table) = &self.table else {
            return pv;
        };
        let mut s = G::apply(state.clone(), first);
        for _ in 1..depth {
            if !matches!(G::terminal_status(&s), TerminalStatus::NotTerminal) {
                break;
            }
            let (canonical, sym) = G::canonical_representation(Real(s.clone()));
            let canonical_state = canonical.into_inner();
            let Some(next) = table
                .read()
                .unwrap()
                .lookup(G::zobrist_hash(&canonical_state), &canonical_state)
                .and_then(|e| e.best_action.clone())
                .map(|a| G::invert_action(Canonical(a), sym).into_inner())
            else {
                break;
            };
            pv.push(next.clone());
            s = G::apply(s, &next);
        }
        pv
    }

    /// One full-width pass over `root_moves` at `depth`, searching each
    /// child `depth - 1` plies deep and returning every move's score,
    /// sorted best-first -- the per-iteration body shared by
    /// `choose_action_serial`'s iterative-deepening loop and
    /// [`Self::bounded_negamax`]'s single-shot search. `None` on timeout
    /// (only reachable when `deadline` is `Some`; [`Self::bounded_negamax`]
    /// always passes `None`).
    fn search_root_once(
        &mut self,
        state: &G::S,
        root_moves: Vec<(G::A, Score)>,
        depth: u32,
        deadline: Option<Instant>,
    ) -> Option<Vec<(G::A, Score)>> {
        let mut alpha = LOSS_SCORE - 1;
        let beta = WIN_SCORE + 1;
        let mut iter_scores = Vec::with_capacity(root_moves.len());
        for (a, _) in root_moves {
            let child = G::apply(state.clone(), &a);
            let child_score = self.negamax_search(
                &child,
                Some(&a),
                depth - 1,
                1,
                Window {
                    alpha: -beta,
                    beta: -alpha,
                },
                deadline,
            )?;
            let score = -child_score;
            if score > alpha {
                alpha = score;
            }
            iter_scores.push((a, score));
        }
        iter_scores.sort_by_key(|b| std::cmp::Reverse(b.1));
        Some(iter_scores)
    }

    /// A single, private, throwaway search rooted at an arbitrary state:
    /// no iterative deepening (searches exactly `depth` plies, once), no
    /// time budget, and this `Negamax`'s own `history`/`countermove`/
    /// `table` are reused as-is
    /// rather than reset -- callers that want a genuinely cold search
    /// (e.g. a fresh call per MCTS node, with no state leaking between
    /// unrelated positions) should construct a dedicated `Negamax` with
    /// `NegamaxOptions::table_bits(0)` for it, exactly like any other
    /// `Negamax` instance; this method adds no new state-isolation
    /// mechanism of its own; it is a thin single-depth entry point,
    /// nothing more.
    ///
    /// Returns `(best action, its score from `state`'s mover's
    /// perspective)`. Panics if `state` has no legal moves -- callers
    /// (MCTS's expansion/simulation/backprop hooks) already have an action
    /// list in hand by the time they'd call this and should not call it on
    /// a state they haven't already checked isn't terminal.
    pub fn bounded_negamax(&mut self, state: &G::S, depth: u32) -> (G::A, Score) {
        debug_assert!(
            supports::<G>(),
            "Negamax assumes a deterministic, perfect-information, two-player \
             alternating-move game -- see `negamax::supports` and the module docs"
        );
        assert!(depth >= 1, "bounded_negamax requires depth >= 1");
        let mut root_actions = Vec::new();
        G::generate_actions(state, &mut root_actions);
        assert!(
            !root_actions.is_empty(),
            "bounded_negamax called on a state with no legal moves"
        );
        let root_moves = root_actions.into_iter().map(|a| (a, DRAW_SCORE)).collect();
        self.singular_extension_cap = depth * 2;
        let scores = self
            .search_root_once(state, root_moves, depth, None)
            .expect("bounded_negamax passes deadline: None, so search_root_once can't time out");
        let (action, score) = scores.into_iter().next().expect("checked non-empty above");
        (action, score)
    }

    /// The single-threaded iterative-deepening search loop, run either
    /// directly (`NegamaxOptions::num_threads == 1`) or, once per worker,
    /// by `choose_action_root_split`. `self.start_depth` (always `1`
    /// outside of root splitting) is where iterative deepening begins.
    fn choose_action_serial(&mut self, state: &G::S) -> G::A {
        let mut root_actions = Vec::new();
        G::generate_actions(state, &mut root_actions);
        assert!(
            !root_actions.is_empty(),
            "choose_action called on a state with no legal moves"
        );

        self.nodes_searched = 0;
        self.depth_reached = 0;
        self.root_scores = root_actions
            .iter()
            .cloned()
            .map(|a| (a, DRAW_SCORE))
            .collect();
        self.history.clear();
        self.countermove.clear();

        let deadline = self.options.max_time.map(|d| Instant::now() + d);
        let mut prev_score = DRAW_SCORE;

        let mut depth = self.start_depth;
        while depth <= self.options.max_depth {
            self.singular_extension_cap = depth * 2;
            if depth > 2 {
                if let Some(margin) = self.options.aspiration_window {
                    let _ = self.negamax_search(
                        state,
                        None,
                        depth,
                        0,
                        Window {
                            alpha: prev_score.saturating_sub(margin),
                            beta: prev_score.saturating_add(margin),
                        },
                        deadline,
                    );
                }
            }

            let Some(iter_scores) =
                self.search_root_once(state, self.root_scores.clone(), depth, deadline)
            else {
                break;
            };

            self.root_scores = iter_scores;
            self.depth_reached = depth;
            prev_score = self.root_scores[0].1;

            self.pv = self.extract_pv(state, &self.root_scores[0].0, depth);

            if self.options.verbose {
                eprintln!(
                    "negamax depth={depth} score={prev_score} nodes={} best={:?}",
                    self.nodes_searched, self.root_scores[0].0
                );
            }

            if prev_score.abs() >= WIN_SCORE - depth as Score {
                // A mate has been fully proven within the searched depth --
                // deepening further can't change the answer.
                break;
            }
            depth += 1;
        }

        self.root_scores[0].0.clone()
    }

    /// Lazy-SMP-style root splitting (see the module docs' "Parallel
    /// search"): spawn `num_threads` workers, each a fresh `Negamax`
    /// sharing this instance's evaluator and transposition table but with
    /// its own private history/countermove tables and a staggered
    /// iterative-deepening start depth, then keep the result from whichever
    /// worker completed the greatest depth (ties broken on score).
    fn choose_action_root_split(&mut self, state: &G::S, num_threads: usize) -> G::A {
        /// One worker's result: `(depth_reached, root_score, best_action,
        /// principal_variation, nodes_searched)`.
        type WorkerResult<G> = (u32, Score, <G as Game>::A, Vec<<G as Game>::A>, u64);

        let mut workers: Vec<Self> = (0..num_threads)
            .map(|i| Self {
                eval: Arc::clone(&self.eval),
                options: NegamaxOptions {
                    num_threads: 1,
                    ..self.options.clone()
                },
                table: self.table.clone(),
                start_depth: 1 + (i as u32) % (num_threads as u32),
                pv: Vec::new(),
                root_scores: Vec::new(),
                history: HashMap::new(),
                countermove: HashMap::new(),
                singular_extension_cap: 0,
                nodes_searched: 0,
                depth_reached: 0,
                name: self.name.clone(),
                game_type: PhantomData,
            })
            .collect();

        let results: Vec<WorkerResult<G>> = std::thread::scope(|scope| {
            let handles: Vec<_> = workers
                .iter_mut()
                .map(|worker| {
                    scope.spawn(move || {
                        let action = worker.choose_action_serial(state);
                        (
                            worker.depth_reached,
                            worker.root_score(),
                            action,
                            worker.pv.clone(),
                            worker.nodes_searched,
                        )
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let best = results
            .iter()
            .max_by_key(|(depth, score, ..)| (*depth, *score))
            .expect("num_threads > 1 implies at least one worker result")
            .clone();

        self.depth_reached = best.0;
        self.root_scores = vec![(best.2.clone(), best.1)];
        self.pv = best.3;
        self.nodes_searched = results.iter().map(|(.., n)| n).sum();

        best.2
    }
}

impl<G: Game, E: Evaluator<G> + Default> Default for Negamax<G, E> {
    fn default() -> Self {
        Self::new(E::default())
    }
}

/// Hand-written rather than `#[derive(Clone)]`: a derive would add an
/// `E: Clone` bound even though `eval` is only ever touched through the
/// `Arc` it's already wrapped in, which would needlessly reject every
/// `Evaluator` that doesn't happen to implement `Clone` itself (e.g. one
/// holding a large precomputed table). A clone shares the same
/// transposition table (`Arc<RwLock<_>>`, same as root-splitting workers
/// already share it deliberately) and evaluator, but gets its own
/// independent `history`/`countermove`/`pv`/`root_scores` -- the same
/// per-worker isolation `choose_action_root_split` relies on.
impl<G: Game, E: Evaluator<G>> Clone for Negamax<G, E> {
    fn clone(&self) -> Self {
        Self {
            eval: Arc::clone(&self.eval),
            options: self.options.clone(),
            table: self.table.clone(),
            start_depth: self.start_depth,
            pv: self.pv.clone(),
            root_scores: self.root_scores.clone(),
            history: self.history.clone(),
            countermove: self.countermove.clone(),
            singular_extension_cap: self.singular_extension_cap,
            nodes_searched: self.nodes_searched,
            depth_reached: self.depth_reached,
            name: self.name.clone(),
            game_type: PhantomData,
        }
    }
}

impl<G: Game, E: Evaluator<G>> Search for Negamax<G, E> {
    type G = G;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.into();
    }

    fn estimated_depth(&self) -> usize {
        self.depth_reached as usize
    }

    fn arena_len(&self) -> usize {
        self.table.as_ref().map_or(0, |t| t.read().unwrap().len())
    }

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A {
        debug_assert!(
            supports::<G>(),
            "Negamax assumes a deterministic, perfect-information, two-player \
             alternating-move game -- see `negamax::supports` and the module docs"
        );
        let num_threads = self.options.num_threads.max(1);
        if num_threads > 1 {
            self.choose_action_root_split(state, num_threads)
        } else {
            self.choose_action_serial(state)
        }
    }

    fn principle_variation(&self) -> Vec<<Self::G as Game>::A> {
        self.pv.clone()
    }

    fn root_report(&self, _state: &<Self::G as Game>::S) -> RootReport<<Self::G as Game>::A> {
        let proven = self
            .root_scores
            .first()
            .is_some_and(|(_, s)| s.abs() >= WIN_SCORE - self.depth_reached as Score);
        RootReport {
            actions: self
                .root_scores
                .iter()
                .map(|(action, score)| ActionReport {
                    action: action.clone(),
                    visits: 1,
                    mean_value: (*score as f64 / WIN_SCORE as f64).clamp(-1., 1.),
                    is_proven: proven && *score == self.root_scores[0].1,
                })
                .collect(),
            principal_variation: self.pv.clone(),
            total_visits: self.nodes_searched.min(u32::MAX as u64) as u32,
        }
    }
}

#[cfg(test)]
mod tests;
