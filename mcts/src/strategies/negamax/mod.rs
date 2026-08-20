//! Iterative-deepening negamax (alpha-beta) as a standalone [`Search`]
//! strategy, independent of the MCTS tree/node machinery in
//! `strategies::mcts`.
//!
//! Negamax only makes sense for deterministic, perfect-information,
//! two-player zero-sum games (the whole formulation relies on
//! `value(state) == -value(state_after_opponents_best_reply)`), so this is
//! a separate `Search` implementation rather than another `mcts::Strategy`
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
//! - A depth-preferred transposition table (`table.rs`) that both catches
//!   transposing move orders within one depth and seeds move ordering on
//!   the next, deeper iteration.
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
use std::time::{Duration, Instant};

use crate::game::{Game, PlayerIndex, TerminalStatus};
use crate::strategies::{ActionReport, RootReport, Search};
use table::{Bound, TranspositionTable};

/// Score type returned by [`Evaluator::evaluate`] and by the search itself,
/// always from the perspective of the player about to move in the state
/// being scored (the "nega" in negamax: a child's value is negated to
/// become its parent's).
pub type Score = i32;

/// A proven win for the player to move. Kept well below `i32::MAX` so
/// mate-distance adjustments (`WIN_SCORE - ply`) and aspiration-window
/// arithmetic (`target +/- window`) can't overflow.
pub const WIN_SCORE: Score = 1_000_000;
pub const LOSS_SCORE: Score = -WIN_SCORE;
pub const DRAW_SCORE: Score = 0;

/// Evaluators should stay within this band so a heuristic score can never
/// be confused with a mate-distance-adjusted `WIN_SCORE`/`LOSS_SCORE`
/// (which live in `[WIN_SCORE - max_depth, WIN_SCORE]` and the mirror image
/// below zero, for any `max_depth` this crate would realistically be
/// configured with).
pub const EVAL_MAGNITUDE_LIMIT: Score = 900_000;

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

/// A static evaluation of a non-terminal state, from the perspective of
/// `Game::player_to_move(state)`. Only consulted at the search's depth
/// cutoff (terminal states are always scored from `Game::terminal_status`
/// instead) -- a game small enough to negamax out to a terminal state at
/// whatever depth it's configured with doesn't need one at all (see
/// [`MaterialBlind`] below).
///
/// This is intentionally not part of `Game` itself: most games plugged
/// into this crate have no static evaluator (they're built for MCTS, whose
/// rollouts don't need one), and folding an `evaluate` method into `Game`
/// would force every one of them to grow a stub. Implement this trait only
/// for the games (and only in the crates) that actually want negamax.
pub trait Evaluator<G: Game>: Sync + Send {
    fn evaluate(&self, state: &G::S) -> Score;
}

/// An [`Evaluator`] that always returns a draw score, for a game whose
/// state space is small enough that `NegamaxOptions::max_depth` can just
/// be set past its longest possible game -- the depth cutoff then never
/// actually fires, so what it returns doesn't matter. Also useful as a
/// placeholder while a real evaluator is still being written.
#[derive(Clone, Copy, Default)]
pub struct MaterialBlind;

impl<G: Game> Evaluator<G> for MaterialBlind {
    fn evaluate(&self, _state: &G::S) -> Score {
        DRAW_SCORE
    }
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
    pub verbose: bool,
}

impl Default for NegamaxOptions {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_time: None,
            table_bits: 20,
            aspiration_window: None,
            principal_variation_search: true,
            history_heuristic: true,
            singular_extension: true,
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

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// Iterative-deepening negamax. See the module docs for the feature list.
pub struct Negamax<G: Game, E: Evaluator<G>> {
    eval: E,
    options: NegamaxOptions,
    table: Option<TranspositionTable<G>>,
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
        let table = (options.table_bits > 0).then(|| TranspositionTable::new(options.table_bits));
        Self {
            eval,
            options,
            table,
            pv: Vec::new(),
            root_scores: Vec::new(),
            history: HashMap::new(),
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
    /// module docs' "mate-distance scoring").
    fn negamax_search(
        &mut self,
        state: &G::S,
        depth: u32,
        ply: u32,
        mut alpha: Score,
        beta: Score,
        deadline: Option<Instant>,
    ) -> Option<Score> {
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

        let hash = G::zobrist_hash(state);
        let alpha_orig = alpha;
        let mut beta = beta;
        let mut tt_move = None;
        if let Some(table) = &self.table {
            if let Some(entry) = table.lookup(hash, state) {
                tt_move = entry.best_action.clone();
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
        if self.options.history_heuristic {
            actions[ordered_start..]
                .sort_by_key(|a| std::cmp::Reverse(self.history.get(a).copied().unwrap_or(0)));
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
                    child_depth,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    deadline,
                )?;
                if probe > alpha && probe < beta {
                    -self.negamax_search(&child, child_depth, ply + 1, -beta, -probe, deadline)?
                } else {
                    probe
                }
            } else {
                -self.negamax_search(&child, child_depth, ply + 1, -beta, -alpha, deadline)?
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
                break;
            }
        }

        if let Some(table) = &mut self.table {
            let bound = if best_score <= alpha_orig {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            table.store(hash, state, depth, best_score, bound, Some(best_action));
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
            let Some(next) = table
                .lookup(G::zobrist_hash(&s), &s)
                .and_then(|e| e.best_action.clone())
            else {
                break;
            };
            pv.push(next.clone());
            s = G::apply(s, &next);
        }
        pv
    }
}

impl<G: Game, E: Evaluator<G> + Default> Default for Negamax<G, E> {
    fn default() -> Self {
        Self::new(E::default())
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
        self.table.as_ref().map_or(0, |t| t.len())
    }

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A {
        debug_assert!(
            supports::<G>(),
            "Negamax assumes a deterministic, perfect-information, two-player \
             alternating-move game -- see `negamax::supports` and the module docs"
        );

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

        let deadline = self.options.max_time.map(|d| Instant::now() + d);
        let mut prev_score = DRAW_SCORE;

        let mut depth = 1u32;
        while depth <= self.options.max_depth {
            self.singular_extension_cap = depth * 2;
            if depth > 2 {
                if let Some(window) = self.options.aspiration_window {
                    let _ = self.negamax_search(
                        state,
                        depth,
                        0,
                        prev_score.saturating_sub(window),
                        prev_score.saturating_add(window),
                        deadline,
                    );
                }
            }

            let mut alpha = LOSS_SCORE - 1;
            let beta = WIN_SCORE + 1;
            let mut iter_scores = Vec::with_capacity(self.root_scores.len());
            let mut complete = true;
            for (a, _) in self.root_scores.clone() {
                let child = G::apply(state.clone(), &a);
                let Some(child_score) =
                    self.negamax_search(&child, depth - 1, 1, -beta, -alpha, deadline)
                else {
                    complete = false;
                    break;
                };
                let score = -child_score;
                if score > alpha {
                    alpha = score;
                }
                iter_scores.push((a, score));
            }

            if !complete {
                break;
            }

            iter_scores.sort_by_key(|b| std::cmp::Reverse(b.1));
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
